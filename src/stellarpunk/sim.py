import sys
import gc
import logging
import contextlib
import time
import math
import curses
import warnings
from typing import List, Optional, Mapping, Any, Tuple, Deque, TextIO, Set
import collections
import heapq

import numpy as np
import cymunk # type: ignore

from stellarpunk import util, core, interface, generate, orders, econ_sim, agenda
from stellarpunk.interface import universe as universe_interface

TICKS_PER_HIST_SAMPLE = 0#10
ECONOMY_LOG_PERIOD_SEC = 2.0
ZERO_ONE = (0,1)

class Simulator:
    def __init__(self, gamestate:core.Gamestate, ui:interface.AbstractInterface, max_dt:Optional[float]=None, economy_log:Optional[TextIO]=None, ticks_per_hist_sample:int=TICKS_PER_HIST_SAMPLE) -> None:
        self.logger = logging.getLogger(util.fullname(self))
        self.gamestate = gamestate
        self.ui = ui

        self.pause_on_collision = False
        self.enable_collisions = False

        # time between ticks, this is the framerate
        self.desired_dt = gamestate.desired_dt
        if not max_dt:
            max_dt = self.desired_dt
        self.max_dt = max_dt
        self.dt_scaleup = 1.5
        self.dt_scaledown = 0.9
        # dt is "best effort" constant but varies between desired_dt and max_dt
        self.dt = self.desired_dt

        # number of ticks we're currently behind
        self.behind_ticks = 0.
        # number of ticks we've been behind by >= 1 tick
        self.behind_length = 0
        # number of ticks we need to be behind to trigger dt scaling
        # note that dt scaling is somewhat disruptive since some things might
        # expect us to have a constant dt.
        self.behind_dt_scale_thresthold = 30
        # a throttle for warning logging when we get behind
        self.behind_message_throttle = 0.

        self.ticktime_alpha = 0.1

        self.sleep_count = 0

        # sample rate for taking SectorEntity history entries
        # we offest by sector so all entities in a sector get history taken on
        # the same tick, but each sector is sampled on different ticks to
        # ammortize the cost of recording history.
        self.ticks_per_hist_sample = ticks_per_hist_sample

        self.next_economy_sample = 0.
        self.economy_log = economy_log

        # a little book-keeping for storing collisions during step callbacks
        # this is not for external consumption
        self._collisions:List[Tuple[core.SectorEntity, core.SectorEntity, Tuple[float, float], float]] = []
        self._colliders:Set[str] = set()
        self._last_colliders:Set[str] = set()

    def _ship_collision_detected(self, arbiter:cymunk.Arbiter) -> bool:#, space:pymunk.Space, data:Mapping[str, Any]) -> bool:
        # which ship(s) are colliding?

        (shape_a, shape_b) = arbiter.shapes

        # ignore collisions between the same bodies on consecutive ticks
        colliders = "".join(sorted(map(str, [shape_a.body.data.entity_id, shape_b.body.data.entity_id])))
        self._colliders.add(colliders)
        if colliders in self._last_colliders:
            return self.enable_collisions

        sector = shape_a.body.data.sector

        tons_of_tnt = arbiter.total_ke / 4.184e9
        self.logger.debug(f'collision detected in {sector.short_id()}, between {shape_a.body.data.address_str()} {shape_b.body.data.address_str()} with {arbiter.total_impulse}N and {arbiter.total_ke}j ({tons_of_tnt} tons of tnt)')

        self._collisions.append((
            shape_a.body.data,
            shape_b.body.data,
            arbiter.total_impulse,
            arbiter.total_ke,
        ))

        # return if the collision should happen
        return self.enable_collisions

    def initialize(self) -> None:
        """ One-time initialize of the simulation. """
        for sector in self.gamestate.sectors.values():
            sector.space.set_default_collision_handler(pre_solve = self._ship_collision_detected)

    def tick(self, dt: float) -> None:
        """ Do stuff to update the universe """

        # update physics simulations
        # do this for all sectors
        for sector in self.gamestate.sectors.values():
            sector.space.step(dt)

            # keep track of this tick collisions (if any) so we can ignore
            # collisions that last over several consecutive ticks
            self._last_colliders = self._colliders
            self._colliders = set()

            if self._collisions:
                self.gamestate.counters[core.Counters.COLLISIONS] += len(self._collisions)
                #TODO: use kinetic energy from the collision to cause damage
                # metals have an impact strength between 0.34e3 and 145e3
                # joules / meter^2
                # so an impact of 17M joules spread over an area of 1000 m^2
                # would be a lot, but not catastrophic
                # spread over 100m^2 would be
                # for comparison, a typical briefcase bomb is comparable to
                # 50 pounds of TNT, which is nearly 100M joules
                self.gamestate.paused = self.pause_on_collision
                for entity_a, entity_b, impulse, ke in self._collisions:
                    self.ui.collision_detected(entity_a, entity_b, impulse, ke)

                # keep _collisions clear for next time
                self._collisions.clear()

        # at this point all physics sim is done for the tick and the gamestate
        # is up to date across the universe

        # do AI stuff, e.g. let ships take action (but don't actually have the
        # actions take effect yet!)

        orders_processed = 0
        for order in self.gamestate.pop_current_orders():
            if order == order.ship.current_order():
                if order.is_complete():
                    ship = order.ship
                    self.logger.debug(f'ship {ship.entity_id} completed {order} in {self.gamestate.timestamp - order.started_at:.2f} est {order.init_eta:.2f}')
                    ship.complete_current_order()

                    next_order = ship.current_order()
                    if not next_order:
                        ship.prepend_order(ship.default_order(self.gamestate))
                    elif not self.gamestate.is_order_scheduled(next_order):
                        #TODO: this is kind of janky, can't we just demand that orders schedule themselves?
                        # what about the order queue being simply a queue?
                        self.gamestate.schedule_order_immediate(next_order)

                    #TODO: seems like we don't want this any more (why does the UI need
                    #to know when every single order is complete? I think this was a
                    #testing hook. but that's not probably the right way to do this
                    self.ui.order_complete(order)
                else:
                    order.act(dt)
            else:
                # else order isn't the front item, so we'll ignore this action
                self.logger.warning(f'got non-front order scheduled action: {order=} vs {order.ship.current_order()=}')
                self.gamestate.counters[core.Counters.NON_FRONT_ORDER_ACTION] += 1
            orders_processed += 1

        self.gamestate.counters[core.Counters.ORDERS_PROCESSED] += orders_processed

        # process effects
        for effect in self.gamestate.pop_current_effects():
            effect.act(dt)

        # let characters act on their (scheduled) agenda items
        for agendum in self.gamestate.pop_current_agenda():
            agendum.act()

        self.gamestate.ticks += 1
        self.gamestate.timestamp += dt

        # record some state about the final state of this tick
        if self.ticks_per_hist_sample > 0:
            for sector in self.gamestate.sectors.values():
                if self.gamestate.ticks % self.ticks_per_hist_sample == sector.entity_id.int % self.ticks_per_hist_sample:
                    for ship in sector.ships:
                        ship.history.append(ship.to_history(self.gamestate.timestamp))

        if self.economy_log is not None and self.gamestate.timestamp > self.next_economy_sample:
            #TODO: revisit economic logging
            #for i, amount in enumerate(self.gamestate.production_chain.resources_mined):
            #    self.economy_log.write(f'{self.gamestate.timestamp}\tMINE\t{i}\t{amount}\n')
            #for i, amount in enumerate(self.gamestate.production_chain.goods_produced):
            #    self.economy_log.write(f'{self.gamestate.timestamp}\tPRODUCE\t{i}\t{amount}\n')

            total_ships = 0
            total_goto_orders = 0
            total_orders_with_ct = 0
            total_orders_with_cac = 0
            total_speed = 0.
            total_neighbors = 0.
            for sector in self.gamestate.sectors.values():
                for ship in sector.ships:
                    total_ships += 1
                    total_speed += ship.phys.speed
                    if len(ship._orders) > 0:
                        order = ship._orders[0]
                        if isinstance(order, orders.movement.GoToLocation):
                            total_goto_orders += 1
                            total_neighbors += order.num_neighbors
                            if order.threat_count:
                                total_orders_with_ct += 1
                            if order.cannot_avoid_collision_hold:
                                total_orders_with_cac += 1

            total_agenda = 0
            total_mining_agenda = 0
            total_idle_mining_agenda = 0
            total_trading_agenda = 0
            total_idle_trading_agenda = 0
            total_snob_trading_agenda = 0
            total_snos_trading_agenda = 0
            for character in self.gamestate.characters.values():
                for agendum in character.agenda:
                    total_agenda += 1
                    if isinstance(agendum, agenda.MiningAgendum):
                        total_mining_agenda += 1
                        if agendum.state == agenda.MiningAgendum.State.IDLE:
                            total_idle_mining_agenda += 1
                    elif isinstance(agendum, agenda.TradingAgendum):
                        total_trading_agenda += 1
                        if agendum.state == agenda.TradingAgendum.State.IDLE:
                            total_idle_trading_agenda += 1
                        elif agendum.state == agenda.TradingAgendum.State.SLEEP_NO_BUYS:
                            total_snob_trading_agenda += 1
                        elif agendum.state == agenda.TradingAgendum.State.SLEEP_NO_SALES:
                            total_snos_trading_agenda += 1

            self.logger.info(f'ships: {total_ships} goto orders: {total_goto_orders} ct: {total_orders_with_ct} cac: {total_orders_with_cac} mean_speed: {total_speed/total_ships:.2f} mean_neighbors: {total_neighbors/total_goto_orders if total_goto_orders > 0 else 0.:.2f}')
            self.logger.info(f'agenda: {total_agenda} mining agenda: {total_mining_agenda} idle: {total_idle_mining_agenda} trading agenda: {total_trading_agenda} idle: {total_idle_trading_agenda} snob: {total_snob_trading_agenda} snos: {total_snos_trading_agenda}')
            self.gamestate.log_econ()

            self.next_economy_sample = self.gamestate.timestamp + ECONOMY_LOG_PERIOD_SEC

    def _handle_behind_ticks(self, last_time_delta:float) -> None:
        # what to do if we miss a tick (or a lot)
        # seems like we should run a tick with a longer dt to make up for
        # it, and stop rendering until we catch up
        # but why would we miss ticks?
        if last_time_delta > self.dt:
            self.gamestate.counters[core.Counters.BEHIND_TICKS] += 1
            #self.logger.debug(f'ticks: {self.gamestate.ticks} sleep_count: {self.sleep_count} gc stats: {gc.get_stats()}')
            self.gamestate.missed_ticks += 1
            behind = (last_time_delta)/self.dt
            if self.behind_length > self.behind_dt_scale_thresthold and behind >= self.behind_ticks:
                if not self.ui.decrease_fps():
                    self.dt = min(self.max_dt, self.dt * self.dt_scaleup)
            self.behind_ticks = behind
            self.behind_length += 1
            self.behind_message_throttle = util.throttled_log(self.gamestate.timestamp, self.behind_message_throttle, self.logger, logging.WARNING, f'behind by {last_time_delta:.4f}s {behind:.2f} ticks dt: {self.dt:.4f} for {self.behind_length} ticks', 3.)
        else:
            if self.behind_ticks > 0:
                self.ui.increase_fps()
                self.logger.debug(f'ticks caught up with realtime ticks dt: {self.dt:.4f} for {self.behind_length} ticks')

            self.behind_ticks = 0
            self.behind_length = 0

    def run(self) -> None:

        next_tick = time.perf_counter()+self.dt

        while self.gamestate.keep_running:
            if self.gamestate.should_raise:
                raise Exception()
            now = time.perf_counter()

            if next_tick - now > self.gamestate.min_tick_sleep:
                if self.dt > self.desired_dt:
                    self.dt = max(self.desired_dt, self.dt * self.dt_scaledown)
                    self.logger.debug(f'dt: {self.dt}')
                time.sleep(next_tick - now)
                self.sleep_count += 1

            self._handle_behind_ticks(now - next_tick)

            starttime = time.perf_counter()
            if not self.gamestate.paused:
                self.tick(self.dt)

            now = time.perf_counter()

            last_tick = next_tick
            if self.gamestate.fast_mode:
                next_tick = now
            else:
                next_tick = next_tick + self.dt / self.gamestate.time_accel_rate

            timeout = next_tick - now
            if not self.gamestate.paused:
                self.gamestate.timeout = self.ticktime_alpha * timeout + (1-self.ticktime_alpha) * self.gamestate.timeout

            self.ui.tick(timeout, self.dt)

            now = time.perf_counter()
            ticktime = now - starttime
            self.gamestate.ticktime = self.ticktime_alpha * ticktime + (1-self.ticktime_alpha) * self.gamestate.ticktime

def main() -> None:
    with contextlib.ExitStack() as context_stack:
        # for reference, config to stderr:
        # logging.basicConfig(stream=sys.stderr, level=logging.INFO)
        logging.basicConfig(
                format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
                filename="/tmp/stellarpunk.log",
                filemode="w",
                level=logging.INFO
        )
        logging.getLogger("numba").level = logging.INFO
        logging.getLogger("stellarpunk").level = logging.DEBUG
        # send warnings to the logger
        logging.captureWarnings(True)
        # turn warnings into exceptions
        warnings.filterwarnings("error")

        mgr = context_stack.enter_context(util.PDBManager())
        gamestate = core.Gamestate()

        data_logger = context_stack.enter_context(econ_sim.EconomyDataLogger(enabled=True, line_buffering=True, gamestate=gamestate))
        gamestate.econ_logger = data_logger

        logging.info("generating universe...")
        generator = generate.UniverseGenerator(gamestate)
        stellar_punk = generator.generate_universe()

        ui = context_stack.enter_context(interface.Interface(gamestate, generator))

        ui.initialize()
        uv = universe_interface.UniverseView(gamestate, ui)
        ui.open_view(uv)

        economy_log = context_stack.enter_context(open("/tmp/economy.log", "wt", 1))

        #logging.info("running simulation...")
        #stellar_punk.run()

        stellar_punk.production_chain.viz().render("/tmp/production_chain", format="pdf")

        sim = Simulator(gamestate, ui, max_dt=1/5, economy_log=economy_log)
        sim.initialize()

        # experimentally chosen so that we don't get multiple gcs during a tick
        # this helps a lot because there's lots of short lived objects during a
        # tick and it's better if they stay in the youngest generation
        #TODO: should we just disable a gc while we're doing a tick?
        gc.set_threshold(700*4, 10*4, 10*4)
        data_logger.begin_simulation()
        sim.run()


        counter_str = "\n".join(map(lambda x: f'{str(x[0])}:\t{x[1]}', zip(list(core.Counters), gamestate.counters)))
        logging.info(f'counters:\n{counter_str}')

        logging.info(f'ticks:\t{gamestate.ticks}')

        logging.info("done.")

if __name__ == "__main__":
    main()
