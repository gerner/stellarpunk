import sys
import gc
import logging
import contextlib
import time
import math
import curses
import warnings
from typing import List, Optional, Mapping, Any, Tuple, Deque
import collections

import numpy as np
import pymunk

from stellarpunk import util, core, interface, generate, orders
from stellarpunk.interface import universe as universe_interface

TICKS_PER_HIST_SAMPLE = 10
ZERO_ONE = (0,1)

class Simulator:
    def __init__(self, gamestate:core.Gamestate, ui:interface.AbstractInterface, dt:float=1/60, max_dt:Optional[float]=None) -> None:
        self.logger = logging.getLogger(util.fullname(self))
        self.gamestate = gamestate
        self.ui = ui

        # time between ticks, this is the framerate
        self.desired_dt = dt
        if not max_dt:
            max_dt = dt
        self.max_dt = max_dt
        self.dt_scaleup = 1.5
        self.dt_scaledown = 0.9
        # dt is "best effort" constant but varies between desired_dt and max_dt
        self.dt = dt

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
        self.min_tick_sleep = self.desired_dt/5

        self.sleep_count = 0

        self.collisions:List[tuple[core.SectorEntity, core.SectorEntity, Tuple[float, float], float]] = []

    def _ship_collision_detected(self, arbiter:pymunk.Arbiter, space:pymunk.Space, data:Mapping[str, Any]) -> None:
        # which ship(s) are colliding?

        (shape_a, shape_b) = arbiter.shapes
        sector = data["sector"]

        self.logger.debug(f'collision detected in {sector.short_id()}, between {shape_a.body.entity.address_str()} {shape_b.body.entity.address_str()} with {arbiter.total_impulse}N and {arbiter.total_ke}j')

        self.collisions.append((
            shape_a.body.entity,
            shape_b.body.entity,
            arbiter.total_impulse,
            arbiter.total_ke,
        ))

    def initialize(self) -> None:
        """ One-time initialize of the simulation. """
        for sector in self.gamestate.sectors.values():
            h = sector.space.add_default_collision_handler()
            h.data["sector"] = sector
            h.post_solve = self._ship_collision_detected

    def tick_order(self, ship: core.Ship, dt: float) -> None:
        if not ship.orders:
            ship.orders.append(ship.default_order(self.gamestate))

        order = ship.orders[0]
        if order.started_at < 0:
            order.begin_order()

        if order.is_complete():
            self.logger.debug(f'{ship.entity_id} completed {order} in {self.gamestate.timestamp - order.started_at:.2f} est {order.init_eta:.2f}')
            order.complete_order()
            ship.orders.popleft()
            self.ui.order_complete(order)
        else:
            order.act(dt)

    def produce_at_station(self, station:core.Station) -> None:
        # waiting for production to finish case
        if station.next_batch_time > 0:
            # check if the batch is ready
            if station.next_batch_time <= self.gamestate.timestamp:
                # add the batch to cargo
                station.cargo[station.resource] += self.gamestate.production_chain.batch_sizes[station.resource]
                station.next_batch_time = 0.
                station.next_production_time = 0.
        # waiting for enough cargo to produce case
        elif station.next_production_time <= self.gamestate.timestamp:
            # check if we have enough resource to start a batch
            resources_needed = self.gamestate.production_chain.adj_matrix[:,station.resource] * self.gamestate.production_chain.batch_sizes[station.resource]
            if np.all(station.cargo >= resources_needed):
                station.cargo -= resources_needed
                station.next_batch_time = self.gamestate.timestamp + self.gamestate.production_chain.production_times[station.resource]
            else:
                # wait a cooling off period to avoid needlesss expensive checks
                station.next_production_time = self.gamestate.timestamp + self.gamestate.production_chain.production_coolingoff_time

    def tick_sector(self, sector:core.Sector, dt:float) -> None:

        # do effects
        effects_complete:Deque[core.Effect] = collections.deque()
        for effect in sector.effects:
            if effect.started_at < 0:
                effect.begin_effect()

            if effect.is_complete():
                # defer completion/removal until other effects have a chance to act
                effects_complete.append(effect)
            else:
                effect.act(dt)

        # notify effect completion
        for effect in effects_complete:
            self.logger.debug(f'effect {effect} in {sector.short_id()} complete in {self.gamestate.timestamp - effect.started_at:.2f}')
            effect.complete_effect()
            sector.effects.remove(effect)
            self.ui.effect_complete(effect)

        # produce goods
        # every batch_time seconds we produce one batch of resource from inputs
        # reset production timer
        # if production timer is off, start producting a batch if we have it,
        # setting production timer
        for station in sector.stations:
            self.produce_at_station(station)

    def tick(self, dt: float) -> None:
        """ Do stuff to update the universe """

        # update physics simulations
        # do this for all sectors
        for sector in self.gamestate.sectors.values():
            self.collisions.clear()

            sector.space.step(dt)

            if self.collisions:
                #TODO: use kinetic energy from the collision to cause damage
                # metals have an impact strength between 0.34e3 and 145e3
                # joules / meter^2
                # so an impact of 17M joules spread over an area of 1000 m^2
                # would be a lot, but not catastrophic
                # spread over 100m^2 would be
                self.gamestate.paused = True
                for entity_a, entity_b, impulse, ke in self.collisions:
                    self.ui.collision_detected(entity_a, entity_b, impulse, ke)

            for ship in sector.ships:
                ship.pre_tick()

        # at this point all physics sim is done for the tick and the gamestate
        # is up to date across the universe

        # do AI stuff, e.g. let ships take action (but don't actually have the
        # actions take effect yet!)
        for sector in self.gamestate.sectors.values():
            for ship in sector.ships:
                self.tick_order(ship, dt)
                if self.gamestate.ticks % TICKS_PER_HIST_SAMPLE == ship.entity_id.int % TICKS_PER_HIST_SAMPLE:
                    ship.history.append(ship.to_history(self.gamestate.timestamp))

        # at this point all AI decisions have happened everywhere

        # update sector state after all ships across universe take action
        for sector in self.gamestate.sectors.values():
            for ship in sector.ships:
                ship.post_tick()
            self.tick_sector(sector, dt)

        self.gamestate.ticks += 1
        self.gamestate.timestamp += dt

    def run(self) -> None:

        next_tick = time.perf_counter()+self.dt

        while self.gamestate.keep_running:
            if self.gamestate.should_raise:
                raise Exception()
            now = time.perf_counter()

            if next_tick - now > self.min_tick_sleep:
                if self.dt > self.desired_dt:
                    self.dt = max(self.desired_dt, self.dt * self.dt_scaledown)
                    self.logger.debug(f'dt: {self.dt}')
                time.sleep(next_tick - now)
                self.sleep_count += 1
            #TODO: what to do if we miss a tick (or a lot)
            # seems like we should run a tick with a longer dt to make up for
            # it, and stop rendering until we catch up
            # but why would we miss ticks?
            if now - next_tick > self.dt:
                #self.logger.debug(f'ticks: {self.gamestate.ticks} sleep_count: {self.sleep_count} gc stats: {gc.get_stats()}')
                self.gamestate.missed_ticks += 1
                behind = (now - next_tick)/self.dt
                if self.behind_length > self.behind_dt_scale_thresthold and behind >= self.behind_ticks:
                    self.dt = min(self.max_dt, self.dt * self.dt_scaleup)
                self.behind_ticks = behind
                self.behind_length += 1
                self.behind_message_throttle = util.throttled_log(self.gamestate.timestamp, self.behind_message_throttle, self.logger, logging.WARNING, f'behind by {now - next_tick:.4f}s {behind:.2f} ticks dt: {self.dt:.4f} for {self.behind_length} ticks', 3.)
            else:
                if self.behind_ticks > 0:
                    self.logger.debug(f'ticks caught up with realtime, behind by {now - next_tick:.4f}s {behind:.2f} ticks dt: {self.dt:.4f} for {self.behind_length} ticks')

                self.behind_ticks = 0
                self.behind_length = 0

            starttime = time.perf_counter()
            if not self.gamestate.paused:
                self.tick(self.dt)

            last_tick = next_tick
            next_tick = next_tick + self.dt / self.gamestate.time_accel_rate

            now = time.perf_counter()

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

        logging.info("generating universe...")
        generator = generate.UniverseGenerator(gamestate)
        stellar_punk = generator.generate_universe()

        ui = context_stack.enter_context(interface.Interface(gamestate, generator))

        ui.initialize()
        uv = universe_interface.UniverseView(gamestate, ui)
        ui.open_view(uv)

        #logging.info("running simulation...")
        #stellar_punk.run()

        stellar_punk.production_chain.viz().render("production_chain", format="pdf")

        dt = 1/60
        sim = Simulator(gamestate, ui, dt=dt, max_dt=1/5)
        sim.initialize()

        # experimentally chosen so that we don't get multiple gcs during a tick
        # this helps a lot because there's lots of short lived objects during a
        # tick and it's better if they stay in the youngest generation
        #TODO: should we just disable a gc while we're doing a tick?
        gc.set_threshold(700*4, 10*4, 10*4)
        sim.run()

        logging.info("done.")

if __name__ == "__main__":
    main()
