import sys
import gc
import logging
import contextlib
import time
import math
import curses
import warnings
import collections
import heapq
from typing import Iterable, List, Optional, Mapping, MutableMapping, Any, Tuple, Deque, TextIO, Set

import numpy as np
import cymunk # type: ignore

from stellarpunk import util, core, interface, generate, orders, econ, econ_sim, agenda, events, narrative, config, effects, sensors, intel
from stellarpunk.core import combat, sector_entity
from stellarpunk.agenda import intel as aintel
from stellarpunk.interface import ui_util, manager as interface_manager
from stellarpunk.serialization import (
    save_game, econ as s_econ,
    gamestate as s_gamestate,
    events as s_events,
    sector as s_sector,
    character as s_character,
    sector_entity as s_sector_entity,
    order as s_order,
    sensors as s_sensors,
    agenda as s_agenda,
    combat as s_combat,
    order_core as s_order_core,
    movement as s_movement,
    effect as s_effect,
    intel as s_intel,
)

TICKS_PER_HIST_SAMPLE = 0#10
ECONOMY_LOG_PERIOD_SEC = 30.0
ZERO_ONE = (0,1)

class Simulator(generate.UniverseGeneratorObserver, core.AbstractGameRuntime):
    def __init__(self, generator:generate.UniverseGenerator, ui:interface.AbstractInterface, *args:Any, max_dt:Optional[float]=None, economy_log:Optional[TextIO]=None, ticks_per_hist_sample:int=TICKS_PER_HIST_SAMPLE, game_saver:Optional[save_game.GameSaver]=None, context_stack:Optional[contextlib.ExitStack]=None, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(util.fullname(self))
        self.context_stack=context_stack
        self.generator = generator
        # we create a dummy gamestate immediately, but we get the real one by
        # watching for UniverseGenerator events
        self.gamestate:core.Gamestate = None # type: ignore
        self.ui = ui

        self.pause_on_collision = False
        self.notify_on_collision = False
        self.enable_collisions = True

        self.startup_running = True
        self.keep_running = False
        self.should_raise= False
        self.should_raise_breakpoint = False

        # time between ticks, this is the framerate
        self.desired_dt = 1/60
        self.dt = self.desired_dt
        self.min_tick_sleep = self.desired_dt/5
        if not max_dt:
            max_dt = self.desired_dt
        self.max_dt = max_dt
        self.dt_scaleup = 1.5
        self.dt_scaledown = 0.9
        # dt is "best effort" constant but varies between desired_dt and max_dt

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

        self.ticktime = 0.
        self.ticktime_alpha = 0.1
        self.timeout = 0.
        self.missed_ticks = 0

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
        self._collisions:list[tuple[core.SectorEntity, core.SectorEntity, tuple[float, float], float]] = []
        self._colliders:set[str] = set()

        # some settings related to time acceleration
        # how many seconds of simulation (as in dt) should elapse per second
        self.time_accel_rate = 1.0
        self.fast_mode = False
        self.reference_realtime = 0.
        self.reference_gametime = 0.

        self.game_saver = game_saver

    def _ship_collision_detected(self, arbiter:cymunk.Arbiter) -> bool:
        return self.enable_collisions

    def _ship_collision_handler(self, arbiter:cymunk.Arbiter) -> None:
        # which ship(s) are colliding?
        (shape_a, shape_b) = arbiter.shapes

        # keep track of collisions in consecutive ticks so we can ignore them
        # later, but still process them for physics purposes
        colliders = "".join(sorted(map(str, [shape_a.body.data.entity_id, shape_b.body.data.entity_id])))
        self._colliders.add(colliders)
        if colliders in self.gamestate.last_colliders:
            return

        sector = shape_a.body.data.sector

        tons_of_tnt = arbiter.total_ke / 4.184e9
        self.logger.info(f'collision detected in {sector.short_id()}, between {shape_a.body.data.address_str()} {shape_b.body.data.address_str()} with {arbiter.total_impulse}N and {arbiter.total_ke}j ({tons_of_tnt} tons of tnt)')

        self._collisions.append((
            shape_a.body.data,
            shape_b.body.data,
            arbiter.total_impulse,
            arbiter.total_ke,
        ))

    # core.AbstractGameRuntime
    def get_missed_ticks(self) -> int:
        return self.missed_ticks

    def get_ticktime(self) -> float:
        return self.ticktime

    def get_time_acceleration(self) -> Tuple[float, bool]:
        return self.time_accel_rate, self.fast_mode

    def time_acceleration(self, accel_rate:float, fast_mode:bool) -> None:
        real_span, game_span, rel_drift, expected_rel_drift = self.compute_timedrift()
        self.logger.debug(f'timedrift: {real_span} vs {game_span} {rel_drift:.3f} vs {expected_rel_drift:.3f}')
        self.reference_realtime = time.perf_counter()
        self.reference_gametime = self.gamestate.timestamp
        self.time_accel_rate = accel_rate
        self.fast_mode = fast_mode

    def exit_startup(self) -> None:
        self.startup_running = False

    def start_game(self) -> None:
        assert(self.ui.gamestate)
        self.keep_running = True

    def game_running(self) -> bool:
        return self.keep_running

    def quit(self) -> None:
        self.startup_running = False
        self.keep_running = False

    def get_desired_dt(self) -> float:
        return self.desired_dt

    def get_dt(self) -> float:
        return self.dt

    def raise_exception(self) -> None:
        self.should_raise = True

    def raise_breakpoint(self) -> None:
        self.should_raise_breakpoint = True

    def should_breakpoint(self) -> bool:
        return self.should_raise_breakpoint

    def initialize_gamestate(self, gamestate:core.Gamestate) -> None:
        """ post-gamestate generation/loading initialization.

        initialize_gamestate is a pattern. it can be called many times with
        different gamestates. we can assume we'll only be operating on one
        gamestate at a time and if we get a new initialize_gamestate call, we
        should toss the old gamestate."""

        self.dt = self.desired_dt
        self.next_economy_sample = 0.
        self._collisions = []
        self._colliders = set()
        self.time_accel_rate = 1.0
        self.fast_mode = False


        self.gamestate = gamestate
        self.gamestate.game_runtime = self

        self.reference_realtime = time.perf_counter()
        self.reference_gametime = self.gamestate.timestamp

        # TODO: this context should only exist for as long as this gamestate
        # does. how do we properly close related resources?
        # similarly for economy_log (and why do we need that if we have this?)
        if self.context_stack is not None:
            data_logger = self.context_stack.enter_context(econ_sim.EconomyDataLogger(enabled=True, line_buffering=True, gamestate=self.gamestate))
            self.gamestate.econ_logger = data_logger
            data_logger.begin_simulation()

        # can only happen after the universe is initialized
        combat.initialize_gamestate(self.gamestate)
        #TODO: intialize_gamestate for other modules dynamically added

        for sector in self.gamestate.sectors.values():
            sector.space.set_default_collision_handler(pre_solve = self._ship_collision_detected, post_solve = self._ship_collision_handler)


    # generate.UniverseGeneratorObserver
    def universe_generated(self, gamestate:core.Gamestate) -> None:
        self.initialize_gamestate(gamestate)

    def universe_loaded(self, gamestate:core.Gamestate) -> None:
        self.initialize_gamestate(gamestate)

    def pre_initialize(self) -> None:
        self.generator.observe(self)

    def _tick_space(self, dt: float) -> None:
        # update physics simulations
        # do this for all sectors
        for sector in self.gamestate.sectors.values():
            sector.space.step(dt)

    def _tick_collisions(self, dt:float) -> None:
        # keep track of this tick collisions (if any) so we can ignore
        # collisions that last over several consecutive ticks
        self.gamestate.last_colliders = self._colliders
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
            if self.notify_on_collision:
                for entity_a, entity_b, impulse, ke in self._collisions:
                    self.ui.collision_detected(entity_a, entity_b, impulse, ke)

            for entity_a, entity_b, impulse, ke in self._collisions:
                if entity_a.sector is None or entity_a.sector != entity_b.sector:
                    raise Exception(f'collision between entities in different or null sectors {entity_a.sector} != {entity_b.sector}')

                sector = entity_a.sector
                if entity_a.entity_id in sector.collision_observers:
                    for observer in sector.collision_observers[entity_a.entity_id].copy():
                        observer.collision(entity_a, entity_b, impulse, ke)
                if entity_b.entity_id in sector.collision_observers:
                    for observer in sector.collision_observers[entity_b.entity_id].copy():
                        observer.collision(entity_b, entity_a, impulse, ke)

            # keep _collisions clear for next time
            self._collisions.clear()

    def _tick_orders(self, dt: float) -> None:
        orders_processed = 0
        for order in self.gamestate.pop_current_orders():
            order.base_act(dt)
            orders_processed += 1
        self.gamestate.counters[core.Counters.ORDERS_PROCESSED] += orders_processed

    def _tick_effects(self, dt: float) -> None:
        # process effects
        for effect in self.gamestate.pop_current_effects():
            effect.act(dt)

    def _tick_agenda(self, dt: float) -> None:
        # let characters act on their (scheduled) agenda items
        for agendum in self.gamestate.pop_current_agenda():
            agendum.act()

    def _tick_tasks(self, dt:float) -> None:
        for task in self.gamestate.pop_current_task():
            task.act()

    def _tick_record(self, dt: float) -> None:
        # record some state about the final state of this tick
        if self.ticks_per_hist_sample > 0:
            for sector in self.gamestate.sectors.values():
                if self.gamestate.ticks % self.ticks_per_hist_sample == sector.entity_id.int % self.ticks_per_hist_sample:
                    for ship in sector.entities_by_type(core.Ship):
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
                for ship in sector.entities_by_type(core.Ship):
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
            total_sleep_trading_agenda = 0
            total_wp_trading_agenda = 0
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
                        elif agendum.state == agenda.TradingAgendum.State.SLEEP:
                            total_sleep_trading_agenda += 1
                        elif agendum.state == agenda.TradingAgendum.State.WAIT_PRIMARY:
                            total_wp_trading_agenda += 1

            self.logger.info(f'ships: {total_ships} goto orders: {total_goto_orders} ct: {total_orders_with_ct} cac: {total_orders_with_cac} mean_speed: {total_speed/total_ships:.2f} mean_neighbors: {total_neighbors/total_goto_orders if total_goto_orders > 0 else 0.:.2f}')
            self.logger.info(f'agenda: {total_agenda} mining agenda: {total_mining_agenda} idle: {total_idle_mining_agenda} trading agenda: {total_trading_agenda} idle: {total_idle_trading_agenda} sleep: {total_sleep_trading_agenda} wp: {total_wp_trading_agenda}')
            self.gamestate.log_econ()

            self.next_economy_sample = self.gamestate.timestamp + ECONOMY_LOG_PERIOD_SEC

    def _tick_destroy(self, dt:float) -> None:
        self.gamestate.handle_destroy_entities()

    def tick(self, dt: float) -> None:
        """ Do stuff to update the universe """

        self._tick_space(dt)
        self._tick_collisions(dt)

        # at this point all physics sim is done for the tick and the gamestate
        # is up to date across the universe

        # do AI stuff, e.g. let ships take action (but don't actually have the
        # actions take effect yet!)

        self._tick_orders(dt)
        self._tick_effects(dt)
        self._tick_agenda(dt)
        self._tick_tasks(dt)

        self.gamestate.event_manager.tick()

        self.gamestate.ticks += 1
        self.gamestate.timestamp += dt

        self._tick_record(dt)
        self._tick_destroy(dt)

        if self.gamestate.one_tick:
            self.gamestate.paused = True
            self.gamestate.one_tick = False

    def compute_timedrift(self) -> Tuple[float, float, float, float]:
        now = time.perf_counter()
        real_span = now - self.reference_realtime
        game_span = self.gamestate.timestamp - self.reference_gametime
        if real_span > 0.:
            rel_drift = game_span/real_span
        else:
            rel_drift = 0.

        return real_span, game_span, rel_drift, self.time_accel_rate if not self.fast_mode else rel_drift


    def _handle_synchronization(self, now:float, next_tick:float) -> None:
        if next_tick - now > self.min_tick_sleep:
            if self.dt > self.desired_dt:
                self.dt = max(self.desired_dt, self.dt * self.dt_scaledown)
                self.logger.debug(f'dt: {self.dt}')
            time.sleep(next_tick - now)
            self.sleep_count += 1
        else:
            # what to do if we miss a tick (or a lot)
            # seems like we should run a tick with a longer dt to make up for
            # it, and stop rendering until we catch up
            # but why would we miss ticks?
            if now - next_tick > self.dt:
                if self.gamestate is not None:
                    self.gamestate.counters[core.Counters.BEHIND_TICKS] += 1
                #self.logger.debug(f'ticks: {self.gamestate.ticks} sleep_count: {self.sleep_count} gc stats: {gc.get_stats()}')
                self.missed_ticks += 1
                behind = (now - next_tick)/self.dt
                if self.behind_length > self.behind_dt_scale_thresthold and behind >= self.behind_ticks:
                    if not self.ui.decrease_fps():
                        self.dt = min(self.max_dt, self.dt * self.dt_scaleup)
                self.behind_ticks = behind
                self.behind_length += 1
                if self.gamestate is not None:
                    self.behind_message_throttle = util.throttled_log(self.gamestate.timestamp, self.behind_message_throttle, self.logger, logging.WARNING, f'behind by {now - next_tick:.4f}s {behind:.2f} ticks dt: {self.dt:.4f} for {self.behind_length} ticks', 3.)
            else:
                if self.behind_length > self.behind_dt_scale_thresthold:
                    self.ui.increase_fps()
                    self.logger.debug(f'ticks caught up with realtime ticks dt: {self.dt:.4f} for {self.behind_length} ticks')

                self.behind_ticks = 0
                self.behind_length = 0

    def run_startup(self) -> None:
        next_tick = time.perf_counter()+self.dt
        while self.startup_running:
            now = time.perf_counter()
            self._handle_synchronization(now, next_tick)
            next_tick = next_tick + self.dt
            timeout = next_tick - now
            self.ui.tick(timeout, self.dt)

    def run(self) -> None:
        next_tick = time.perf_counter()+self.dt
        while self.keep_running:
            if self.should_raise:
                raise Exception("debug breakpoint in Simulator at start of tick")
            now = time.perf_counter()

            self._handle_synchronization(now, next_tick)

            starttime = time.perf_counter()
            if not self.gamestate.paused:
                self.tick(self.dt)

            now = time.perf_counter()

            last_tick = next_tick
            if self.fast_mode:
                next_tick = now
            else:
                next_tick = next_tick + self.dt / self.time_accel_rate

            timeout = next_tick - now
            if not self.gamestate.paused:
                self.timeout = self.ticktime_alpha * timeout + (1-self.ticktime_alpha) * self.timeout

            self.ui.tick(timeout, self.dt)

            now = time.perf_counter()
            ticktime = now - starttime
            self.ticktime = util.update_ema(self.ticktime, self.ticktime_alpha, ticktime)

def initialize_intel_director() -> aintel.IntelCollectionDirector:
    intel_director = aintel.IntelCollectionDirector()
    intel_director.register_gatherer(intel.SectorHexPartialCriteria, aintel.SectorHexIntelGatherer())
    intel_director.register_gatherer(intel.SectorEntityPartialCriteria, aintel.SectorEntityIntelGatherer())
    intel_director.register_gatherer(intel.EconAgentSectorEntityPartialCriteria, aintel.EconAgentSectorEntityIntelGatherer())

    return intel_director

def initialize_save_game(generator:generate.UniverseGenerator, event_manager:events.EventManager, intel_director:aintel.IntelCollectionDirector, debug:bool=True) -> save_game.GameSaver:
    sg = save_game.GameSaver(generator, event_manager, intel_director, debug=debug)

    # top level stuff
    sg.register_saver(events.EventState, s_events.EventStateSaver(sg))
    sg.register_saver(core.Gamestate, s_gamestate.GamestateSaver(sg))
    sg.register_saver(core.StarfieldLayer, s_gamestate.StarfieldLayerSaver(sg))

    # Sector stuff
    sg.register_saver(core.SectorWeatherRegion, s_sector.SectorWeatherRegionSaver(sg))

    # entities (live in Gamestate)
    sg.register_saver(core.Entity, save_game.DispatchSaver[core.Entity](sg))
    sg.register_saver(core.Player, s_character.PlayerSaver(sg))
    sg.register_saver(core.Sector, s_sector.SectorSaver(sg))
    sg.register_saver(core.Character, s_character.CharacterSaver(sg))
    sg.register_saver(econ.PlayerAgent, s_econ.PlayerAgentSaver(sg))
    sg.register_saver(econ.StationAgent, s_econ.StationAgentSaver(sg))
    sg.register_saver(econ.ShipTraderAgent, s_econ.ShipTraderAgentSaver(sg))
    sg.register_saver(core.Message, s_character.MessageSaver(sg))
    sg.register_saver(core.Ship, s_sector_entity.ShipSaver(sg))
    sg.register_saver(sector_entity.Asteroid, s_sector_entity.AsteroidSaver(sg))
    sg.register_saver(sector_entity.TravelGate, s_sector_entity.TravelGateSaver(sg))
    sg.register_saver(sector_entity.Planet, s_sector_entity.PlanetSaver(sg))
    sg.register_saver(sector_entity.Station, s_sector_entity.StationSaver(sg))
    sg.register_saver(combat.Missile, s_sector_entity.MissileSaver(sg))

    # intel
    sg.register_saver(intel.SectorEntityIntel, s_intel.SectorEntityIntelSaver(sg))
    sg.register_saver(intel.AsteroidIntel, s_intel.AsteroidIntelSaver(sg))
    sg.register_saver(intel.StationIntel, s_intel.StationIntelSaver(sg))
    sg.register_saver(intel.EconAgentIntel, s_intel.EconAgentIntelSaver(sg))
    sg.register_saver(intel.SectorHexIntel, s_intel.SectorHexIntelSaver(sg))

    # intel match criteria
    sg.register_saver(core.IntelMatchCriteria, save_game.DispatchSaver[core.IntelMatchCriteria](sg))
    sg.register_saver(intel.EntityIntelMatchCriteria, s_intel.EntityIntelMatchCriteriaSaver(sg))
    sg.register_saver(intel.SectorHexMatchCriteria, s_intel.SectorHexMatchCriteriaSaver(sg))
    sg.register_saver(intel.SectorEntityPartialCriteria, s_intel.SectorEntityPartialCriteriaSaver(sg))
    sg.register_saver(intel.AsteroidIntelPartialCriteria, s_intel.AsteroidIntelPartialCriteriaSaver(sg))
    sg.register_saver(intel.StationIntelPartialCriteria, s_intel.StationIntelPartialCriteriaSaver(sg))
    sg.register_saver(intel.SectorHexPartialCriteria, s_intel.SectorHexPartialCriteriaSaver(sg))
    sg.register_saver(intel.EconAgentSectorEntityPartialCriteria, s_intel.EconAgentSectorEntityPartialCriteriaSaver(sg))

    # agenda
    sg.register_saver(core.AbstractAgendum, save_game.DispatchSaver[core.AbstractAgendum](sg))
    sg.register_saver(agenda.StationManager, s_agenda.StationManagerSaver(sg))
    sg.register_saver(agenda.PlanetManager, s_agenda.PlanetManagerSaver(sg))
    sg.register_saver(agenda.CaptainAgendum, s_agenda.CaptainAgendumSaver(sg))
    sg.register_saver(agenda.TradingAgendum, s_agenda.TradingAgendumSaver(sg))
    sg.register_saver(agenda.MiningAgendum, s_agenda.MiningAgendumSaver(sg))
    sg.register_saver(aintel.IntelCollectionAgendum, s_agenda.IntelCollectionAgendumSaver(sg))

    # orders
    sg.register_saver(core.Order, save_game.DispatchSaver[core.Order](sg))
    sg.register_saver(core.NullOrder, s_order.NullOrderSaver(sg))
    sg.register_saver(sensors.SensorScanOrder, s_sensors.SensorScanOrderSaver(sg))

    # core orders
    sg.register_saver(orders.core.MineOrder, s_order_core.MineOrderSaver(sg))
    sg.register_saver(orders.core.TransferCargo, s_order_core.TransferCargoSaver[orders.core.TransferCargo](sg))
    sg.register_saver(orders.core.TradeCargoToStation, s_order_core.TradeCargoToStationSaver(sg))
    sg.register_saver(orders.core.TradeCargoFromStation, s_order_core.TradeCargoFromStationSaver(sg))
    sg.register_saver(orders.core.DisembarkToEntity, s_order_core.DisembarkToEntitySaver(sg))
    sg.register_saver(orders.core.TravelThroughGate, s_order_core.TravelThroughGateSaver(sg))
    sg.register_saver(orders.core.DockingOrder, s_order_core.DockingOrderSaver(sg))
    sg.register_saver(orders.core.LocationExploreOrder, s_order_core.LocationExploreOrderSaver(sg))

    # steering orders
    sg.register_saver(orders.movement.KillRotationOrder, s_movement.KillRotationOrderSaver(sg))
    sg.register_saver(orders.movement.RotateOrder, s_movement.RotateOrderSaver(sg))
    sg.register_saver(orders.movement.KillVelocityOrder, s_movement.KillVelocityOrderSaver(sg))
    sg.register_saver(orders.movement.GoToLocation, s_movement.GoToLocationSaver(sg))
    sg.register_saver(orders.movement.EvadeOrder, s_movement.EvadeOrderSaver(sg))
    sg.register_saver(orders.movement.PursueOrder, s_movement.PursueOrderSaver(sg))
    sg.register_saver(orders.movement.WaitOrder, s_movement.WaitOrderSaver(sg))

    # combat orders
    sg.register_saver(combat.MissileOrder, s_combat.MissileOrderSaver(sg))
    sg.register_saver(combat.HuntOrder, s_combat.HuntOrderSaver(sg))
    sg.register_saver(combat.AttackOrder, s_combat.AttackOrderSaver(sg))
    sg.register_saver(combat.FleeOrder, s_combat.FleeOrderSaver(sg))

    # effects
    sg.register_saver(core.Effect, save_game.DispatchSaver[core.Effect](sg))
    sg.register_saver(effects.TransferCargoEffect, s_effect.TransferCargoEffectSaver[effects.TransferCargoEffect](sg))
    sg.register_saver(effects.TradeTransferEffect, s_effect.TradeTransferEffectSaver(sg))
    sg.register_saver(effects.MiningEffect, s_effect.MiningEffectSaver(sg))
    sg.register_saver(effects.WarpOutEffect, s_effect.WarpOutEffectSaver(sg))
    sg.register_saver(effects.WarpInEffect, s_effect.WarpInEffectSaver(sg))
    sg.register_saver(combat.PointDefenseEffect, s_combat.PointDefenseEffectSaver(sg))

    # scheduled tasks (live in Gamestate)
    sg.register_saver(core.ScheduledTask, save_game.DispatchSaver[core.ScheduledTask](sg))
    sg.register_saver(combat.TimedOrderTask, s_combat.TimedOrderTaskSaver(sg))
    sg.register_saver(intel.ExpireIntelTask, s_intel.ExpireIntelTaskSaver(sg))

    # sensor settings (live in SectorEntity)
    sg.register_saver(core.AbstractSensorSettings, s_sensors.SensorSettingsSaver(sg))
    sg.register_saver(sensors.SensorSettings, s_sensors.SensorSettingsSaver(sg))

    # sensor images (live in SensorSettings)
    sg.register_saver(core.AbstractSensorImage, s_sensors.SensorImageSaver(sg))
    sg.register_saver(sensors.SensorImage, s_sensors.SensorImageSaver(sg))

    # other stuff
    sg.register_saver(combat.ThreatTracker, s_combat.ThreatTrackerSaver(sg))
    sg.register_saver(intel.IntelManager, s_intel.IntelManagerSaver(sg))

    return sg

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
        #logging.getLogger("stellarpunk").level = logging.DEBUG
        #logging.getLogger("stellarpunk.sensors").level = logging.DEBUG
        #logging.getLogger("stellarpunk.events").level = logging.DEBUG
        # send warnings to the logger
        logging.captureWarnings(True)
        # turn warnings into exceptions
        warnings.filterwarnings("error")

        # Note: the construction/initialization order here is a little fragile
        # there are some circular dependencies which we resolve through
        # interleaving construction and initialization

        #TODO: how can code register stuff if we don't create the EventManager prior to this point?
        # code must have registered events, context keys, actions and the event manager must be fully initialized prior to this point because we're about to populate contexts and trigger events

        mgr = context_stack.enter_context(util.PDBManager())

        event_manager = events.EventManager()
        events.register_events(event_manager)
        sensors.pre_initialize(event_manager)
        intel.pre_initialize(event_manager)
        intel_director = initialize_intel_director()

        generator = generate.UniverseGenerator()
        sg = initialize_save_game(generator, event_manager, intel_director)
        ui = context_stack.enter_context(interface_manager.InterfaceManager(generator, sg))

        generator.pre_initialize(event_manager, intel_director)

        ui_util.initialize()
        ui.pre_initialize(event_manager)

        # note: universe generator is handled by the ui if the player chooses
        # a new game or to load the game

        #TODO: should this be tied to the gamestate?
        economy_log = context_stack.enter_context(open("/tmp/economy.log", "wt", 1))
        sim = Simulator(generator, ui.interface, max_dt=1/5, economy_log=economy_log, game_saver=sg, context_stack=context_stack)
        sim.pre_initialize()

        ui.interface.runtime = sim

        # this event_manager pre-initialization must happen as late as possible
        # prior to gamestate creation so code has a chance to register events
        # and such
        event_manager.pre_initialize(config.Events)

        sim.run_startup()

        if sim.gamestate:
            # experimentally chosen so that we don't get multiple gcs during a tick
            # this helps a lot because there's lots of short lived objects during a
            # tick and it's better if they stay in the youngest generation
            #TODO: should we just disable a gc while we're doing a tick?
            gc.set_threshold(700*4, 10*4, 10*4)
            sim.run()

            counter_str = "\n".join(map(lambda x: f'{str(x[0])}:\t{x[1]}', zip(list(core.Counters), sim.gamestate.counters)))
            logging.info(f'counters:\n{counter_str}')

            assert all(x == y for x,y in zip(sim.gamestate.entities.values(), sim.gamestate.entities_short.values()))
            logging.info(f'entities:\t{len(sim.gamestate.entities)}')
            logging.info(f'ticks:\t{sim.gamestate.ticks}')
            logging.info(f'timestamp:\t{sim.gamestate.timestamp}')
            real_span, game_span, rel_drift, expected_rel_drift = sim.compute_timedrift()
            logging.info(f'timedrift: {real_span} vs {game_span} {rel_drift:.3f} vs {expected_rel_drift:.3f}')

        logging.info("done.")

if __name__ == "__main__":
    main()
