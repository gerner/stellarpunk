import sys
import logging
import contextlib
import time
import math
import curses
import warnings
from typing import List, Optional

import ipdb # type: ignore
import numpy as np

from stellarpunk import util, core, interface, generate, orders
from stellarpunk.interface import universe as universe_interface

class IPDBManager:
    def __init__(self):
        self.logger = logging.getLogger(util.fullname(self))

    def __enter__(self):
        self.logger.info("entering IPDBManager")

        return self

    def __exit__(self, e, m, tb):
        self.logger.info("exiting IPDBManager")
        if e is not None:
            self.logger.info(f'handling exception {e} {m}')
            print(m.__repr__(), file=sys.stderr)
            ipdb.post_mortem(tb)

class Simulator:
    def __init__(self, gamestate, ui, dt:float=1/60, max_dt:Optional[float]=None) -> None:
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
        # number of tickse we need to be behind to trigger dt scaling
        # note that dt scaling is somewhat disruptive since some things might
        # expect us to have a constant dt.
        self.behind_dt_scale_thresthold = 30

        self.ticktime_alpha = 0.1
        self.min_tick_sleep = self.desired_dt/5
        self.min_ui_timeout = 0.

        self.collisions:List[tuple[core.SectorEntity, core.SectorEntity, float, float]] = []

    def _ship_collision_detected(self, arbiter, space, data):
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
        if order.is_complete():
            self.logger.debug(f'{ship.entity_id} completed {order}')
            ship.orders.popleft()
        else:
            order.act(dt)

    def tick(self, dt: float) -> None:
        """ Do stuff to update the universe """

        # update physics simulations
        #self.logger.debug(f'{len(self.gamestate.sectors.values())} sectors')
        for i,sector in enumerate(self.gamestate.sectors.values()):
            self.collisions.clear()

            sector.space.step(dt)

            if self.collisions:
                #TODO: use kinetic energy from the collision to cause damage
                # metals hav an impact strength between 0.34e3 and 145e3
                # joules / meter^2
                # so an impact of 17M joules spread over an area of 1000 m^2
                # would be a lot, but not catastrophic
                # spread over 100m^2 would be
                self.gamestate.paused = True
                self.ui.status_message(f'collision detected {self.collisions[0][0].address_str()}, {self.collisions[0][1].address_str()}', attr=curses.color_pair(1))

            for ship in sector.ships:
                # update ship positions from physics sim
                ship.loc = np.array(ship.phys.position)

                ship.angle = ship.phys.angle
                ship.velocity = np.array(ship.phys.velocity)
                ship.angular_velocity = ship.phys.angular_velocity

            for ship in sector.ships:
                self.tick_order(ship, dt)
                ship.history.append(ship.to_history(self.gamestate.timestamp))

            #TODO: do resource and production stuff
            #TODO: do AI stuff
        self.gamestate.ticks += 1
        self.gamestate.timestamp += dt

    def run(self) -> None:

        next_tick = time.perf_counter()+self.dt

        while self.gamestate.keep_running:
            now = time.perf_counter()

            if next_tick - now > self.min_tick_sleep:
                if self.dt > self.desired_dt:
                    self.dt = max(self.desired_dt, self.dt * self.dt_scaledown)
                    self.logger.debug(f'dt: {self.dt}')
                time.sleep(next_tick - now)
            #TODO: what to do if we miss a tick (or a lot)
            # seems like we should run a tick with a longer dt to make up for
            # it, and stop rendering until we catch up
            # but why would we miss ticks?
            if now - next_tick > self.dt:
                self.gamestate.missed_ticks += 1
                behind = (now - next_tick)/self.dt
                if self.behind_length > self.behind_dt_scale_thresthold and behind >= self.behind_ticks:
                    self.dt = min(self.max_dt, self.dt * self.dt_scaleup)
                self.behind_ticks = behind
                self.behind_length += 1
                self.logger.warning(f'behind by {behind} ticks dt: {self.dt} for {self.behind_length} ticks')
            else:
                self.behind_ticks = 0
                self.behind_length = 0


            starttime = time.perf_counter()
            if not self.gamestate.paused:
                self.tick(self.dt)

            last_tick = next_tick
            next_tick = next_tick + self.dt

            now = time.perf_counter()

            ticktime = now - starttime
            self.gamestate.ticktime = self.ticktime_alpha * ticktime + (1-self.ticktime_alpha) * self.gamestate.ticktime

            timeout = next_tick - now
            # only render a frame if there's enough time
            if timeout > self.min_ui_timeout:
                if not self.gamestate.paused:
                    self.gamestate.timeout = self.ticktime_alpha * timeout + (1-self.ticktime_alpha) * self.gamestate.timeout

                try:
                    self.ui.tick(timeout)
                except interface.QuitError:
                    self.logger.info("quitting")
                    self.gamestate.quit()

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

        mgr = context_stack.enter_context(IPDBManager())
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

        sim.run()

        logging.info("done.")

if __name__ == "__main__":
    main()
