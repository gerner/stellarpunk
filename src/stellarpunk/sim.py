import sys
import logging
import contextlib
import time
import math
import warnings
import cProfile
import pstats

import ipdb # type: ignore

from stellarpunk import util, core, interface, generate
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

class StellarPunkSim:
    def __init__(self, gamestate, ui, dt=1/60):
        self.logger = logging.getLogger(util.fullname(self))
        self.gamestate = gamestate
        self.ui = ui

        # time between ticks, this is the framerate
        self.dt = dt

        self.ticktime_alpha = 0.2
        self.min_tick_sleep = self.dt/5

        self.collisions = []

    def _ship_collision_detected(self, arbiter, space, data):
        # which ship(s) are colliding?

        (shape_a, shape_b) = arbiter.shapes
        sector = data["sector"]

        self.logger.debug(f'collision detected in {sector}, between {shape_a.body.entity} {shape_b.body.entity} with {arbiter.total_impulse}N and {arbiter.total_ke}j')

        self.collisions.append((
            shape_a.body.entity,
            shape_b.body.entity,
            arbiter.total_impulse,
            arbiter.total_ke,
        ))

    def initialize(self):
        """ One-time initialize of the simulation. """
        for sector in self.gamestate.sectors.values():
            h = sector.space.add_default_collision_handler()#add_wildcard_collision_handler(core.ObjectType.SHIP)
            h.data["sector"] = sector
            h.post_solve = self._ship_collision_detected

    def tick_order(self, ship, dt):
        if ship.order:
            #self.logger.debug(f'evaluating {ship.order}')
            if ship.order.is_complete():
                self.logger.debug(f'{ship.entity_id} completed {ship.order}')
                ship.order = ship.default_order()
            else:
                ship.order.act(dt)
        else:
            raise Exception(f'{ship} unexpectedly had no order!')


    def tick(self, dt):
        """ Do stuff to update the universe """

        # update physics simulations
        #self.logger.debug(f'{len(self.gamestate.sectors.values())} sectors')
        for i,sector in enumerate(self.gamestate.sectors.values()):
            self.collisions.clear()

            sector.space.step(dt)

            if self.collisions:
                self.gamestate.paused = True

            for ship in sector.ships:
                # update ship positions from physics sim
                ship.x, ship.y = ship.phys.position
                #new_x, new_y = ship.phys.position
                #if (new_x, new_y) != (ship.x, ship.y):
                #    ship.sector.spatial.delete(ship.short_id_int(), (ship.x, ship.y, ship.x, ship.y))
                #    ship.x, ship.y = ship.phys.position
                #    ship.sector.spatial.insert(ship.short_id_int(), (ship.x, ship.y, ship.x, ship.y), obj=ship.entity_id)

                ship.angle = ship.phys.angle
                ship.velocity = ship.phys.velocity
                ship.angular_velocity = ship.phys.angular_velocity

            #sector.reindex_locations()

            #self.logger.debug(f'{sector} has {len(sector.ships)}')
            for ship in sector.ships:
                self.tick_order(ship, dt)

            #TODO: do resource and production stuff
            #TODO: do AI stuff
        self.gamestate.ticks += 1
        self.gamestate.timestamp += dt

    def run(self):
        keep_running = True
        next_tick = time.perf_counter()+self.dt

        while keep_running:
            now = time.perf_counter()

            if next_tick - now > self.min_tick_sleep:
                time.sleep(next_tick - now)
            #TODO: what to do if we miss a tick (or a lot)
            # seems like we should run a tick with a longer dt to make up for
            # it, and stop rendering until we catch up
            # but why would we miss ticks?
            if now - next_tick > self.dt:
                self.gamestate.missed_ticks += 1
                self.logger.debug(f'behind by {(now - next_tick)/self.dt} ticks')

            starttime = time.perf_counter()
            if not self.gamestate.paused:
                self.tick(self.dt)

            last_tick = next_tick
            next_tick = next_tick + self.dt

            now = time.perf_counter()

            ticktime = now - starttime
            self.gamestate.ticktime = self.ticktime_alpha * ticktime + (1-self.ticktime_alpha) * self.gamestate.ticktime

            timeout = next_tick - now
            if timeout > 0: # only render a frame if there's enough time
                if not self.gamestate.paused:
                    self.gamestate.timeout = self.ticktime_alpha * timeout + (1-self.ticktime_alpha) * self.gamestate.timeout

                try:
                    self.ui.tick(next_tick - time.perf_counter())
                except interface.QuitError:
                    self.logger.info("quitting")
                    keep_running = False

def main():
    profile = False
    with contextlib.ExitStack() as context_stack:
        # for reference, config to stderr:
        # logging.basicConfig(stream=sys.stderr, level=logging.INFO)
        logging.basicConfig(
                format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
                filename="/tmp/stellarpunk.log",
                level=logging.DEBUG
        )
        # send warnings to the logger
        logging.captureWarnings(True)
        # turn warnings into exceptions
        warnings.filterwarnings("error")

        mgr = context_stack.enter_context(IPDBManager())
        gamestate = core.StellarPunk()

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
        if profile:
            dt = 1/40
        sim = StellarPunkSim(gamestate, ui, dt=dt)
        sim.initialize()

        if profile:
            pr = context_stack.enter_context(cProfile.Profile())
        sim.run()

        logging.info("done.")

    if profile:
        pstats.Stats(pr).dump_stats("/tmp/profile.prof")

if __name__ == "__main__":
    main()
