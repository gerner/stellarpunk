import sys
import logging
import contextlib
import time

import ipdb

from stellarpunk import util, core, interface, generate

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
    def __init__(self, gamestate, ui):
        self.gamestate = gamestate
        self.ui = ui

        # time between ticks, this is the framerate
        self.dt = 1/10

        self.ticktime_alpha = 0.5

    def tick(self):
        """ Do stuff to update the universe """

        # update physics simulations
        for sector in self.gamestate.sectors.values():
            sector.space.step(self.dt)

            #TODO: hack for pymunk
            # update ship positions from physics sim
            for ship in sector.ships:
                ship.x, ship.y = ship.phys.position
            sector.reindex_locations()

            #TODO: do resource and production stuff
            #TODO: do AI stuff

    def run(self):
        keep_running = True
        next_tick = time.time()+self.dt

        #TODO: get rid of this hack
        # hack to test out some physics
        import pymunk
        import numpy as np
        dt = 100 / 1000 # 100ms
        ship_mass = 2 * 1e6
        ship_radius = 30
        ship_moment = pymunk.moment_for_circle(ship_mass, 0, ship_radius)
        r = np.random.default_rng()
        for sector in self.gamestate.sectors.values():
            sector.space = pymunk.Space()
            for ship in sector.ships:
                ship_body = pymunk.Body(ship_mass, ship_moment)
                ship_shape = pymunk.Circle(ship_body, ship_radius)
                ship_body.position = ship.x, ship.y
                v = tuple(r.normal(0, 50, 2))
                ship_body.velocity = v
                sector.space.add(ship_body, ship_shape)
                ship.phys = ship_body
                ship.velocity = v

        while keep_running:
            starttime = time.time()
            self.tick()
            ticktime = time.time() - starttime

            last_tick = next_tick
            next_tick = next_tick + self.dt
            now = time.time()
            timeout = next_tick - now
            assert timeout > 0
            self.gamestate.ticktime = self.ticktime_alpha * ticktime + (1-self.ticktime_alpha) * self.gamestate.ticktime

            self.ui.tick(next_tick - time.time())

            self.gamestate.ticks += 1

def main():
    with contextlib.ExitStack() as context_stack:
        logging.basicConfig(
                format="PID %(process)d %(asctime)s %(name)-12s %(levelname)-8s %(message)s",
                filename="/tmp/stellarpunk.log",
                level=logging.DEBUG
        )
        #logging.basicConfig(stream=sys.stderr, level=logging.INFO)

        mgr = context_stack.enter_context(IPDBManager())
        gamestate = core.StellarPunk()
        ui = context_stack.enter_context(interface.Interface(gamestate))

        ui.initialize()
        uv = interface.UniverseView(gamestate, ui)
        uv.initialize()
        ui.view_focus.append(uv)
        ui.input_focus.append(uv)
        generation_listener = ui.generation_listener()

        logging.info("generating universe...")
        generator = generate.UniverseGenerator(gamestate, listener=generation_listener)
        stellar_punk = generator.generate_universe()

        #logging.info("running simulation...")
        #stellar_punk.run()

        stellar_punk.production_chain.viz().render("production_chain", format="pdf")

        sim = StellarPunkSim(gamestate, ui)
        sim.run()

        logging.info("done.")

if __name__ == "__main__":
    main()
