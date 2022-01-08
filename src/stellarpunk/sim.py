import sys
import logging
import contextlib
import time
import math

import ipdb

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
    def __init__(self, gamestate, ui):
        self.logger = logging.getLogger(util.fullname(self))
        self.gamestate = gamestate
        self.ui = ui

        # time between ticks, this is the framerate
        self.dt = 1/60

        self.ticktime_alpha = 0.01
        self.min_tick_sleep = self.dt/5

    def tick(self, dt):
        """ Do stuff to update the universe """

        # update physics simulations
        for sector in self.gamestate.sectors.values():
            for ship in sector.ships:
                if ship.order:
                    if ship.order.is_complete():
                        self.logger.debug(f'{ship.entity_id} completed {ship.order}')
                        ship.order = None
                    else:
                        ship.order.act(dt)

            sector.space.step(dt)

            # update ship positions from physics sim
            for ship in sector.ships:
                ship.x, ship.y = ship.phys.position
                ship.angle = ship.phys.angle
                ship.velocity = ship.phys.velocity

            sector.reindex_locations()

            #TODO: do resource and production stuff
            #TODO: do AI stuff
        self.gamestate.ticks += 1

    def run(self):
        keep_running = True
        next_tick = time.perf_counter()+self.dt

        #TODO: get rid of this hack
        # hack to test out some physics
        import pymunk
        import numpy as np
        # soyuz 5000 - 10000kg
        # dragon capsule 4000kg
        # shuttle orbiter 78000kg
        ship_mass = 2 * 1e3

        # soyuz: 7-10m long
        # shuttle orbiter: 37m long
        # spacex dragon: 6.1m
        # spacex starship: 120m long
        ship_radius = 30

        # one raptor: 1.81 kN
        # one SSME: 2.279 kN
        # OMS main engine: 26.7 kN
        # KTDU-80 main engine: 2.95 kN
        max_thrust = 0.5 * 1e6

        # one draco: 400 N (x16 on Dragon)
        # OMS aft RCS: 3.87 kN
        # KTDU-80 11D428A-16: 129.16 N (x16 on the Soyuz)
        # note about g-forces:
        # assuming circle of radius 30m, mass 2e3 kg
        # mass moment 18,000,000 kg m^2
        # centriptal acceleration = r * w^2
        # 1g at 30m with angular acceleration of 0.57 rad/sec
        # 5000 * 30 N m can get 2e3kg, 30m circle up to half a g in 60 seconds
        # starting from zero
        # space shuttle doesn't exeed 3g during ascent
        max_torque = 5000 * ship_radius

        ship_moment = pymunk.moment_for_circle(ship_mass, 0, ship_radius)
        r = np.random.default_rng()
        for sector in self.gamestate.sectors.values():
            sector.space = pymunk.Space()
            for ship in sector.ships:
                ship_body = pymunk.Body(ship_mass, ship_moment)
                ship_shape = pymunk.Circle(ship_body, ship_radius)
                ship_shape.friction=0.5
                ship_body.position = ship.x, ship.y
                v = pymunk.vec2d.Vec2d(*(r.normal(0, 50, 2)))
                ship_body.velocity = v
                ship_body.angle = v.angle

                sector.space.add(ship_body, ship_shape)

                #TODO: quick hack to give some interest
                ship_body.angular_velocity = r.normal(0, 0.08)

                ship.phys = ship_body
                ship.max_thrust = max_thrust
                ship.max_torque = max_torque

        while keep_running:
            now = time.perf_counter()

            if next_tick - now > self.min_tick_sleep:
                time.sleep(next_tick - now)
            #TODO: what to do if we miss a tick (or a lot)
            # seems like we should run a tick with a longer dt to make up for
            # it, and stop rendering until we catch up
            # but why would we miss ticks?
            if now - next_tick > self.dt:
                self.gamestate.missed_ticks += int((now - next_tick)/self.dt)

            starttime = time.perf_counter()
            if not self.gamestate.paused:
                self.tick(self.dt)

            last_tick = next_tick
            next_tick = next_tick + self.dt

            now = time.perf_counter()

            timeout = next_tick - now
            if timeout > 0: # only render a frame if there's enough time
                if not self.gamestate.paused:
                    self.gamestate.timeout = self.ticktime_alpha * timeout + (1-self.ticktime_alpha) * self.gamestate.timeout

                try:
                    self.ui.tick(next_tick - time.perf_counter())
                except interface.QuitError:
                    self.logger.info("quitting")
                    keep_running = False

            now = time.perf_counter()
            ticktime = now - starttime
            self.gamestate.ticktime = self.ticktime_alpha * ticktime + (1-self.ticktime_alpha) * self.gamestate.ticktime

def main():
    with contextlib.ExitStack() as context_stack:
        logging.basicConfig(
                format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
                filename="/tmp/stellarpunk.log",
                level=logging.DEBUG
        )
        #logging.basicConfig(stream=sys.stderr, level=logging.INFO)

        mgr = context_stack.enter_context(IPDBManager())
        gamestate = core.StellarPunk()
        ui = context_stack.enter_context(interface.Interface(gamestate))

        ui.initialize()
        uv = universe_interface.UniverseView(gamestate, ui)
        ui.open_view(uv)
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
