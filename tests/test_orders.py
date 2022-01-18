""" Tests for ship orders and behaviors. """

import logging

import pytest
import pymunk
import numpy as np

from stellarpunk import core, sim, generate, orders, util

class TestUI:
    def tick(self, timeout):
        pass

@pytest.fixture
def testui():
    return TestUI()

@pytest.fixture
def gamestate():
    return core.Gamestate()

@pytest.fixture
def simulator(gamestate, testui):
    simulation = sim.Simulator(gamestate, testui)
    simulation.min_tick_sleep = np.inf
    simulation.min_ui_timeout = -np.inf

    return simulation

@pytest.fixture
def generator(gamestate):
    return generate.UniverseGenerator(gamestate)

@pytest.fixture
def sector(gamestate):
    sector_radius=1e5
    sector_name = "Sector"

    sector = core.Sector(0, 0, sector_radius, sector_name)
    sector.space = pymunk.Space()
    gamestate.sectors[(0,0)] = sector

    return sector

# collision avoidance interesting cases:
# no collision, just go wherever you're going
# tangential collision, avoid
# head-on collision, need to pick a side to avoid collision
# tangential collision to one side, really want to go in that direction tho
#   should have a way to express that an avoid in a helpful direction

def test_basic_gotolocation(gamestate, generator, sector, testui, simulator):
    ship_driver = generator.spawn_ship(sector, -400, 15000, v=(0,0), w=0, theta=0)

    goto_order = orders.GoToLocation(np.array((0,0)), ship_driver, gamestate)
    ship_driver.orders.append(goto_order)

    distance = np.linalg.norm(ship_driver.loc)
    eta = 2 * np.sqrt( 2 * (distance/2) / ship_driver.max_acceleration()) + orders.rotation_time(2*np.pi, ship_driver.max_angular_acceleration())

    starttime = gamestate.timestamp
    def tick(timeout):
        logging.debug(f'{ship_driver.loc} {ship_driver.velocity} {ship_driver.angle} {gamestate.timestamp - starttime}s vs {eta}s')
        assert not simulator.collisions
        if goto_order.is_complete():
            gamestate.quit()
        assert not goto_order.cannot_stop
        assert gamestate.timestamp - starttime < eta

    testui.tick = tick
    simulator.run()

def test_gotolocation_with_entity_target(gamestate, generator, sector, testui, simulator):
    ship_driver = generator.spawn_ship(sector, -400, 15000, v=(0,0), w=0, theta=0)
    ship_blocker = generator.spawn_ship(sector, 0, 0, v=(0,0), w=0, theta=0)

    goto_order = orders.GoToLocation(np.array((0,0)), ship_driver, gamestate)
    ship_driver.orders.append(goto_order)

    distance = np.linalg.norm(ship_driver.loc)
    eta = 2 * np.sqrt( 2 * (distance/2) / ship_driver.max_acceleration()) + orders.rotation_time(2*np.pi, ship_driver.max_angular_acceleration())

    starttime = gamestate.timestamp
    def tick(timeout):
        logging.debug(f'{ship_driver.loc} {ship_driver.velocity} {ship_driver.angle} {gamestate.timestamp - starttime}s vs {eta}s')
        assert not simulator.collisions
        if goto_order.is_complete():
            gamestate.quit()
        assert not goto_order.cannot_stop
        assert not goto_order.cannot_avoid_collision
        assert not goto_order.collision_threat
        assert gamestate.timestamp - starttime < eta

    testui.tick = tick
    simulator.run()
def test_gotolocation_with_sympathetic_starting_velocity(gamestate, generator, sector, testui, simulator):
    ship_driver = generator.spawn_ship(sector, -400, 15000, v=(0,0), w=0, theta=0)
    ship_driver.velocity = np.array((0, -10)) * 50
    ship_driver.phys.velocity = tuple(ship_driver.velocity)

    goto_order = orders.GoToLocation(np.array((0,0)), ship_driver, gamestate)
    ship_driver.orders.append(goto_order)

    distance = np.linalg.norm(ship_driver.loc)
    eta = 2 * np.sqrt( 2 * (distance/2) / ship_driver.max_acceleration()) + orders.rotation_time(2*np.pi, ship_driver.max_angular_acceleration())

    starttime = gamestate.timestamp
    def tick(timeout):
        logging.debug(f'{ship_driver.loc} {ship_driver.velocity} {ship_driver.angle} {gamestate.timestamp - starttime}s vs {eta}s')
        nonlocal distance
        assert not simulator.collisions
        if goto_order.is_complete():
            gamestate.quit()
        assert not goto_order.cannot_stop
        assert gamestate.timestamp - starttime < eta

        d = np.linalg.norm(ship_driver.loc)
        #assert d < distance or d < goto_order.arrival_distance
        distance = d

    testui.tick = tick
    simulator.run()

def test_gotolocation_with_deviating_starting_velocity(gamestate, generator, sector, testui, simulator):
    ship_driver = generator.spawn_ship(sector, 0, 15000, v=(0,0), w=0, theta=0)
    ship_driver.velocity = np.array((-4, -10)) * 50
    ship_driver.phys.velocity = tuple(ship_driver.velocity)

    goto_order = orders.GoToLocation(np.array((0,0)), ship_driver, gamestate)
    ship_driver.orders.append(goto_order)

    distance = np.linalg.norm(ship_driver.loc)
    eta = 2 * np.sqrt( 2 * (distance/2) / ship_driver.max_acceleration()) + orders.rotation_time(2*np.pi, ship_driver.max_angular_acceleration())

    starttime = gamestate.timestamp
    def tick(timeout):
        logging.debug(f'{ship_driver.loc} {ship_driver.velocity} {ship_driver.angle} {gamestate.timestamp - starttime}s vs {eta}s')
        nonlocal distance
        assert not simulator.collisions

        if goto_order.is_complete():
            gamestate.quit()
        assert not goto_order.cannot_stop
        assert gamestate.timestamp - starttime < eta

        d = np.linalg.norm(ship_driver.loc)
        #assert d < distance or d < goto_order.arrival_distance
        distance = d

    testui.tick = tick
    simulator.run()

def test_basic_collision_avoidance(gamestate, generator, sector, testui, simulator, caplog):
    # set up a sector, including space
    # add two ships in an offset (relative to desired course) arrangement
    # travel along course and observe no collision

    ship_blocker = generator.spawn_ship(sector, -300, 1200, v=(0,0), w=0, theta=0)
    ship_driver = generator.spawn_ship(sector, 0, 2400, v=(0,0), w=0, theta=0)

    goto_order = orders.GoToLocation(np.array((0,0)), ship_driver, gamestate)
    ship_driver.orders.append(goto_order)

    # d = v_i*t + 1/2 a * t**2
    # v_i = 0
    # d = 1/2 a * t**2
    # t = sqrt( 2*d/a )
    # expect path to be accelerate half-way, turn around, decelerate
    # this is approximate
    distance = np.linalg.norm(ship_driver.loc)
    eta = 2 * np.sqrt( 2 * (distance) / ship_driver.max_acceleration()) + orders.rotation_time(2*np.pi, ship_driver.max_angular_acceleration())

    starttime = gamestate.timestamp
    def tick(timeout):
        logging.debug(f'{ship_driver.loc} {ship_driver.velocity} {ship_driver.angle} {gamestate.timestamp - starttime}s vs {eta}s')
        assert not simulator.collisions
        if goto_order.is_complete():
            gamestate.quit()
        assert not goto_order.cannot_stop
        assert not goto_order.cannot_avoid_collision
        assert gamestate.timestamp - starttime < eta*1.5

    testui.tick = tick
    simulator.run()

def test_head_on_collision_avoidance(gamestate, generator, sector, testui, simulator):
    ship_blocker = generator.spawn_ship(sector, 0, 1200, v=(0,0), w=0, theta=0)
    ship_driver = generator.spawn_ship(sector, 0, 2400, v=(0,0), w=0, theta=0)

    goto_order = orders.GoToLocation(np.array((0,0)), ship_driver, gamestate)
    ship_driver.orders.append(goto_order)

    distance = 2400
    eta = 2 * np.sqrt( 2 * (distance) / ship_driver.max_acceleration()) + orders.rotation_time(2*np.pi, ship_driver.max_angular_acceleration())

    starttime = gamestate.timestamp
    def tick(timeout):
        assert not simulator.collisions
        if goto_order.is_complete():
            gamestate.quit()
        assert gamestate.timestamp - starttime < eta*2.7

    testui.tick = tick
    simulator.run()

def test_desired_velocity_collision_avoidance(gamestate, generator, sector, testui, simulator):
    # set up a sector, including space
    # add two ships in an offset (relative to desired course) arrangement
    # travel along course and observe no collision

    # a "wall" of blockers to the left of our target
    generator.spawn_ship(sector, -300, 12000, v=(0,0), w=0, theta=0)
    generator.spawn_ship(sector, -300, 11000, v=(0,0), w=0, theta=0)
    generator.spawn_ship(sector, -300, 10000, v=(0,0), w=0, theta=0)
    generator.spawn_ship(sector, -300, 9000, v=(0,0), w=0, theta=0)
    generator.spawn_ship(sector, -300, 6000, v=(0,0), w=0, theta=0)
    generator.spawn_ship(sector, -300, 2000, v=(0,0), w=0, theta=0)
    generator.spawn_ship(sector, -300, 1000, v=(0,0), w=0, theta=0)
    generator.spawn_ship(sector, -300, 500, v=(0,0), w=0, theta=0)

    ship_driver = generator.spawn_ship(sector, -400, 15000, v=(0,0), w=0, theta=0)
    ship_driver.velocity = np.array((0, -10)) * 50
    ship_driver.phys.velocity = tuple(ship_driver.velocity)

    goto_order = orders.GoToLocation(np.array((0,0)), ship_driver, gamestate)
    ship_driver.orders.append(goto_order)

    # d = v_i*t + 1/2 a * t**2
    # v_i = 0
    # d = 1/2 a * t**2
    # t = sqrt( 2*d/a )
    # expect path to be accelerate half-way, turn around, decelerate
    # this is approximate
    distance = np.linalg.norm(ship_driver.loc)
    eta = 2 * np.sqrt( 2 * (distance) / ship_driver.max_acceleration()) + orders.rotation_time(2*np.pi, ship_driver.max_angular_acceleration())

    starttime = gamestate.timestamp
    def tick(timeout):
        nonlocal distance
        d = np.linalg.norm(ship_driver.loc)
        #assert d < distance
        distance = d

        assert not simulator.collisions
        neighbor_distance = np.inf
        neighbor = None
        for hit in sector.spatial_point(ship_driver.loc):
            if hit == ship_driver:
                continue
            d = np.linalg.norm(ship_driver.loc - hit.loc) - hit.radius - ship_driver.radius
            if d < neighbor_distance:
                neighbor_distance = d
                neighbor = hit

        logging.debug(f'{ship_driver.loc} {ship_driver.velocity} {ship_driver.angle}, {d:.2e} {neighbor}, {goto_order.collision_threat}')
        if goto_order.is_complete():
            gamestate.quit()
        #assert not goto_order.cannot_stop
        #assert not goto_order.cannot_avoid_collision
        assert gamestate.timestamp - starttime < eta*1.5

    testui.tick = tick
    simulator.run()

    util.write_history_to_file(ship_driver, "/tmp/stellarpunk_test.history")
