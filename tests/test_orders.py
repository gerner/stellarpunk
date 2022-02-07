""" Tests for ship orders and behaviors. """

import logging
import os

import pytest
import numpy as np

from stellarpunk import core, sim, generate, orders, util
from . import write_history, nearest_neighbor

TESTDIR = os.path.dirname(__file__)

def test_coalesce():
    """ Test that coalescing threats actually covers them. """

    # set up: ship heading toward several static points all within twice the
    # collision margin

    pos = np.array((0.,0.))
    v = np.array((10.,0.))

    hits_l = np.array((
        ((2000., 0.)),
        ((2500., 500.)),
        ((2500., -500.)),
        ((5000., 0.)), # not part of the group, but a threat
        ((2500., -5000.)), # not part of the group
    ))
    hits_v = np.array([(0.,0.)]*len(hits_l))
    hits_r = np.array([30.]*len(hits_l))

    (
            idx,
            approach_time,
            rel_pos,
            rel_vel,
            min_sep,
            threat_count,
            coalesced_threats,
            threat_radius,
            threat_loc,
            threat_velocity,
            nearest_neighbor_idx,
            nearest_neighbor_dist,
            neighborhood_density,
            coalesed_idx,
    ) = orders._analyze_neighbors(
            hits_l, hits_v, hits_r, pos, v,
            max_distance=1e4,
            ship_radius=30.,
            margin=5e2,
            neighborhood_radius=1e4)

    assert idx == 0
    assert np.allclose(rel_pos, (2000., 0.))
    assert np.allclose(rel_vel, v*-1)
    assert min_sep == 0.
    assert threat_count == 4
    assert coalesced_threats == 3

    for i in range(coalesced_threats):
        assert np.linalg.norm(hits_l[i] - threat_loc)+hits_r[i] <= threat_radius

def test_goto_entity(gamestate, generator, sector):
    ship_driver = generator.spawn_ship(sector, -10000, 0, v=(0,0), w=0, theta=0)
    station = generator.spawn_station(sector, 0, 0, resource=0)

    arrival_distance = 1.5e3
    collision_margin = 1e3
    goto_order = orders.GoToLocation.goto_entity(station, ship_driver, gamestate, arrival_distance - station.radius, collision_margin)

    assert np.linalg.norm(station.loc - goto_order.target_location)+goto_order.arrival_distance <= arrival_distance + orders.VELOCITY_EPS
    assert np.linalg.norm(station.loc - goto_order.target_location)-station.radius-goto_order.arrival_distance >= collision_margin - orders.VELOCITY_EPS
    assert goto_order.arrival_distance >= (arrival_distance*0.1)/2 - orders.VELOCITY_EPS
    assert goto_order.min_distance == 0.

@write_history
def test_zero_rotation_time(gamestate, generator, sector, testui, simulator):
    ship_driver = generator.spawn_ship(sector, -400, 15000, v=(0,0), w=0, theta=0)

    rotate_order = orders.RotateOrder(np.pi, ship_driver, gamestate)
    ship_driver.orders.append(rotate_order)

    eta = orders.rotation_time(np.pi, 0, ship_driver.max_angular_acceleration(), rotate_order.safety_factor)

    testui.eta = eta
    testui.orders = [rotate_order]

    simulator.run()
    assert rotate_order.is_complete()
    assert ship_driver.angular_velocity == 0
    assert ship_driver.angle == np.pi

@write_history
def test_non_zero_rotation_time(gamestate, generator, sector, testui, simulator):
    ship_driver = generator.spawn_ship(sector, -400, 15000, v=(0,0), w=-2, theta=0)

    rotate_order = orders.RotateOrder(np.pi/2, ship_driver, gamestate)
    ship_driver.orders.append(rotate_order)

    eta = orders.rotation_time(rotate_order.target_angle, ship_driver.angular_velocity, ship_driver.max_angular_acceleration(), rotate_order.safety_factor)

    testui.eta = eta
    testui.orders = [rotate_order]

    simulator.run()
    assert rotate_order.is_complete()
    assert ship_driver.angular_velocity == 0
    assert ship_driver.angle == rotate_order.target_angle

@write_history
def test_basic_gotolocation(gamestate, generator, sector, testui, simulator):
    ship_driver = generator.spawn_ship(sector, -400, 15000, v=(0,0), w=0, theta=0)

    goto_order = orders.GoToLocation(np.array((0.,0.)), ship_driver, gamestate)
    ship_driver.orders.append(goto_order)

    distance = np.linalg.norm(ship_driver.loc)
    eta = goto_order.eta()
    starttime = gamestate.timestamp
    def tick(timeout):
        logging.debug(f'{ship_driver.loc} {ship_driver.velocity} {ship_driver.angle} {gamestate.timestamp - starttime}s vs {eta}s')
        assert not simulator.collisions
        neighbor, neighbor_dist = nearest_neighbor(sector, ship_driver)
        assert neighbor_dist >= goto_order.collision_margin

        if goto_order.is_complete():
            gamestate.quit()
        assert not goto_order.cannot_stop
        assert gamestate.timestamp - starttime < eta

    testui.tick = tick
    simulator.run()
    assert goto_order.is_complete()

@write_history
def test_gotolocation_with_entity_target(gamestate, generator, sector, testui, simulator):
    ship_driver = generator.spawn_ship(sector, -400, 15000, v=(0,0), w=0, theta=0)
    ship_blocker = generator.spawn_ship(sector, 0, 0, v=(0,0), w=0, theta=0)

    goto_order = orders.GoToLocation(np.array((0.,0.)), ship_driver, gamestate)
    ship_driver.orders.append(goto_order)

    distance = np.linalg.norm(ship_driver.loc)
    eta = goto_order.eta()

    starttime = gamestate.timestamp
    def tick(timeout):
        logging.debug(f'{ship_driver.loc} {ship_driver.velocity} {ship_driver.angle} {gamestate.timestamp - starttime}s vs {eta}s')
        assert not simulator.collisions
        neighbor, neighbor_dist = nearest_neighbor(sector, ship_driver)
        assert neighbor_dist >= goto_order.collision_margin

        if goto_order.is_complete():
            gamestate.quit()
        assert not goto_order.cannot_stop
        assert not goto_order.cannot_avoid_collision
        assert gamestate.timestamp - starttime < eta

    testui.tick = tick
    simulator.run()
    assert goto_order.is_complete()

@write_history
def test_gotolocation_with_sympathetic_starting_velocity(gamestate, generator, sector, testui, simulator):
    ship_driver = generator.spawn_ship(sector, -400, 15000, v=(0,0), w=0, theta=0)
    ship_driver.velocity = np.array((0., -10.)) * 50.
    ship_driver.phys.velocity = tuple(ship_driver.velocity)

    goto_order = orders.GoToLocation(np.array((0.,0.)), ship_driver, gamestate)
    ship_driver.orders.append(goto_order)

    distance = np.linalg.norm(ship_driver.loc)
    eta = goto_order.eta()
    starttime = gamestate.timestamp
    def tick(timeout):
        logging.debug(f'{ship_driver.loc} {ship_driver.velocity} {ship_driver.angle} {gamestate.timestamp - starttime}s vs {eta}s')
        assert not simulator.collisions
        neighbor, neighbor_dist = nearest_neighbor(sector, ship_driver)
        assert neighbor_dist >= goto_order.collision_margin

        if goto_order.is_complete():
            gamestate.quit()
        assert not goto_order.cannot_stop
        assert gamestate.timestamp - starttime < eta

    testui.tick = tick
    simulator.run()
    assert goto_order.is_complete()

@write_history
def test_gotolocation_with_deviating_starting_velocity(gamestate, generator, sector, testui, simulator):
    ship_driver = generator.spawn_ship(sector, 0, 15000, v=(0,0), w=0, theta=0)
    ship_driver.velocity = np.array((-4., -10.)) * 50.
    ship_driver.phys.velocity = tuple(ship_driver.velocity)

    goto_order = orders.GoToLocation(np.array((0.,0.)), ship_driver, gamestate)
    ship_driver.orders.append(goto_order)

    distance = np.linalg.norm(ship_driver.loc)
    eta = goto_order.eta()

    starttime = gamestate.timestamp
    def tick(timeout):
        logging.debug(f'{ship_driver.loc} {ship_driver.velocity} {ship_driver.angle} {gamestate.timestamp - starttime}s vs {eta}s')
        nonlocal distance
        assert not simulator.collisions

        neighbor, neighbor_dist = nearest_neighbor(sector, ship_driver)
        assert neighbor_dist >= goto_order.collision_margin

        if goto_order.is_complete():
            gamestate.quit()
        assert not goto_order.cannot_stop
        assert gamestate.timestamp - starttime < eta

        d = np.linalg.norm(ship_driver.loc)
        #assert d < distance or d < goto_order.arrival_distance
        distance = d

    testui.tick = tick
    simulator.run()
    assert goto_order.is_complete()
