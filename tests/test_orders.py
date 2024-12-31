""" Tests for ship orders and behaviors. """

import logging
import os

import pytest
import numpy as np

from stellarpunk import core, sim, generate, orders, util
from stellarpunk.orders import steering, collision
from . import write_history, nearest_neighbor

TESTDIR = os.path.dirname(__file__)

def test_goto_entity(gamestate, generator, sector):
    ship_driver = generator.spawn_ship(sector, -10000, 0, v=(0,0), w=0, theta=0)
    station = generator.spawn_station(sector, 0, 0, resource=0)

    arrival_distance = 1.5e3
    collision_margin = 1e3
    goto_order = orders.GoToLocation.goto_entity(station, ship_driver, gamestate, arrival_distance - station.radius, collision_margin)

    assert np.linalg.norm(station.loc - goto_order._target_location)+goto_order.arrival_distance <= arrival_distance + steering.VELOCITY_EPS
    assert np.linalg.norm(station.loc - goto_order._target_location)-station.radius-goto_order.arrival_distance >= collision_margin - steering.VELOCITY_EPS
    assert goto_order.arrival_distance >= (arrival_distance*0.1)/2 - steering.VELOCITY_EPS
    assert goto_order.min_distance == 0.

def test_compute_eta(generator, sector):
    ship = generator.spawn_ship(sector, -400, 15000, v=(0,0), w=-2, theta=0)
    assert orders.GoToLocation.compute_eta(ship, np.array((10000,0)), 2.0) > 0
    assert orders.GoToLocation.compute_eta(ship, np.array((100000,0)), 2.0) > 0
    assert orders.GoToLocation.compute_eta(ship, np.array((1000000,0)), 2.0) > 0

@write_history
def test_zero_rotation_time(gamestate, generator, sector, testui, simulator):
    """ Tests rotation starting with zero angular velocity """
    ship_driver = generator.spawn_ship(sector, -400, 15000, v=(0,0), w=0, theta=-np.pi/4)

    rotate_order = orders.RotateOrder.create_rotate_order(np.pi, ship_driver, gamestate)
    ship_driver.prepend_order(rotate_order)

    eta = collision.rotation_time(np.pi, 0, ship_driver.max_angular_acceleration())

    testui.eta = eta
    testui.orders = [rotate_order]

    simulator.run()
    assert rotate_order.is_complete()
    assert ship_driver.angular_velocity == 0
    assert util.isclose(ship_driver.angle, np.pi)

@write_history
def test_non_zero_rotation_time(gamestate, generator, sector, testui, simulator):
    """ Tests rotation starting with non zero angular velocity """
    ship_driver = generator.spawn_ship(sector, -400, 15000, v=(0,0), w=-2, theta=0)

    rotate_order = orders.RotateOrder.create_rotate_order(np.pi/2, ship_driver, gamestate)
    ship_driver.prepend_order(rotate_order)

    eta = collision.rotation_time(rotate_order.target_angle, ship_driver.angular_velocity, ship_driver.max_angular_acceleration())

    testui.eta = eta*1.15
    testui.orders = [rotate_order]

    simulator.run()
    assert rotate_order.is_complete()
    assert ship_driver.angular_velocity == 0
    assert util.isclose(ship_driver.angle, rotate_order.target_angle)

    # make sure our eta estimate is within 15% of the estimate after backing
    # out the safety margin
    assert np.isclose(gamestate.timestamp, eta, rtol=0.15)

@write_history
def test_basic_gotolocation(gamestate, generator, sector, testui, simulator):
    ship_driver = generator.spawn_ship(sector, -400, 15000, v=(0,0), w=0, theta=0)

    goto_order = orders.GoToLocation.create_go_to_location(np.array((0.,0.)), ship_driver, gamestate)
    ship_driver.prepend_order(goto_order)

    distance = np.linalg.norm(ship_driver.loc)
    eta = goto_order.estimate_eta()
    starttime = gamestate.timestamp
    def tick(timeout, dt):
        logging.debug(f'{ship_driver.loc} {ship_driver.velocity} {ship_driver.angle} {gamestate.timestamp - starttime}s vs {eta}s')
        assert not testui.collisions
        neighbor, neighbor_dist = nearest_neighbor(sector, ship_driver)
        assert neighbor_dist >= goto_order.collision_margin

        if goto_order.is_complete():
            simulator.quit()
        assert not goto_order.cannot_stop
        assert gamestate.timestamp - starttime < eta

    testui.tick = tick
    simulator.run()
    assert goto_order.is_complete()

@write_history
def test_gotolocation_with_entity_target(gamestate, generator, sector, testui, simulator):
    ship_driver = generator.spawn_ship(sector, -400, 15000, v=(0,0), w=0, theta=0)
    ship_blocker = generator.spawn_ship(sector, 0, 0, v=(0,0), w=0, theta=0)

    goto_order = orders.GoToLocation.create_go_to_location(np.array((0.,0.)), ship_driver, gamestate)
    ship_driver.prepend_order(goto_order)

    distance = np.linalg.norm(ship_driver.loc)
    eta = goto_order.estimate_eta()

    starttime = gamestate.timestamp
    def tick(timeout, dt):
        logging.debug(f'{ship_driver.loc} {ship_driver.velocity} {ship_driver.angle} {gamestate.timestamp - starttime}s vs {eta}s')
        assert not testui.collisions
        neighbor, neighbor_dist = nearest_neighbor(sector, ship_driver)
        assert neighbor_dist >= goto_order.collision_margin

        if goto_order.is_complete():
            simulator.quit()
        assert not goto_order.cannot_stop
        assert not goto_order.cannot_avoid_collision
        assert gamestate.timestamp - starttime < eta

    testui.tick = tick
    simulator.run()
    assert goto_order.is_complete()

@write_history
def test_gotolocation_with_sympathetic_starting_velocity(gamestate, generator, sector, testui, simulator):
    ship_driver = generator.spawn_ship(sector, -400, 15000, v=(0,0), w=0, theta=0)
    ship_driver.set_velocity(np.array((0., -10.)) * 50.)

    goto_order = orders.GoToLocation.create_go_to_location(np.array((0.,0.)), ship_driver, gamestate)
    ship_driver.prepend_order(goto_order)

    distance = np.linalg.norm(ship_driver.loc)
    eta = goto_order.estimate_eta()
    starttime = gamestate.timestamp
    def tick(timeout, dt):
        logging.debug(f'{ship_driver.loc} {ship_driver.velocity} {ship_driver.angle} {gamestate.timestamp - starttime}s vs {eta}s')
        assert not testui.collisions
        neighbor, neighbor_dist = nearest_neighbor(sector, ship_driver)
        assert neighbor_dist >= goto_order.collision_margin

        if goto_order.is_complete():
            simulator.quit()
        assert not goto_order.cannot_stop
        assert gamestate.timestamp - starttime < eta

    testui.tick = tick
    simulator.run()
    assert goto_order.is_complete()

@write_history
def test_gotolocation_with_deviating_starting_velocity(gamestate, generator, sector, testui, simulator):
    ship_driver = generator.spawn_ship(sector, 0, 15000, v=(0,0), w=0, theta=0)
    ship_driver.set_velocity(np.array((-4., -10.)) * 50.)

    goto_order = orders.GoToLocation.create_go_to_location(np.array((0.,0.)), ship_driver, gamestate)
    ship_driver.prepend_order(goto_order)

    distance = np.linalg.norm(ship_driver.loc)
    eta = goto_order.estimate_eta()

    starttime = gamestate.timestamp
    def tick(timeout, dt):
        logging.debug(f'{ship_driver.loc} {ship_driver.velocity} {ship_driver.angle} {gamestate.timestamp - starttime}s vs {eta}s')
        nonlocal distance
        assert not testui.collisions

        neighbor, neighbor_dist = nearest_neighbor(sector, ship_driver)
        assert neighbor_dist >= goto_order.collision_margin

        if goto_order.is_complete():
            simulator.quit()
        assert not goto_order.cannot_stop
        assert gamestate.timestamp - starttime < eta

        d = np.linalg.norm(ship_driver.loc)
        #assert d < distance or d < goto_order.arrival_distance
        distance = d

    testui.tick = tick
    simulator.run()
    assert goto_order.is_complete()

@write_history
def test_disembark_skip_disembark(gamestate, generator, sector, testui, simulator):
    # ship starts near nothing, go to an entity
    ship_driver = generator.spawn_ship(sector, -10000, 0, v=(0,0), w=0, theta=0)
    blocker = generator.spawn_station(sector, 0, 0, resource=0)

    disembark_order = orders.DisembarkToEntity.disembark_to(blocker, ship_driver, gamestate)
    ship_driver.prepend_order(disembark_order)

    testui.orders = [disembark_order]
    testui.margin_neighbors = [ship_driver]

    simulator.run()
    assert disembark_order.is_complete()
    assert np.linalg.norm(blocker.loc - ship_driver.loc) < 2.3e3
    assert disembark_order in testui.complete_orders

@write_history
def test_basic_disembark(gamestate, generator, sector, testui, simulator):
    # ship starts near an entity, go to another
    ship_driver = generator.spawn_ship(sector, -10000, 0, v=(0,0), w=0, theta=0)
    start_blocker = generator.spawn_station(sector, -8500, -300, resource=0)
    blocker = generator.spawn_station(sector, 5000, 0, resource=0)

    disembark_order = orders.DisembarkToEntity.disembark_to(blocker, ship_driver, gamestate)
    ship_driver.prepend_order(disembark_order)

    testui.orders = [disembark_order]
    testui.margin_neighbors = [ship_driver]

    simulator.run()
    assert disembark_order.is_complete()
    assert np.linalg.norm(blocker.loc - ship_driver.loc) < 2.3e3
    assert disembark_order.disembark_from == start_blocker
    assert disembark_order.embark_to == blocker

@write_history
def test_basic_mining_order(gamestate, generator, sector, testui, simulator):
    # ship and asteroid
    resource = 0
    ship = generator.spawn_ship(sector, -3000, 0, v=(0,0), w=0, theta=0)
    asteroid = generator.spawn_asteroid(sector, 0, 0, resource, 5e2)
    # ship mines the asteroid
    mining_order = orders.MineOrder.create_mine_order(asteroid, 3.5e2, ship, gamestate)
    ship.prepend_order(mining_order)

    testui.orders = [mining_order]
    testui.margin_neighbors = [ship]

    assert ship.cargo[0] == 0.
    assert np.isclose(asteroid.cargo[asteroid.resource], 5e2)

    simulator.run()
    assert mining_order.is_complete()

    # make sure ship ends up near enough to the asteroid
    assert np.linalg.norm(ship.loc - asteroid.loc) < 2e3 + asteroid.radius + steering.VELOCITY_EPS
    # make sure we got the resources
    assert np.isclose(ship.cargo[0], 3.5e2)
    # make sure asteroid lost the resources
    assert np.isclose(asteroid.cargo[asteroid.resource], 5e2 - 3.5e2)

@write_history
def test_over_mine(gamestate, generator, sector, testui, simulator):
    # ship and asteroid
    resource = 0
    ship = generator.spawn_ship(sector, -3000, 0, v=(0,0), w=0, theta=0)
    asteroid = generator.spawn_asteroid(sector, 0, 0, resource, 2.5e2)
    # ship mines the asteroid
    mining_order = orders.MineOrder.create_mine_order(asteroid, 3.5e2, ship, gamestate)
    ship.prepend_order(mining_order)

    testui.orders = [mining_order]
    testui.margin_neighbors = [ship]

    assert ship.cargo[0] == 0.
    assert np.isclose(asteroid.cargo[asteroid.resource], 2.5e2)

    simulator.run()
    assert mining_order.is_complete()

    # make sure ship ends up near enough to the asteroid
    assert np.linalg.norm(ship.loc - asteroid.loc) < 2e3 + asteroid.radius + steering.VELOCITY_EPS
    # make sure we got the resources
    assert np.isclose(ship.cargo[0], 2.5e2)
    # make sure asteroid lost the resources
    assert np.isclose(asteroid.cargo[asteroid.resource], 0)

@write_history
def test_basic_transfer_order(gamestate, generator, sector, testui, simulator):
    # two ships
    resource = 0
    ship_a = generator.spawn_ship(sector, -3000, 0, v=(0,0), w=0, theta=0)
    ship_b = generator.spawn_ship(sector, 0, 0, v=(0,0), w=0, theta=0)
    ship_a.cargo[0] = 5e2

    # ship mines the asteroid
    transfer_order = orders.TransferCargo.create_transfer_cargo(ship_b, 0, 3.5e2, ship_a, gamestate)
    ship_a.prepend_order(transfer_order)

    testui.orders = [transfer_order]
    testui.margin_neighbors = [ship_a]

    assert ship_a.cargo[0] == 5e2
    assert ship_b.cargo[0] == 0.

    simulator.run()
    assert transfer_order.is_complete()

    # make sure ship ends up near enough to the asteroid
    assert np.linalg.norm(ship_a.loc - ship_b.loc) < 2e3 + steering.VELOCITY_EPS
    # make sure we transferred cargo
    assert np.isclose(ship_b.cargo[0], 3.5e2)
    assert np.isclose(ship_a.cargo[0], 5e2 - 3.5e2)

@write_history
def test_over_transfer(gamestate, generator, sector, testui, simulator):
    # two ships
    resource = 0
    ship_a = generator.spawn_ship(sector, -3000, 0, v=(0,0), w=0, theta=0)
    ship_b = generator.spawn_ship(sector, 0, 0, v=(0,0), w=0, theta=0)
    ship_a.cargo[0] = 2.5e2

    transfer_order = orders.TransferCargo.create_transfer_cargo(ship_b, 0, 3.5e2, ship_a, gamestate)
    ship_a.prepend_order(transfer_order)

    testui.orders = [transfer_order]
    testui.margin_neighbors = [ship_a]

    assert ship_a.cargo[0] == 2.5e2
    assert ship_b.cargo[0] == 0.

    simulator.run()
    assert transfer_order.is_complete()

    # make sure ship ends up near enough to the asteroid
    assert np.linalg.norm(ship_a.loc - ship_b.loc) < 2e3 + steering.VELOCITY_EPS
    # make sure we transferred cargo
    assert np.isclose(ship_b.cargo[0], 2.5e2)
    assert np.isclose(ship_a.cargo[0], 0)

"""
@write_history
def test_basic_harvest(gamestate, generator, sector, testui, simulator):
    # two ships
    resource = 0
    ship_a = generator.spawn_ship(sector, -3000, 0, v=(0,0), w=0, theta=0)
    ship_a.cargo_capacity = 5e2
    ship_b = generator.spawn_ship(sector, 0, 0, v=(0,0), w=0, theta=0)
    ship_b.cargo_capacity = 5e3

    sector.radius = 3000.

    asteroid = generator.spawn_asteroid(sector, 0, 5000, resource, 12.5e2)

    harvest_order = orders.HarvestOrder(ship_b, 0, ship_a, gamestate, max_trips=2)
    ship_a.orders.append(harvest_order)

    testui.orders = [harvest_order]
    testui.margin_neighbors = [ship_a]
    testui.eta = 200

    assert ship_a.cargo[0] == 0.
    assert ship_b.cargo[0] == 0.
    assert asteroid.cargo[asteroid.resource] == 12.5e2

    simulator.run()
    assert harvest_order.init_eta < 200
    assert harvest_order.is_complete()
    assert harvest_order.trips == 2

    # make sure we transferred cargo
    assert np.isclose(ship_b.cargo[0], 10e2)
    assert np.isclose(ship_a.cargo[0], 0)
    assert np.isclose(asteroid.cargo[asteroid.resource], 12.5e2 - 10e2)
"""

def test_docking_order_compute_eta(generator, sector):
    ship_driver = generator.spawn_ship(sector, -10000, 0, v=(0,0), w=0, theta=0)
    station = generator.spawn_station(sector, 0, 0, resource=0)

    computed_eta = orders.DockingOrder.compute_eta(ship_driver, station)

    assert computed_eta < np.inf
    assert computed_eta < orders.GoToLocation.compute_eta(ship_driver, station.loc) + 20

@write_history
def test_docking_order(gamestate, generator, sector, testui, simulator):
    ship_driver = generator.spawn_ship(sector, -10000, 0, v=(0,0), w=0, theta=0)
    station = generator.spawn_station(sector, 0, 0, resource=0)

    arrival_distance = 1.5e3
    goto_order = orders.DockingOrder.create_docking_order(station, ship_driver, gamestate, surface_distance=arrival_distance - station.radius)


    ship_driver.prepend_order(goto_order)
    testui.orders = [goto_order]

    simulator.run()
    assert goto_order.is_complete()
    distance = util.distance(ship_driver.loc, station.loc)
    assert distance < arrival_distance + station.radius
    assert distance > 7e2 + station.radius + ship_driver.radius

    assert all(np.isclose(ship_driver.velocity, np.array((0., 0.))))
