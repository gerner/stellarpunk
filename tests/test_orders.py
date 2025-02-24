""" Tests for ship orders and behaviors. """

import logging
import math
import os

import pytest
import numpy as np

from stellarpunk import core, sim, generate, orders, util, intel
from stellarpunk.core import sector_entity
from stellarpunk.orders import steering, collision
from . import write_history, nearest_neighbor, add_sector_intel

TESTDIR = os.path.dirname(__file__)

def test_goto_entity(gamestate, generator, sector):
    ship_driver = generator.spawn_ship(sector, -10000, 0, v=(0,0), w=0, theta=0)
    station = generator.spawn_station(sector, 0, 0, resource=0)

    arrival_distance = 1.5e3
    collision_margin = 1e3
    station_image = sector.sensor_manager.target(station, ship_driver)
    goto_order = orders.GoToLocation.goto_entity(station_image, ship_driver, gamestate, arrival_distance - station.radius, collision_margin)

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

"""
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
"""

@write_history
def test_basic_mining_order(gamestate, generator, sector, testui, simulator):
    # ship and asteroid
    resource = 0
    ship = generator.spawn_ship(sector, -3000, 0, v=(0,0), w=0, theta=0)
    ship_owner = generator.spawn_character(ship)
    ship_owner.take_ownership(ship)
    ship.captain = ship_owner
    asteroid = generator.spawn_asteroid(sector, 0, 0, resource, 5e2)
    add_sector_intel(ship, sector, ship_owner, gamestate)
    asteroid_intel = ship_owner.intel_manager.get_intel(intel.EntityIntelMatchCriteria(asteroid.entity_id), intel.AsteroidIntel)
    assert asteroid_intel
    # ship mines the asteroid
    mining_order = orders.MineOrder.create_mine_order(asteroid_intel, 3.5e2, ship, gamestate)
    ship.prepend_order(mining_order)

    testui.orders = [mining_order]
    testui.margin_neighbors = [ship]

    assert ship.cargo[0] == 0.
    assert np.isclose(asteroid.cargo[asteroid.resource], 5e2)

    simulator.run()

    # save/load support
    gamestate = testui.gamestate
    mining_order = testui.orders[0]
    ship, asteroid = gamestate.recover_objects((ship, asteroid))

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
    ship_owner = generator.spawn_character(ship)
    ship_owner.take_ownership(ship)
    ship.captain = ship_owner
    asteroid = generator.spawn_asteroid(sector, 0, 0, resource, 2.5e2)
    add_sector_intel(ship, sector, ship_owner, gamestate)
    asteroid_intel = ship_owner.intel_manager.get_intel(intel.EntityIntelMatchCriteria(asteroid.entity_id), intel.AsteroidIntel)
    assert asteroid_intel
    # ship mines the asteroid
    mining_order = orders.MineOrder.create_mine_order(asteroid_intel, 3.5e2, ship, gamestate)
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

    ship_owner = generator.spawn_character(ship_a)
    ship_owner.take_ownership(ship_a)
    ship_a.captain = ship_owner
    add_sector_intel(ship_a, sector, ship_owner, gamestate)

    ship_b_intel = ship_owner.intel_manager.get_intel(intel.EntityIntelMatchCriteria(ship_b.entity_id), intel.SectorEntityIntel)
    transfer_order = orders.TransferCargo.create_transfer_cargo(ship_b_intel, 0, 3.5e2, ship_a, gamestate)
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

    ship_owner = generator.spawn_character(ship_a)
    ship_owner.take_ownership(ship_a)
    ship_a.captain = ship_owner
    add_sector_intel(ship_a, sector, ship_owner, gamestate)

    ship_b_intel = ship_owner.intel_manager.get_intel(intel.EntityIntelMatchCriteria(ship_b.entity_id), intel.SectorEntityIntel)
    transfer_order = orders.TransferCargo.create_transfer_cargo(ship_b_intel, 0, 3.5e2, ship_a, gamestate)
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

    computed_eta = orders.DockingOrder.compute_eta(ship_driver, station.loc)

    assert computed_eta < np.inf
    assert computed_eta < orders.GoToLocation.compute_eta(ship_driver, station.loc) + 20

@write_history
def test_docking_order(gamestate, generator, sector, testui, simulator):
    ship_driver = generator.spawn_ship(sector, -10000, 0, v=(0,0), w=0, theta=0)
    station = generator.spawn_station(sector, 0, 0, resource=0)

    station_image = sector.sensor_manager.target(station, ship_driver)
    arrival_distance = 1.5e3
    goto_order = orders.DockingOrder.create_docking_order(ship_driver, gamestate, target_image=station_image, surface_distance=arrival_distance - station.radius)


    ship_driver.prepend_order(goto_order)
    testui.orders = [goto_order]

    simulator.run()

    # save/load support
    gamestate = testui.gamestate
    goto_order = testui.orders[0]
    ship_driver, station = gamestate.recover_objects((ship_driver, station))

    assert goto_order.is_complete()
    distance = util.distance(ship_driver.loc, station.loc)
    assert distance < arrival_distance + station.radius
    assert distance > 7e2 + station.radius + ship_driver.radius

    assert all(np.isclose(ship_driver.velocity, np.array((0., 0.))))

def test_travel_through_gate(gamestate, generator, sector, connecting_sector, testui, simulator):
    ship_driver = generator.spawn_ship(sector, -10000, 0, v=(0,0), w=0, theta=0)
    ship_owner = generator.spawn_character(ship_driver)
    ship_owner.take_ownership(ship_driver)
    ship_driver.captain = ship_owner
    add_sector_intel(ship_driver, sector, ship_owner, gamestate)
    gate_intel = ship_owner.intel_manager.get_intel(intel.SectorEntityPartialCriteria(sector_id=sector.entity_id, cls=sector_entity.TravelGate), intel.TravelGateIntel)

    assert gate_intel.intel_entity_id == next(sector.entities_by_type(sector_entity.TravelGate)).entity_id

    order = orders.TravelThroughGate.create_travel_through_gate(gate_intel, ship_driver, gamestate)
    ship_driver.prepend_order(order)
    assert order.estimate_eta() < 200.0

    testui.eta = order.estimate_eta()*1.1
    testui.orders = [order]
    simulator.run()

    # some support to use this with save/load
    gamestate = testui.gamestate
    order = testui.orders[0]
    ship_driver, connecting_sector = gamestate.recover_objects((ship_driver, connecting_sector))

    assert order.is_complete()
    assert ship_driver.sector is not None and ship_driver.sector == connecting_sector
    assert util.isclose(util.magnitude(*ship_driver.velocity), 0.0)
    # we should be close to where the corresponding travel gate is
    # this depends on generation logic placing the gate here
    destination_gate = next(connecting_sector.entities_by_type(sector_entity.TravelGate))
    r, theta = util.cartesian_to_polar(*ship_driver.loc)
    assert abs(theta - destination_gate.direction) < math.radians(5.0)
    assert r > connecting_sector.radius*2.0+2e3
    assert r < connecting_sector.radius*2.5+2e3

    # the above parameters should guarantee this, although this depends on how
    # gates a re spawned in generate
    assert util.distance(destination_gate.loc, ship_driver.loc) < connecting_sector.radius*0.5

def test_simple_physics(gamestate, generator, sector, testui, simulator):
    ship_driver = generator.spawn_ship(sector, -10000, 0, v=(0,0), w=0, theta=0)

    # start out with some velocity
    ship_driver.phys.velocity = np.array([12458.33398438,     0.        ])

    # apply some force
    travel_thrust = 5000000.0
    ship_driver.apply_force((-travel_thrust, 0.0), True)
    #calculate where we should end up
    # s = u * t + 1/2 * a * t
    # f = m * a
    # a = f / m
    travel_time = 5.0
    expected_travel_distance = util.magnitude(*ship_driver.velocity) * travel_time + 0.5 * (-travel_thrust / ship_driver.mass) * travel_time * travel_time
    start_loc = np.array(ship_driver.loc)

    testui.max_timestamp=travel_time
    simulator.run()

    assert abs(expected_travel_distance - util.distance(ship_driver.loc, start_loc)) < 2e2

def test_location_explore_order(gamestate, generator, sector, testui, simulator):
    ship_driver = generator.spawn_ship(sector, -10000, 0, v=(0,0), w=0, theta=0)
    ship_owner = generator.spawn_character(ship_driver)
    ship_owner.take_ownership(ship_driver)
    ship_driver.captain = ship_owner

    target_loc = np.array((sector.hex_size*2.0, 0))
    order = orders.LocationExploreOrder.create_order(ship_driver, gamestate, sector.entity_id, target_loc)
    ship_driver.prepend_order(order)

    target_hex = util.axial_round(util.pixel_to_pointy_hex(target_loc, sector.hex_size))

    # make sure we don't have the hex intel before we start
    assert util.int_coords(util.axial_round(util.pixel_to_pointy_hex(ship_driver.loc, sector.hex_size))) != util.int_coords(target_hex)
    assert len(ship_owner.intel_manager.intel(intel.SectorHexPartialCriteria(sector_id=sector.entity_id, is_static=True, hex_loc=target_hex), intel.SectorHexIntel)) == 0

    testui.eta = order.estimate_eta()*1.1
    testui.orders = [order]
    simulator.run()

    # save/load support
    order = testui.orders[0]
    gamestate = testui.gamestate
    ship_driver, ship_owner, sector = gamestate.recover_objects((ship_driver, ship_owner, sector))

    assert order.is_complete()

    # make sure we end up in the appropriate hex, with hex intel about that hex
    assert util.int_coords(util.axial_round(util.pixel_to_pointy_hex(ship_driver.loc, sector.hex_size))) == util.int_coords(target_hex)
    assert len(ship_owner.intel_manager.intel(intel.SectorHexPartialCriteria(sector_id=sector.entity_id, is_static=True, hex_loc=target_hex), intel.SectorHexIntel)) == 1

def test_multi_sector_location_explore(gamestate, generator, sector, connecting_sector, testui, simulator):
    ship_driver = generator.spawn_ship(sector, -10000, 0, v=(0,0), w=0, theta=0)
    ship_owner = generator.spawn_character(ship_driver)
    ship_owner.take_ownership(ship_driver)
    ship_driver.captain = ship_owner

    add_sector_intel(ship_driver, sector, ship_owner, gamestate)

    target_loc = np.array((25000, 0))
    order = orders.LocationExploreOrder.create_order(ship_driver, gamestate, connecting_sector.entity_id, target_loc)
    ship_driver.prepend_order(order)

    target_hex = util.axial_round(util.pixel_to_pointy_hex(target_loc, connecting_sector.hex_size))

    # make sure we don't have the hex intel before we start
    assert ship_driver.sector.entity_id != connecting_sector.entity_id
    assert len(ship_owner.intel_manager.intel(intel.SectorHexPartialCriteria(sector_id=connecting_sector.entity_id, is_static=True, hex_loc=target_hex), intel.SectorHexIntel)) == 0

    testui.eta = order.estimate_eta()*1.1
    testui.orders = [order]
    simulator.run()

    # save/load support
    order = testui.orders[0]
    gamestate = testui.gamestate
    ship_driver, ship_owner, sector, connecting_sector = gamestate.recover_objects((ship_driver, ship_owner, sector, connecting_sector))

    assert order.is_complete()

    # make sure we end up in the appropriate hex, with hex intel about that hex
    assert ship_driver.sector.entity_id == connecting_sector.entity_id
    assert util.int_coords(util.axial_round(util.pixel_to_pointy_hex(ship_driver.loc, connecting_sector.hex_size))) == util.int_coords(target_hex)
    assert len(ship_owner.intel_manager.intel(intel.SectorHexPartialCriteria(sector_id=sector.entity_id, is_static=True, hex_loc=target_hex), intel.SectorHexIntel)) == 1

def test_navigate_order(gamestate, generator, sector, connecting_sector, testui, simulator):
    ship_driver = generator.spawn_ship(sector, -10000, 0, v=(0,0), w=0, theta=0)
    ship_owner = generator.spawn_character(ship_driver)
    ship_owner.take_ownership(ship_driver)
    ship_driver.captain = ship_owner

    add_sector_intel(ship_driver, sector, ship_owner, gamestate)

    order = orders.NavigateOrder.create_order(ship_driver, gamestate, connecting_sector.entity_id)
    ship_driver.prepend_order(order)

    assert ship_driver.sector.entity_id != connecting_sector.entity_id

    testui.eta = order.estimate_eta()*1.1
    testui.orders = [order]
    simulator.run()

    # save/load support
    order = testui.orders[0]
    gamestate = testui.gamestate
    ship_driver, ship_owner, sector, connecting_sector = gamestate.recover_objects((ship_driver, ship_owner, sector, connecting_sector))

    assert order.is_complete()
    assert ship_driver.sector.entity_id == connecting_sector.entity_id

