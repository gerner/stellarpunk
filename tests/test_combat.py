import logging
import weakref
import collections
import functools
from typing import Optional, MutableMapping

import numpy as np

from stellarpunk import core, agenda, util
from stellarpunk.orders import movement
from stellarpunk.core import combat

def test_compute_thrust(gamestate, generator, sector):
    ship = generator.spawn_ship(sector, -3000, 0, v=(0,0), w=0, theta=0, initial_sensor_power_ratio=0.0, initial_transponder=False)

    # note, there's numerical instability as we approach 11.2 for this distance
    # and these coefficients
    target_profile = 11.3
    t = sector.sensor_manager.compute_thrust_for_profile(ship, (3e5)**2, target_profile)

    ship.sensor_settings.set_thrust(t)
    assert target_profile == sector.sensor_manager.compute_target_profile(ship, 3e5**2)

def test_missile_attack(gamestate, generator, sector, testui, simulator):
    ship = generator.spawn_ship(sector, -3000, 0, v=(0,0), w=0, theta=0, initial_sensor_power_ratio=0.0, initial_transponder=False)
    target = generator.spawn_ship(sector, 0, 0, v=(0,0), w=0, theta=0, initial_sensor_power_ratio=0.0, initial_transponder=False)


    missile:Optional[combat.Missile] = combat.MissileOrder.spawn_missile(ship, gamestate, sector.sensor_manager.target(target, ship))
    assert missile
    missile_order = missile.current_order()
    assert missile_order
    attack_order:Optional[combat.AttackOrder] = combat.AttackOrder.create_attack_order(sector.sensor_manager.target(target, ship), ship, gamestate)
    assert attack_order
    ship.prepend_order(attack_order)

    testui.orders = [attack_order, missile_order]
    testui.eta = 30

    simulator.run()
    assert attack_order.is_complete()
    assert missile_order.is_complete()

    # target got destroyed
    assert target.entity_id not in sector.entities
    assert target.entity_id not in gamestate.entities
    assert target.sector is None

    # missile got destroyed
    assert missile.entity_id not in sector.entities
    assert missile.entity_id not in gamestate.entities
    assert missile.sector is None

    # make sure we're cleaning things up properly and don't have any dangling
    # references like observer registrations or entity tracking, etc.
    testui.orders.clear()
    testui.complete_orders.clear()
    target_ref = weakref.ref(target)
    missile_ref = weakref.ref(missile)
    missile_order_ref = weakref.ref(missile_order)
    attack_order_ref = weakref.ref(attack_order)

    target = None
    assert target_ref() is None
    missile = None
    # missile order still holds on to this
    assert missile_ref() is not None

    missile_order = None
    assert missile_order_ref() is None
    assert missile_ref() is None

    attack_order = None
    assert missile_order_ref() is None
    assert attack_order_ref() is None

def test_two_missiles(gamestate, generator, sector, testui, simulator):
    missile1 = generator.spawn_missile(sector, -3000, 0, v=(0,0), w=0, theta=0)
    missile2 = generator.spawn_missile(sector, 3000, 0, v=(0,0), w=0, theta=0)

    missile_order1 = combat.MissileOrder.create_missile_order(sector.sensor_manager.target(missile2, missile1), missile1, gamestate)
    missile1.prepend_order(missile_order1)
    missile_order2 = combat.MissileOrder.create_missile_order(sector.sensor_manager.target(missile1, missile2), missile2, gamestate)
    missile2.prepend_order(missile_order2)

    testui.orders = [missile_order1, missile_order2]
    testui.eta = 20

    simulator.run()
    assert missile_order1.is_complete()
    assert missile_order2.is_complete()
    assert missile1.sector is None
    assert missile2.sector is None

def test_attack_and_defend(gamestate, generator, sector, testui, simulator):
    #TODO: this test can be flaky and sometimes keeps running.
    #  what is non-determinitistic about it???

    # simulates an attack run by a single ship on another single ship
    attacker = generator.spawn_ship(sector, -300000, 0, v=(0,0), w=0, theta=0, initial_sensor_power_ratio=1.0, initial_transponder=False)
    defender = generator.spawn_ship(sector, 0, 0, v=(0,0), w=0, theta=0, initial_sensor_power_ratio=0.0, initial_transponder=False)

    defender_owner = generator.spawn_character(defender)
    defender_owner.take_ownership(defender)
    defender_owner.add_agendum(agenda.CaptainAgendum.create_eoa(defender, defender_owner, gamestate))

    attack_order = combat.AttackOrder.create_attack_order(sector.sensor_manager.target(defender, attacker), attacker, gamestate, max_missiles=15)
    attacker.prepend_order(attack_order)

    testui.orders = [attack_order]
    testui.eta = 3600
    testui.collisions_allowed = True

    state_ticks:MutableMapping[combat.AttackOrder.State, int] = collections.defaultdict(int)
    attack_ticks = 0
    age_sum = 0.
    a_zero_forces = 0
    a_non_zero_forces = 0
    d_zero_forces = 0
    d_non_zero_forces = 0
    evade_max_thrust_sum = 0.
    ticks_evading = 0
    ticks_fleeing = 0
    distinct_flee_orders = 0
    dist_sum = 0
    max_thrust_sum = 0.

    last_loc = defender.loc
    last_force = np.array(defender.phys.force)
    last_velocity = defender.velocity

    def tick_callback():
        nonlocal attacker, state_ticks, a_zero_forces, a_non_zero_forces, d_zero_forces, d_non_zero_forces, age_sum, attack_ticks
        nonlocal defender, ticks_fleeing, distinct_flee_orders, ticks_evading, evade_max_thrust_sum, dist_sum, max_thrust_sum
        nonlocal last_loc, last_force, last_velocity
        if util.distance(last_loc, defender.loc) > max(np.linalg.norm(defender.velocity)*simulator.dt*3.0, 1.):
            raise Exception()
        if util.magnitude(*defender.velocity) > 15000*1.5:
            raise Exception()
        last_loc = defender.loc
        last_velocity = defender.velocity
        last_force = np.array(defender.phys.force)
        assert attacker.sector
        if not attack_order.is_complete():
            state_ticks[attack_order.state] += 1
            age_sum += attack_order.target.age
            attack_ticks += 1
        if defender.phys.force.length == 0.:
            d_zero_forces += 1
        else:
            d_non_zero_forces += 1
        if attacker.phys.force.length == 0.:
            a_zero_forces += 1
        else:
            a_non_zero_forces += 1
        defender_top_order = defender.top_order()
        if isinstance(defender_top_order, combat.FleeOrder):
            ticks_fleeing += 1
            dist_sum += util.distance(defender.loc, attacker.loc)
            max_thrust_sum += defender_top_order.max_thrust
            if defender_top_order not in testui.orders:
                testui.add_order(defender_top_order)
                distinct_flee_orders += 1
        defender_current_order = defender.current_order()
        if isinstance(defender_current_order, movement.EvadeOrder):
            ticks_evading += 1
            evade_max_thrust_sum += defender_current_order.max_thrust

        if attack_order.is_complete():
            simulator.raise_breakpoint()
    testui.tick_callback = tick_callback

    simulator.notify_on_collision = True
    simulator.run()

    # either the defender is destroyed or we had to give up
    assert not attack_order.target.is_active() or attack_order.state == combat.AttackOrder.State.GIVEUP

    assert attacker not in set(functools.reduce(lambda x, y: x + [y[0], y[1]], testui.collisions, list()))

    # should only have the attack order and a flee order
    assert len(testui.orders) == 2
    flee_order = testui.orders[1]
    assert isinstance(flee_order, combat.FleeOrder)

    logging.info(f'{attacker.sector=} {defender.sector=} in {gamestate.timestamp}s')
    logging.info(f'target avg age: {age_sum/attack_ticks}s avg dist: {util.human_distance(dist_sum/ticks_fleeing)}')
    logging.info(f'missiles fired: {attack_order.missiles_fired}')
    logging.info(f'target active: {attack_order.target.is_active()}')
    logging.info(f'threats destroyed: {flee_order.point_defense.targets_destroyed}')

def test_discover_missing(gamestate, generator, sector, testui, simulator):
    """ tests that we eventually discover a target is inactive

    Even if we do not witness the target being destroyed (or migrating), if we
    get close enough we should detect that they are missing and mark the image
    as such. And that should end the attack order. """
    attacker = generator.spawn_ship(sector, -300000, 0, v=(0,0), w=0, theta=0, initial_sensor_power_ratio=1.0, initial_transponder=False)
    defender = generator.spawn_ship(sector, 0, 0, v=(0,0), w=0, theta=0, initial_sensor_power_ratio=0.0, initial_transponder=False)

    defender_owner = generator.spawn_character(defender)
    defender_owner.take_ownership(defender)
    defender_owner.add_agendum(agenda.CaptainAgendum.create_eoa(defender, defender_owner, gamestate))

    target_image = sector.sensor_manager.target(defender, attacker)
    attack_order = combat.AttackOrder.create_attack_order(target_image, attacker, gamestate, max_missiles=15)
    attacker.prepend_order(attack_order)

    testui.orders = [attack_order]
    testui.eta = 3600

    undetected_ticks = None
    stillactive_ticks = None
    inactive_ticks = None
    def tick_callback():
        nonlocal gamestate, sector, attacker, defender, target_image, undetected_ticks, stillactive_ticks, inactive_ticks
        if not undetected_ticks and not target_image.detected():
            undetected_ticks = gamestate.ticks
            gamestate.destroy_entity(defender)
            assert target_image.is_active()
        if target_image.is_active():
            stillactive_ticks = gamestate.ticks
        else:
            inactive_ticks = gamestate.ticks

    testui.tick_callback = tick_callback
    simulator.run()

    assert undetected_ticks
    assert stillactive_ticks
    assert inactive_ticks
    # make sure we still saw the target as active for a couple of ticks
    assert stillactive_ticks - undetected_ticks > 1
    assert inactive_ticks - stillactive_ticks > 1
    assert not target_image.is_active()
    assert target_image.inactive_reason == core.SensorImageInactiveReason.MISSING

