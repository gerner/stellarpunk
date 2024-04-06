import weakref
from typing import Optional

from stellarpunk.core import combat

def test_missile_attack(gamestate, generator, sector, testui, simulator):
    ship = generator.spawn_ship(sector, -3000, 0, v=(0,0), w=0, theta=0)
    target = generator.spawn_ship(sector, 0, 0, v=(0,0), w=0, theta=0)


    missile:Optional[combat.Missile] = combat.MissileOrder.spawn_missile(ship, gamestate, target=target)
    assert missile
    missile_order = missile.current_order()
    assert missile_order
    attack_order:Optional[combat.AttackOrder] = combat.AttackOrder(target, ship, gamestate)
    assert attack_order
    ship.prepend_order(attack_order)

    testui.orders = [attack_order]
    testui.eta = 20

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

    missile_order1 = combat.MissileOrder(missile1, gamestate, target=missile2)
    missile1.prepend_order(missile_order1)
    missile_order2 = combat.MissileOrder(missile2, gamestate, target=missile1)
    missile2.prepend_order(missile_order2)

    testui.orders = [missile_order1, missile_order2]
    testui.eta = 20

    simulator.run()
    assert missile_order1.is_complete()
    assert missile_order2.is_complete()
    assert missile1.sector is None
    assert missile2.sector is None
