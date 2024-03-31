from typing import Optional

from stellarpunk.core import combat

def test_missile_attack(gamestate, generator, sector, testui, simulator):
    ship = generator.spawn_ship(sector, -3000, 0, v=(0,0), w=0, theta=0)
    target = generator.spawn_ship(sector, 0, 0, v=(0,0), w=0, theta=0)


    missile:Optional[combat.Missile] = combat.MissileOrder.spawn_missile(ship, target, gamestate)
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
    import weakref
    target_ref = weakref.ref(target)
    missile_ref = weakref.ref(missile)
    missile_order_ref = weakref.ref(missile_order)
    attack_order_ref = weakref.ref(attack_order)

    target = None
    missile = None
    # missile and attack orders still hold on to these
    assert target_ref() is not None
    assert missile_ref() is not None

    missile_order = None
    assert target_ref() is not None
    assert missile_ref() is None
    attack_order = None

    assert target_ref() is None
    assert missile_order_ref() is None
    assert attack_order_ref() is None

