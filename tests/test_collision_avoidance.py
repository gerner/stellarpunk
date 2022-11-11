""" Tests for ship orders and behaviors. """

import logging
import os
import uuid

import pytest
import numpy as np
import cymunk # type: ignore

from stellarpunk import core, sim, generate, orders, util
from stellarpunk.orders import steering, collision
from . import write_history, nearest_neighbor, ship_from_history, station_from_history, asteroid_from_history, order_from_history, history_from_file

TESTDIR = os.path.dirname(__file__)

def test_collision_dv_sanity():
    current_threat_loc = cymunk.Vec2d(0,0)
    threat_velocity = cymunk.Vec2d(0., 0.)
    loc = cymunk.Vec2d(-1e2, 1e2)
    velocity = cymunk.Vec2d(100., -100.)
    desired_margin = 10
    desired_direction = cymunk.Vec2d(-10.,10.)
    collision_cbdr = False
    delta_v_budget = 1000

    dv = collision.collision_dv(
            current_threat_loc, threat_velocity,
            loc, velocity,
            desired_margin,
            desired_direction - velocity,
            collision_cbdr, 2.,
            delta_v_budget)

    assert not any(np.isclose(dv, desired_direction-velocity))

    # sending back in the return from the prior call should give same divert
    dv2 = collision.collision_dv(
            current_threat_loc, threat_velocity,
            loc, velocity,
            desired_margin,
            dv,
            collision_cbdr, 2.,
            delta_v_budget)

    assert all(np.isclose(dv2, dv))

def test_collision_dv_solution_assert():
    current_threat_loc = cymunk.Vec2d(-198584.96875  , -106311.2421875)
    threat_velocity = cymunk.Vec2d( 11.43229866, -99.34436035)
    ship_loc = cymunk.Vec2d(-198324.96875, -106061.25   )
    ship_velocity = cymunk.Vec2d( 11.13045979, -99.37863159)
    desired_margin = 260.0
    desired_delta_velocity = cymunk.Vec2d( 7.59949604e-07, -2.16841866e-06)
    collision_cbdr = False
    cbdr_bias = -2.0
    delta_v_budget = 94181.78058053002

    delta_velocity = collision.collision_dv(
            current_threat_loc, threat_velocity,
            ship_loc, ship_velocity,
            desired_margin, desired_delta_velocity,
            collision_cbdr, cbdr_bias,
            delta_v_budget,
    )
    assert all(np.isclose(delta_velocity, cymunk.Vec2d(0.003617, 0.092102)))

    desired_delta_velocity = cymunk.Vec2d(15., 25.)
    delta_velocity = collision.collision_dv(
            current_threat_loc, threat_velocity,
            ship_loc, ship_velocity,
            desired_margin, desired_delta_velocity,
            collision_cbdr, cbdr_bias,
            delta_v_budget,
    )
    assert all(np.isclose(delta_velocity, cymunk.Vec2d(0.603678, 25.000000)))

@write_history
def test_basic_collision_avoidance(gamestate, generator, sector, testui, simulator, caplog):
    # set up a sector, including space
    # add two ships in an offset (relative to desired course) arrangement
    # travel along course and observe no collision

    ship_driver = generator.spawn_ship(sector, 0, 2400, v=(0,0), w=0, theta=0)
    ship_blocker = generator.spawn_ship(sector, -300, 1200, v=(0,0), w=0, theta=0)

    goto_order = orders.GoToLocation(np.array((0.,0.)), ship_driver, gamestate)
    ship_driver.prepend_order(goto_order)

    # d = v_i*t + 1/2 a * t**2
    # v_i = 0
    # d = 1/2 a * t**2
    # t = sqrt( 2*d/a )
    # expect path to be accelerate half-way, turn around, decelerate
    # this is approximate
    distance = np.linalg.norm(ship_driver.loc)

    #testui.eta = goto_order.estimate_eta()
    testui.orders = [goto_order]
    testui.cannot_stop_orders = [goto_order]
    testui.cannot_avoid_collision_orders = [goto_order]
    testui.margin_neighbors = [ship_driver]

    simulator.run()
    assert goto_order.is_complete()

@write_history
def test_head_on_static_collision_avoidance(gamestate, generator, sector, testui, simulator):
    ship_driver = generator.spawn_ship(sector, 0, 3500, v=(0,0), w=0, theta=0)
    ship_blocker = generator.spawn_ship(sector, 0, 1700, v=(0,0), w=0, theta=0)

    goto_order = orders.GoToLocation(np.array((0.,0.)), ship_driver, gamestate)
    ship_driver.prepend_order(goto_order)

    testui.eta = goto_order.estimate_eta() * 1.1
    testui.orders = [goto_order]
    testui.cannot_stop_orders = [goto_order]
    testui.cannot_avoid_collision_orders = [goto_order]
    testui.margin_neighbors = [ship_driver]

    simulator.run()
    assert goto_order.is_complete()

@write_history
def test_blocker_wall_collision_avoidance(gamestate, generator, sector, testui, simulator):
    """ Initial state is travelling along course west of a blocker, but ideal
    path is east. Do we travel to the east of the blocker, a smaller overall
    maneuver, even though it's a bigger maneuver to avoid the collision. """

    # set up a sector, including space
    # add two ships in an offset (relative to desired course) arrangement
    # travel along course and observe no collision

    ship_driver = generator.spawn_ship(sector, -400, 20000, v=(0.,0.), w=0., theta=0.)
    ship_driver.set_velocity(np.array((0., 0.)))

    goto_order = orders.GoToLocation(np.array((0.,0.)), ship_driver, gamestate)
    ship_driver.prepend_order(goto_order)

    # a "wall" of blockers to the left of our target
    generator.spawn_ship(sector, -300, 10000, v=(0,0), w=0, theta=0)
    generator.spawn_ship(sector, -300, 9000, v=(0,0), w=0, theta=0)
    generator.spawn_ship(sector, -300, 6000, v=(0,0), w=0, theta=0)
    generator.spawn_ship(sector, -300, 2000, v=(0,0), w=0, theta=0)
    generator.spawn_ship(sector, -300, 1000, v=(0,0), w=0, theta=0)

    distance = np.linalg.norm(ship_driver.loc)
    eta = goto_order.estimate_eta()*1.7

    testui.eta = eta
    testui.orders = [goto_order]
    testui.cannot_stop_orders = [goto_order]
    testui.cannot_avoid_collision_orders = [goto_order]
    testui.margin_neighbors = [ship_driver]

    simulator.run()
    assert goto_order.is_complete()

@write_history
def test_simple_ships_intersecting(gamestate, generator, sector, testui, simulator):
    ship_a = generator.spawn_ship(sector, -5000, 0, v=(0,0), w=0, theta=0)
    ship_b = generator.spawn_ship(sector, 0, -5000, v=(0,0), w=0, theta=np.pi/2)

    goto_a = orders.GoToLocation(np.array((5000.,0.)), ship_a, gamestate)
    ship_a.prepend_order(goto_a)
    goto_b = orders.GoToLocation(np.array((0.,5000.)), ship_b, gamestate)
    ship_b.prepend_order(goto_b)

    a_cbdr = False
    b_cbdr = False

    testui.eta = max((goto_a.estimate_eta(), goto_b.estimate_eta()))*1.3
    testui.orders = [goto_a, goto_b]
    testui.cannot_stop_orders = [goto_a, goto_b]
    testui.cannot_avoid_collision_orders = [goto_a, goto_b]
    testui.margin_neighbors = [ship_a]

    simulator.run()
    assert goto_a.is_complete()
    assert not goto_a.collision_cbdr
    assert goto_b.is_complete()
    assert not goto_b.collision_cbdr

    assert any(False if hist_entry.order_hist is None else hist_entry.order_hist.get("cbdr", False) for hist_entry in ship_a.history), "ship_a never detected CBDR"
    assert any(False if hist_entry.order_hist is None else hist_entry.order_hist.get("cbdr", False) for hist_entry in ship_b.history), "ship_b never detected CBDR"



@write_history
def test_headon_ships_intersecting(gamestate, generator, sector, testui, simulator):
    # got a weird CBDR issue when ship_a was lesser uuid, so let's force it to be the bigger one
    # this does suggest that there's some behavioral issues that come up
    # OTOH, CBDR should be completely symmetric in this case
    uuids = sorted([ uuid.uuid4(), uuid.uuid4()])
    ship_a = generator.spawn_ship(sector, -5000, 0, v=(0,0), w=0, theta=0, entity_id=uuids[1])
    ship_b = generator.spawn_ship(sector, 5000, 0, v=(0,0), w=0, theta=np.pi, entity_id=uuids[0])

    goto_a = orders.GoToLocation(np.array((10000.,0.)), ship_a, gamestate)
    ship_a.prepend_order(goto_a)
    goto_b = orders.GoToLocation(np.array((-10000.,0.)), ship_b, gamestate)
    ship_b.prepend_order(goto_b)

    eta = max(goto_a.estimate_eta(), goto_b.estimate_eta())

    a_had_cbdr = False
    b_had_cbdr = False
    def tick(timeout, dt):
        nonlocal goto_a, goto_b, a_had_cbdr, b_had_cbdr

        assert not testui.collisions

        # only need to check one, they are symmetric
        neighbor, neighbor_dist = nearest_neighbor(sector, ship_a)
        assert neighbor_dist >= goto_a.collision_margin

        if goto_a.is_complete() and goto_b.is_complete():
            gamestate.quit()
        assert not goto_a.cannot_stop
        #assert not goto_a.cannot_avoid_collision
        assert not goto_b.cannot_stop
        #assert not goto_b.cannot_avoid_collision

        assert gamestate.timestamp < eta

        #assert (goto_a.collision_threat is None) == (goto_b.collision_threat is None)
        #assert goto_a.collision_cbdr == goto_b.collision_cbdr

        a_had_cbdr = a_had_cbdr or goto_a.collision_cbdr
        b_had_cbdr = b_had_cbdr or goto_b.collision_cbdr


    testui.tick = tick
    simulator.run()
    assert a_had_cbdr
    assert b_had_cbdr
    assert goto_a.is_complete()
    assert not goto_a.collision_cbdr
    assert goto_b.is_complete()
    assert not goto_b.collision_cbdr

@pytest.mark.skip(reason="this test is pretty slow because it takes a while to reach the destinations. test_simple_ships_intersecting basically covers this scenario")
@write_history
def test_ships_intersecting_collision(gamestate, generator, sector, testui, simulator):
    # two ships headed on intersecting courses collide
    # testcase from gameplay logs

    a = {"eid": "bebefe43-24b3-4588-9b42-4f5504de5903", "ts": 9.833333333333401, "loc": [140973.20332888863, 37746.464152281136], "a": 3.697086234730296, "v": [-1517.033904995614, -872.3805171209457], "av": 0.016666666666666663, "o": {"o": "stellarpunk.orders.GoToLocation", "ct": "e5e92226-9c6b-4dd4-a9dd-8e90f4ce43f4", "ct_loc": [128406.45773998012, -4516.575837068859], "ct_ts": 9.816666666666734, "cac": False, "nnd": 44133.001729957134, "t_loc": [-132817.46981686977, -119690.80145835043], "cs": False, "ad":1.5e3, "md":1.35e3}}

    b = {"eid": "e5e92226-9c6b-4dd4-a9dd-8e90f4ce43f4", "ts": 9.833333333333401, "loc": [128406.45773998012, -4516.575837068859], "a": 2.0580617926511264, "v": [-795.4285804691448, 1491.5775925290025], "av": -0.016666666666666663, "o": {"o": "stellarpunk.orders.GoToLocation", "ct": "bebefe43-24b3-4588-9b42-4f5504de5903", "ct_loc": [140998.48779162427, 37761.003422847934], "ct_ts": 9.816666666666734, "cac": False, "nnd": 35079.805827675904, "t_loc": [18374.44894231548, 201848.06161807684], "cs": False, "ad":1.5e3, "md":1.35e3}}

    ship_a = ship_from_history(a, generator, sector)
    ship_b = ship_from_history(b, generator, sector)

    goto_a = order_from_history(a, ship_a, gamestate)
    goto_b = order_from_history(b, ship_b, gamestate)

    eta = max(goto_a.estimate_eta(), goto_b.estimate_eta())

    def tick(timeout, dt):
        assert not testui.collisions

        # only need to check one, they are symmetric
        neighbor, neighbor_dist = nearest_neighbor(sector, ship_a)
        assert neighbor_dist >= goto_a.collision_margin

        if goto_a.is_complete() and goto_b.is_complete():
            gamestate.quit()
        assert not goto_a.cannot_stop
        assert not goto_a.cannot_avoid_collision
        assert not goto_b.cannot_stop
        assert not goto_b.cannot_avoid_collision

        assert gamestate.timestamp < eta

    testui.tick = tick
    simulator.run()
    assert goto_a.is_complete()
    assert goto_b.is_complete()

@write_history
def test_ship_existing_velocity(gamestate, generator, sector, testui, simulator):
    # ship headed in one direciton, wants to go 90 deg to it, almost collides
    # with a distant object
    # testcase from gameplay logs

    ship_driver = generator.spawn_ship(sector, -61548.10777914036, -122932.75622689343, v=[130.58825256350576, -20.791840524660724], w=-0.4600420747138861, theta=-0.10231674372628569)
    ship_blocker = generator.spawn_station(sector, -45858.953065820686, -126065.49162802949, resource=0)

    goto_order = orders.GoToLocation(np.array([-61165.07884422924, -152496.78251442552]), ship_driver, gamestate)
    ship_driver.prepend_order(goto_order)

    distance = np.linalg.norm(ship_driver.loc)
    eta = goto_order.estimate_eta()

    def tick(timeout, dt):
        assert not testui.collisions

        neighbor, neighbor_dist = nearest_neighbor(sector, ship_driver)
        assert neighbor_dist >= goto_order.collision_margin
        if goto_order.is_complete():
            gamestate.quit()
        assert not goto_order.cannot_stop
        assert not goto_order.cannot_avoid_collision
        assert gamestate.timestamp < eta

    testui.tick = tick
    simulator.run()
    assert goto_order.is_complete()

@write_history
def test_collision_flapping(gamestate, generator, sector, testui, simulator):
    """ Illustrates "flapping" in collision detection between the target and
    the collision threat, which makes avoiding the collision very slow. """

    log_entry = {"eid": "54ac288f-f321-4a5d-b681-06304946c1c5", "ts": 24.316986544634826, "loc": [-33555.48438201977, 26908.30401095389], "a": -1.6033951624880438, "v": [-30.158483917339932, -196.17277081634103], "av": -0.303054933539972, "o": {"o": "stellarpunk.orders.GoToLocation", "ct": "5d23b4c8-7fcd-463c-b46e-bce1f5daf1ff", "ct_loc": [-40857.126658436646, -16386.73414552246], "ct_ts": 24.30031987796816, "cac": False, "cbdr": False, "nnd": 43909.734240760576, "t_loc": [-58968.88094427537, -50074.22099620187], "cs": False, "ad":1.5e3, "md":1.35e3}}

    ship_driver = ship_from_history(log_entry, generator, sector)
    blocker = generator.spawn_station(sector, -40857.126658436646, -16386.73414552246, resource=0)

    goto_order = order_from_history(log_entry, ship_driver, gamestate, load_ct=False)

    starttime = gamestate.timestamp
    distance = np.linalg.norm(ship_driver.loc - goto_order._target_location)
    eta = goto_order.estimate_eta()

    def tick(timeout, dt):
        assert not testui.collisions

        # only need to check one, they are symmetric
        neighbor, neighbor_dist = nearest_neighbor(sector, ship_driver)
        assert neighbor_dist >= goto_order.collision_margin

        if goto_order.is_complete():
            gamestate.quit()
        assert not goto_order.cannot_stop
        assert not goto_order.cannot_avoid_collision
        assert gamestate.timestamp - starttime < eta*1.1

    testui.tick = tick
    simulator.run()
    assert goto_order.is_complete()

@write_history
def test_double_threat(gamestate, generator, sector, testui, simulator):
    """ Illustrates two threats close together on opposite sides of the desired
    vector. Collision detection will potentially ignore one and steer into it
    while trying to avoid the other."""

    a = {"eid":"c99bf358-5ed8-45b6-94ae-c109de01fd6d","ts":173.88333333336385,"loc":[11238.533139689527,2642.1486435851307],"a":-4.462215755014741,"v":[8.673617379884035e-19,-1.734723475976807e-18],"av":-0.5601751626636616,"f":[-2171.128156091397,-4504.0207070824135],"t":900000,"o":{"o":"stellarpunk.orders.GoToLocation","ct":"230d82ca-b3a5-4d65-90c4-9ec4b561afa8","ct_loc":[10900.178689763814,1713.1145124524808],"ct_ts":173.88333333336385,"ct_dv":[9.1603417240449e-18,-4.415671492188985e-18],"cac":False,"cbdr":False,"nnd":988.7305753307772,"t_loc":[-88985.9230279687,-205274.1941428623],"t_v":[-434.22563121827943,-900.8041414164829],"cs":False, "ad":1.5e3, "md":1.35e3}}

    b = {"eid":"8bba20d1-2f53-40b7-bea1-a758a1447a77","ts":173.88333333336385,"loc":[11885.659312275971,-629.4315137146432],"a":-1.0588787644372917,"v":[-97.96382930038575,232.86584616015844],"av":-0.09934969312887348,"f":[-1938.8568302904862,4608.7779499164335],"t":900000,"o":{"o":"stellarpunk.orders.GoToLocation","nnd":2541.396061628005,"t_loc":[10900.178689763814,1713.1145124524808],"t_v":[-115.13388981445071,273.6801007557891],"cs":False, "ad":1.5e3, "md":1.35e3}}

    c = {"eid":"230d82ca-b3a5-4d65-90c4-9ec4b561afa8","ts":0,"loc":[10900.178689763814,1713.1145124524808],"a":0,"v":[0,0],"av":0,"f":0,"t":0,"o":None}

    ship_a = ship_from_history(a, generator, sector)
    ship_b = ship_from_history(b, generator, sector)
    station = station_from_history(c, generator, sector)

    goto_a = order_from_history(a, ship_a, gamestate, load_ct=False)
    goto_a._target_location = (ship_a.phys.position + (goto_a._target_location  - ship_a.phys.position)/25)
    goto_b = order_from_history(b, ship_b, gamestate, load_ct=False)

    eta = goto_a.estimate_eta()

    testui.eta = eta * 1.5
    testui.orders = [goto_a]
    testui.cannot_stop_orders = [goto_a]
    #testui.cannot_avoid_collision_orders = [goto_a, goto_b]
    testui.margin_neighbors = [ship_a]

    simulator.run()
    assert goto_a.is_complete()

@write_history
def test_ct_near_target(gamestate, generator, sector, testui, simulator):
    # This case caused a collision while running, but I think it was because of
    # changing dt, perhaps because of a mouse click. it doesn't repro in test.
    a = {"eid": "a06358ed-5d1c-4026-b978-c6d05b65b971", "ts": 73.23596008924422, "loc": [33817.46867325524, -2802.702863489674], "a": 0.6501890587823068, "v": [-1516.455517907544, -865.0005079951259], "av": 1.4302311626798327, "o": {"o": "stellarpunk.orders.GoToLocation", "nnd": 16950.413686796317, "t_loc": [19117.170259352486, -11241.763871430074], "t_v": [-1419.7763852577352, -815.0568917357202], "cs": False, "ad":1.5e3, "md":1.35e3}}
    b = {"eid": "30ece38b-26e7-470e-8791-9096b0a9fd33", "ts": 73.23596008924422, "loc": [15113.769651997736, -3413.9408398308624], "a": 7.021124149518508, "v": [1540.1197961995574, -830.7283085974165], "av": -0.6830200566843943, "o": {"o": "stellarpunk.orders.GoToLocation", "nnd": 8792.157291416223, "t_loc": [62361.39816239622, -27754.575109759873], "t_v": [1555.6927812270774, -801.4486698709783], "cs": False, "ad":1.5e3, "md":1.35e3}}
    c = {"eid": "41c3a7aa-6d60-420a-b89d-362792d74283", "ts": 73.23596008924422, "loc": [57385.92081958368, -25954.613633593606], "a": 2.821461155469132, "v": [1098.0665620846898, -397.24379669298503], "av": 0.03190852110518538, "o": {"o": "stellarpunk.orders.GoToLocation", "nnd": 5291.052457169889, "t_loc": [62361.39816239622, -27754.575109759873], "t_v": [1091.9167479758394, -395.0189994084582], "cs": False, "ad":1.5e3, "md":1.35e3}}


    ship_a = ship_from_history(a, generator, sector)
    ship_b = ship_from_history(b, generator, sector)
    ship_c = ship_from_history(c, generator, sector)

    goto_a = order_from_history(a, ship_a, gamestate)
    goto_b = order_from_history(b, ship_b, gamestate)
    goto_c = order_from_history(c, ship_c, gamestate)

    blocker = generator.spawn_station(sector, 19117.170259352486, -11241.763871430074, resource=0)

    eta = goto_a.estimate_eta()

    testui.eta = eta*1.2
    testui.orders = [goto_a]
    #testui.cannot_avoid_collision_orders = [goto_a]
    # the setup for this test predates the latest thrust settings, so it's very
    # hard for it to slow down in time, but it should avoid collisions
    #testui.cannot_stop_orders = [goto_a]
    testui.margin_neighbors = [ship_a]
    # this test starts off at a very high speed that we should not have in
    # practice, so we allow violating the margin
    testui.margin = 10

    simulator.run()
    assert goto_a.is_complete()

@write_history
def test_many_threats(gamestate, generator, sector, testui, simulator):
    a = {"eid": "c2066f5f-80b0-4972-be15-86731721d0ac", "ts": 181.00000000003618, "loc": [-156664.6196115718, 15103.774316939725], "a": 6.706346854557575, "v": [-566.413704366243, -427.2391616051364], "av": 0.22045526531291454, "f": [3991.7686372857047, 3010.943896252839], "t": 900000.0, "o": {"o": "stellarpunk.orders.GoToLocation", "nnd": np.inf, "t_loc": [-162728.94555641068, 10529.524943379436], "t_v": [-527.0746776061375, -397.5661987481488], "cs": False, "ad":1.5e3, "md":1.35e3}}
    b = {"eid": "efd8dd82-59e9-4f71-97fc-96d16be37101", "ts": 181.00000000003618, "loc": [-155648.10021655622, 14496.034982381125], "a": 6.631761022100543, "v": [-633.9212228300115, -355.10659725031456], "av": 0.2932266489261978, "f": [436220.52590004867, 244359.6791278892], "t": -233347.4140426577, "o": {"o": "stellarpunk.orders.GoToLocation", "nnd": np.inf, "t_loc": [-162728.94555641068, 10529.524943379436],
"t_v": [-607.3950624974589, -340.2473147486867], "cs": False, "ad":1.5e3, "md":1.35e3}}
    c = {"eid": "1688d213-9d49-4222-aa0b-95dc090c59fa", "ts": 181.00000000003618, "loc": [-161271.80180938027, 11201.79460589419], "a": 0.17012052360683455, "v": [-123.61634016109224, -57.03178938300922], "av": 0.04141546936831053, "f": [4540.10299648228, 2094.627599677953], "t": 900000.0, "o": {"o": "stellarpunk.orders.GoToLocation", "nnd": np.inf, "t_loc": [-162728.94555641068, 10529.524943379436], "t_v": [-120.13404078695115, -55.42519138620854], "cs": False, "ad":1.5e3, "md":1.35e3}}
    d = {"eid": "f9cb7b45-858e-4422-a3b1-9513343b2fb7", "ts": 0, "loc": [-162728.94555641068, 10529.524943379436], "a": 0.0, "v": [0.0, 0.0], "av": 0.0, "f": [0.0, 0.0], "t": 0, "o": None}


    ship_a = ship_from_history(a, generator, sector)
    ship_b = ship_from_history(b, generator, sector)
    ship_c = ship_from_history(c, generator, sector)

    goto_a = order_from_history(a, ship_a, gamestate)
    goto_b = order_from_history(b, ship_b, gamestate)
    goto_c = order_from_history(c, ship_c, gamestate)

    station = station_from_history(d, generator, sector)

    eta = max(goto_a.estimate_eta(),goto_b.estimate_eta(),goto_c.estimate_eta())*1.2

    testui.eta = eta * 5.5#2.5
    testui.orders = [goto_a, goto_b, goto_c]
    #testui.cannot_avoid_collision_orders = [goto_a, goto_b, goto_c]
    #testui.cannot_stop_orders = [goto_a, goto_b, goto_c]
    # we might exceed the margin, but we just want to avoid collisions
    #testui.margin_neighbors = [ship_a, ship_b, ship_c]

    simulator.run()
    assert goto_a.is_complete()
    assert goto_b.is_complete()
    assert goto_c.is_complete()

@write_history
def test_followers(gamestate, generator, sector, testui, simulator):
    """ Shows two ships following in the same track to the same destination.
    Collision will happen when one slows down. """

    # this test was pulled from a live run (commented out lines), I updated the
    # position of the station to make the setup a little more forgiving.

    #a = {"eid": "c17c3726-a3d0-4734-9bb4-69e673b0ae5e", "ts": 2090.016666665668, "loc": [-8574.378687731325, -92946.33143588615], "a": -9.2819632330169, "v": [591.4913700392813, -46.297259041083194], "av": 0.14590014800404383, "f": [-4984.753723543476, 390.1670355366494], "t": -900000.0, "o": {"o": "stellarpunk.orders.GoToLocation", "nnd": np.inf, "t_loc": [-2280.7298265650393, -93438.9484145915], "t_v": [589.0932900475955, -46.10955633534738], "cs": False}}
    #b = {"eid": "527e4811-fca7-4e30-a259-d44fbc8f7bc2", "ts": 2090.016666665668, "loc": [-6951.495359661687, -93022.48310584611], "a": -3.1004619612519555, "v": [459.9857860425358, -41.01428792459363], "av": -0.2260183468057087, "f": [4980.242073810298, -444.0595525936751], "t": 900000.0, "o": {"o": "stellarpunk.orders.GoToLocation", "nnd": np.inf, "t_loc": [-2280.7298265650393, -93438.9484145915], "t_v": [477.8824725058597, -42.610032562310124], "cs": False}}
    #c = {"eid": "749a5424-31c7-4724-869b-bd2b17ff14b6", "ts": 0, "loc": [-2280.7298265650393, -93438.9484145915], "a": 0.0, "v": [0.0, 0.0], "av": 0.0, "f": [0.0, 0.0], "t": 0, "o": None}
    a = {"eid": "c17c3726-a3d0-4734-9bb4-69e673b0ae5e", "ts": 2090.016666665668, "loc": [-8574.378687731325, -92946.33143588615], "a": -9.2819632330169, "v": [591.4913700392813, -46.297259041083194], "av": 0.14590014800404383, "f": [-4984.753723543476, 390.1670355366494], "t": -900000.0, "o": {"o": "stellarpunk.orders.GoToLocation", "nnd": np.inf, "t_loc": [0., -93438.9484145915], "t_v": [589.0932900475955, -46.10955633534738], "cs": False, "ad":1.5e3, "md":1.35e3}}
    b = {"eid": "527e4811-fca7-4e30-a259-d44fbc8f7bc2", "ts": 2090.016666665668, "loc": [-6951.495359661687, -93022.48310584611], "a": -3.1004619612519555, "v": [459.9857860425358, -41.01428792459363], "av": -0.2260183468057087, "f": [4980.242073810298, -444.0595525936751], "t": 900000.0, "o": {"o": "stellarpunk.orders.GoToLocation", "nnd": np.inf, "t_loc": [0., -93438.9484145915], "t_v": [477.8824725058597, -42.610032562310124], "cs": False, "ad":1.5e3, "md":1.35e3}}
    c = {"eid": "749a5424-31c7-4724-869b-bd2b17ff14b6", "ts": 0, "loc": [0., -93438.9484145915], "a": 0.0, "v": [0.0, 0.0], "av": 0.0, "f": [0.0, 0.0], "t": 0, "o": None}

    ship_a = ship_from_history(a, generator, sector)
    ship_b = ship_from_history(b, generator, sector)

    goto_a = order_from_history(a, ship_a, gamestate)
    goto_b = order_from_history(b, ship_b, gamestate)

    station = station_from_history(c, generator, sector)

    testui.order_eta_error_factor = 1.3
    testui.orders = [goto_a]
    #testui.cannot_avoid_collision_orders = [goto_a, goto_b]
    #testui.cannot_stop_orders = [goto_a, goto_b]
    #testui.margin_neighbors = [ship_a, ship_b]

    simulator.run()
    assert goto_a.is_complete()

@write_history
def test_complicated_approach(gamestate, generator, sector, testui, simulator):
    a = {"eid": "c4c9abf9-6ab9-49d0-8521-5c213d84b649", "ts": 4014.000000023499, "loc": [77333.26634645685, -87886.94054689727], "a": -18.482272340387922, "v": [933.3460037580764, 358.9780456641165], "av": 0.0, "f": [0.0, 0.0], "t": 0.0, "o": {"o": "stellarpunk.orders.GoToLocation", "nnd": np.inf, "t_loc": [172664.93589409246, -51221.0349917073], "t_v": [933.3460037580766, 358.9780456641166], "cs": False, "ad":1.5e3, "md":1.35e3}}
    b = {"eid": "384be8ba-5b7e-4c72-bacb-c71ffc2741d5", "ts": 4014.000000023499, "loc": [85952.11093122436, -81661.92484174395], "a": -3.476194829731272, "v": [337.9608980594086, -222.67825369389368], "av": 0.08982571339053197, "f": [4175.180969873127, -2750.9750760065594], "t":900000.0, "o": {"o": "stellarpunk.orders.GoToLocation", "nnd": np.inf, "t_loc": [89663.85581987163, -84107.54761854198], "t_v": [339.01440182054654, -223.3723942857104], "cs": False, "ad":1.5e3, "md":1.35e3}}
    c = {"eid": "34eb3bbd-8bae-4214-a18d-dc5495826f34", "ts": 0, "loc": [89663.85581987163, -84107.54761854198], "a": 0.0, "v": [0.0, 0.0], "av": 0.0, "f": [0.0, 0.0], "t": 0, "o": None}

    ship_a = ship_from_history(a, generator, sector)
    ship_b = ship_from_history(b, generator, sector)

    goto_a = order_from_history(a, ship_a, gamestate)
    goto_b = order_from_history(b, ship_b, gamestate)

    station = station_from_history(c, generator, sector)

    eta = goto_a.estimate_eta()

    testui.eta = eta * 1.3
    testui.orders = [goto_a]
    #testui.cannot_avoid_collision_orders = [goto_a, goto_b]
    testui.cannot_stop_orders = [goto_a]
    #testui.margin_neighbors = [ship_a, ship_b]

    simulator.run()
    assert goto_a.is_complete()

@write_history
def test_perpendicular_threat(gamestate, generator, sector, testui, simulator):
    """ Illustrates two threats, one is nearby and 90 deg from desired
    direction. This might cause collision avoidance to steer into it, making a
    collision unavoidable. Hopefully it recognizes this and avoids it.
    """

    a = {"eid": "4caeca9f-55a4-4506-86a1-60c9ac097b89", "ts": 4425.59586668641, "loc": [-31776.544065050453, -54412.2677790681], "a": -21.506125835205, "v": [0.0, 1.1102230246251565e-16], "av": -0.28222264587779555, "f": [-1205.3543123910688, 4852.537581678297], "t": -900000.0, "o": {"o": "stellarpunk.orders.GoToLocation", "nnd": np.inf, "t_loc": [-57942.647404913, 50927.712770622566], "t_v": [-241.07086247821385, 970.5075163356594], "cs": False, "ad":1.5e3, "md":1.35e3}}
    b = {"eid": "76f1873d-6ace-492e-b6ea-bffc254ef416", "ts": 4425.59586668641, "loc": [-23534.115217425642, -44324.10863108866], "a": 10.257188790090208, "v": [-673.1223862010328, -739.5311036021592], "av": 0.0, "f": [0.0, 0.0], "t": 0.0, "o": {"o": "stellarpunk.orders.GoToLocation", "nnd": np.inf, "t_loc": [-74064.09143095232, -99839.25690164384], "t_v": [-673.1223862010329, -739.5311036021593], "cs": False, "ad":1.5e3, "md":1.35e3}}
    c = {"eid": "d7d0cc2b-549b-4bf5-853d-ad35f3ba3c80", "ts": 0, "loc": [-30368.148687458197, -53914.09999586431], "a": 0.0, "v": [0.0, 0.0], "av": 0.0, "f": [0.0, 0.0], "t": 0, "o": None}

    ship_a = ship_from_history(a, generator, sector)
    ship_b = ship_from_history(b, generator, sector)

    goto_a = order_from_history(a, ship_a, gamestate)
    # make the target location close-ish to ship_a
    goto_a._target_location = (ship_a.phys.position + (goto_a._target_location - ship_a.phys.position)/5)

    goto_b = order_from_history(b, ship_b, gamestate)

    station = station_from_history(c, generator, sector)

    testui.orders = [goto_a]
    #testui.cannot_avoid_collision_orders = [goto_a, goto_b]
    testui.cannot_stop_orders = [goto_a, goto_b]
    #testui.margin_neighbors = [ship_a, ship_b]

    simulator.run()
    assert goto_a.is_complete()

@write_history
def test_dense_neighborhood(gamestate, generator, sector, testui, simulator):
    # this test is randomized and can be kind of flaky
    ship_driver = generator.spawn_ship(sector, -1e4, 0., v=[0., 0.], w=0., theta=0.)

    num_blockers = 20
    generator.spawn_resource_field(sector, 0., 0., 0, num_blockers, width=5e3, mean_per_asteroid=1, variance_per_asteroid=0)

    goto_order = orders.GoToLocation(np.array([0., 0.]), ship_driver, gamestate)
    ship_driver.prepend_order(goto_order)

    eta = goto_order.estimate_eta()

    testui.eta = eta*4
    testui.orders = [goto_order]
    testui.cannot_avoid_collision_orders = [goto_order]
    testui.cannot_stop_orders = [goto_order]
    testui.margin_neighbors = [ship_driver]

    simulator.run()
    assert goto_order.is_complete()

@write_history
def test_more_headon(gamestate, generator, sector, testui, simulator):
    a = {"eid": "d876e0d8-ce5c-40ce-9411-dd52e92fd040", "ts": 531.4958666665119, "loc": [-8553.459552706245, 145290.59627284817], "a": 3.215596961832528, "v": [-997.2473217543096, -74.1470110918603], "av": 0.0, "f": [0.0, 0.0], "t": 0.0, "o": {"o": "stellarpunk.orders.GoToLocation", "nnd": np.inf, "t_loc": [-211457.22301090288, 130204.36112182347], "t_v": [-997.2473217543095, -74.14701109186028], "cs": False, "ad":1.5e3, "md":1.35e3}}
    b = {"eid": "a548766f-ab07-467e-a48b-eecd26050ec9", "ts": 531.4958666665119, "loc": [-128147.07960002865, 135773.3305332], "a": 0.08454138538747719, "v": [996.4189546142318, 84.55333751828712], "av": 0.0, "f": [0.0, 0.0], "t": 0.0, "o": {"o": "stellarpunk.orders.GoToLocation", "nnd": np.inf, "t_loc": [72804.09083505644, 152825.48721870864], "t_v": [996.4189546142318, 84.55333751828721], "cs": False, "ad":1.5e3, "md":1.35e3}}

    ship_a = ship_from_history(a, generator, sector)
    ship_b = ship_from_history(b, generator, sector)

    goto_a = order_from_history(a, ship_a, gamestate)
    goto_b = order_from_history(b, ship_b, gamestate)

    testui.orders = [goto_a]
    testui.cannot_avoid_collision_orders = [goto_a, goto_b]
    testui.cannot_stop_orders = [goto_a, goto_b]
    testui.margin_neighbors = [ship_a, ship_b]

    simulator.run()
    assert goto_a.is_complete()

@write_history
def test_either_side(gamestate, generator, sector, testui, simulator):
    """ Illustrates flapping between two threats on either side of the velocity
    direction. The issue is that sometimes we try to avoid one (into the other)
    and vice versa and sometimes they are coalesced. All of that creates
    discontinuities that cause the avoidance direction to flip dramatically."""

    entities = history_from_file(os.path.join(TESTDIR, "data/collision_either_side.history"), generator, sector, gamestate)

    ship_a = entities["8ccb3abc-b940-453c-82f8-2d108117312e"]
    logging.debug(f'{ship_a.entity_id}')
    goto_a = ship_a.current_order()

    testui.orders = [goto_a]
    #testui.cannot_avoid_collision_orders = [goto_a]
    testui.cannot_stop_orders = [goto_a]
    testui.margin_neighbors = [ship_a]

    simulator.run()
    assert goto_a.is_complete()

@write_history
def test_complicated_departure(gamestate, generator, sector, testui, simulator):
    entities = history_from_file(os.path.join(TESTDIR, "data/collision_complicated_departure.history"), generator, sector, gamestate, load_ct=False)

    ship_a = entities["951081e3-6253-4a9c-8be7-6a21cbf31feb"]
    logging.debug(f'{ship_a.entity_id}')
    goto_a = ship_a.current_order()

    eta = goto_a.estimate_eta()

    testui.eta = eta*1.3
    testui.orders = [goto_a]
    #testui.cannot_avoid_collision_orders = [goto_a]
    testui.cannot_stop_orders = [goto_a]
    #testui.margin_neighbors = [ship_a]

    simulator.run()
    assert goto_a.is_complete()

@write_history
def test_20220331(gamestate, generator, sector, testui, simulator):
    entities = history_from_file(os.path.join(TESTDIR, "data/collision_20220331.history"), generator, sector, gamestate, load_ct=False)

    ship_a = entities["670fff17-7333-4b69-be59-98d00286dc6f"]
    logging.debug(f'{ship_a.entity_id}')
    goto_a = ship_a.current_order()

    eta = goto_a.estimate_eta()

    testui.eta = eta * 1.3
    testui.orders = [goto_a]
    #testui.cannot_avoid_collision_orders = [goto_a]
    testui.cannot_stop_orders = [goto_a]
    #testui.margin_neighbors = [ship_a]

    simulator.run()
    assert goto_a.is_complete()

@write_history
def test_too_fast_small_margin(gamestate, generator, sector, testui, simulator):
    """ A ship flying very fast doesn't have enough time to course correct """
    entities = history_from_file(os.path.join(TESTDIR, "data/too_fast_small_margin.history"), generator, sector, gamestate)

    ship_a = entities["267924bc-6490-4561-984b-cb47dce385d5"]
    logging.debug(f'{ship_a.entity_id}')
    goto_a = ship_a.current_order()

    eta = goto_a.estimate_eta()

    testui.eta = eta * 1.3
    testui.orders = [goto_a]
    testui.cannot_stop_orders = [goto_a]
    #testui.margin_neighbors = [ship_a]

    simulator.run()
    assert goto_a.is_complete()

@write_history
def test_coalesced_cbdr(gamestate, generator, sector, testui, simulator):
    """ CBDR can flap a lot when coalescing targets.

    This test sees what happens when CBDR might otherwise get triggered. """
    entities = history_from_file(os.path.join(TESTDIR, "data/coalesced_cbdr.history"), generator, sector, gamestate, load_ct=False)

    ship_a = entities["2b20dd4b-b5e9-4562-84ab-74d8f40db525"]
    logging.debug(f'{ship_a.entity_id}')
    goto_a = ship_a.current_order()

    eta = goto_a.estimate_eta()

    testui.eta = eta
    testui.orders = [goto_a]
    testui.cannot_stop_orders = [goto_a]

    simulator.run()
    assert goto_a.is_complete()

@write_history
def test_fast_speed_asteroid_field(gamestate, generator, sector, testui, simulator):
    """ Tests a ship heading near a large asteroid field at high speed.

    Should avoid all of them."""

    entities = history_from_file(os.path.join(TESTDIR, "data/fast_speed_asteroid_field.history"), generator, sector, gamestate)

    ship_a = entities["b695ac11-0021-4c52-b72f-288f715c08c5"]
    logging.debug(f'{ship_a.entity_id}')
    goto_a = ship_a.current_order()

    eta = goto_a.estimate_eta()

    testui.eta = eta * 1.3
    testui.orders = [goto_a]
    #testui.cannot_avoid_collision_orders = [goto_a]
    testui.cannot_stop_orders = [goto_a]
    #testui.margin_neighbors = [ship_a]
    testui.max_timestamp = 45

    starting_distance = ship_a.phys.position.get_distance(goto_a._target_location)
    simulator.run()
    assert starting_distance - ship_a.phys.position.get_distance(goto_a._target_location) > 1.5e4

@write_history
def test_respond_to_new(gamestate, generator, sector, testui, simulator):
    """ Tests a ship avoiding one target to turn into another.

    The new threat could lead to discontinuities in collision avoidance, but we
    need to respond to the new threat quickly. """

    entities = history_from_file(os.path.join(TESTDIR, "data/respond_to_new.history"), generator, sector, gamestate, load_ct=False)

    ship_a = entities["dcf07fc2-5a4e-4acc-8674-2f4d871bdaf0"]
    logging.debug(f'{ship_a.entity_id}')
    goto_a = ship_a.current_order()

    eta = goto_a.estimate_eta()

    testui.eta = eta*1.3
    testui.orders = [goto_a]
    #testui.cannot_avoid_collision_orders = [goto_a]
    testui.cannot_stop_orders = [goto_a]
    testui.margin_neighbors = [ship_a]
    # this test would run for a long time, we just want to avoid the immediate
    # collision
    testui.max_timestamp = 30

    simulator.run()
    # see above
    #assert goto_a.is_complete()

@write_history
def test_failed_to_divert(gamestate, generator, sector, testui, simulator):
    """ Tests a ship avoiding a threat in a field of others

    for some reason it didn't divert, but this seems like a simple scenario."""

    entities = history_from_file(os.path.join(TESTDIR, "data/failed_to_divert.history"), generator, sector, gamestate, load_ct=False)

    ship_a = entities["2aac25bb-2dea-4f7b-a6cc-c44f3b18ed70"]
    logging.debug(f'{ship_a.entity_id}')
    goto_a = ship_a.current_order()

    eta = goto_a.estimate_eta()

    testui.eta = eta*1.3
    testui.orders = [goto_a]
    testui.cannot_avoid_collision_orders = [goto_a]
    testui.cannot_stop_orders = [goto_a]
    #testui.margin_neighbors = [ship_a]
    # this test would run for a long time, we just want to avoid the immediate
    # collision
    testui.max_timestamp = 45

    starting_distance = ship_a.phys.position.get_distance(goto_a._target_location)
    simulator.run()
    asteroid = entities["90acd111-7fb0-4e89-ac2a-98dabb8c7d10"]
    assert util.distance(ship_a.loc, asteroid.loc) > 1.5e3
    assert starting_distance - ship_a.phys.position.get_distance(goto_a._target_location) > 1.5e3
    #assert goto_a.is_complete()

@write_history
def test_threading_needle(gamestate, generator, sector, testui, simulator):
    """ Tests a ship heading between two threat.

    Should avoid all of them, and this should be easy, but collided in game"""

    entities = history_from_file(os.path.join(TESTDIR, "data/threading_needle.history"), generator, sector, gamestate, load_ct=False)

    ship_a = entities["bd9a76fa-96d4-42a3-a469-c000e5e38ebb"]
    logging.debug(f'{ship_a.entity_id}')
    goto_a = ship_a.current_order()

    eta = goto_a.estimate_eta()

    testui.eta = eta
    testui.orders = [goto_a]
    #testui.cannot_avoid_collision_orders = [goto_a]
    testui.cannot_stop_orders = [goto_a]
    #testui.margin_neighbors = [ship_a]

    simulator.run()
    assert goto_a.is_complete()

@write_history
def test_collision_field(gamestate, generator, sector, testui, simulator):
    """ Tests a ship heading in a field of many potential threats.

    Can get confused and steer into a future threat."""

    entities = history_from_file(os.path.join(TESTDIR, "data/collision_field.history"), generator, sector, gamestate, load_ct=False)

    ship_a = entities["c12de111-949a-47bc-8570-a3d7cadac9f1"]
    logging.debug(f'{ship_a.entity_id}')
    goto_a = ship_a.current_order()

    eta = goto_a.estimate_eta()

    testui.eta = eta
    testui.orders = [goto_a]
    #testui.cannot_avoid_collision_orders = [goto_a]
    testui.cannot_stop_orders = [goto_a]
    testui.margin_neighbors = [ship_a]

    simulator.run()
    assert goto_a.is_complete()

@write_history
def test_histeresis(gamestate, generator, sector, testui, simulator):
    """ Tests a ship that drops coalesced threats.

    Histeresis should avoid discontinuities."""

    entities = history_from_file(os.path.join(TESTDIR, "data/histeresis.history"), generator, sector, gamestate, load_ct=False)

    ship_a = entities["116f6f98-865b-4690-bfd1-73c8b7cd22a0"]
    logging.debug(f'{ship_a.entity_id}')
    goto_a = ship_a.current_order()

    eta = goto_a.estimate_eta()

    testui.eta = eta
    testui.orders = [goto_a]
    #testui.cannot_avoid_collision_orders = [goto_a]
    testui.cannot_stop_orders = [goto_a]
    #testui.margin_neighbors = [ship_a]

    simulator.run()
    assert goto_a.is_complete()

@write_history
def test_overlapping(gamestate, generator, sector, testui, simulator):
    """ Tests a ship travels throw many threats that nearly coalesc.

    The overlapping coalesced circles should not flap."""

    entities = history_from_file(os.path.join(TESTDIR, "data/overlapping.history"), generator, sector, gamestate, load_ct=False)

    ship_a = entities["d64692e0-2206-4f24-b512-a21d90f53189"]
    logging.debug(f'{ship_a.entity_id}')
    goto_a = ship_a.current_order()

    eta = goto_a.estimate_eta()

    testui.eta = eta
    testui.orders = [goto_a]
    #testui.cannot_avoid_collision_orders = [goto_a]
    testui.cannot_stop_orders = [goto_a]
    #testui.margin_neighbors = [ship_a]

    simulator.run()
    assert goto_a.is_complete()

@write_history
def test_cross_traffic(gamestate, generator, sector, testui, simulator):
    """ Tests a ship avoiding cross traffic then avoiding a farther threat.

    once the farther threat is added, the coalesced threat is right in front of
    us and we cannot avoid collision. """

    entities = history_from_file(os.path.join(TESTDIR, "data/cross_traffic.history"), generator, sector, gamestate)

    ship_a = entities["5e9685d2-9272-4923-9bb2-5a0a2d07692f"]
    logging.debug(f'{ship_a.entity_id}')
    goto_a = ship_a.current_order()

    eta = goto_a.estimate_eta()

    testui.eta = eta
    testui.orders = [goto_a]
    #testui.cannot_avoid_collision_orders = [goto_a]
    testui.cannot_stop_orders = [goto_a]
    #testui.margin_neighbors = [ship_a]

    simulator.run()
    assert goto_a.is_complete()

@write_history
def test_traffic_lane(gamestate, generator, sector, testui, simulator):
    """ Tests a ship travelling in a popular travel lane.

    Lots of moving targets can confusing collision avoidance: cbdr, coalescing,
    moving threat locations, etc.. """

    entities = history_from_file(os.path.join(TESTDIR, "data/traffic_lane.history"), generator, sector, gamestate, load_ct=False)

    ship_a = entities["88bf287a-921f-4b19-bd6d-4deb6100c834"]
    logging.debug(f'{ship_a.entity_id}')
    goto_a = ship_a.current_order()

    eta = goto_a.estimate_eta()

    testui.eta = eta
    testui.orders = [goto_a]
    #testui.cannot_avoid_collision_orders = [goto_a]
    testui.cannot_stop_orders = [goto_a]
    #testui.margin_neighbors = [ship_a]

    simulator.run()
    assert goto_a.is_complete()

@write_history
def test_arrival_occupied(gamestate, generator, sector, testui, simulator):
    """ Tests a ship arriving in a spot already occupied, partially.

    should stop as soon as possible. """

    entities = history_from_file(os.path.join(TESTDIR, "data/arrival_occupied.history"), generator, sector, gamestate)

    ship_a = entities["1d68c162-c8eb-418f-b379-1bb0674742ae"]
    logging.debug(f'{ship_a.entity_id}')
    goto_a = ship_a.current_order()

    eta = goto_a.estimate_eta()

    testui.eta = eta
    testui.orders = [goto_a]
    testui.cannot_avoid_collision_orders = [goto_a]
    testui.cannot_stop_orders = [goto_a]
    testui.margin_neighbors = [ship_a]

    simulator.run()
    assert goto_a.is_complete()

@write_history
def test_navigate_field(gamestate, generator, sector, testui, simulator):
    """ Tests a ship navigating an asteroid field at high speed.

    should stop as soon as possible. """

    entities = history_from_file(os.path.join(TESTDIR, "data/navigate_field.history"), generator, sector, gamestate, load_ct=False)

    ship_a = entities["f95f25b2-c2e9-48cc-bc13-15d4c7393e72"]
    logging.debug(f'{ship_a.entity_id}')
    goto_a = ship_a.current_order()

    eta = goto_a.estimate_eta()

    testui.eta = eta
    testui.orders = [goto_a]
    #testui.cannot_avoid_collision_orders = [goto_a]
    testui.cannot_stop_orders = [goto_a]
    #testui.margin_neighbors = [ship_a]
    testui.max_timestamp = 45

    starting_distance =ship_a.phys.position.get_distance(goto_a._target_location)
    simulator.run()
    assert starting_distance - ship_a.phys.position.get_distance(goto_a._target_location) > 1.5e4

    #assert goto_a.is_complete()

@write_history
def test_actual_vs_desired_velocity(gamestate, generator, sector, testui, simulator):
    """ Tests a ship headed just to one side of a threat with a desired velocity just to the other side.

    should be able to avoid the threat, but the confusion between the two might cause issues. """

    entities = history_from_file(os.path.join(TESTDIR, "data/actual_vs_desired_velocity.history"), generator, sector, gamestate)

    ship_a = entities["d0c48ae3-b438-463a-9a6f-1f6f8ad69e4e"]
    logging.debug(f'{ship_a.entity_id}')
    goto_a = ship_a.current_order()

    eta = goto_a.estimate_eta()

    testui.eta = eta
    testui.orders = [goto_a]
    #testui.cannot_avoid_collision_orders = [goto_a]
    testui.cannot_stop_orders = [goto_a]
    #testui.margin_neighbors = [ship_a]

    simulator.run()
    assert goto_a.is_complete()

@write_history
def test_overeager_arrival(gamestate, generator, sector, testui, simulator):
    """ Tests a ship coming in for a busy arrival.

    should be able to arrive without collision. """

    entities = history_from_file(os.path.join(TESTDIR, "data/overeager_arrival.history"), generator, sector, gamestate)

    ship_a = entities["9fac0d05-f96a-4410-8167-da3921aa22e8"]
    logging.debug(f'{ship_a.entity_id}')
    goto_a = ship_a.current_order()

    eta = goto_a.estimate_eta()

    testui.eta = eta*6
    testui.orders = [goto_a]
    #testui.cannot_avoid_collision_orders = [goto_a]
    testui.cannot_stop_orders = [goto_a]
    #testui.margin_neighbors = [ship_a]

    simulator.run()
    assert goto_a.is_complete()

@write_history
def test_arrival_occupied2(gamestate, generator, sector, testui, simulator):
    """ Tests a ship arriving in a spot already occupied, partially.

    should stop as soon as possible. """

    entities = history_from_file(os.path.join(TESTDIR, "data/arrival_occupied2.history"), generator, sector, gamestate)

    ship_a = entities["b97d8d21-ec2a-476c-bc84-fce9dff8a29b"]
    logging.debug(f'{ship_a.entity_id}')
    goto_a = ship_a.current_order()

    eta = goto_a.estimate_eta()

    testui.eta = eta * 2.5
    testui.orders = [goto_a]
    #testui.cannot_avoid_collision_orders = [goto_a]
    testui.cannot_stop_orders = [goto_a]
    #testui.margin_neighbors = [ship_a]

    simulator.run()
    assert goto_a.is_complete()

@write_history
def test_busy_intersection(gamestate, generator, sector, testui, simulator):
    """ Tests two ships nearly parallel that intersect.

    should detect cbdr and avoid collision """

    entities = history_from_file(os.path.join(TESTDIR, "data/busy_intersection.history"), generator, sector, gamestate)

    ship_a = entities["caf26807-a636-4408-9825-7138b9f559e1"]
    logging.debug(f'{ship_a.entity_id}')
    goto_a = ship_a.current_order()

    eta = goto_a.estimate_eta()

    testui.eta = eta
    testui.orders = [goto_a]
    #testui.cannot_avoid_collision_orders = [goto_a]
    testui.cannot_stop_orders = [goto_a]
    #testui.margin_neighbors = [ship_a]

    testui.max_timestamp = 30

    starting_distance = ship_a.phys.position.get_distance(goto_a._target_location)
    simulator.run()
    assert starting_distance - ship_a.phys.position.get_distance(goto_a._target_location) > 1.5e4
    #assert goto_a.is_complete()

@write_history
def test_rotate_lag(gamestate, generator, sector, testui, simulator):
    """ Tests a ship that may need to rotate 180 deg to avoid collision.

    should avoid somehow. """

    entities = history_from_file(os.path.join(TESTDIR, "data/rotate_lag.history"), generator, sector, gamestate)

    ship_a = entities["0e9e97d6-2a23-4cbe-b976-a1ff9e7440e0"]
    logging.debug(f'{ship_a.entity_id}')
    goto_a = ship_a.current_order()

    eta = goto_a.estimate_eta()

    testui.eta = eta
    testui.orders = [goto_a]
    #testui.cannot_avoid_collision_orders = [goto_a]
    testui.cannot_stop_orders = [goto_a]
    #testui.margin_neighbors = [ship_a]

    simulator.run()
    assert goto_a.is_complete()

@write_history
def test_through_asteroid_field(gamestate, generator, sector, testui, simulator):
    """ Tests a ship travelling through a tight asteroid field. """

    entities = history_from_file(os.path.join(TESTDIR, "data/through_asteroid_field.history"), generator, sector, gamestate)

    ship_a = entities["5f56e223-38ba-4c86-b055-585535b3caa6"]
    logging.debug(f'{ship_a.entity_id}')
    goto_a = ship_a.current_order()

    eta = goto_a.estimate_eta()

    testui.eta = eta
    testui.orders = [goto_a]
    #testui.cannot_avoid_collision_orders = [goto_a]
    testui.cannot_stop_orders = [goto_a]
    #testui.margin_neighbors = [ship_a]
    testui.max_timestamp = 30

    starting_distance = ship_a.phys.position.get_distance(goto_a._target_location)
    simulator.run()
    assert starting_distance - ship_a.phys.position.get_distance(goto_a._target_location) > 1.5e4

    #assert goto_a.is_complete()

@write_history
def test_more_busy_lane(gamestate, generator, sector, testui, simulator):
    """ Tests a ship in a busy traffic lane in an asteroid field. """

    entities = history_from_file(os.path.join(TESTDIR, "data/more_busy_lane.history"), generator, sector, gamestate)

    ship_a = entities["537958f9-536c-485d-8ca4-dfea883fc65b"]
    logging.debug(f'{ship_a.entity_id}')
    goto_a = ship_a.current_order()

    eta = goto_a.estimate_eta()

    testui.eta = eta
    testui.orders = [goto_a]
    #testui.cannot_avoid_collision_orders = [goto_a]
    testui.cannot_stop_orders = [goto_a]
    #testui.margin_neighbors = [ship_a]

    simulator.run()
    assert goto_a.is_complete()

@write_history
def test_target_behind_asteroid(gamestate, generator, sector, testui, simulator):
    """ Tests a ship going to a location behind an asteroid, in an asteroid
    field. """

    entities = history_from_file(os.path.join(TESTDIR, "data/target_behind_asteroid.history"), generator, sector, gamestate)

    ship_a = entities["a4c71ae5-f3cd-47ff-a266-5a8ed4facef5"]
    logging.debug(f'{ship_a.entity_id}')
    goto_a = ship_a.current_order()

    eta = goto_a.estimate_eta()

    testui.eta = eta
    testui.orders = [goto_a]
    #testui.cannot_avoid_collision_orders = [goto_a]
    testui.cannot_stop_orders = [goto_a]
    #testui.margin_neighbors = [ship_a]

    simulator.run()
    assert goto_a.is_complete()

@write_history
def test_more_asteroid_nav(gamestate, generator, sector, testui, simulator):
    """ Tests a ship going to a location behind an asteroid, in an asteroid
    field. """

    entities = history_from_file(os.path.join(TESTDIR, "data/more_asteroid_nav.history"), generator, sector, gamestate)

    ship_a = entities["28ce5698-a1bf-43df-89f6-0f0b07ee957e"]
    logging.debug(f'{ship_a.entity_id}')
    goto_a = ship_a.current_order()

    eta = goto_a.estimate_eta()

    testui.eta = eta
    testui.orders = [goto_a]
    #testui.cannot_avoid_collision_orders = [goto_a]
    testui.cannot_stop_orders = [goto_a]
    #testui.margin_neighbors = [ship_a]
    testui.max_timestamp = 30

    starting_distance = ship_a.phys.position.get_distance(goto_a._target_location)
    simulator.run()
    assert starting_distance - ship_a.phys.position.get_distance(goto_a._target_location) > 1.5e4
    #assert goto_a.is_complete()

@write_history
def test_headon_asteroid_field(gamestate, generator, sector, testui, simulator):
    """ Tests a ship aimed headon at another in an asteroid field. """

    entities = history_from_file(os.path.join(TESTDIR, "data/headon_asteroid_field.history"), generator, sector, gamestate)

    ship_a = entities["922e3718-1d71-4c70-ba8f-e7a5085336a9"]
    logging.debug(f'{ship_a.entity_id}')
    goto_a = ship_a.current_order()

    eta = goto_a.estimate_eta()

    testui.eta = eta
    testui.orders = [goto_a]
    #testui.cannot_avoid_collision_orders = [goto_a]
    testui.cannot_stop_orders = [goto_a]
    #testui.margin_neighbors = [ship_a]

    simulator.run()
    assert goto_a.is_complete()

@write_history
def test_ship_and_asteroid(gamestate, generator, sector, testui, simulator):
    """ Tests a ship headed for collision to an asteroid and cross travelling
    ship """

    entities = history_from_file(os.path.join(TESTDIR, "data/asteroid_and_ship.history"), generator, sector, gamestate)

    ship_a = entities["85bff636-aa4b-45cc-afee-edbc0577e16a"]
    logging.debug(f'{ship_a.entity_id}')
    goto_a = ship_a.current_order()

    eta = goto_a.estimate_eta()

    testui.eta = eta
    testui.orders = [goto_a]
    #testui.cannot_avoid_collision_orders = [goto_a]
    testui.cannot_stop_orders = [goto_a]
    #testui.margin_neighbors = [ship_a]

    simulator.run()
    assert goto_a.is_complete()

@write_history
def test_more_cbdr(gamestate, generator, sector, testui, simulator):
    """ Tests another case of cbdr """

    entities = history_from_file(os.path.join(TESTDIR, "data/more_cbdr.history"), generator, sector, gamestate)

    ship_a = entities["d9361e71-7095-4b77-97f5-af27d00f67f5"]
    logging.debug(f'{ship_a.entity_id}')
    goto_a = ship_a.current_order()

    eta = goto_a.estimate_eta()

    testui.eta = eta
    testui.orders = [goto_a]
    #testui.cannot_avoid_collision_orders = [goto_a]
    testui.cannot_stop_orders = [goto_a]
    #testui.margin_neighbors = [ship_a]

    simulator.run()
    assert goto_a.is_complete()

