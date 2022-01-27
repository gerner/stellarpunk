""" Tests for ship orders and behaviors. """

import logging
import functools

import pytest
import pymunk
import numpy as np

from stellarpunk import core, sim, generate, orders, util

class MonitoringUI:
    def __init__(self, gamestate, sector):
        self.simulator = None
        self.gamestate = gamestate
        self.sector = sector

        self.orders = []
        self.cannot_stop_orders = []
        self.cannot_avoid_collision_orders = []
        self.margin_neighbors = []
        self.eta = np.inf

    def status_message(self, string, attr=None):
        pass

    def get_color(self, color):
        return None

    def margin_neighbors(self, margin_neighbors):
        self.margin_neighbors = margin_neighbors

    def tick(self, timeout):
        assert not self.simulator.collisions

        assert self.gamestate.timestamp < self.eta

        assert all(map(lambda x: not x.cannot_stop, self.cannot_stop_orders))
        assert all(map(lambda x: not x.cannot_avoid_collision, self.cannot_avoid_collision_orders))
        for margin_neighbor in self.margin_neighbors:
            neighbor, neighbor_dist = nearest_neighbor(self.sector, margin_neighbor)
            assert neighbor_dist >= 0 - orders.VELOCITY_EPS

        if all(map(lambda x: x.is_complete(), self.orders)):
            self.gamestate.quit()

def write_history(func):
    """ Decorator that writes sector history to file when an exception is
    raised in a test. """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        sector = kwargs["sector"]
        try:
            return func(*args, **kwargs)
        except Exception as e:
            core.write_history_to_file(sector, f'/tmp/stellarpunk_test.{func.__name__}.history.gz')
            raise
    return wrapper

@pytest.fixture
def testui(gamestate, sector):
    return MonitoringUI(gamestate, sector)

@pytest.fixture
def gamestate():
    return core.Gamestate()

@pytest.fixture
def simulator(gamestate, testui):
    simulation = sim.Simulator(gamestate, testui)
    testui.simulator = simulation
    simulation.min_tick_sleep = np.inf
    simulation.min_ui_timeout = -np.inf

    simulation.initialize()

    return simulation

@pytest.fixture
def generator(gamestate):
    ug = generate.UniverseGenerator(gamestate)
    gamestate.random = ug.r
    return ug

@pytest.fixture
def sector(gamestate):
    sector_radius=1e5
    sector_name = "Sector"

    sector = core.Sector(0, 0, sector_radius, sector_name)
    sector.space = pymunk.Space()
    gamestate.sectors[(0,0)] = sector

    return sector

def nearest_neighbor(sector, entity):
    neighbor_distance = np.inf
    neighbor = None
    for hit in sector.spatial_point(entity.loc):
        if hit == entity:
            continue
        d = np.linalg.norm(entity.loc - hit.loc) - hit.radius - entity.radius
        if d < neighbor_distance:
            neighbor_distance = d
            neighbor = hit
    return neighbor, neighbor_distance

def ship_from_history(history_entry, generator, sector):
    x, y = history_entry["loc"]
    v = history_entry["v"]
    w = history_entry["av"]
    theta = history_entry["a"]
    ship = generator.spawn_ship(sector, x, y, v=v, w=w, theta=theta)
    return ship

def station_from_history(history_entry, generator, sector):
    x, y = history_entry["loc"]
    station = generator.spawn_station(sector, x, y, 0)
    return station

def order_from_history(history_entry, ship, gamestate):
    if history_entry["o"]["o"] != "stellarpunk.orders.GoToLocation":
        raise ValueError(f'can only support stellarpunk.orders.GoToLocation, not {history_entry["o"]["o"]}')
    order = orders.GoToLocation(np.array(history_entry["o"]["t_loc"]), ship, gamestate)
    ship.orders.append(order)
    return order

# collision avoidance interesting cases:
# no collision, just go wherever you're going
# tangential collision, avoid
# head-on collision, need to pick a side to avoid collision
# tangential collision to one side, really want to go in that direction tho
#   should have a way to express that an avoid in a helpful direction

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

    core.write_history_to_file(rotate_order.ship, "/tmp/stellarpunk_test.history")

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

    core.write_history_to_file(rotate_order.ship, "/tmp/stellarpunk_test.history")

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

    core.write_history_to_file(goto_order.ship, "/tmp/stellarpunk_test.history")

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

@write_history
def test_basic_collision_avoidance(gamestate, generator, sector, testui, simulator, caplog):
    # set up a sector, including space
    # add two ships in an offset (relative to desired course) arrangement
    # travel along course and observe no collision

    ship_driver = generator.spawn_ship(sector, 0, 2400, v=(0,0), w=0, theta=0)
    ship_blocker = generator.spawn_ship(sector, -300, 1200, v=(0,0), w=0, theta=0)

    goto_order = orders.GoToLocation(np.array((0.,0.)), ship_driver, gamestate)
    ship_driver.orders.append(goto_order)

    # d = v_i*t + 1/2 a * t**2
    # v_i = 0
    # d = 1/2 a * t**2
    # t = sqrt( 2*d/a )
    # expect path to be accelerate half-way, turn around, decelerate
    # this is approximate
    distance = np.linalg.norm(ship_driver.loc)
    eta = goto_order.eta()

    testui.eta = eta*2
    testui.orders = [goto_order]
    testui.cannot_stop_orders = [goto_order]
    testui.cannot_avoid_collision_orders = [goto_order]
    testui.margin_neighbors = [ship_driver]

    simulator.run()
    assert goto_order.is_complete()

def test_head_on_static_collision_avoidance(gamestate, generator, sector, testui, simulator):
    ship_blocker = generator.spawn_ship(sector, 0, 1200, v=(0,0), w=0, theta=0)
    ship_driver = generator.spawn_ship(sector, 0, 2400, v=(0,0), w=0, theta=0)

    goto_order = orders.GoToLocation(np.array((0.,0.)), ship_driver, gamestate)
    ship_driver.orders.append(goto_order)

    distance = 2400
    eta = goto_order.eta()

    starttime = gamestate.timestamp
    def tick(timeout):
        assert not simulator.collisions
        neighbor, neighbor_dist = nearest_neighbor(sector, ship_driver)
        assert neighbor_dist >= goto_order.collision_margin
        if goto_order.is_complete():
            gamestate.quit()
        assert gamestate.timestamp - starttime < eta * 2.6
        assert not goto_order.cannot_stop
        assert not goto_order.cannot_avoid_collision
        assert not goto_order.collision_cbdr

    testui.tick = tick
    simulator.run()
    assert goto_order.is_complete()

    #core.write_history_to_file(goto_order.ship, "/tmp/stellarpunk_test.history")

def test_blocker_wall_collision_avoidance(gamestate, generator, sector, testui, simulator):
    """ Initial state is travelling along course west of a blocker, but ideal
    path is east. Do we travel to the east of the blocker, a smaller overall
    maneuver, even though it's a bigger maneuver to avoid the collision. """

    # set up a sector, including space
    # add two ships in an offset (relative to desired course) arrangement
    # travel along course and observe no collision

    # a "wall" of blockers to the left of our target
    #generator.spawn_ship(sector, -300, 12000, v=(0,0), w=0, theta=0)
    #generator.spawn_ship(sector, -300, 11000, v=(0,0), w=0, theta=0)
    generator.spawn_ship(sector, -300, 10000, v=(0,0), w=0, theta=0)
    generator.spawn_ship(sector, -300, 9000, v=(0,0), w=0, theta=0)
    generator.spawn_ship(sector, -300, 6000, v=(0,0), w=0, theta=0)
    generator.spawn_ship(sector, -300, 2000, v=(0,0), w=0, theta=0)
    generator.spawn_ship(sector, -300, 1000, v=(0,0), w=0, theta=0)
    #generator.spawn_ship(sector, -300, 500, v=(0,0), w=0, theta=0)

    ship_driver = generator.spawn_ship(sector, -400, 20000, v=(0.,0.), w=0., theta=0.)
    ship_driver.velocity = np.array((0., 0.))
    ship_driver.phys.velocity = tuple(ship_driver.velocity)

    goto_order = orders.GoToLocation(np.array((0.,0.)), ship_driver, gamestate)
    ship_driver.orders.append(goto_order)

    # d = v_i*t + 1/2 a * t**2
    # v_i = 0
    # d = 1/2 a * t**2
    # t = sqrt( 2*d/a )
    # expect path to be accelerate half-way, turn around, decelerate
    # this is approximate
    distance = np.linalg.norm(ship_driver.loc)
    eta = goto_order.eta()

    starttime = gamestate.timestamp
    def tick(timeout):
        assert not simulator.collisions

        neighbor, neighbor_dist = nearest_neighbor(sector, ship_driver)
        #logging.debug(f'{ship_driver.loc} {ship_driver.velocity} {ship_driver.angle}, {d:.2e} {neighbor}, {goto_order.collision_threat}')
        assert neighbor_dist >= goto_order.collision_margin
        if goto_order.is_complete():
            gamestate.quit()

        #TODO: we should not hit this condition!
        #assert not goto_order.cannot_stop
        assert not goto_order.cannot_avoid_collision
        assert goto_order.collision_dv[0] >= 0.
        assert gamestate.timestamp - starttime < eta*10

    testui.tick = tick
    simulator.run()
    assert goto_order.is_complete()
    core.write_history_to_file(goto_order.ship, "/tmp/stellarpunk_test.history")

@write_history
def test_simple_ships_intersecting(gamestate, generator, sector, testui, simulator):
    ship_a = generator.spawn_ship(sector, -5000, 0, v=(0,0), w=0, theta=0)
    ship_b = generator.spawn_ship(sector, 0, -5000, v=(0,0), w=0, theta=np.pi/2)

    goto_a = orders.GoToLocation(np.array((5000.,0.)), ship_a, gamestate)
    ship_a.orders.append(goto_a)
    goto_b = orders.GoToLocation(np.array((0.,5000.)), ship_b, gamestate)
    ship_b.orders.append(goto_b)

    a_cbdr = False
    b_cbdr = False

    eta = max(goto_a.eta(), goto_b.eta())

    testui.eta = eta*1.1
    testui.orders = [goto_a, goto_b]
    testui.cannot_stop_orders = [goto_a, goto_b]
    testui.cannot_avoid_collision_orders = [goto_a, goto_b]
    testui.margin_neighbors = [ship_a]

    simulator.run()
    assert goto_a.is_complete()
    assert not goto_a.collision_cbdr
    assert goto_b.is_complete()
    assert not goto_b.collision_cbdr

    assert any(False if hist_entry.order_hist is None else hist_entry.order_hist.get("cbdr", False) for hist_entry in ship_a.history)
    assert any(False if hist_entry.order_hist is None else hist_entry.order_hist.get("cbdr", False) for hist_entry in ship_b.history)

def test_headon_ships_intersecting(gamestate, generator, sector, testui, simulator):
    ship_a = generator.spawn_ship(sector, -5000, 0, v=(0,0), w=0, theta=0)
    ship_b = generator.spawn_ship(sector, 5000, 0, v=(0,0), w=0, theta=np.pi)

    goto_a = orders.GoToLocation(np.array((10000.,0.)), ship_a, gamestate)
    ship_a.orders.append(goto_a)
    goto_b = orders.GoToLocation(np.array((-10000.,0.)), ship_b, gamestate)
    ship_b.orders.append(goto_b)

    eta = max(goto_a.eta(), goto_b.eta())

    def tick(timeout):
        assert not simulator.collisions

        # only need to check one, they are symmetric
        neighbor, neighbor_dist = nearest_neighbor(sector, ship_a)
        assert neighbor_dist >= goto_a.collision_margin

        if goto_a.is_complete() and goto_b.is_complete():
            gamestate.quit()
        assert not goto_a.cannot_stop
        assert not goto_a.cannot_avoid_collision
        assert not goto_b.cannot_stop
        assert not goto_b.cannot_avoid_collision

        assert gamestate.timestamp < eta*1.5

    testui.tick = tick
    simulator.run()
    assert goto_a.is_complete()
    assert not goto_a.collision_cbdr
    assert goto_b.is_complete()
    assert not goto_b.collision_cbdr

    core.write_history_to_file(goto_a.ship, "/tmp/stellarpunk_test.history")
    core.write_history_to_file(goto_b.ship, "/tmp/stellarpunk_test.history", mode="a")
@pytest.mark.skip(reason="this test is pretty slow because it takes a while to reach the destinations")
def test_ships_intersecting_collision(gamestate, generator, sector, testui, simulator):
    # two ships headed on intersecting courses collide
    # testcase from gameplay logs

    a = {"eid": "bebefe43-24b3-4588-9b42-4f5504de5903", "ts": 9.833333333333401, "loc": [140973.20332888863, 37746.464152281136], "a": 3.697086234730296, "v": [-1517.033904995614, -872.3805171209457], "av": 0.016666666666666663, "o": {"o": "stellarpunk.orders.GoToLocation", "ct": "e5e92226-9c6b-4dd4-a9dd-8e90f4ce43f4", "ct_loc": [128406.45773998012, -4516.575837068859], "ct_ts": 9.816666666666734, "cac": False, "nnd": 44133.001729957134, "t_loc": [-132817.46981686977, -119690.80145835043], "cs": False}}

    b = {"eid": "e5e92226-9c6b-4dd4-a9dd-8e90f4ce43f4", "ts": 9.833333333333401, "loc": [128406.45773998012, -4516.575837068859], "a": 2.0580617926511264, "v": [-795.4285804691448, 1491.5775925290025], "av": -0.016666666666666663, "o": {"o": "stellarpunk.orders.GoToLocation", "ct": "bebefe43-24b3-4588-9b42-4f5504de5903", "ct_loc": [140998.48779162427, 37761.003422847934], "ct_ts": 9.816666666666734, "cac": False, "nnd": 35079.805827675904, "t_loc": [18374.44894231548, 201848.06161807684], "cs": False}}

    ship_a = ship_from_history(a, generator, sector)
    ship_b = ship_from_history(b, generator, sector)

    goto_a = order_from_history(a, ship_a, gamestate)
    goto_b = order_from_history(b, ship_b, gamestate)

    eta = max(goto_a.eta(), goto_b.eta())

    def tick(timeout):
        assert not simulator.collisions

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

    #core.write_history_to_file(goto_a.ship, "/tmp/stellarpunk_test.history")
    #core.write_history_to_file(goto_b.ship, "/tmp/stellarpunk_test.history", mode="a")

def test_ship_existing_velocity(gamestate, generator, sector, testui, simulator):
    # ship headed in one direciton, wants to go 90 deg to it, almost collides
    # with a distant object
    # testcase from gameplay logs

    ship_driver = generator.spawn_ship(sector, -61548.10777914036, -122932.75622689343, v=[130.58825256350576, -20.791840524660724], w=-0.4600420747138861, theta=-0.10231674372628569)
    ship_blocker = generator.spawn_station(sector, -45858.953065820686, -126065.49162802949, resource=0)

    goto_order = orders.GoToLocation(np.array([-61165.07884422924, -152496.78251442552]), ship_driver, gamestate)
    ship_driver.orders.append(goto_order)

    distance = np.linalg.norm(ship_driver.loc)
    eta = goto_order.eta()

    def tick(timeout):
        assert not simulator.collisions

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

def test_collision_flapping(gamestate, generator, sector, testui, simulator):
    """ Illustrates "flapping" in collision detection between the target and
    the collision threat, which makes avoiding the collision very slow. """

    log_entry = {"eid": "54ac288f-f321-4a5d-b681-06304946c1c5", "ts": 24.316986544634826, "loc": [-33555.48438201977, 26908.30401095389], "a": -1.6033951624880438, "v": [-30.158483917339932, -196.17277081634103], "av": -0.303054933539972, "o": {"o": "stellarpunk.orders.GoToLocation", "ct": "5d23b4c8-7fcd-463c-b46e-bce1f5daf1ff", "ct_loc": [-40857.126658436646, -16386.73414552246], "ct_ts": 24.30031987796816, "cac": False, "cbdr": False, "nnd": 43909.734240760576, "t_loc": [-58968.88094427537, -50074.22099620187], "cs": False}}

    ship_driver = ship_from_history(log_entry, generator, sector)
    blocker = generator.spawn_station(sector, -40857.126658436646, -16386.73414552246, resource=0)

    goto_order = order_from_history(log_entry, ship_driver, gamestate)

    starttime = gamestate.timestamp
    distance = np.linalg.norm(ship_driver.loc - goto_order.target_location)
    eta = goto_order.eta()

    def tick(timeout):
        assert not simulator.collisions

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

    core.write_history_to_file(goto_order.ship, "/tmp/stellarpunk_test.history")

@write_history
def test_double_threat(gamestate, generator, sector, testui, simulator):
    """ Illustrates two threats close together on opposite sides of the desired
    vector. Collision detection will potentially ignore one and steer into it
    while trying to avoid the other."""

    a = {"eid":"c99bf358-5ed8-45b6-94ae-c109de01fd6d","ts":173.88333333336385,"loc":[11238.533139689527,2642.1486435851307],"a":-4.462215755014741,"v":[8.673617379884035e-19,-1.734723475976807e-18],"av":-0.5601751626636616,"f":[-2171.128156091397,-4504.0207070824135],"t":900000,"o":{"o":"stellarpunk.orders.GoToLocation","ct":"230d82ca-b3a5-4d65-90c4-9ec4b561afa8","ct_loc":[10900.178689763814,1713.1145124524808],"ct_ts":173.88333333336385,"ct_dv":[9.1603417240449e-18,-4.415671492188985e-18],"cac":False,"cbdr":False,"nnd":988.7305753307772,"t_loc":[-88985.9230279687,-205274.1941428623],"t_v":[-434.22563121827943,-900.8041414164829],"cs":False}}

    b = {"eid":"8bba20d1-2f53-40b7-bea1-a758a1447a77","ts":173.88333333336385,"loc":[11885.659312275971,-629.4315137146432],"a":-1.0588787644372917,"v":[-97.96382930038575,232.86584616015844],"av":-0.09934969312887348,"f":[-1938.8568302904862,4608.7779499164335],"t":900000,"o":{"o":"stellarpunk.orders.GoToLocation","nnd":2541.396061628005,"t_loc":[10900.178689763814,1713.1145124524808],"t_v":[-115.13388981445071,273.6801007557891],"cs":False}}

    c = {"eid":"230d82ca-b3a5-4d65-90c4-9ec4b561afa8","ts":0,"loc":[10900.178689763814,1713.1145124524808],"a":0,"v":[0,0],"av":0,"f":0,"t":0,"o":None}

    ship_a = ship_from_history(a, generator, sector)
    ship_b = ship_from_history(b, generator, sector)
    station = station_from_history(c, generator, sector)

    goto_a = order_from_history(a, ship_a, gamestate)
    goto_a.target_location = ship_a.loc + (goto_a.target_location  - ship_a.loc)/10
    goto_b = order_from_history(b, ship_b, gamestate)

    eta = goto_a.eta()

    testui.eta = eta
    testui.orders = [goto_a]
    testui.cannot_stop_orders = [goto_a]
    testui.cannot_avoid_collision_orders = [goto_a, goto_b]
    testui.margin_neighbors = [ship_a]
    simulator.run()
    assert goto_a.is_complete()

@write_history
def test_ct_near_target(gamestate, generator, sector, testui, simulator):
    # This case caused a collision while running, but I think it was because of
    # changing dt, perhaps because of a mouse click. it doesn't repro in test.
    a = {"eid": "a06358ed-5d1c-4026-b978-c6d05b65b971", "ts": 73.23596008924422, "loc": [33817.46867325524, -2802.702863489674], "a": 0.6501890587823068, "v": [-1516.455517907544, -865.0005079951259], "av": 1.4302311626798327, "o": {"o": "stellarpunk.orders.GoToLocation", "nnd": 16950.413686796317, "t_loc": [19117.170259352486, -11241.763871430074], "t_v": [-1419.7763852577352, -815.0568917357202], "cs": False}}
    b = {"eid": "30ece38b-26e7-470e-8791-9096b0a9fd33", "ts": 73.23596008924422, "loc": [15113.769651997736, -3413.9408398308624], "a": 7.021124149518508, "v": [1540.1197961995574, -830.7283085974165], "av": -0.6830200566843943, "o": {"o": "stellarpunk.orders.GoToLocation", "nnd": 8792.157291416223, "t_loc": [62361.39816239622, -27754.575109759873], "t_v": [1555.6927812270774, -801.4486698709783], "cs": False}}
    c = {"eid": "41c3a7aa-6d60-420a-b89d-362792d74283", "ts": 73.23596008924422, "loc": [57385.92081958368, -25954.613633593606], "a": 2.821461155469132, "v": [1098.0665620846898, -397.24379669298503], "av": 0.03190852110518538, "o": {"o": "stellarpunk.orders.GoToLocation", "nnd": 5291.052457169889, "t_loc": [62361.39816239622, -27754.575109759873], "t_v": [1091.9167479758394, -395.0189994084582], "cs": False}}


    ship_a = ship_from_history(a, generator, sector)
    ship_b = ship_from_history(b, generator, sector)
    ship_c = ship_from_history(c, generator, sector)

    goto_a = order_from_history(a, ship_a, gamestate)
    goto_b = order_from_history(b, ship_b, gamestate)
    goto_c = order_from_history(c, ship_c, gamestate)

    blocker = generator.spawn_station(sector, 19117.170259352486, -11241.763871430074, resource=0)

    eta = goto_a.eta()

    testui.eta = eta
    testui.orders = [goto_a]
    testui.cannot_avoid_collision_orders = [goto_a]
    testui.cannot_stop_orders = [goto_a]
    testui.margin_neighbors = [ship_a]

    simulator.run()
    assert goto_a.is_complete()

@write_history
def test_many_threats(gamestate, generator, sector, testui, simulator):
    a = {"eid": "c2066f5f-80b0-4972-be15-86731721d0ac", "ts": 181.00000000003618, "loc": [-156664.6196115718, 15103.774316939725], "a": 6.706346854557575, "v": [-566.413704366243, -427.2391616051364], "av": 0.22045526531291454, "f": [3991.7686372857047, 3010.943896252839], "t": 900000.0, "o": {"o": "stellarpunk.orders.GoToLocation", "nnd": np.inf, "t_loc": [-162728.94555641068, 10529.524943379436], "t_v": [-527.0746776061375, -397.5661987481488], "cs": False}}
    b = {"eid": "efd8dd82-59e9-4f71-97fc-96d16be37101", "ts": 181.00000000003618, "loc": [-155648.10021655622, 14496.034982381125], "a": 6.631761022100543, "v": [-633.9212228300115, -355.10659725031456], "av": 0.2932266489261978, "f": [436220.52590004867, 244359.6791278892], "t": -233347.4140426577, "o": {"o": "stellarpunk.orders.GoToLocation", "nnd": np.inf, "t_loc": [-162728.94555641068, 10529.524943379436],
"t_v": [-607.3950624974589, -340.2473147486867], "cs": False}}
    c = {"eid": "1688d213-9d49-4222-aa0b-95dc090c59fa", "ts": 181.00000000003618, "loc": [-161271.80180938027, 11201.79460589419], "a": 0.17012052360683455, "v": [-123.61634016109224, -57.03178938300922], "av": 0.04141546936831053, "f": [4540.10299648228, 2094.627599677953], "t": 900000.0, "o": {"o": "stellarpunk.orders.GoToLocation", "nnd": np.inf, "t_loc": [-162728.94555641068, 10529.524943379436], "t_v": [-120.13404078695115, -55.42519138620854], "cs": False}}
    d = {"eid": "f9cb7b45-858e-4422-a3b1-9513343b2fb7", "ts": 0, "loc": [-162728.94555641068, 10529.524943379436], "a": 0.0, "v": [0.0, 0.0], "av": 0.0, "f": [0.0, 0.0], "t": 0, "o": None}


    ship_a = ship_from_history(a, generator, sector)
    ship_b = ship_from_history(b, generator, sector)
    ship_c = ship_from_history(c, generator, sector)

    goto_a = order_from_history(a, ship_a, gamestate)
    goto_b = order_from_history(b, ship_b, gamestate)
    goto_c = order_from_history(c, ship_c, gamestate)

    station = station_from_history(d, generator, sector)

    eta = goto_a.eta()

    testui.eta = eta
    testui.orders = [goto_a]
    #testui.cannot_avoid_collision_orders = [goto_a]
    #testui.cannot_stop_orders = [goto_a]
    testui.margin_neighbors = [ship_a]

    simulator.run()
    assert goto_a.is_complete()
