""" Tests for ship orders and behaviors. """

import logging

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
        self.margin_neighbors = []
        self.eta = np.inf

    def margin_neighbors(self, margin_neighbors):
        self.margin_neighbors = margin_neighbors

    def tick(self, timeout):
        assert not self.simulator.collisions

        for margin_neighbor in self.margin_neighbors:
            neighbor, neighbor_dist = nearest_neighbor(sector, margin_neighbor)
            assert neighbor_dist >= 300

        if all(map(lambda x: x.is_complete(), self.orders)):
            self.gamestate.quit()

        assert self.gamestate.timestamp < self.eta

        assert all(map(lambda x: not x.cannot_stop, self.cannot_stop_orders))
        assert all(map(lambda x: not x.cannot_avoid_collision, self.orders))

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

    util.write_history_to_file(rotate_order.ship, "/tmp/stellarpunk_test.history")

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

    util.write_history_to_file(rotate_order.ship, "/tmp/stellarpunk_test.history")

def test_basic_gotolocation(gamestate, generator, sector, testui, simulator):
    ship_driver = generator.spawn_ship(sector, -400, 15000, v=(0,0), w=0, theta=0)

    goto_order = orders.GoToLocation(np.array((0,0)), ship_driver, gamestate)
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

    util.write_history_to_file(goto_order.ship, "/tmp/stellarpunk_test.history")

def test_gotolocation_with_entity_target(gamestate, generator, sector, testui, simulator):
    ship_driver = generator.spawn_ship(sector, -400, 15000, v=(0,0), w=0, theta=0)
    ship_blocker = generator.spawn_ship(sector, 0, 0, v=(0,0), w=0, theta=0)

    goto_order = orders.GoToLocation(np.array((0,0)), ship_driver, gamestate)
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
    ship_driver.velocity = np.array((0, -10)) * 50
    ship_driver.phys.velocity = tuple(ship_driver.velocity)

    goto_order = orders.GoToLocation(np.array((0,0)), ship_driver, gamestate)
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
    ship_driver.velocity = np.array((-4, -10)) * 50
    ship_driver.phys.velocity = tuple(ship_driver.velocity)

    goto_order = orders.GoToLocation(np.array((0,0)), ship_driver, gamestate)
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

def test_basic_collision_avoidance(gamestate, generator, sector, testui, simulator, caplog):
    # set up a sector, including space
    # add two ships in an offset (relative to desired course) arrangement
    # travel along course and observe no collision

    ship_driver = generator.spawn_ship(sector, 0, 2400, v=(0,0), w=0, theta=0)
    ship_blocker = generator.spawn_ship(sector, -300, 1200, v=(0,0), w=0, theta=0)

    goto_order = orders.GoToLocation(np.array((0,0)), ship_driver, gamestate)
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
        #logging.debug(f'{ship_driver.loc} {ship_driver.velocity} {ship_driver.angle} {gamestate.timestamp - starttime}s vs {eta}s')
        assert not simulator.collisions
        neighbor, neighbor_dist = nearest_neighbor(sector, ship_driver)
        assert neighbor_dist >= goto_order.collision_margin
        if goto_order.is_complete():
            gamestate.quit()
        assert not goto_order.cannot_stop
        assert not goto_order.cannot_avoid_collision
        assert not goto_order.collision_cbdr
        assert gamestate.timestamp - starttime < eta*1.6

    testui.tick = tick
    simulator.run()
    assert goto_order.is_complete()

    #util.write_history_to_file(goto_order.ship, "/tmp/stellarpunk_test.history")

def test_head_on_static_collision_avoidance(gamestate, generator, sector, testui, simulator):
    ship_blocker = generator.spawn_ship(sector, 0, 1200, v=(0,0), w=0, theta=0)
    ship_driver = generator.spawn_ship(sector, 0, 2400, v=(0,0), w=0, theta=0)

    goto_order = orders.GoToLocation(np.array((0,0)), ship_driver, gamestate)
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

    #util.write_history_to_file(goto_order.ship, "/tmp/stellarpunk_test.history")

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

    ship_driver = generator.spawn_ship(sector, -400, 20000, v=(0,0), w=0, theta=0)
    ship_driver.velocity = np.array((0, 0))
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
    util.write_history_to_file(goto_order.ship, "/tmp/stellarpunk_test.history")

def test_simple_ships_intersecting(gamestate, generator, sector, testui, simulator):
    ship_a = generator.spawn_ship(sector, -5000, 0, v=(0,0), w=0, theta=0)
    ship_b = generator.spawn_ship(sector, 0, -5000, v=(0,0), w=0, theta=np.pi/2)

    goto_a = orders.GoToLocation(np.array((5000,0)), ship_a, gamestate)
    ship_a.orders.append(goto_a)
    goto_b = orders.GoToLocation(np.array((0,5000)), ship_b, gamestate)
    ship_b.orders.append(goto_b)

    a_cbdr = False
    b_cbdr = False

    eta = max(goto_a.eta(), goto_b.eta())

    def tick(timeout):
        nonlocal a_cbdr
        nonlocal b_cbdr
        r,theta = util.cartesian_to_polar(*goto_a.collision_dv)
        logging.debug(f'collision_dv: {r} {theta} {goto_a.ship.angle}')
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
        a_cbdr = a_cbdr or goto_a.collision_cbdr
        b_cbdr = b_cbdr or goto_b.collision_cbdr

        assert gamestate.timestamp < eta

    testui.tick = tick
    simulator.run()
    assert goto_a.is_complete()
    assert not goto_a.collision_cbdr
    assert goto_b.is_complete()
    assert not goto_b.collision_cbdr
    assert a_cbdr
    assert b_cbdr

    util.write_history_to_file(goto_a.ship, "/tmp/stellarpunk_test.history")
    util.write_history_to_file(goto_b.ship, "/tmp/stellarpunk_test.history", mode="a")

def test_headon_ships_intersecting(gamestate, generator, sector, testui, simulator):
    ship_a = generator.spawn_ship(sector, -5000, 0, v=(0,0), w=0, theta=0)
    ship_b = generator.spawn_ship(sector, 5000, 0, v=(0,0), w=0, theta=np.pi)

    goto_a = orders.GoToLocation(np.array((10000,0)), ship_a, gamestate)
    ship_a.orders.append(goto_a)
    goto_b = orders.GoToLocation(np.array((-10000,0)), ship_b, gamestate)
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

    util.write_history_to_file(goto_a.ship, "/tmp/stellarpunk_test.history")
    util.write_history_to_file(goto_b.ship, "/tmp/stellarpunk_test.history", mode="a")
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

    #util.write_history_to_file(goto_a.ship, "/tmp/stellarpunk_test.history")
    #util.write_history_to_file(goto_b.ship, "/tmp/stellarpunk_test.history", mode="a")

@pytest.mark.skip
def test_ship_existing_velocity(gamestate, generator, sector, testui, simulator):
    # ship headed in one direciton, wants to go 90 deg to it, almost collides
    # with a distant object
    # testcase from gameplay logs

    ship_driver = generator.spawn_ship(sector, -61548.10777914036, -122932.75622689343, v=[130.58825256350576, -20.791840524660724], w=-0.4600420747138861, theta=-0.10231674372628569)
    ship_blocker = generator.spawn_station(sector, -45858.953065820686, -126065.49162802949, resource=0)

    goto_order = orders.GoToLocation(np.array([-61165.07884422924, -152496.78251442552]), ship_driver, gamestate)
    ship_driver.orders.append(goto_order)

    distance = np.linalg.norm(ship_driver.loc)
    eta = goto_order.eta

    def tick(timeout):
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

    util.write_history_to_file(goto_order.ship, "/tmp/stellarpunk_test.history")

def test_double_threat(gamestate, generator, sector, testui, simulator):
    """ Illustrates two threats close together on opposite sides of the desired
    vector. Collision detection will potentially ignore one and steer into it
    while trying to avoid the other."""

    a = {"eid": "930b8a10-031c-441a-856a-cb2fb1290cd0", "ts": 30.008905357101778, "loc": [-72143.76532798856, -78624.06548147832], "a": 2.9133531117749483, "v": [1463.036867755909, -996.9419093935754], "av": 0.4587402180730085, "o": {"o": "stellarpunk.orders.GoToLocation", "ct": "21983a29-1531-46bc-b6c8-0ef07425b7c1", "ct_loc": [-40624.746151124345, -100866.35271782603], "ct_ts": 30.008905357101778, "cac": False, "cbdr": False, "nnd": 36974.430981702404, "t_loc": [-36920.864211910965, -103518.43254502042], "cs": False}}
    b = {"eid": "16da5d1f-7f10-404d-9d72-8a619a0e6dcb", "ts": 30.008905357101778, "loc": [-43429.36600606786, -53155.60502547812], "a": 1.6291834169699106, "v": [235.54041525110756, -4006.892165292321], "av": 0.0208950231759309, "o": {"o": "stellarpunk.orders.GoToLocation", "nnd": 10321.52131236344, "t_loc": [-40624.746151124345, -100866.35271782603], "cs": False}}

    ship_a = ship_from_history(a, generator, sector)
    ship_b = ship_from_history(b, generator, sector)

    goto_a = order_from_history(a, ship_a, gamestate)
    goto_b = order_from_history(b, ship_b, gamestate)

    planet = generator.spawn_planet(sector, -40624.746151124345, -100866.35271782603)

    starttime = gamestate.timestamp
    distance = np.linalg.norm(ship_a.loc - goto_a.target_location)

    #we really only care about a here. b might run into the planet or something
    eta = goto_a.eta()

    def tick(timeout):
        assert not simulator.collisions

        # only need to check one, they are symmetric
        neighbor, neighbor_dist = nearest_neighbor(sector, ship_a)
        assert neighbor_dist >= goto_a.collision_margin

        if goto_a.is_complete():
            gamestate.quit()
        #assert not goto_a.cannot_stop
        #assert not goto_a.cannot_avoid_collision
        #assert not goto_b.cannot_stop
        #assert not goto_b.cannot_avoid_collision
        assert gamestate.timestamp - starttime < eta# * 1.5

    testui.tick = tick
    simulator.run()
    assert goto_a.is_complete()
    #assert goto_b.is_complete()

    util.write_history_to_file(goto_a.ship, "/tmp/stellarpunk_test.history")
    util.write_history_to_file(goto_b.ship, "/tmp/stellarpunk_test.history", mode="a")
