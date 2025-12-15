""" Tests for the collision helper lib """

from typing import Tuple

import numpy as np
import cymunk # type: ignore

from stellarpunk import util, task_schedule, collision

def test_analyze_neighbors(gamestate, generator, sector, testui, simulator):
    ship_a = generator.spawn_ship(sector, 0, 0, v=(0,0), w=0, theta=0)
    ship_b = generator.spawn_ship(sector, 100, 100, v=(0,0), w=0, theta=0)
    ship_c = generator.spawn_ship(sector, 1000, 1000, v=(0,0), w=0, theta=0)

    neighbor_analyzer = collision.Navigator(
            sector.space, ship_a.phys,
            ship_a.radius,
            ship_a.max_thrust, ship_a.max_torque, ship_a.max_speed(),
            5e2, 1e4,
    )
    (
            threat,
            approach_time,
            rel_pos,
            rel_vel,
            min_sep,
            threat_count,
            coalesced_threats,
            non_coalesced_threats,
            threat_radius,
            threat_loc,
            threat_velocity,
            nearest_neighbor_idx,
            nearest_neighbor_dist,
            neighborhood_density,
            num_neighbors,
            prior_threat_count,
    ) = neighbor_analyzer.analyze_neighbors(
            0.,
            1e4,
    )

    assert threat.data == ship_b
    assert threat_count == 1
    assert num_neighbors == 2
    assert approach_time == 0
    assert np.isclose(min_sep, np.linalg.norm(np.array((100,100))))
    assert rel_pos == np.array((100,100))
    assert rel_vel == np.array((0,0))

def test_coalesce(generator, sector):
    """ Test that coalescing threats actually covers them. """

    # set up: ship heading toward several static points all within twice the
    # collision margin

    v = np.array((10.,0.))
    ship_a = generator.spawn_ship(sector, 0., 0., v=v, w=0, theta=0)

    other_ships = []
    for pos in (
        ((2000., 0.)),
        ((2500., 500.)),
        ((2500., -500.)),
        ((5000., 0.)), # not part of the group, but a threat
        ((2500., -5000.)), # not part of the group
        ):
        other_ships.append(generator.spawn_ship(sector, pos[0], pos[1], v=(0,0), w=0, theta=0))

    neighbor_analyzer = collision.Navigator(
            sector.space, ship_a.phys,
            ship_a.radius,
            ship_a.max_thrust, ship_a.max_torque, ship_a.max_speed(),
            5e2, 1e4,
    )
    (
            threat,
            approach_time,
            rel_pos,
            rel_vel,
            min_sep,
            threat_count,
            coalesced_threats,
            non_coalesced_threats,
            threat_radius,
            threat_loc,
            threat_velocity,
            nearest_neighbor_idx,
            nearest_neighbor_dist,
            neighborhood_density,
            num_neighbors,
            prior_threat_count,
    ) = neighbor_analyzer.analyze_neighbors(
            0.,
            max_distance=1e4,
    )

    assert threat.data == other_ships[0]
    assert np.allclose(np.array(rel_pos), (2000., 0.))
    assert np.allclose(np.array(rel_vel), v*-1)
    assert min_sep == 0.
    assert threat_count == 4
    assert coalesced_threats == 3

    for neighbor_loc in neighbor_analyzer.coalesced_neighbor_locations():
        neighbor_radius = ship_a.radius # assume all ships have same radius
        assert np.linalg.norm(np.array(neighbor_loc) - np.array(threat_loc))+neighbor_radius <= threat_radius

def test_force_for_delta_velocity():
    # dv = f/m * t

    # time constrained (timestep is long enough that we have to reduce our force)
    (force, time) = collision.force_for_delta_velocity(cymunk.Vec2d(10,0), 2, 100, 20)
    # 10 = f/2 * 20
    # 1 = f
    assert force.length <= 100
    assert force.x == 1.0
    assert force.y == 0.0
    assert time == 20

    # force constrained (timestep is short enough that we want to exceed max thrust)
    (force, time) = collision.force_for_delta_velocity(cymunk.Vec2d(10,0), 2, 10, 1)
    # 10 = 10/2 * t
    # 2 = t
    assert force.length <= 10
    assert force.x == 10
    assert force.y == 0.0
    assert time == 2

def test_enclosing_circle():
    def contains(c1, r1, c2, r2):
        d = util.distance(np.array(c1),np.array(c2))
        return r1+5e-1 >= d+r2
    c1 = cymunk.Vec2d(-5, 0)
    r1 = 10.
    c2 = cymunk.Vec2d(5,0)
    r2 = 10.
    c, r = collision.make_enclosing_circle(c1, r1, c2, r2)
    assert contains(c, r, c1, r1)
    assert contains(c, r, c2, r2)
    assert r == 15
    assert c[0] == 0
    assert c[1] == 0

    c1 = cymunk.Vec2d(-15235.3324, 2342348.53)
    r1 = 30
    c2 = cymunk.Vec2d(-16225.5422, 23432641.48)
    r2 = 88.7789
    c, r = collision.make_enclosing_circle(c1, r1, c2, r2)
    assert contains(c, r, c1, r1)
    assert contains(c, r, c2, r2)

def test_collision_dv():
    expected_dv = cymunk.Vec2d(385.087433, 49.343849)

    ct_loc = cymunk.Vec2d(4708.002441, -11.092171)
    ct_v = cymunk.Vec2d(-243.264709, 1.169649)
    loc = cymunk.Vec2d(-4708.002441, 0.038988)
    vel = cymunk.Vec2d(243.264709, 1.169649)
    margin = 352.9983135407874
    desired_v = cymunk.Vec2d(386.92272949,  -1.17131988)
    cbdr = False
    cbdr_bias = 2
    delta_v_budget = 1790.6210182137115

    #from stellarpunk.orders import steering
    #delta_velocity = steering._collision_dv(
    delta_velocity = collision.collision_dv(
            ct_loc, ct_v,
            loc, vel,
            margin, desired_v,
            cbdr, cbdr_bias,
            delta_v_budget,
    )

    assert all(np.isclose(delta_velocity, expected_dv))

def test_intercept_heading():
    start_loc = cymunk.Vec2d(0., 0.)
    start_v = cymunk.Vec2d(0.0, 0.0)
    target_loc = cymunk.Vec2d(10., 10.)
    target_v = cymunk.Vec2d(0., 0.)
    muzzle_velocity = 1000.

    intercept_time, intercept_loc, intercept_heading = collision.find_intercept_heading(start_loc, start_v, target_loc, target_v, muzzle_velocity)

    assert np.isclose(intercept_time, np.linalg.norm(target_loc) / muzzle_velocity)
    assert np.isclose(intercept_heading, np.pi/4.)
    assert all(np.isclose(intercept_loc, np.array((10., 10.))))

    start_loc = cymunk.Vec2d(0., 0.)
    start_v = cymunk.Vec2d(10., 10.)
    target_loc = cymunk.Vec2d(10., 10.)
    target_v = cymunk.Vec2d(10., 10.)
    muzzle_velocity = 1000.

    intercept_time, intercept_loc, intercept_heading = collision.find_intercept_heading(start_loc, start_v, target_loc, target_v, muzzle_velocity)

    assert np.isclose(intercept_time, np.linalg.norm(target_loc) / muzzle_velocity)
    assert np.isclose(intercept_heading, np.pi/4.)
    assert all(np.isclose(intercept_loc, np.array((10., 10.))+target_v*intercept_time))
    start_loc = cymunk.Vec2d(0., 0.)
    start_v = cymunk.Vec2d(0., 0.)
    target_loc = cymunk.Vec2d(10., 10.)
    target_v = cymunk.Vec2d(-10., -10.)
    muzzle_velocity = 1000.

    intercept_time, intercept_loc, intercept_heading = collision.find_intercept_heading(start_loc, start_v, target_loc, target_v, muzzle_velocity)

    assert np.isclose(intercept_time, np.linalg.norm(target_loc) / (muzzle_velocity + np.linalg.norm(target_v)))
    assert np.isclose(intercept_heading, np.pi/4.)
    assert all(np.isclose(intercept_loc, np.array((10., 10.))+target_v*intercept_time))
    start_loc = cymunk.Vec2d(0., 0.)
    start_v = cymunk.Vec2d(20.0, 0.0)
    target_loc = cymunk.Vec2d(10., 10.)
    target_v = cymunk.Vec2d(20., 0.)
    muzzle_velocity = 1000.

    intercept_time, intercept_loc, intercept_heading = collision.find_intercept_heading(start_loc, start_v, target_loc, target_v, muzzle_velocity)

    assert np.isclose(intercept_time, np.linalg.norm(target_loc) / muzzle_velocity)
    assert np.isclose(intercept_heading, np.pi/4.)
    assert all(np.isclose(intercept_loc, np.array((10., 10.)+target_v*intercept_time)))


def test_task_schedule():
    schedule:task_schedule.TaskSchedule[Tuple[str, int]] = task_schedule.TaskSchedule()
    schedule.push_task(5, ("a", 47))
    assert schedule.top() == ("a", 47)
    schedule.push_task(3, ("b", 42))
    assert schedule.top() == ("b", 42)
    schedule.push_task(7, ("c", 41))
    assert schedule.top() == ("b", 42)

    assert schedule.empty(2)
    assert not schedule.empty(4)
    assert not schedule.empty(9)

    tasks = schedule.pop_current_tasks(2)
    assert tasks == []

    tasks = schedule.pop_current_tasks(6)
    assert tasks == [("b", 42), ("a", 47)]

    tasks = schedule.pop_current_tasks(7)
    assert tasks == [("c", 41)]

    tasks = schedule.pop_current_tasks(8)
    assert tasks == []

def test_cancel_task():
    schedule:task_schedule.TaskSchedule[Tuple[str, int]] = task_schedule.TaskSchedule()
    schedule.push_task(5, ("a", 47))
    assert schedule.top() == ("a", 47)
    schedule.push_task(3, ("b", 42))
    assert schedule.top() == ("b", 42)
    schedule.push_task(7, ("c", 41))
    assert schedule.top() == ("b", 42)

    schedule.cancel_task(("b", 42))
    assert schedule.top() == ("a", 47)

    tasks = schedule.pop_current_tasks(6)
    assert tasks == [("a", 47)]

    tasks = schedule.pop_current_tasks(7)
    assert tasks == [("c", 41)]

    tasks = schedule.pop_current_tasks(8)
    assert tasks == []

def test_get_set_navigator_parameters(generator, sector):
    ship_a = generator.spawn_ship(sector, 0, 0, v=(0,0), w=0, theta=0)
    ship_b = generator.spawn_ship(sector, 100, 100, v=(0,0), w=0, theta=0)
    ship_c = generator.spawn_ship(sector, 1000, 1000, v=(0,0), w=0, theta=0)

    neighbor_analyzer = collision.Navigator(
            sector.space, ship_a.phys,
            ship_a.radius,
            ship_a.max_thrust, ship_a.max_torque, ship_a.max_speed(),
            5e2, 1e4,
    )
    (
            threat,
            approach_time,
            rel_pos,
            rel_vel,
            min_sep,
            threat_count,
            coalesced_threats,
            non_coalesced_threats,
            threat_radius,
            threat_loc,
            threat_velocity,
            nearest_neighbor_idx,
            nearest_neighbor_dist,
            neighborhood_density,
            num_neighbors,
            prior_threat_count,
    ) = neighbor_analyzer.analyze_neighbors(
            0.,
            1e4,
    )

    assert threat.data == ship_b
    assert threat_count == 1
    assert num_neighbors == 2
    assert approach_time == 0
    assert np.isclose(min_sep, np.linalg.norm(np.array((100,100))))
    assert rel_pos == np.array((100,100))
    assert rel_vel == np.array((0,0))

    neighbor_analyzer2 = collision.Navigator(
            sector.space, ship_a.phys,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    )

    neighbor_analyzer2.set_navigator_parameters(neighbor_analyzer.get_navigator_parameters())

    (
            threat,
            approach_time,
            rel_pos,
            rel_vel,
            min_sep,
            threat_count,
            coalesced_threats,
            non_coalesced_threats,
            threat_radius,
            threat_loc,
            threat_velocity,
            nearest_neighbor_idx,
            nearest_neighbor_dist,
            neighborhood_density,
            num_neighbors,
            prior_threat_count,
    ) = neighbor_analyzer2.analyze_neighbors(
            0.,
            1e4,
    )

    assert threat.data == ship_b
    assert threat_count == 1
    assert num_neighbors == 2
    assert approach_time == 0
    assert np.isclose(min_sep, np.linalg.norm(np.array((100,100))))
    assert rel_pos == np.array((100,100))
    assert rel_vel == np.array((0,0))

