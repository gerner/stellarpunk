""" Tests for the collision helper lib """

import numpy as np
import cymunk # type: ignore

from stellarpunk.orders import collision

def test_analyze_neighbors(gamestate, generator, sector, testui, simulator):
    ship_a = generator.spawn_ship(sector, 0, 0, v=(0,0), w=0, theta=0)
    ship_b = generator.spawn_ship(sector, 100, 100, v=(0,0), w=0, theta=0)
    ship_c = generator.spawn_ship(sector, 1000, 1000, v=(0,0), w=0, theta=0)

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
            coalesced_neighbors,
    ) = collision.analyze_neighbors(
            ship_a.phys, sector.space,
            1e4, ship_a.radius, 5e2,
            ship_a.phys.position, 1e4,
            ship_a.max_acceleration())

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
            coalesced_neighbors,
    ) = collision.analyze_neighbors(
            ship_a.phys, sector.space,
            max_distance=1e4,
            ship_radius=30.,
            margin=5e2,
            neighborhood_loc=ship_a.phys.position,
            neighborhood_radius=1e4,
            maximum_acceleration=100.)

    assert threat.data == other_ships[0]
    assert np.allclose(np.array(rel_pos), (2000., 0.))
    assert np.allclose(np.array(rel_vel), v*-1)
    assert min_sep == 0.
    assert threat_count == 4
    assert coalesced_threats == 3

    for neighbor in coalesced_neighbors:
        assert np.linalg.norm(np.array(neighbor.position) - np.array(threat_loc))+neighbor.data.radius <= threat_radius

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
