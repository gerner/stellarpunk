from typing import Tuple
import cython
from libcpp.vector cimport vector
from libcpp.set cimport set

import numpy.typing as npt
import numpy as np
import cymunk
cimport cymunk.cymunk as ccymunk
cimport libc.math as math

cdef struct CollisionThreat:
    ccymunk.cpShape *threat_shape
    ccymunk.cpVect c_loc

cdef struct NeighborAnalysis:
    double neighborhood_radius
    double ship_radius
    double margin
    double max_distance
    double maximum_acceleration
    ccymunk.cpBody *body

    double worst_ultimate_separation
    double nearest_neighbor_dist
    double approach_time
    ccymunk.cpVect relative_position
    ccymunk.cpVect relative_velocity
    double minimum_separation
    ccymunk.cpVect collision_loc
    int neighborhood_size
    int threat_count
    ccymunk.cpShape *threat_shape

    ccymunk.cpShape *nearest_neighbor_shape
    vector[CollisionThreat] collision_threats
    vector[ccymunk.cpShape *] coalesced_threats
    set[ccymunk.cpHashValue] considered_shapes

    double threat_radius
    ccymunk.cpVect threat_loc
    ccymunk.cpVect threat_velocity

cdef struct AnalyzedNeighbor:
    double rel_dist
    double approach_t
    ccymunk.cpVect rel_pos
    ccymunk.cpVect rel_vel
    double min_sep
    ccymunk.cpVect c_loc
    double collision_distance

cdef ccymunk.cpVect ZERO_VECTOR = ccymunk.cpv(0.,0.)
cdef ccymunk.Vec2d PY_ZERO_VECTOR = ccymunk.Vec2d(0.,0.)
cdef double VELOCITY_EPS = 5e-1

# cymunk fails to export this type from chipmunk
ctypedef struct cpCircleShape :
    ccymunk.cpShape shape;
    ccymunk.cpVect c
    ccymunk.cpVect tc
    ccymunk.cpFloat r

cdef ccymunk.Vec2d cpvtoVec2d(ccymunk.cpVect v):
    return ccymunk.Vec2d(v.x, v.y)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.infer_types(False)
@cython.nonecheck(False)
cdef AnalyzedNeighbor _analyze_neighbor(NeighborAnalysis *analysis, ccymunk.cpShape *shape, double margin):
    # we make the assumption that all shapes are circles here
    cdef cpCircleShape *circle_shape = <cpCircleShape *>shape
    cdef double entity_radius = circle_shape.r
    cdef ccymunk.cpVect entity_pos = shape.body.p
    cdef ccymunk.cpVect entity_v = shape.body.v
    cdef ccymunk.cpVect rel_pos = ccymunk.cpvsub(entity_pos ,analysis.body.p)
    cdef ccymunk.cpVect rel_vel = ccymunk.cpvsub(entity_v, analysis.body.v)

    cdef double rel_speed = ccymunk.cpvlength(rel_vel)
    cdef double rel_dist = ccymunk.cpvlength(rel_pos)

    # check for parallel paths
    if rel_speed == 0:
        if rel_dist < margin + entity_radius:
            # this can cause discontinuities in approach_time
            return AnalyzedNeighbor(rel_dist, 0., rel_pos, rel_vel, rel_dist, entity_pos, 0.)
        return AnalyzedNeighbor(rel_dist, ccymunk.INFINITY, rel_pos, rel_vel, ccymunk.INFINITY, ZERO_VECTOR, ccymunk.INFINITY)

    cdef ccymunk.cpVect rel_tangent = ccymunk.cpvmult(rel_vel, 1. / rel_speed)
    cdef double approach_t = -1 * ccymunk.cpvdot(rel_tangent, rel_pos) / rel_speed

    if approach_t <= 0:
        if rel_dist < margin + entity_radius:
            # this can cause discontinuities in approach_time
            return AnalyzedNeighbor(rel_dist, 0., rel_pos, rel_vel, rel_dist, entity_pos, 0.)
        return AnalyzedNeighbor(rel_dist, ccymunk.INFINITY, rel_pos, rel_vel, ccymunk.INFINITY, ZERO_VECTOR, ccymunk.INFINITY)

    cdef double speed = ccymunk.cpvlength(analysis.body.v)
    # compute the closest approach within max_distance
    cdef double collision_distance = speed * approach_t
    if collision_distance > analysis.max_distance:
        approach_t = analysis.max_distance / speed
        collision_distance = analysis.max_distance - VELOCITY_EPS

    cdef ccymunk.cpVect sep_vec = ccymunk.cpvsub(
            ccymunk.cpvadd(
                analysis.body.p,
                ccymunk.cpvmult(analysis.body.v, approach_t)
            ),
            ccymunk.cpvadd(
                entity_pos,
                ccymunk.cpvmult(entity_v, approach_t)
            )
    )
    cdef double min_sep = ccymunk.cpvlength(sep_vec)
    cdef ccymunk.cpVect collision_loc = ccymunk.cpvadd(entity_pos, ccymunk.cpvmult(entity_v, approach_t))

    return AnalyzedNeighbor(rel_dist, approach_t, rel_pos, rel_vel, min_sep, collision_loc, collision_distance)

cdef void _sensor_point_callback(ccymunk.cpShape *shape, ccymunk.cpFloat distance, ccymunk.cpVect point, void *data):
    _analyze_neighbor_callback(shape, data)

cdef void _sensor_shape_callback(ccymunk.cpShape *shape, ccymunk.cpContactPointSet *points, void *data):
    _analyze_neighbor_callback(shape, data)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.infer_types(False)
@cython.nonecheck(False)
cdef void _analyze_neighbor_callback(ccymunk.cpShape *shape, void *data):
    cdef NeighborAnalysis *analysis = <NeighborAnalysis *>data

    # ignore ourself
    if shape.body == analysis.body:
        return

    # ignore stuff we've already procesed
    if analysis.considered_shapes.count(shape.hashid_private) > 0:
        return

    # keep track of shapes we've considered (we can compose multiple queries)
    analysis.considered_shapes.insert(shape.hashid_private)

    cdef AnalyzedNeighbor neighbor = _analyze_neighbor(analysis, shape, analysis.ship_radius+analysis.margin)

    #if neighbor.rel_dist < analysis.neighborhood_radius:
    analysis.neighborhood_size += 1

    if neighbor.rel_dist < analysis.nearest_neighbor_dist:
        analysis.nearest_neighbor_shape = shape
        analysis.nearest_neighbor_dist = neighbor.rel_dist

    # this neighbor isn't going to collide with us
    if not (neighbor.min_sep < ccymunk.INFINITY):
        return

    # we need to keep track of all collision threats for coalescing later
    analysis.collision_threats.push_back(CollisionThreat(shape, neighbor.c_loc))
    cdef cpCircleShape *circle_shape = <cpCircleShape *>shape
    cdef double entity_radius = circle_shape.r
    if neighbor.min_sep > entity_radius + analysis.ship_radius + analysis.margin:
        return

    analysis.threat_count += 1

    # keep track of "worst margin offender", prioritizing a tradeoff
    # between minimum separation and approach time by assuming we
    # accelerate constantly to maximize the separation
    cdef double ultimate_sep = neighbor.min_sep - entity_radius - analysis.ship_radius + 0.5 * analysis.maximum_acceleration * neighbor.approach_t ** 2
    if ultimate_sep < analysis.worst_ultimate_separation:
        analysis.worst_ultimate_separation = ultimate_sep
        analysis.approach_time = neighbor.approach_t
        analysis.threat_shape = shape
        analysis.relative_position = neighbor.rel_pos
        analysis.relative_velocity = neighbor.rel_vel
        analysis.minimum_separation = neighbor.min_sep
        analysis.collision_loc = neighbor.c_loc

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.infer_types(False)
@cython.nonecheck(False)
cdef void coalesce_threats(NeighborAnalysis *analysis):
    cdef cpCircleShape *circle_shape
    cdef int coalesced_threats
    cdef double t_radius, t_dist, t_angle, coalesced_radius
    cdef CollisionThreat collision_threat
    cdef ccymunk.cpVect t_loc, t_velocity, t_rel_pos, c_rel

    if analysis.threat_count > 0:
        circle_shape = <cpCircleShape *>analysis.threat_shape
        analysis.threat_radius = circle_shape.r
        analysis.threat_loc = analysis.collision_loc
        analysis.threat_velocity = analysis.threat_shape.body.v

        # coalesce nearby threats
        # this avoids flapping in collision targets
        coalesced_threats = 1
        analysis.coalesced_threats.push_back(analysis.threat_shape)
        if analysis.threat_count > 1:
            for collision_threat in analysis.collision_threats:
                if collision_threat.threat_shape == analysis.threat_shape:
                    continue
                t_loc = collision_threat.threat_shape.body.p
                t_velocity = collision_threat.threat_shape.body.v
                circle_shape = <cpCircleShape *>collision_threat.threat_shape
                t_radius = circle_shape.r
                t_rel_pos = ccymunk.cpvsub(analysis.threat_loc, t_loc)
                t_dist = ccymunk.cpvlength(t_rel_pos)
                t_angle = ccymunk.cpvtoangle(t_rel_pos)
                if t_dist + t_radius < analysis.threat_radius:
                    # the old radius completely covers the new one
                    analysis.threat_velocity = ccymunk.cpvmult(
                            ccymunk.cpvadd(ccymunk.cpvmult(analysis.threat_velocity, coalesced_threats), t_velocity),
                            1/(coalesced_threats + 1)
                    )
                    coalesced_threats += 1
                    analysis.coalesced_threats.push_back(collision_threat.threat_shape)
                elif t_dist < analysis.threat_radius + t_radius + 2*analysis.margin:
                    # new is within coalesce dist, but not already covered
                    # coalesced threat should just cover both
                    # diameter = 2*threat_radius + 2*t_radius + (t_dist - threat_radius - t_radius)
                    # diameter = t_dist + threat_radius + t_radius
                    coalesced_radius = (t_dist + analysis.threat_radius + t_radius)/2

                    c_rel = ccymunk.cpvmult(ccymunk.cpvforangle(t_angle+math.pi), coalesced_radius - analysis.threat_radius)

                    analysis.threat_loc = ccymunk.cpvadd(analysis.threat_loc, c_rel)
                    analysis.threat_radius = coalesced_radius
                    analysis.threat_velocity = ccymunk.cpvmult(
                            ccymunk.cpvadd(ccymunk.cpvmult(analysis.threat_velocity, coalesced_threats), t_velocity),
                            1/(coalesced_threats + 1)
                    )
                    coalesced_threats += 1
                    analysis.coalesced_threats.push_back(collision_threat.threat_shape)
    else:
        # no threat found, return some default values
        analysis.threat_radius = 0.
        analysis.threat_loc = ZERO_VECTOR
        analysis.threat_velocity = ZERO_VECTOR

    return

def analyze_neighbors(
        body:cymunk.Body,
        space:cymunk.Space,
        max_distance:float,
        ship_radius:float,
        margin:float,
        neighborhood_loc:cymunk.Vec2d,
        neighborhood_radius:float,
        maximum_acceleration:float
        ) -> Tuple[
            int,
            float,
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            float,
            int,
            int,
            int,
            float,
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            int,
            float,
            float,
            npt.NDArray[np.int64],
        ]:

    if not isinstance(space, ccymunk.Space):
        raise TypeError()

    if not isinstance(body, ccymunk.Body):
        raise TypeError()

    cdef ccymunk.Space cyspace = <ccymunk.Space?>space
    cdef ccymunk.Body cybody = <ccymunk.Body?>body
    cdef ccymunk.cpShape *ct

    cdef NeighborAnalysis analysis
    analysis.neighborhood_radius = neighborhood_radius
    analysis.ship_radius = ship_radius
    analysis.margin = margin
    analysis.max_distance = max_distance
    analysis.maximum_acceleration = maximum_acceleration
    analysis.body = cybody._body
    analysis.worst_ultimate_separation = ccymunk.INFINITY
    analysis.approach_time = ccymunk.INFINITY
    analysis.nearest_neighbor_dist = ccymunk.INFINITY
    analysis.neighborhood_size = 0
    analysis.threat_count = 0

    # look for threats in a circle
    ccymunk.cpSpaceNearestPointQuery(cyspace._space, neighborhood_loc.v, neighborhood_radius, 1, 0, _sensor_point_callback, &analysis)

    # look for threats in a cone facing the direction of our velocity
    cdef v_normalized = ccymunk.cpvnormalize(cybody._body.v)
    cdef v_perp = ccymunk.cpvperp(v_normalized)
    cdef ccymunk.cpVect start_point = ccymunk.cpvadd(cybody._body.p, ccymunk.cpvmult(v_normalized, ship_radius+margin))
    cdef ccymunk.cpVect end_point = ccymunk.cpvadd(cybody._body.p, ccymunk.cpvmult(v_normalized, neighborhood_radius*5))

    ccymunk.cpSpaceSegmentQuery(cyspace._space, start_point, end_point, 1, 0, _sensor_point_callback, &analysis)
    #for i in range(1, 200):
    #    ccymunk.cpSpaceSegmentQuery(cyspace._space, start_point,  ccymunk.cpvadd(end_point, ccymunk.cpvmult(v_perp, i*ship_radius)), 1, 0, _sensor_point_callback, &analysis)
    #    ccymunk.cpSpaceSegmentQuery(cyspace._space, start_point, ccymunk.cpvadd(end_point, ccymunk.cpvmult(v_perp, -i*ship_radius)), 1, 0, _sensor_point_callback, &analysis)

    #cdef ccymunk.cpVect[4] sensor_cone;
    # points are ordered to get a convex shape with the proper winding
    #sensor_cone[1] = ccymunk.cpvadd(start_point, ccymunk.cpvmult(v_perp, (ship_radius+margin)))
    #sensor_cone[0] = ccymunk.cpvadd(start_point, ccymunk.cpvmult(v_perp, -(ship_radius+margin)))
    #sensor_cone[2] = ccymunk.cpvadd(end_point, ccymunk.cpvmult(v_perp, (ship_radius+margin)*100))
    #sensor_cone[3] = ccymunk.cpvadd(end_point, ccymunk.cpvmult(v_perp, -(ship_radius+margin)*100))

    #print("looking for hits in")
    #print(sensor_cone[1])
    #print(sensor_cone[0])
    #print(sensor_cone[2])
    #print(sensor_cone[3])
    #cdef ccymunk.cpShape *sensor_cone_shape = ccymunk.cpPolyShapeNew(NULL, 4, sensor_cone, ZERO_VECTOR)
    #print(ccymunk.cpShapeUpdate(sensor_cone_shape, ZERO_VECTOR, ZERO_VECTOR))
    #print(sensor_cone_shape.bb)
    #ccymunk.cpSpaceShapeQuery(cyspace._space, sensor_cone_shape, _sensor_shape_callback, &analysis)
    #TODO: do we need to deallocate the cone shape?

    coalesce_threats(&analysis)

    # to return:
    #        idx, #neighbor
    #        approach_time,
    #        relative_position,
    #        relative_velocity,
    #        minimum_separation,
    #        threat_count,
    #        coalesced_threats,
    #        non_coalesced_threats,
    #        threat_radius,
    #        threat_loc,
    #        threat_velocity,
    #        nearest_neighbor_idx, #nearest_neighbor
    #        nearest_neighbor_dist,
    #        neighborhood_density,
    #        num_neighbors,
    #        coalesced_idx, #coalesced_neighbors

    if analysis.nearest_neighbor_dist < ccymunk.INFINITY:
        nearest_neighbor = space.shapes[analysis.nearest_neighbor_shape.hashid_private].body
    else:
        nearest_neighbor = None
    neighborhood_density = analysis.neighborhood_size / (math.pi * analysis.neighborhood_radius ** 2)

    if analysis.threat_count == 0:
        return tuple((
                None,
                ccymunk.INFINITY,
                PY_ZERO_VECTOR,
                PY_ZERO_VECTOR,
                ccymunk.INFINITY,
                0,
                0,
                0,
                0,
                PY_ZERO_VECTOR,
                PY_ZERO_VECTOR,
                nearest_neighbor,
                analysis.nearest_neighbor_dist,
                neighborhood_density,
                analysis.neighborhood_size,
                [],
        ))
    else:
        threat = space.shapes[analysis.threat_shape.hashid_private].body
        coalesced_neighbors = []
        for ct in analysis.coalesced_threats:
            coalesced_neighbors.append(
                space.shapes[ct.hashid_private].body
            )

        return tuple((
                threat,
                analysis.approach_time,
                cpvtoVec2d(analysis.relative_position),
                cpvtoVec2d(analysis.relative_velocity),
                analysis.minimum_separation,
                analysis.threat_count,
                analysis.coalesced_threats.size(),
                analysis.threat_count - analysis.coalesced_threats.size(),
                analysis.threat_radius,
                cpvtoVec2d(analysis.threat_loc),
                cpvtoVec2d(analysis.threat_velocity),
                nearest_neighbor,
                analysis.nearest_neighbor_dist,
                neighborhood_density,
                analysis.neighborhood_size,
                coalesced_neighbors,
        ))
