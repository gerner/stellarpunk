from typing import Tuple, List
import cython
from libcpp.vector cimport vector
from libcpp.set cimport set
from libcpp.list cimport list
from libcpp cimport bool
from libc.stdlib cimport malloc
from libc.math cimport fabs, sqrt, pi

import numpy.typing as npt
import numpy as np
import cymunk
cimport cymunk.cymunk as ccymunk
cimport libc.math as math

# some utils

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.infer_types(False)
@cython.nonecheck(False)
cdef bool isclose(double a, double b, double rtol=1e-05, double atol=1e-08):
    # numba gets confused with default parameters sometimes, so we have this
    # "overload"
    return fabs(a-b) <= (atol + rtol * fabs(b))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.infer_types(False)
@cython.nonecheck(False)
cdef double normalize_angle(double angle, bool shortest=0):
    angle = angle % (2*pi)
    angle = (angle + 2*pi) if angle < 0 else angle
    if not shortest or angle <= pi:
        return angle
    else:
        return angle - 2*pi

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.infer_types(False)
@cython.nonecheck(False)
cdef int sgn(double val):
    return (0 < val) - (val < 0);

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.infer_types(False)
@cython.nonecheck(False)
cdef double interpolate(double x1, double y1, double x2, double y2, double x):
    """ interpolates the y given x and two points on a line. """
    cdef double m = (y2 - y1) / (x2 - x1)
    cdef double b = y1 - m * x1
    return m * x + b

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.infer_types(False)
@cython.nonecheck(False)
cdef double clip(double d, double m, double M):
    cdef double t
    if d < m:
        t = m
    else:
        t = d
    if t > M:
        return M
    else:
        return t

# collision detection types

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
cdef ccymunk.cpVect ONE_VECTOR = ccymunk.cpv(1.,0.)
cdef ccymunk.Vec2d PY_ZERO_VECTOR = ccymunk.Vec2d(0.,0.)
cdef double VELOCITY_EPS = 5e-1

# neighborhood location interpolation parameters to move the center of our
# neighborhood search in the direction of our velocity depending on our speed
cdef double NOFF_LOW = 0
cdef double NOFF_SPEED_LOW = 0
cdef double NOFF_HIGH = 2.5e4
cdef double NOFF_SPEED_HIGH = 2500

# CBDR parameters
cdef double CBDR_MIN_HIST_SEC = 0.5
cdef double CBDR_MAX_HIST_SEC = 1.1
cdef double CBDR_ANGLE_EPS = 1e-1 # about 6 degrees
cdef double CBDR_DIST_EPS = 5

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

cdef void _body_shape_callback(ccymunk.cpBody *body, ccymunk.cpShape *shape, void *data):
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

cdef struct RelPosHistoryEntry:
    double timestamp
    ccymunk.cpVect rel_pos

cdef class NeighborAnalyzer:
    cdef ccymunk.Space space
    cdef ccymunk.Body body

    # some state we keep from one analysis call to the next
    cdef ccymunk.cpHashValue last_threat_id
    # list of prior threat shape ids, be careful to check that these are still
    # in the space before
    cdef vector[ccymunk.cpHashValue] prior_threat_ids
    # list of relative positions of the last threat
    cdef list[RelPosHistoryEntry] rel_pos_history

    def __cinit__(self, space:cymunk.Space, body:cymunk.Body) -> None:
        self.space = <ccymunk.Space?> space
        self.body = <ccymunk.Body?> body

    def add_neighbor_shape(self, shape:cymunk.Shape) -> None:
        """ Testing helper to set up NeighborAnalyzer """
        self.prior_threat_ids.push_back((<ccymunk.Shape?>shape)._shape.hashid_private)

    def coalesced_neighbor_locations(self) -> List[Tuple[float, float]]:
        cdef ccymunk.Shape cyshape

        locs = []
        for shape_id in self.prior_threat_ids:
            shape = self.space._shapes.get(shape_id)
            if shape_id is not None:
                # we strongly assume that all values in space._shape are Shapes
                cyshape = <ccymunk.Shape> shape
                locs.append((cyshape._shape.body.p.x, cyshape._shape.body.p.y))

        return locs

    def cbdr_history_summary(self) -> List[Tuple[float,float]]:
        if self.rel_pos_history.size() > 1:
            return [
                (self.rel_pos_history.front().rel_pos.x, self.rel_pos_history.front().rel_pos.y),
                (self.rel_pos_history.back().rel_pos.x, self.rel_pos_history.back().rel_pos.y),
            ]
        elif self.rel_pos_history.size() > 0:
            return [
                (self.rel_pos_history.front().rel_pos.x, self.rel_pos_history.front().rel_pos.y),
            ]
        else:
            return[]

    def analyze_neighbors(
            self,
            current_timestamp:float,
            max_distance:float,
            ship_radius:float,
            margin:float,
            neighborhood_radius:float,
            maximum_acceleration:float,
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
                int,
            ]:

        cdef ccymunk.cpShape *ct
        cdef ccymunk.Body cythreat_body
        cdef double speed = ccymunk.cpvlength(self.body._body.v)
        cdef double neighborhood_offset
        cdef ccymunk.cpVect cneighborhood_loc
        cdef set[ccymunk.cpHashValue] prior_shapes
        cdef int prior_threat_count
        cdef ccymunk.cpHashValue prior_threat_id

        if speed > 0:
            # offset looking for threats in the direction we're travelling,
            # depending on our speed
            neighborhood_offset = clip(
                    interpolate(
                        NOFF_SPEED_LOW, NOFF_LOW, NOFF_SPEED_HIGH, NOFF_HIGH,
                        speed
                    ),
                    0, neighborhood_radius - margin)
            cneighborhood_loc = ccymunk.cpvadd(self.body._body.p, ccymunk.cpvmult(self.body._body.v, neighborhood_offset / speed))
        else:
            cneighborhood_loc = self.body._body.p


        cdef NeighborAnalysis analysis
        analysis.neighborhood_radius = neighborhood_radius
        analysis.ship_radius = ship_radius
        analysis.margin = margin
        analysis.max_distance = max_distance
        analysis.maximum_acceleration = maximum_acceleration
        analysis.body = self.body._body
        analysis.worst_ultimate_separation = ccymunk.INFINITY
        analysis.approach_time = ccymunk.INFINITY
        analysis.nearest_neighbor_dist = ccymunk.INFINITY
        analysis.neighborhood_size = 0
        analysis.threat_count = 0

        # start by considering prior threats
        for shape_id in self.prior_threat_ids:
            shape = self.space._shapes.get(shape_id)
            if shape_id is not None:
                _analyze_neighbor_callback((<ccymunk.Shape>shape)._shape, &analysis)

        # grab a copy of the shape ids for prior shapes
        prior_shape_ids = analysis.considered_shapes

        # look for threats in a circle
        ccymunk.cpSpaceNearestPointQuery(self.space._space, cneighborhood_loc, neighborhood_radius, 1, 0, _sensor_point_callback, &analysis)

        # look for threats in a cone facing the direction of our velocity
        # cone is truncated, starts at the edge of our nearest point query circle
        # goes until another 4 neighborhood radii in direction of our velocity
        # cone starts at margin
        cdef ccymunk.cpVect v_normalized = ccymunk.cpvnormalize(self.body._body.v)
        cdef ccymunk.cpVect v_perp = ccymunk.cpvperp(v_normalized)
        cdef ccymunk.cpVect start_point = ccymunk.cpvadd(cneighborhood_loc, ccymunk.cpvmult(v_normalized, neighborhood_radius-margin))
        cdef ccymunk.cpVect end_point = ccymunk.cpvadd(cneighborhood_loc, ccymunk.cpvmult(v_normalized, neighborhood_radius*3))

        cdef ccymunk.cpVect sensor_cone[4]
        # points are ordered to get a convex shape with the proper winding
        sensor_cone[1] = ccymunk.cpvadd(start_point, ccymunk.cpvmult(v_perp, (ship_radius+margin*2)))
        sensor_cone[0] = ccymunk.cpvadd(start_point, ccymunk.cpvmult(v_perp, -(ship_radius+margin*2)))
        sensor_cone[2] = ccymunk.cpvadd(end_point, ccymunk.cpvmult(v_perp, (ship_radius+margin)*5))
        sensor_cone[3] = ccymunk.cpvadd(end_point, ccymunk.cpvmult(v_perp, -(ship_radius+margin)*5))

        cdef ccymunk.cpShape *sensor_cone_shape = ccymunk.cpPolyShapeNew(NULL, 4, sensor_cone, ZERO_VECTOR)
        ccymunk.cpShapeUpdate(sensor_cone_shape, ZERO_VECTOR, ONE_VECTOR)
        ccymunk.cpSpaceShapeQuery(self.space._space, sensor_cone_shape, _sensor_shape_callback, &analysis)
        ccymunk.cpShapeFree(sensor_cone_shape)
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
            nearest_neighbor = self.space.shapes[analysis.nearest_neighbor_shape.hashid_private].body
        else:
            nearest_neighbor = None
        neighborhood_density = analysis.neighborhood_size / (math.pi * analysis.neighborhood_radius ** 2)

        if analysis.threat_count == 0:
            self.last_threat_id = 0
            self.prior_threat_ids.clear()
            self.rel_pos_history.clear()
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
                    False
            ))
        else:
            prior_threat_id = self.last_threat_id
            self.last_threat_id = analysis.threat_shape.hashid_private
            if self.last_threat_id == prior_threat_id:
                self.rel_pos_history.push_back(RelPosHistoryEntry(current_timestamp, analysis.relative_position))
                # nuke history entries that are too old
                while current_timestamp - self.rel_pos_history.front().timestamp > CBDR_MAX_HIST_SEC:
                    self.rel_pos_history.pop_front()

            else:
                self.rel_pos_history.clear()
                self.rel_pos_history.push_back(RelPosHistoryEntry(current_timestamp, analysis.relative_position))

            threat = self.space.shapes[analysis.threat_shape.hashid_private].body
            self.prior_threat_ids.clear()
            prior_threat_count = 0
            for ct in analysis.coalesced_threats:
                self.prior_threat_ids.push_back(ct.hashid_private)
                prior_threat_count += prior_shape_ids.count(ct.hashid_private)

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
                    prior_threat_count,
            ))

    def detect_cbdr(self, current_timestamp:float):
        if self.rel_pos_history.size() < 2:
            return False
        if current_timestamp - self.rel_pos_history.front().timestamp < CBDR_MIN_HIST_SEC:
            return False

        cdef double oldest_distance = ccymunk.cpvlength(self.rel_pos_history.front().rel_pos)
        cdef double oldest_bearing = ccymunk.cpvtoangle(self.rel_pos_history.front().rel_pos)

        cdef double latest_distance = ccymunk.cpvlength(self.rel_pos_history.back().rel_pos)
        cdef double latest_bearing = ccymunk.cpvtoangle(self.rel_pos_history.back().rel_pos)

        return fabs(normalize_angle(oldest_bearing - latest_bearing, shortest=1)) < CBDR_ANGLE_EPS and oldest_distance - latest_distance > CBDR_DIST_EPS


cdef double ANGLE_EPS = 5e-2 # about 3 degrees
cdef double COARSE_VELOCITY_MATCH = 2e0
cdef double COARSE_ANGLE_MATCH = pi/16 # about 11 degrees
cdef double ARRIVAL_ANGLE = 5e-1 # about 29 degrees

cdef struct TorqueResult:
    double torque
    double continue_time

cdef struct ForceResult:
    ccymunk.cpVect force
    double continue_time

cdef struct ForceTorqueResult:
    ccymunk.cpVect force
    double torque
    double continue_time

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.infer_types(False)
@cython.nonecheck(False)
cdef double _rotation_time(
        double delta_angle, double angular_velocity,
        double max_angular_acceleration):
    # theta_f = theta_0 + omega_0*t + 1/2 * alpha * t^2
    # assume omega_0 = 0 <--- assumes we're not currently rotating!
    # assume we constantly accelerate half way, constantly accelerate the
    # other half
    return (fabs(angular_velocity)/max_angular_acceleration + 2*sqrt(fabs(delta_angle + 0.5*angular_velocity**2/max_angular_acceleration)/max_angular_acceleration))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.infer_types(False)
@cython.nonecheck(False)
cdef TorqueResult _torque_for_angle(
        double target_angle,
        double angle, double w, double moment,
        double max_torque, double dt):
    """ What torque to apply to achieve target angle """

    cdef double difference_angle = normalize_angle(target_angle - angle, shortest=1)
    cdef double desired_w
    cdef double difference_w
    cdef double t
    cdef double w_dampener

    if fabs(w) < ANGLE_EPS and fabs(difference_angle) < ANGLE_EPS:
        # bail if we're basically already there
        # caller can handle this, e.g. set rotation to target and w to 0
        return TorqueResult(0.0, ccymunk.INFINITY)
    else:
        # add torque in the desired direction to get
        # accel = tau / moment
        # dw = accel * dt
        # desired w is w such that braking_angle = difference_angle
        # braking_angle =  -1 * np.sign(w) * -0.5 * w*w * moment / max_torque
        # sqrt(difference_angle * max_torque / (0.5 * moment)) = w
        if fabs(difference_angle) < ANGLE_EPS:
            desired_w = 0.
        else:
            # w_f**2 = w_i**2 + 2 * a (d_theta)
            desired_w =  sgn(difference_angle) * sqrt(fabs(difference_angle) * max_torque/moment * 2) * 0.8

        if fabs(difference_angle) < ARRIVAL_ANGLE:
            w_dampener = interpolate(
                ARRIVAL_ANGLE, 1.0,
                0, 0.6,
                fabs(difference_angle)
            )
            desired_w = desired_w * w_dampener

        difference_w = fabs(desired_w - w)

        if difference_w < ANGLE_EPS:
            return TorqueResult(0., ccymunk.INFINITY)

        t = (desired_w - w)*moment/dt

        if t < -max_torque:
            return TorqueResult(-max_torque, difference_w * moment / max_torque)
        elif t > max_torque:
            return TorqueResult(max_torque, difference_w * moment / max_torque)
        else:
            return TorqueResult(t, dt)

def torque_for_angle(target_angle:float, angle:float, w:float, moment:float, max_torque:float, dt:float) -> Tuple[float, float]:
    """ Exposes _torque_for_angle to python """
    cdef TorqueResult ret = _torque_for_angle(target_angle, angle, w, moment, max_torque, dt)
    return (ret.torque, ret.continue_time)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.infer_types(False)
@cython.nonecheck(False)
cdef ForceResult _force_for_delta_velocity(
        ccymunk.cpVect dv, double mass, double max_thrust, double dt):
    """ What force to apply to get dv change in velocity. Ignores heading. """

    if isclose(0, ccymunk.cpvlength(dv)):
        return ForceResult(ZERO_VECTOR, ccymunk.INFINITY)

    cdef double dv_magnitude = ccymunk.cpvlength(dv)
    if dv_magnitude < VELOCITY_EPS:
        return ForceResult(ZERO_VECTOR, ccymunk.INFINITY)

    # f = ma
    # dv = a * t
    # dv = f/m * t
    # f = m * dv / t
    # two unknowns, force and time are free, but we have constraints
    # time can be no less that dt and force can be no more than max_thrust
    # minimize time >= dt
    # maximize force <= max_thrust
    cdef double desired_thrust = mass * dv_magnitude / dt
    if desired_thrust > max_thrust:
        return ForceResult(
            ccymunk.cpvmult(dv, max_thrust / dv_magnitude),
            mass * dv_magnitude/max_thrust
        )
    else:
        return ForceResult(
            ccymunk.cpvmult(dv, desired_thrust / dv_magnitude),
            dt
        )

def force_for_delta_velocity(dv:cymunk.Vec2d, mass:float, max_thrust:float, dt:float) -> Tuple[cymunk.Vec2d, float]:
    """ Exposes _force_for_delta_velocity to python for testing. """
    cdef ccymunk.Vec2d cydv = <ccymunk.Vec2d?>dv
    cdef ForceResult ret = _force_for_delta_velocity(dv.v, mass, max_thrust, dt)
    return (cpvtoVec2d(ret.force), ret.continue_time)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.infer_types(False)
@cython.nonecheck(False)
cdef ForceTorqueResult _force_torque_for_delta_velocity(
        ccymunk.cpVect target_velocity, ccymunk.cpBody *body,
        double max_speed, double max_torque,
        double max_thrust, double max_fine_thrust,
        double dt):
    """ Given target velocity, a timestep size  and parameters about the ship,
    return force, torque, target velocity, and desired speed difference and
    time to hold those values before calling me again. """

    cdef ccymunk.cpVect dv = ccymunk.cpvsub(target_velocity, body.v)
    cdef double difference_mag = ccymunk.cpvlength(dv)
    cdef double difference_angle = ccymunk.cpvtoangle(dv)

    if difference_mag < VELOCITY_EPS and fabs(body.w) < ANGLE_EPS:
        return ForceTorqueResult(ZERO_VECTOR, 0., ccymunk.INFINITY)

    cdef double delta_heading = normalize_angle(body.a-difference_angle, shortest=1)
    cdef double rot_time = _rotation_time(delta_heading, body.w, max_torque/body.i)

    # while we've got a lot of thrusting to do, we can tolerate only
    # approximately matching our desired angle
    # this should have something to do with how much we expect this angle
    # to change in dt time, but order of magnitude seems reasonable approx

    cdef TorqueResult torque_result
    cdef ForceResult force_result
    cdef double continue_time

    if (difference_mag * body.m / max_fine_thrust > rot_time and abs(delta_heading) > COARSE_ANGLE_MATCH) or difference_mag < COARSE_VELOCITY_MATCH:
        # we need to rotate in direction of thrust
        torque_result = _torque_for_angle(difference_angle, body.a, body.w, body.i, max_torque, dt)

        # also apply thrust depending on where we're pointed
        force_result = _force_for_delta_velocity(dv, body.m, max_fine_thrust, dt)
    else:
        torque_result = _torque_for_angle(difference_angle, body.a, body.w, body.i, max_torque, dt)

        # we should apply thrust, however we can with the current heading
        # max thrust is main engines if we're pointing in the desired
        # direction, otherwise use fine thrusters
        if fabs(delta_heading) < COARSE_ANGLE_MATCH:
            max_thrust = max_thrust
        else:
            max_thrust = max_fine_thrust

        force_result = _force_for_delta_velocity(dv, body.m, max_thrust, dt)

    if isclose(0, rot_time):
        continue_time = force_result.continue_time
    else:
        continue_time = min(torque_result.continue_time, force_result.continue_time)

    return ForceTorqueResult(force_result.force, torque_result.torque, continue_time)

def rotate_to(
        body:cymunk.Body, target_angle:float, dt:float,
        max_torque:float) -> float:
    """ Applies torque to rotate the given body to the desired angle

    returns the time to continue applying that torque. """

    # given current angle and angular_velocity and max torque, choose
    # torque to apply for dt now to hit target angle

    cdef ccymunk.Body cybody = <ccymunk.Body?> body
    cdef double w = cybody._body.w
    cdef double moment = cybody._body.i
    cdef double angle = cybody._body.a

    cdef TorqueResult torque_result = _torque_for_angle(
            target_angle, angle, w, moment,
            max_torque, dt)

    cdef double difference_angle = normalize_angle(angle - target_angle, shortest=1)
    if torque_result.torque == 0 and fabs(difference_angle) <= ANGLE_EPS:
        cybody._body.a = target_angle
        cybody._body.w = 0
        cybody._body.t = 0.
    else:
        cybody._body.t = torque_result.torque

    return torque_result.continue_time

cdef struct DeltaVResult:
    ccymunk.cpVect target_velocity,
    double distance
    double distance_estimate
    bool cannot_stop
    double delta_speed

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.infer_types(False)
@cython.nonecheck(False)
cdef DeltaVResult _find_target_v(ccymunk.cpBody *body, ccymunk.cpVect target_location, double arrival_distance, double min_distance, double max_acceleration, double max_angular_acceleration, double max_speed, double dt, double safety_factor):
    """ Given goto location params, determine the desired velocity.

    returns a tuple:
        target velocity vector
        distance to the target location
        an estimate of the distance to the target location after dt
        boolean indicator if we cannot stop before reaching location
        delta speed between current and target
    """

    cdef ccymunk.cpVect target_v
    cdef double desired_speed
    cdef double a
    cdef double s
    cdef double rot_time

    cdef ccymunk.cpVect course = ccymunk.cpvsub(target_location, body.p)
    cdef double distance = ccymunk.cpvlength(course)
    cdef double target_angle = ccymunk.cpvtoangle(course)

    #TODO: conservative estimate of distance?
    cdef double distance_estimate = distance - max_speed*dt

    # if we were to cancel the velocity component in the direction of the
    # target, will we travel enough so that we cross min_distance?
    cdef double d = (ccymunk.cpvdot(body.v, course) / distance)**2 / (2* max_acceleration)

    cdef bool cannot_stop
    if d > distance-min_distance:
        cannot_stop = 1
    else:
        cannot_stop = 0

    if distance < arrival_distance + VELOCITY_EPS:
        target_v = ZERO_VECTOR
        desired_speed = 0.
    else:
        rot_time = _rotation_time(fabs(normalize_angle(body.a-(target_angle+pi), shortest=1)), body.w, max_angular_acceleration)*safety_factor

        # choose a desired speed such that if we were at that speed right
        # now we would have enough time to rotate 180 degrees and
        # decelerate to a stop at full thrust by the time we reach arrival
        # distance
        a = max_acceleration
        s = distance - min_distance
        if s < 0:
            s = 0

        desired_speed = (-2 * a * rot_time + sqrt((2 * a  * rot_time) ** 2 + 8 * a * s))/2
        desired_speed = clip(desired_speed/safety_factor, 0, max_speed)

        target_v = ccymunk.cpvmult(ccymunk.cpvmult(course, 1./distance), desired_speed)

    return DeltaVResult(target_v, distance, distance_estimate, cannot_stop, fabs(ccymunk.cpvlength(body.v) - desired_speed))


def find_target_v(
        body:cymunk.Body,
        target_location:cymunk.Vec2d, arrival_distance:float, min_distance:float,
        max_acceleration:float, max_angular_acceleration:float, max_speed: float,
        dt: float, safety_factor:float):
    """ Given goto location params, determine the desired velocity.

    returns a tuple:
        target velocity vector
        distance to the target location
        an estimate of the distance to the target location after dt
        boolean indicator if we cannot stop before reaching location
        delta speed between current and target
    """

    cdef ccymunk.Body cybody = <ccymunk.Body?> body
    cdef ccymunk.Vec2d cytarget_location = <ccymunk.Vec2d?> target_location

    cdef DeltaVResult result = _find_target_v(cybody._body, cytarget_location.v, arrival_distance, min_distance, max_acceleration, max_angular_acceleration, max_speed, dt, safety_factor)

    return (cpvtoVec2d(result.target_velocity), result.distance, result.distance_estimate, result.cannot_stop, result.delta_speed)

def accelerate_to(
        body:cymunk.Body, target_velocity:cymunk.Vec2d, dt:float,
        max_speed:float, max_torque:float, max_thrust:float, max_fine_thrust:float) -> float:

    cdef ccymunk.Vec2d cyvelocity = <ccymunk.Vec2d?>target_velocity
    cdef ccymunk.Body cybody = <ccymunk.Body?>body

    # compute force/torque
    cdef ForceTorqueResult ft_result = _force_torque_for_delta_velocity(
                cyvelocity.v,
                cybody._body,
                max_speed, max_torque, max_thrust, max_fine_thrust,
                dt
        )

    assert ft_result.continue_time > 0.

    if ccymunk.cpvlength(ft_result.force) < VELOCITY_EPS:
        cybody._body.v = cyvelocity.v
        cybody._body.f = ZERO_VECTOR
    else:
        cybody._body.f = ft_result.force

    if ft_result.torque != 0.:
        cybody._body.t = ft_result.torque
    else:
        cybody._body.t = 0.

    return ft_result.continue_time
