# cython: boundscheck=False, wraparound=False, cdivision=True, infer_types=False, nonecheck=False

from typing import Tuple, List, Any, Dict
import cython
from libcpp.vector cimport vector
from libcpp.set cimport set
from libcpp.list cimport list
from libcpp.queue cimport priority_queue
from libcpp cimport bool
from libc.stdlib cimport malloc
from libc.stdio cimport fprintf, fflush, stderr, stdout
from libc.math cimport fabs, sqrt, pi, isnan, isinf, atan2
import math as pymath

import cymunk
cimport cymunk.cymunk as ccymunk
cimport libc.math as math

DEF LOGGING_ENABLED=1

# some utils

cdef bool isclose(double a, double b, double rtol=1e-05, double atol=1e-08):
    # numba gets confused with default parameters sometimes, so we have this
    # "overload"
    return fabs(a-b) <= (atol + rtol * fabs(b))

cdef double normalize_angle(double angle, bool shortest=0):
    angle = angle % (2*pi)
    angle = (angle + 2*pi) if angle < 0 else angle
    if not shortest or angle <= pi:
        return angle
    else:
        return angle - 2*pi

cdef int sgn(double val):
    return (0 < val) - (val < 0);

cdef double interpolate(double x1, double y1, double x2, double y2, double x):
    """ interpolates the y given x and two points on a line. """
    cdef double m = (y2 - y1) / (x2 - x1)
    cdef double b = y1 - m * x1
    return m * x + b

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

cdef struct Circle:
    ccymunk.cpVect center
    double radius

cdef Circle enclosing_circle(
        ccymunk.cpVect c1, double r1,
        ccymunk.cpVect c2, double r2):
    """ Finds the smallest circle enclosing two other circles.

    courtesey: https://stackoverflow.com/a/36736270/553580
    """
    cdef double dx = c2.x - c1.x
    cdef double dy = c2.y - c1.y
    #center-center distance
    cdef double dc = math.sqrt(dx**2. + dy**2.)
    cdef double rmin = min(r1, r2)
    cdef double rmax = max(r1, r2)

    cdef ccymunk.cpVect cret
    cdef double R

    if rmin + dc < rmax:
        if r1 < r2:
            cret = c2
            R = r2
        else:
            cret = c1
            R = r1
    else:
        R = 0.5 * (r1 + r2 + dc)
        cret = ccymunk.cpv(
            c1.x + (R - r1) * dx / dc,
            c1.y + (R - r1) * dy / dc
        )

    return Circle(cret, R)

def make_enclosing_circle(
        c1:cymunk.Vec2d, r1:float,
        c2:cymunk.Vec2d, r2:float) -> Tuple[cymunk.Vec2d, float]:
    cdef Circle ret = enclosing_circle(c1.v, r1, c2.v, r2)
    return (cpvtoVec2d(ret.center), ret.radius)

cdef void log(ccymunk.Body body, message, eid_prefix=""):
    if LOGGING_ENABLED:
        if str(body.data.entity_id).startswith(eid_prefix):
            print(f'{body.data.entity_id}\t{message}')

# collision detection types

cdef struct CollisionThreat:
    ccymunk.cpShape *threat_shape
    ccymunk.cpVect c_loc

cdef struct NeighborAnalysis:
    # params about doing the analysis
    double neighborhood_radius
    double ship_radius
    double margin
    double max_distance
    double maximum_acceleration
    ccymunk.cpBody *body
    double timestamp

    # results of the analysis
    double worst_ultimate_separation
    double nearest_neighbor_dist
    double approach_time
    ccymunk.cpVect relative_position
    ccymunk.cpVect relative_velocity
    double minimum_separation
    ccymunk.cpVect collision_loc
    size_t neighborhood_size
    size_t threat_count
    ccymunk.cpShape *threat_shape
    double detection_timestamp

    # current (not projected) threat location and distance
    ccymunk.cpVect current_threat_loc
    double distance_to_threat

    ccymunk.cpShape *nearest_neighbor_shape
    vector[CollisionThreat] collision_threats
    vector[ccymunk.cpShape *] coalesced_threats
    set[ccymunk.cpHashValue] considered_shapes

    # coalesced threat
    double threat_radius
    ccymunk.cpVect threat_loc
    ccymunk.cpVect threat_velocity

    bool cannot_avoid_collision

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

# the scale (per tick) we use to scale down threat radii if the new threat is
# still covered by the previous threat radius
cdef double THREAT_RADIUS_SCALE_FACTOR = 0.995

cdef double COLLISION_MARGIN_HISTERESIS_FACTOR = 0.55

# cymunk fails to export this type from chipmunk
ctypedef struct cpCircleShape :
    ccymunk.cpShape shape;
    ccymunk.cpVect c
    ccymunk.cpVect tc
    ccymunk.cpFloat r

cdef ccymunk.Vec2d cpvtoVec2d(ccymunk.cpVect v):
    return ccymunk.Vec2d(v.x, v.y)

def cpvtoTuple(ccymunk.cpVect v):
    return (v.x, v.y)

cdef ccymunk.cpVect tupletocpv(v):
    return ccymunk.cpv(v[0], v[1])

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
    cdef double ultimate_sep = neighbor.min_sep - entity_radius - analysis.ship_radius + 0.5 * analysis.maximum_acceleration * neighbor.approach_t ** 2.
    if ultimate_sep < analysis.worst_ultimate_separation:
        analysis.worst_ultimate_separation = ultimate_sep
        analysis.approach_time = neighbor.approach_t
        analysis.threat_shape = shape
        analysis.relative_position = neighbor.rel_pos
        analysis.relative_velocity = neighbor.rel_vel
        analysis.minimum_separation = neighbor.min_sep
        analysis.collision_loc = neighbor.c_loc

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

cdef ccymunk.cpVect _collision_dv(
        ccymunk.cpVect entity_pos, ccymunk.cpVect entity_vel,
        ccymunk.cpVect pos, ccymunk.cpVect vel, double angle,
        double margin, ccymunk.cpVect v_d,
        bool cbdr, double cbdr_bias, double delta_v_budget, bool smaller_angle=False) except *:
    """ Computes a divert vector (as in accelerate_to(v + dv)) to avoid a
    collision by at least distance m. This divert will be of minimum size
    relative to the desired velocity.

    entity_pos: location of the threat
    entity_pos: velocity of the threat
    pos: our position
    v: our velocity
    v_d: the desired delta velocity
    """

    # rel pos
    cdef ccymunk.cpVect r = ccymunk.cpvsub(entity_pos, pos)
    # rel vel
    cdef ccymunk.cpVect v = ccymunk.cpvsub(entity_vel, vel)
    # margin, including radii
    cdef double m = margin

    # desired diversion from v
    cdef ccymunk.cpVect a = ccymunk.cpvneg(v_d)

    # check if the desired divert is already viable
    cdef double x = a.x
    cdef double y = a.y

    cdef double do_nothing_margin_sq
    cdef double p
    cdef double x1, y1, x2, y2
    cdef double s_1x, s_1y, s_2x, s_2y, s_x, s_y
    cdef double cost1, cost2, cost
    cdef double q_a, q_b, q_c
    cdef double cross1, cross2
    cdef bool horizonal_rel_pos
    cdef double da1, da2

    if isclose(v.x, 0.) and isclose(v.y, 0.):
        do_nothing_margin_sq = r.x**2.+r.y**2.
    else:
        do_nothing_margin_sq = r.x**2.+r.y**2. - (r.x*x+r.y*y+(2*r.x*v.x+2*r.y*v.y))**2./((2*v.x+x)**2.+(2*v.y+y)**2.)
    if do_nothing_margin_sq > 0 and do_nothing_margin_sq >= m**2.:
        return v_d

    if ccymunk.cpvlength(r) <= margin:
        raise ValueError("already inside margin")

    # given divert (x,y):
    # (r.x**2.+r.y**2.)-(2*(r.x*v.x+r.y*v.y)+(r.x*x+r.y*y))**2./((2*v.x+x)**2. + (2*v.y+y)**2.) > m**2.
    # this forms a pair of intersecting lines with viable diverts between them

    # given divert (x,y):
    # cost_from desired = (a.x-x)**2. +(a.y-y)**2.
    # see https://www.desmos.com/calculator/qvk8fpbw3k

    # to understand the margin, we end up with two circles whose intersection
    # points are points on the tangent lines that form the boundary of our
    # viable diverts
    # (x+2*v.x)**2. + (y+2*v.y)**2. = r.x**2.+r.y**2.-m**2.
    # (x+2*v.x-r.x)**2. + (y+2*v.y-r.y)**2. = m**2.
    # a couple of simlifying substitutions:
    # we can translate the whole system to the origin:
    # let s_x,s_y = (x + 2*v.x, y + 2*v.y)
    # let p = r.x**2. + r.y**2. - m**2.
    # giving
    # s_x**2 + s_y**2 = p
    # (s_x-r.x)**2 + (s_y-r.y)**2 = m**2
    # having done this we can subtract the one from the other, solve for y,
    # plug back into one of the equations
    # solve the resulting quadratic eq for x (two roots)
    # plug back into one of the equations to get y (two sets of two roots)
    # for the y roots, only one will satisfy the other equation, pick that one
    # also, we'll divide by r.y below. So if that's zero we have an alternate
    # form where there's a single value for x

    p = r.x**2. + r.y**2. - m**2.

    horizonal_rel_pos = isclose(r.y**2., 0, rtol=1e-05, atol=1e-5)
    if horizonal_rel_pos:
        # this case would cause a divide by zero when computing the
        # coefficients of the quadratic equation below
        s_1x = s_2x = p/r.x
    else:
        # note that r.x and r.y cannot both be zero (assuming m>0)
        q_a = r.x**2./r.y**2.+1
        q_b = -2*p*r.x/r.y**2.
        q_c = p**2./r.y**2. - p

        # quadratic formula
        # note that we get real roots as long as the problem is feasible (i.e.
        # we're not already inside the margin

        # numerical stability approach from
        # https://people.csail.mit.edu/bkph/articles/Quadratics.pdf
        if q_b >= 0:
            s_1x = (-q_b-sqrt(q_b**2.-4*q_a*q_c)) / (2*q_a)
            s_2x = 2*q_c / (-q_b-sqrt(q_b**2.-4*q_a*q_c))
        else:
            s_1x = 2*q_c / (-q_b+sqrt(q_b**2.-4*q_a*q_c))
            s_2x = (-q_b+sqrt(q_b**2.-4*q_a*q_c)) / (2*q_a)

        assert not isnan(s_1x)
        assert not isnan(s_2x)

    # y roots are y_i and -y_i, but only one each for i=0,1 will be on the curve

    # if p and s_2x**2. are close, this can appear to go negative
    if p - s_1x**2. < 0.:
        s_1y = 0.
    else:
        s_1y = sqrt(p-s_1x**2.)
    if not isclose((s_1x - r.x)**2. + (s_1y - r.y)**2., m**2.):
        s_1y = -s_1y

    # if p and s_2x**2. are close, this can appear to go negative
    if p-s_2x**2. < 0.:
        s_2y = 0.
    else:
        s_2y = sqrt(p-s_2x**2.)
    if not isclose((s_2x - r.x)**2. + (s_2y - r.y)**2., m**2.):
        s_2y = -s_2y

    if horizonal_rel_pos:
        s_2y = -s_1y

    # subbing back in our x_hat,y_hat above,
    # these determine the slope of the boundry lines of our viable region
    # (1) y+2*v.y = s_iy/s_ix * (x+2*v.x) for i = 0,1 (careful if x_i = 0)
    # with perpendiculars going through the desired_divert point
    # (2) y-a.y = -s_ix/s_iy * (x-a.x) (careful if y_i == 0)
    # so find the intersection of each of these pairs of equations

    if isclose(s_1x, 0):
        # tangent line is vertical
        # implies perpendicular is horizontal
        y1 = a.y
        x1 = -2*v.x
    elif isclose(s_1y, 0):
        # tangent line is horizontal
        # implies perpendicular is vertical
        x1 = a.x
        y1 = -2*v.y
    else:
        # solve (1) for y in terms of x and plug into (2), solve for x
        x1 = (s_1x/s_1y*a.x+a.y - s_1y/s_1x*2*v.x + 2*v.y) / (s_1y/s_1x + s_1x/s_1y)
        # plug back into (1)
        y1 = s_1y/s_1x * (x1+2*v.x) - 2*v.y

    if isclose(s_2x, 0):
        y2 = a.y
        x2 = -2*v.x
    elif isclose(s_2y, 0):
        x2 = a.x
        y2 = -2*v.y
    else:
        x2 = (s_2x/s_2y*a.x+a.y - s_2y/s_2x*2*v.x + 2*v.y) / (s_2y/s_2x + s_2x/s_2y)
        y2 = s_2y/s_2x * (x2+2*v.x) - 2*v.y

    cost1 = (a.x-x1)**2. +(a.y-y1)**2.
    cost2 = (a.x-x2)**2. +(a.y-y2)**2.

    assert not (isnan(cost1) or isinf(cost1))
    assert not (isnan(cost2) or isinf(cost2))

    s_x = 0.
    s_y = 0.
    cost = 0.
    if not cost2 < cost1:
        x = x1
        y = y1
        s_x = s_1x
        s_y = s_1y
        cost = cost1
    elif not cost1 < cost2:
        x = x2
        y = y2
        s_x = s_2x
        s_y = s_2y
        cost = cost2
    else:
        # not exactly sure why either would be nan, but hopefully one is not
        assert not isnan(cost1) or not isnan(cost2)

    if smaller_angle and sqrt(cost) > delta_v_budget:
        # pick the one with smaller delta angle
        da1 = atan2(x1 - v.x, y1 - v.y)
        da2 = atan2(x2 - v.x, y2 - v.y)
        if fabs(normalize_angle(da1-angle, shortest=1)) < fabs(normalize_angle(da2-angle, shortest=1)):
            x = x1
            y = y1
            s_x = s_1x
            s_y = s_1y
            cost = cost1
        else:
            x = x2
            y = y2
            s_x = s_2x
            s_y = s_2y
            cost = cost2

    if cbdr:
        # prefer diversion in the same direction in case of cbdr
        # the sign of cross1 and cross2 indicate the direction of the divert
        # (clockwise or counter-clockwise)
        cross1 = v.x*y1-v.y*x1
        cross2 = v.x*y2-v.y*x2
        if cross1 > cross2:
            x = x1
            y = y1
            s_x = s_1x
            s_y = s_1y
            cost = cost1
        else:
            x = x2
            y = y2
            s_x = s_2x
            s_y = s_2y
            cost = cost2

    #TODO: this assumes the other guy will move which seems very risky
    # we should come back to this in the future and do something more
    # proactive, but still cooperative
    #if cbdr and cbdr_bias < 0:
    #    return np.array((0.,0.))

    #if cbdr:
    #    # swap our choices if our bias is negative
    #    if cbdr_bias < 0:
    #        if not cost2 < cost1:
    #            x = x2
    #            y = y2
    #            s_x = s_2x
    #            s_y = s_2y
    #            cost = cost2
    #        elif not cost1 < cost2:
    #            x = x1
    #            y = y1
    #            s_x = s_1x
    #            s_y = s_1y
    #            cost = cost1

    # useful assert when testing
    # this asserts that the resulting x,y point matches the the contraint on
    # the margin
    assert isclose(
            (r.x**2.+r.y**2.)-(2*(r.x*v.x+r.y*v.y)+(r.x*x+r.y*y))**2./((2*v.x+x)**2. + (2*v.y+y)**2.),
            m**2.,
            rtol=1e-3)
    return ccymunk.cpv(-x, -y)

def collision_dv(entity_pos:cymunk.Vec2d, entity_vel:cymunk.Vec2d, pos:cymunk.Vec2d, vel:cymunk.Vec2d, margin:float, v_d:cymunk.Vec2d, cbdr:bool, cbdr_bias:float, delta_v_budget:float) -> cymunk.Vec2d:
    return cpvtoVec2d(_collision_dv(
        entity_pos.v, entity_vel.v,
        pos.v, vel.v, 0.,
        margin, v_d.v, cbdr, cbdr_bias, delta_v_budget
    ))

cdef double _update_collision_margin_histeresis(double collision_margin_histeresis, double margin_histeresis, bool any_prior_threats, bool cannot_avoid_collision_hold):

    cdef double y,b,m,y_next

    if cannot_avoid_collision_hold:
        # if likely to collide, we want to collapse the margin as small as possible
        return 0.
    elif any_prior_threats:
        # if there's any overlap, keep the margin extra big
        #self.collision_margin_histeresis = margin_histeresis

        # expand margin up to margin histeresis
        # this is the iterative form of the inverse of expoential decay
        # which we use below to decay the margin histeresis when there's no
        # overlap
        if collision_margin_histeresis < margin_histeresis:
            y = collision_margin_histeresis
            b = margin_histeresis
            m = COLLISION_MARGIN_HISTERESIS_FACTOR
            y_next = (b - y)*(1-m)+y
            # numerical precision means we might exceed the upper bound
            if y_next > margin_histeresis:
                y_next = margin_histeresis
            return y_next
        elif collision_margin_histeresis > margin_histeresis:
            collision_margin_histeresis *= COLLISION_MARGIN_HISTERESIS_FACTOR
            if collision_margin_histeresis < margin_histeresis:
                return margin_histeresis
        return collision_margin_histeresis
    else:
        # if there's no overlap, start collapsing collision margin
        return collision_margin_histeresis * COLLISION_MARGIN_HISTERESIS_FACTOR

cdef struct RelPosHistoryEntry:
    double timestamp
    ccymunk.cpVect rel_pos
    ccymunk.cpVect velocity

cdef class Navigator:
    cdef ccymunk.Space space
    cdef ccymunk.Body body

    # some parameters about us
    cdef double radius
    cdef double max_thrust
    cdef double max_torque
    cdef double max_acceleration
    cdef double max_angular_acceleration
    cdef double worst_case_rot_time

    # some parameters about the navigation we'll be doing

    cdef double base_neighborhood_radius
    cdef double neighborhood_radius
    cdef double full_neighborhood_radius_period
    cdef double full_neighborhood_radius_ts

    # we'll calculate a desired max speed based on conditions (e.g. how crowded
    # things are)
    cdef double base_max_speed
    cdef double max_speed

    # a cap on max speed to apply histeresis
    # we'll keep track of the lowest it gets and when we last *dropped* it
    # the actual cap will be computed as
    # cap = max_speed_cap * (1+alpha)^(time_since_drop)
    cdef double max_speed_cap
    cdef double max_speed_cap_ts
    cdef double max_speed_cap_alpha
    cdef double min_max_speed
    cdef double max_speed_cap_max_expiration

    # the margin we'd like to stay away from other vessels
    # also a margin we'll actually use that we might scale up or down with the
    # situation to stay safe, e.g. scale up as we go faster
    cdef double base_margin
    cdef double margin

    # a target location we are navigating to
    # the goal is to land between min_radius and arrival_radius
    cdef ccymunk.cpVect target_location
    cdef double arrival_radius
    cdef double min_radius

    # some state we keep from one analysis call to the next

    # shape hash id of the most recent threat (the shape might have been
    # removed from the space since we found it)
    cdef ccymunk.cpHashValue last_threat_id
    cdef double collision_margin_histeresis
    cdef bool cannot_avoid_collision_hold
    cdef bool collision_cbdr

    # the most recent analysis we've performed
    cdef NeighborAnalysis analysis

    # list of prior threat shape ids, be careful to check that these are still
    # in the space before
    cdef vector[ccymunk.cpHashValue] prior_threat_ids
    # list of relative positions of the last threat
    cdef list[RelPosHistoryEntry] rel_pos_history

    def __cinit__(
            self, space:cymunk.Space, body:cymunk.Body,
            radius:float, max_thrust:float, max_torque:float,
            max_speed:float,
            base_margin:float,
            base_neighborhood_radius:float,
            ) -> None:
        self.space = <ccymunk.Space?> space
        self.body = <ccymunk.Body?> body

        self.radius = radius
        self.max_thrust = max_thrust
        self.max_torque = max_torque
        self.max_acceleration = self.max_thrust / self.body._body.m
        self.max_angular_acceleration = self.max_torque / self.body._body.i
        self.worst_case_rot_time = sqrt(2. * pi / (self.max_torque/self.body._body.i))

        self.base_neighborhood_radius = base_neighborhood_radius
        self.neighborhood_radius = base_neighborhood_radius
        self.full_neighborhood_radius_period = 5.
        self.full_neighborhood_radius_ts = -self.full_neighborhood_radius_period*2.

        self.base_max_speed = max_speed
        self.max_speed = max_speed
        self.max_speed_cap = max_speed
        self.max_speed_cap_ts = 0.
        self.max_speed_cap_alpha = 0.2
        self.min_max_speed = 100
        # max speed cap decays, the longest that could take to be irrelevant is
        # this long
        self.max_speed_cap_max_expiration = math.log(max_speed/self.min_max_speed)/math.log(1+self.max_speed_cap_alpha)

        self.base_margin = base_margin
        self.target_location = ZERO_VECTOR
        self.arrival_radius = 0.
        self.min_radius = 0.

        # initialize some parameters as if we had done an analysis
        self.collision_margin_histeresis = 0.
        self.margin = self.base_margin
        self.last_threat_id = 0
        self.analysis.threat_count = 0
        self.analysis.neighborhood_size = 0
        self.analysis.nearest_neighbor_dist = ccymunk.INFINITY
        self.cannot_avoid_collision_hold = False
        self.analysis.cannot_avoid_collision = False
        self.collision_cbdr = False

    cdef void log(self, message, eid_prefix=""):
        log(self.body, f'{self.analysis.timestamp}\t'+message, eid_prefix)

    def set_location_params(self, target_location:cymunk.Vec2d, arrival_radius:float, min_radius:float) -> None:
        self.target_location = target_location.v
        self.arrival_radius = arrival_radius
        self.min_radius = min_radius

    def get_collision_margin(self): return self.base_margin

    def set_cannot_avoid_collision_hold(self, cach):
        self.cannot_avoid_collision_hold = cach

    #def set_max_speed_cap_params(self, max_speed_cap:float, max_speed_cap_ts:float, max_speed_cap_alpha:float):
    #    self.max_speed_cap = max_speed_cap
    #    self.max_speed_cap_ts = max_speed_cap_ts
    #    self.max_speed_cap_alpha = max_speed_cap_alpha

    #def set_neighbor_params(self, neighborhood_size:int, nearest_neighbor_dist:float):
    #    self.analysis.neighborhood_size = neighborhood_size
    #    self.analysis.nearest_neighbor_dist = nearest_neighbor_dist

    def get_threat_count(self): return self.analysis.threat_count
    def get_cannot_avoid_collision(self): return self.analysis.cannot_avoid_collision
    def get_cannot_avoid_collision_hold(self): return self.cannot_avoid_collision_hold
    def get_collision_cbdr(self): return self.collision_cbdr
    def get_num_neighbors(self): return self.analysis.neighborhood_size
    def get_nearest_neighbor_dist(self): return self.analysis.nearest_neighbor_dist
    def get_margin(self): return self.margin
    def get_collision_margin_histeresis(self): return self.collision_margin_histeresis
    def get_neighborhood_radius(self): return self.neighborhood_radius
    def get_max_speed(self): return self.max_speed
    def get_max_speed_cap(self): return self.max_speed_cap

    def get_telemetry(self) -> Dict[str, Any]:
        """ Gets a dict of telemetry data about navigation """
        telemetry = {

            "msc": self.max_speed_cap,
            "msc_ts": self.max_speed_cap_ts,
            "msc_a": self.max_speed_cap_alpha,

            "nn": self.analysis.neighborhood_size,
            "nnd": self.analysis.nearest_neighbor_dist
        }
        if self.analysis.threat_count > 0:
            telemetry.update({
                "ct": self.space.shapes[self.analysis.threat_shape.hashid_private].body,
                "ct_ms": self.analysis.minimum_separation,
                "ct_loc": cpvtoTuple(self.analysis.current_threat_loc),
                "ct_v": cpvtoTuple(self.analysis.threat_velocity),
                "ct_ts": self.analysis.detection_timestamp,
                "ct_at": self.analysis.approach_time,
                "ct_ct": self.analysis.coalesced_threats.size(),
                "ct_cloc": cpvtoTuple(self.analysis.threat_loc),
                "ct_cradius": self.analysis.threat_radius,
                "ct_cn": self.coalesced_neighbor_locations(),
                "ct_mh": self.collision_margin_histeresis,

                "cac": self.analysis.cannot_avoid_collision,
                "cach": self.cannot_avoid_collision_hold,
                "cbdr": self.collision_cbdr,
                "cbdr_hist": self.cbdr_history_summary()
            })

        return telemetry

    def set_telemetry(self, telemetry:Dict[str, Any], telemetry_ts:float) -> None:
        """ For testing purposes """
        self.max_speed_cap = telemetry.get("msc", self.base_max_speed)
        self.max_speed_cap_ts = telemetry.get("msc_ts", telemetry_ts) - telemetry_ts
        self.max_speed_cap_alpha = telemetry.get("msc_a", self.max_speed_cap_alpha)

        self.analysis.neighborhood_size = telemetry.get("nn", 0)
        self.analysis.nearest_neighbor_dist = telemetry.get("nnd", pymath.inf)

        if "ct" in telemetry:
            self.analysis.minimum_separation = telemetry.get("ct_ms", pymath.inf)
            self.analysis.current_threat_loc = tupletocpv(telemetry.get("ct_loc", (0.,0.)))
            self.analysis.threat_velocity = tupletocpv(telemetry.get("ct_v", (0.,0.)))
            self.analysis.timestamp = telemetry.get("ct_ts", telemetry_ts)-telemetry_ts
            self.analysis.approach_time = telemetry.get("ct_at", pymath.inf)

            #TODO: reconstitute the coalesced threats?

            self.analysis.threat_loc = tupletocpv(telemetry.get("ct_cloc", (0.,0.)))
            self.analysis.threat_radius = telemetry.get("ct_cradius", 0.)

            self.collision_margin_histeresis = telemetry.get("ct_mh", 0.)

            self.analysis.cannot_avoid_collision = telemetry.get("cac", False)
            self.cannot_avoid_collision_hold = telemetry.get("cach", False)
            self.collision_cbdr = telemetry.get("cbdr", False)

            #TODO: reconstitute cbdr history?

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

    cdef double _calculate_margin(self):
        cdef double scaled_collision_margin

        #TODO: can we take this as computed somewhere else?
        cdef ccymunk.cpVect course = ccymunk.cpvsub(self.target_location, self.body._body.p)
        cdef double distance = ccymunk.cpvlength(course)
        cdef double cm_low, cm_high, cm_speed_low, cm_speed_high

        if self.cannot_avoid_collision_hold:
            scaled_collision_margin = self.radius * 2
        elif distance < self.arrival_radius and distance > self.min_radius:
            scaled_collision_margin = self.radius*1.5
        elif distance < self.arrival_radius * 5:
            scaled_collision_margin = self.base_margin
        else:
            # scale collision margin with speed, more speed = more margin
            cm_low = self.base_margin
            cm_high = self.base_margin*5
            cm_speed_low = 100
            cm_speed_high = 1500
            scaled_collision_margin = clip(
                interpolate(
                    cm_speed_low, cm_low, cm_speed_high, cm_high,
                    ccymunk.cpvlength(self.body._body.v)
                ),
                cm_low, cm_high
            )

        scaled_collision_margin = min(self.analysis.nearest_neighbor_dist*0.8, scaled_collision_margin)

        return scaled_collision_margin

    cdef bool _calculate_cannot_avoid_collision(self, double neighbor_margin):
        cdef double desired_margin = neighbor_margin + self.analysis.threat_radius + self.radius
        cdef double required_acceleration, required_thrust
        cdef double min_clearance = self.analysis.threat_radius# + self.radius
        cdef double t = max(self.analysis.approach_time - self.worst_case_rot_time, 0.1)
        cdef bool cannot_avoid_collision = 0
        if self.analysis.distance_to_threat <= desired_margin + VELOCITY_EPS:
            if self.analysis.minimum_separation < min_clearance:
                if self.analysis.distance_to_threat < min_clearance:
                    cannot_avoid_collision = 1
                elif self.analysis.approach_time > 0.:
                    required_acceleration = 2.*(min_clearance-self.analysis.minimum_separation)/(t ** 2.)
                    required_thrust = self.body._body.m * required_acceleration
                    cannot_avoid_collision = required_thrust > self.max_thrust
        else:
            if self.analysis.minimum_separation < min_clearance:
                if self.analysis.approach_time > 0.:
                    # check if we can avoid collision
                    # s = u*t + 1/2 a * t^2
                    # s = threat_radius - minimum_separation
                    # ignore current velocity, we want to get displacement on
                    #   top of current situation
                    # t = approach_time
                    # a = 2 * s / t^2
                    required_acceleration = 2.*(min_clearance-self.analysis.minimum_separation)/(t ** 2.)
                    required_thrust = self.body._body.m * required_acceleration
                    cannot_avoid_collision = required_thrust > self.max_thrust
                else:
                    cannot_avoid_collision = 1

        return cannot_avoid_collision

    def prepare_analysis(self, timestamp:float):
        cdef double s_low, nr_low, s_high, nr_high
        cdef double d_low, d_high, density_max_speed
        cdef double nn_d_high, nn_s_high, nn_d_low, nn_s_low, nn_max_speed
        cdef double ct_dt_low, ct_s_low, ct_dt_high, ct_s_high, ct_max_speed
        cdef double max_speed_cap

        cdef double max_speed = self.base_max_speed

        if timestamp - self.full_neighborhood_radius_ts > self.full_neighborhood_radius_period:
            # periodically do a "full" ping of the neighborhood
            self.neighborhood_radius = self.base_neighborhood_radius
            self.full_neighborhood_radius_ts = timestamp
        else:
            # choose a neighborhood_radius depending on our speed
            s_low = 100
            nr_low = 1e3
            s_high= max_speed
            nr_high = self.base_neighborhood_radius
            self.neighborhood_radius = clip(
                interpolate(s_low, nr_low, s_high, nr_high,
                    ccymunk.cpvlength(self.body._body.v)
                ),
                nr_low, nr_high
            )

        # ramp down speed as nearby density increases
        # ramp down with inverse of the density: max_speed = m / (density + b)
        # d_low, s_high is one point we want to hit (speed at low density)
        # d_high, s_low is another (speed at high density
        d_low = 1.
        s_high = max_speed
        d_high = 30.
        s_low = self.min_max_speed
        density_max_speed = clip(
            interpolate(
                d_low, s_high, d_high, s_low,
                self.analysis.neighborhood_size,
            ),
            s_low, max_speed
        )

        # also ramp down speed with distance to nearest neighbor
        # nn_d_high, nn_speed_high is one point
        # nn_d_low, nn_speed_low is another
        nn_d_high = 2e3
        nn_s_high = 1000#max_speed
        nn_d_low = 5e2
        nn_s_low = self.min_max_speed
        nn_max_speed = clip(
            interpolate(
                nn_d_high, nn_s_high, nn_d_low, nn_s_low,
                self.analysis.nearest_neighbor_dist),
            nn_s_low, max_speed
        )

        if self.analysis.threat_count <= self.analysis.coalesced_threats.size():
            ct_max_speed = max_speed
        else:
            ct_max_speed = self.min_max_speed
            #ct_dt_low = 0.07
            #ct_s_high = max_speed
            #ct_dt_high = 1.5
            #ct_s_low = self.min_max_speed
            #ct_max_speed = clip(
            #    interpolate(
            #        ct_dt_low, ct_s_high, ct_dt_high, ct_s_low,
            #        self.analysis.timestamp - self.analysis.detection_timestamp
            #    ),
            #    ct_s_low, max_speed
            #)

        max_speed = min(max_speed, density_max_speed, nn_max_speed, ct_max_speed)

        # keep track of how low max speed gets to so we can apply histeresis
        if timestamp - self.max_speed_cap_ts > self.max_speed_cap_max_expiration:
            # this avoids overflow in the exponentiation when it's irrelevant
            max_speed_cap = max_speed
        else:
            # max_speed_cap decays back up to the max speed
            max_speed_cap = self.max_speed_cap * (1.+self.max_speed_cap_alpha)**(timestamp - self.max_speed_cap_ts)

        if max_speed < max_speed_cap:
            # keep track of the lowest our max speed is capped to
            self.max_speed_cap = max_speed
            self.max_speed_cap_ts = timestamp
        else:
            # apply our historic (decaying) max speed cap
            max_speed = min(max_speed_cap, max_speed)

        self.max_speed = max_speed

    def find_target_v(self, dt: float, safety_factor:float):
        """ Given goto location params, determine the desired velocity.

        returns a tuple:
            target velocity vector
            distance to the target location
            an estimate of the distance to the target location after dt
            boolean indicator if we cannot stop before reaching location
            delta speed between current and target
        """

        cdef DeltaVResult result = _find_target_v(self.body._body, self.target_location, self.arrival_radius, self.min_radius, self.max_acceleration, self.max_angular_acceleration, self.max_speed, dt, safety_factor)

        return (cpvtoVec2d(result.target_velocity), result.distance, result.distance_estimate, result.cannot_stop, result.delta_speed)

    def analyze_neighbors(
            self,
            current_timestamp:float,
            max_distance:float,
            ) -> Tuple[
                cymunk.Body,
                float,
                cymunk.Vec2d,
                cymunk.Vec2d,
                float,
                int,
                int,
                int,
                float,
                cymunk.Vec2d,
                cymunk.Vec2d,
                cymunk.Body,
                float,
                float,
                int,
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
        cdef Circle migrated_threat

        # calculate (and update) the margin we want
        self.margin = self._calculate_margin()

        # and expand based on if we're trying to avoid a target
        cdef double margin = self.margin + self.collision_margin_histeresis

        if speed > 0:
            # offset looking for threats in the direction we're travelling,
            # depending on our speed
            neighborhood_offset = clip(
                    interpolate(
                        NOFF_SPEED_LOW, NOFF_LOW, NOFF_SPEED_HIGH, NOFF_HIGH,
                        speed
                    ),
                    0, self.neighborhood_radius - margin)
            cneighborhood_loc = ccymunk.cpvadd(self.body._body.p, ccymunk.cpvmult(self.body._body.v, neighborhood_offset / speed))
        else:
            cneighborhood_loc = self.body._body.p

        # stash the prior threat circle to smooth threat location transitions
        cdef ccymunk.cpVect prior_threat_location = self.analysis.threat_loc
        cdef double prior_threat_radius = self.analysis.threat_radius

        self.analysis.neighborhood_radius = self.neighborhood_radius
        self.analysis.ship_radius = self.radius
        self.analysis.margin = margin
        self.analysis.max_distance = max_distance
        self.analysis.maximum_acceleration = self.max_acceleration
        self.analysis.body = self.body._body
        self.analysis.worst_ultimate_separation = ccymunk.INFINITY
        self.analysis.approach_time = ccymunk.INFINITY
        self.analysis.nearest_neighbor_dist = ccymunk.INFINITY
        self.analysis.neighborhood_size = 0
        self.analysis.threat_count = 0
        self.analysis.threat_radius = 0.
        self.analysis.timestamp = current_timestamp

        self.analysis.collision_threats.clear()
        self.analysis.coalesced_threats.clear()
        self.analysis.considered_shapes.clear()

        # start by considering prior threats
        for shape_id in self.prior_threat_ids:
            shape = self.space._shapes.get(shape_id)
            if shape_id is not None:
                _analyze_neighbor_callback((<ccymunk.Shape>shape)._shape, &self.analysis)

        # grab a copy of the shape ids for prior shapes
        cdef set[ccymunk.cpHashValue] prior_shape_ids = self.analysis.considered_shapes

        # look for threats in a circle
        ccymunk.cpSpaceNearestPointQuery(self.space._space, cneighborhood_loc, self.neighborhood_radius, 1, 0, _sensor_point_callback, &self.analysis)

        # look for threats in a cone facing the direction of our velocity
        # cone is truncated, starts at the edge of our nearest point query circle
        # goes until another 4 neighborhood radii in direction of our velocity
        # cone starts at margin
        cdef ccymunk.cpVect v_normalized = ccymunk.cpvnormalize(self.body._body.v)
        cdef ccymunk.cpVect v_perp = ccymunk.cpvperp(v_normalized)
        cdef ccymunk.cpVect start_point = ccymunk.cpvadd(cneighborhood_loc, ccymunk.cpvmult(v_normalized, self.neighborhood_radius-margin))
        cdef ccymunk.cpVect end_point = ccymunk.cpvadd(cneighborhood_loc, ccymunk.cpvmult(v_normalized, self.neighborhood_radius*3))

        cdef ccymunk.cpVect sensor_cone[4]
        # points are ordered to get a convex shape with the proper winding
        sensor_cone[1] = ccymunk.cpvadd(start_point, ccymunk.cpvmult(v_perp, (self.radius+margin*2)))
        sensor_cone[0] = ccymunk.cpvadd(start_point, ccymunk.cpvmult(v_perp, -(self.radius+margin*2)))
        sensor_cone[2] = ccymunk.cpvadd(end_point, ccymunk.cpvmult(v_perp, (self.radius+margin)*5))
        sensor_cone[3] = ccymunk.cpvadd(end_point, ccymunk.cpvmult(v_perp, -(self.radius+margin)*5))

        cdef ccymunk.cpShape *sensor_cone_shape = ccymunk.cpPolyShapeNew(NULL, 4, sensor_cone, ZERO_VECTOR)
        ccymunk.cpShapeUpdate(sensor_cone_shape, ZERO_VECTOR, ONE_VECTOR)
        ccymunk.cpSpaceShapeQuery(self.space._space, sensor_cone_shape, _sensor_shape_callback, &self.analysis)
        ccymunk.cpShapeFree(sensor_cone_shape)
        #TODO: do we need to deallocate the cone shape?

        coalesce_threats(&self.analysis)

        # we want to smooth transitions between threats if possible
        # but don't do this if we're in an emergency situation
        # i.e. cannot_avoid_collision_hold
        if not self.cannot_avoid_collision_hold and self.analysis.threat_count > 0 and prior_threat_radius > 0:
            migrated_threat = _migrate_threat_location(
                    self.body._body.p, self.radius,
                    prior_threat_location, prior_threat_radius,
                    self.analysis.threat_loc, self.analysis.threat_radius
            )
            self.analysis.threat_loc = migrated_threat.center
            self.analysis.threat_radius = migrated_threat.radius

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

        if self.analysis.nearest_neighbor_dist < ccymunk.INFINITY:
            nearest_neighbor = self.space.shapes[self.analysis.nearest_neighbor_shape.hashid_private].body
        else:
            nearest_neighbor = None
        neighborhood_density = self.analysis.neighborhood_size / (math.pi * self.analysis.neighborhood_radius ** 2)

        if self.analysis.threat_count == 0:
            self.analysis.distance_to_threat = ccymunk.INFINITY
            self.analysis.cannot_avoid_collision = False
            self.last_threat_id = 0
            self.prior_threat_ids.clear()
            self.rel_pos_history.clear()
            self.collision_cbdr = False

            self.collision_margin_histeresis = _update_collision_margin_histeresis(
                    self.collision_margin_histeresis, self.margin,
                    False, self.cannot_avoid_collision_hold
            )
            self.cannot_avoid_collision_hold = False

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
                    self.analysis.nearest_neighbor_dist,
                    neighborhood_density,
                    self.analysis.neighborhood_size,
                    0
            ))
        else:
            self.collision_cbdr = self._detect_cbdr(current_timestamp)
            # we back into current threat location from where the collision is
            # projected to happen and the threat velocity
            # we can't use the known current location of the threat neighbor since
            # the threat might be coalesced from may threats
            # this phantom current threat location is a simplication we  make
            self.analysis.current_threat_loc = ccymunk.cpvsub(
                    self.analysis.threat_loc,
                    ccymunk.cpvmult(
                        self.analysis.threat_velocity,
                        self.analysis.approach_time
                    )
            )
            self.analysis.distance_to_threat = ccymunk.cpvdist(
                    self.analysis.current_threat_loc, self.body._body.p)

            # figure out if we can get away from the threat
            self.analysis.cannot_avoid_collision = self._calculate_cannot_avoid_collision(margin)

            prior_threat_id = self.last_threat_id
            self.last_threat_id = self.analysis.threat_shape.hashid_private
            if self.last_threat_id == prior_threat_id:
                self.rel_pos_history.push_back(RelPosHistoryEntry(current_timestamp, self.analysis.relative_position, self.analysis.threat_velocity))
                # nuke history entries that are too old
                while current_timestamp - self.rel_pos_history.front().timestamp > CBDR_MAX_HIST_SEC:
                    self.rel_pos_history.pop_front()

            else:
                self.analysis.detection_timestamp = current_timestamp
                self.rel_pos_history.clear()
                self.rel_pos_history.push_back(RelPosHistoryEntry(current_timestamp, self.analysis.relative_position, self.analysis.threat_velocity))

            threat = self.space.shapes[self.analysis.threat_shape.hashid_private].body
            self.prior_threat_ids.clear()
            prior_threat_count = 0
            for ct in self.analysis.coalesced_threats:
                self.prior_threat_ids.push_back(ct.hashid_private)
                prior_threat_count += prior_shape_ids.count(ct.hashid_private)

            self.collision_margin_histeresis = _update_collision_margin_histeresis(
                    self.collision_margin_histeresis, self.margin,
                    prior_threat_count > 0, self.cannot_avoid_collision_hold
            )

            # if we cannot currently avoid a collision, flip the flag, but don't
            # clear it just because we currently are ok, that happens elsewhere.
            self.cannot_avoid_collision_hold = (prior_threat_count > 0 and self.cannot_avoid_collision_hold) or self.analysis.cannot_avoid_collision

            return tuple((
                    threat,
                    self.analysis.approach_time,
                    cpvtoVec2d(self.analysis.relative_position),
                    cpvtoVec2d(self.analysis.relative_velocity),
                    self.analysis.minimum_separation,
                    self.analysis.threat_count,
                    self.analysis.coalesced_threats.size(),
                    self.analysis.threat_count - self.analysis.coalesced_threats.size(),
                    self.analysis.threat_radius,
                    cpvtoVec2d(self.analysis.threat_loc),
                    cpvtoVec2d(self.analysis.threat_velocity),
                    nearest_neighbor,
                    self.analysis.nearest_neighbor_dist,
                    neighborhood_density,
                    self.analysis.neighborhood_size,
                    prior_threat_count,
            ))

    cdef bool _detect_cbdr(self, double current_timestamp):
        if self.rel_pos_history.size() < 2:
            return False
        if current_timestamp - self.rel_pos_history.front().timestamp < CBDR_MIN_HIST_SEC:
            return False

        if ccymunk.cpvlength(self.analysis.threat_velocity) == 0:
            return False

        cdef double oldest_distance = ccymunk.cpvlength(self.rel_pos_history.front().rel_pos)
        cdef double oldest_bearing = ccymunk.cpvtoangle(self.rel_pos_history.front().rel_pos)

        cdef double latest_distance = ccymunk.cpvlength(self.rel_pos_history.back().rel_pos)
        cdef double latest_bearing = ccymunk.cpvtoangle(self.rel_pos_history.back().rel_pos)

        return fabs(normalize_angle(oldest_bearing - latest_bearing, shortest=1)) < CBDR_ANGLE_EPS and oldest_distance - latest_distance > CBDR_DIST_EPS

    def collision_dv(self, current_timestamp:float, indesired_direction:cymunk.Vec2d) -> Tuple[Any]:
        """ Compute the delta velocity to avoid collision

        returns the delta velocity and whether the collision can be avoided
        """
        cdef double desired_margin = self.margin + self.collision_margin_histeresis + self.analysis.threat_radius + self.radius
        cdef ccymunk.cpVect desired_direction = indesired_direction.v

        # return values
        cdef ccymunk.cpVect delta_velocity

        cdef double required_acceleration, required_thrust
        cdef double cbdr_bias = 2. #TODO: get rid of cbdr_bias or actually use it!
        cdef ccymunk.cpVect desired_delta_velocity
        cdef double ddv_mag, max_dv_available, delta_v_budget

        if self.analysis.distance_to_threat <= desired_margin + VELOCITY_EPS:
            delta_velocity = ccymunk.cpvmult(
                    ccymunk.cpvsub(self.analysis.current_threat_loc, self.body._body.p),
                    self.analysis.distance_to_threat * self.base_max_speed * -1
            )
        else:
            if self.analysis.minimum_separation > self.analysis.threat_radius and self.analysis.approach_time > 0:
                # check that the desired margin is feasible given our delta-v budget
                # if not, set the margin to whatever our delta-v budget permits
                required_acceleration = 2.*(desired_margin-self.analysis.minimum_separation)/(self.analysis.approach_time ** 2.)
                required_thrust = self.body._body.m * required_acceleration
                if required_thrust > self.max_thrust:
                    desired_margin = (self.max_acceleration * self.analysis.approach_time ** 2. + 2. * self.analysis.minimum_separation)/2.

            desired_delta_velocity = ccymunk.cpvsub(desired_direction, self.body._body.v)
            ddv_mag = ccymunk.cpvlength(desired_delta_velocity)
            max_dv_available = self.max_thrust / self.body._body.m * self.analysis.approach_time
            if ddv_mag > max_dv_available:
                desired_delta_velocity = ccymunk.cpvmult(desired_delta_velocity, max_dv_available / ddv_mag)

            delta_v_budget = self.max_thrust / self.body._body.m * (self.analysis.approach_time - self.worst_case_rot_time)

            delta_velocity = _collision_dv(
                    self.analysis.current_threat_loc, self.analysis.threat_velocity,
                    self.body._body.p, self.body._body.v, self.body._body.a,
                    desired_margin, desired_delta_velocity,
                    self.collision_cbdr, cbdr_bias,
                    delta_v_budget, self.cannot_avoid_collision_hold
            )

        return tuple((cpvtoVec2d(delta_velocity), self.collision_cbdr, self.analysis.cannot_avoid_collision))

cdef double ANGLE_EPS = 8e-2 # about 4 degrees
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

cdef double _rotation_time(
        double delta_angle, double angular_velocity,
        double max_angular_acceleration):
    # theta_f = theta_0 + omega_0*t + 1/2 * alpha * t^2
    # assume omega_0 = 0 <--- assumes we're not currently rotating!
    # assume we constantly accelerate half way, constantly accelerate the
    # other half
    return (fabs(angular_velocity)/max_angular_acceleration + 2*sqrt(fabs(delta_angle + 0.5*angular_velocity**2./max_angular_acceleration)/max_angular_acceleration))

def rotation_time(
        delta_angle:float, angular_velocity:float,
        max_acceleration:float) -> float:
    """ Estimates the time to rotate by a given delta angle amount. """
    return _rotation_time(delta_angle, angular_velocity, max_acceleration)

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
            desired_w =  sgn(difference_angle) * sqrt(fabs(difference_angle) * max_torque/moment * 2)

        if fabs(difference_angle) < ARRIVAL_ANGLE:
            w_dampener = interpolate(
                ARRIVAL_ANGLE, 0.8,
                0, 0.6,
                fabs(difference_angle)
            )
        else:
            w_dampener = 0.8
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

    if difference_mag < VELOCITY_EPS:
        difference_angle = body.a
        if fabs(body.w) < ANGLE_EPS:
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
    cdef double d = (ccymunk.cpvdot(body.v, course) / distance)**2. / (2* max_acceleration)

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

        desired_speed = (-2. * a * rot_time + sqrt((2. * a  * rot_time) ** 2. + 8 * a * s))/2.
        desired_speed = clip(desired_speed/safety_factor, 0, max_speed)

        target_v = ccymunk.cpvmult(ccymunk.cpvmult(course, 1./distance), desired_speed)

    return DeltaVResult(target_v, distance, distance_estimate, cannot_stop, fabs(ccymunk.cpvlength(body.v) - desired_speed))


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
        if ft_result.torque == 0. and cybody._body.w < ANGLE_EPS:
            cybody._body.w = 0.
    else:
        cybody._body.f = ft_result.force

    if ft_result.torque != 0.:
        cybody._body.t = ft_result.torque
    else:
        cybody._body.t = 0.

    return ft_result.continue_time

cdef Circle _migrate_threat_location(
        ccymunk.cpVect ref_loc, double ref_radius,
        ccymunk.cpVect old_loc, double old_radius,
        ccymunk.cpVect new_loc, double new_radius):
    cdef double new_old_dist
    cdef Circle enclosing_c
    cdef ccymunk.cpVect new_old_vec
    cdef ccymunk.cpVect migrated_loc
    cdef double migrated_radius

    new_old_dist = ccymunk.cpvdist(old_loc, new_loc)
    if new_old_dist < 2*new_radius or new_old_dist < 2*old_radius:
        if new_old_dist + new_radius > old_radius + VELOCITY_EPS:
            # the new circle does not completely eclipse the old
            # find the smallest enclosing circle for both
            enclosing_c = enclosing_circle(
                    new_loc, new_radius, old_loc, old_radius)
            old_loc = enclosing_c.center
            old_radius = enclosing_c.radius
            new_old_dist = ccymunk.cpvdist(old_loc, new_loc)

        new_old_vec = ccymunk.cpvsub(new_loc, old_loc)

        # the new threat is smaller than the old one and completely
        # contained in the old one. let's scale and translate the old
        # one toward the new one so that it still contains it, but
        # asymptotically approaches it. this will avoid
        # discontinuities.
        migrated_radius = clip(
            old_radius * THREAT_RADIUS_SCALE_FACTOR,
            new_radius,
            old_radius
        )
        if isclose(0, new_old_dist):
            migrated_loc = old_loc
        else:
            migrated_loc = ccymunk.cpvadd(
                ccymunk.cpvmult(
                    new_old_vec,
                    1./new_old_dist * (old_radius-migrated_radius)
                ),
                old_loc
            )
        # useful assert during testing
        #assert ccymunk.cpvdist(new_loc, migrated_loc) + new_radius >= migrated_radius + VELOCITY_EPS:

        # if we're already inside the margin for this migrated location stick
        # with the one computed for the actual threats we have, discontinuities
        # be damned
        if ccymunk.cpvdist(ref_loc, migrated_loc) < migrated_radius + ref_radius:
            migrated_loc = new_loc
            migrated_radius = new_radius

        return Circle(migrated_loc, migrated_radius)
    else:
        return Circle(new_loc, new_radius)

def migrate_threat_location(
        ref_loc:cymunk.Vec2d, ref_radius:float,
        old_loc:cymunk.Vec2d, old_radius:float,
        new_loc:cymunk.Vec2d, new_radius:float) -> Tuple[cymunk.Vec2d, float]:
    """ Migrate from the old circle towards the new circle.

    This happens such that the migrated circle always contains the new circle.
    Migration is aborted if the migrated circle would include the reference
    circle. """

    cdef Circle migrated_circle = _migrate_threat_location(
            ref_loc, ref_radius,
            old_loc, old_radius,
            new_loc, new_radius
    )

    return (cpvtoVec2d(migrated_circle.center), migrated_circle.radius)
