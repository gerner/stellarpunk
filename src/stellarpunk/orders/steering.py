""" Steering behaviors that can be used in orders. """ 

from __future__ import annotations

import collections
import math
from typing import Optional, Deque, Any, Tuple

import numpy as np
import numpy.typing as npt
import numba as nb # type: ignore
from numba import jit # type: ignore

from stellarpunk import util, core

ANGLE_EPS = 2e-3 # about .06 degrees
PARALLEL_EPS = 0.5e-1
VELOCITY_EPS = 1e-1

CBDR_HIST_SEC = 0.5
CBDR_ANGLE_EPS = 1e-1 # about 6 degrees
CBDR_DIST_EPS = 15

# think of this as the gimballing angle (?)
# pi/16 is ~11 degrees
COARSE_ANGLE_MATCH = np.pi/16

# the scale (per tick) we use to scale down threat radii if the new threat is
# still covered by the previous threat radius
THREAT_RADIUS_SCALE_FACTOR = 0.99
THREAT_LOCATION_ALPHA = 0.001

# a convenient zero vector to avoid needless array creations
ZERO_VECTOR = np.array((0.,0.))
ZERO_VECTOR.flags.writeable = False

@jit(cache=True, nopython=True)
def rotation_time(delta_angle: float, angular_velocity: float, max_angular_acceleration: float, safety_factor:float) -> float:
    # theta_f = theta_0 + omega_0*t + 1/2 * alpha * t^2
    # assume omega_0 = 0 <--- assumes we're not currently rotating!
    # assume we constantly accelerate half way, constantly accelerate the
    # other half
    return (abs(angular_velocity)/max_angular_acceleration + 2*np.sqrt(abs(delta_angle + 0.5*angular_velocity**2/max_angular_acceleration)/max_angular_acceleration))*safety_factor


@jit(cache=True, nopython=True)
def torque_for_angle(target_angle: float, angle:float, w:float, max_torque:float, moment:float, dt:float, safety_factor:float) -> float:
    """ What torque to apply to achieve target angle """

    difference_angle = util.normalize_angle(target_angle - angle, shortest=True)

    if abs(w) < ANGLE_EPS and abs(difference_angle) < ANGLE_EPS:
        # bail if we're basically already there
        # caller can handle this, e.g. set rotation to target and w to 0
        t = 0.0
    else:
        # add torque in the desired direction to get
        # accel = tau / moment
        # dw = accel * dt
        # desired w is w such that braking_angle = difference_angle
        # braking_angle =  -1 * np.sign(w) * -0.5 * w*w * moment / max_torque
        # sqrt(difference_angle * max_torque / (0.5 * moment)) = w
        arrival_angle = ANGLE_EPS
        if abs(difference_angle) < ANGLE_EPS:
            desired_w = 0.
        else:
            # w_f**2 = w_i**2 + 2 * a (d_theta)
            #desired_w = np.sign(difference_angle) * np.sqrt(abs(difference_angle + w*dt) * max_torque / (0.5 * moment))/safety_factor
            desired_w =  np.sign(difference_angle) * np.sqrt(np.abs(difference_angle) * max_torque/moment * 2) * 0.90

        t = (desired_w - w)*moment/dt

    return util.clip(t, -1.0*max_torque, max_torque)
    #return np.clip(t, -1*max_torque, max_torque)

@jit(cache=True, nopython=True)
def force_for_delta_velocity(dv:np.ndarray, max_thrust:float, mass:float, dt:float) -> Tuple[np.ndarray, float]:
    """ What force to apply to get dv change in velocity. Ignores heading. """

    if util.both_almost_zero(dv):
        return ZERO_VECTOR, np.inf

    dv_magnitude = util.magnitude(dv[0], dv[1])
    desired_thrust = mass * dv_magnitude / dt
    if desired_thrust > max_thrust:
        return dv / dv_magnitude * max_thrust, mass * dv_magnitude/max_thrust
    else:
        return dv / dv_magnitude * desired_thrust, dt

@jit(cache=True, nopython=True)
def force_torque_for_delta_velocity(
        target_velocity:np.ndarray, mass:float, moment:float, angle:float,
        w:float, v:np.ndarray, max_speed:float, max_torque:float,
        max_thrust:float, max_fine_thrust:float, dt:float,
        safety_factor:float) -> tuple[np.ndarray, float, np.ndarray, float, float]:
    """ Given target velocity, a timestep size  and parameters about the ship,
    return force, torque, target velocity, and desired speed difference and
    time to hold those values before calling me again. """

    dv = target_velocity - v
    difference_mag, difference_angle = util.cartesian_to_polar(dv[0], dv[1])

    if difference_mag < VELOCITY_EPS and abs(w) < ANGLE_EPS:
        return ZERO_VECTOR, 0., target_velocity, difference_mag, np.inf

    delta_heading = util.normalize_angle(angle-difference_angle, shortest=True)
    rot_time = rotation_time(delta_heading, w, max_torque/moment, safety_factor)

    # while we've got a lot of thrusting to do, we can tolerate only
    # approximately matching our desired angle
    # this should have something to do with how much we expect this angle
    # to change in dt time, but order of magnitude seems reasonable approx

    if (difference_mag * mass / max_fine_thrust > rot_time and abs(delta_heading) > COARSE_ANGLE_MATCH) or difference_mag < VELOCITY_EPS:
        # we need to rotate in direction of thrust
        torque = torque_for_angle(difference_angle, angle, w, max_torque, moment, dt, safety_factor)

        # also apply thrust depending on where we're pointed
        force, thrust_time = force_for_delta_velocity(dv, max_fine_thrust, mass, dt)
    else:
        torque = torque_for_angle(difference_angle, angle, w, max_torque, moment, dt, safety_factor)

        # we should apply thrust, however we can with the current heading
        # max thrust is main engines if we're pointing in the desired
        # direction, otherwise use fine thrusters
        if abs(delta_heading) < COARSE_ANGLE_MATCH:
            max_thrust = max_thrust
        else:
            max_thrust = max_fine_thrust

        force, thrust_time = force_for_delta_velocity(dv, max_thrust, mass, dt)

    if util.isclose(rot_time, 0.):
        continue_time = thrust_time
    else:
        continue_time = min(rot_time, thrust_time)

    return force, torque, target_velocity, difference_mag, continue_time

@jit(cache=True, nopython=True)
def _analyze_neighbor(pos:np.ndarray, v:np.ndarray, entity_radius:float, entity_pos:np.ndarray, entity_v:np.ndarray, max_distance:float, max_approach_time:float, margin:float) -> tuple[float, float, np.ndarray, np.ndarray, float, np.ndarray, float]:
    speed = util.magnitude(v[0], v[1])
    rel_pos = entity_pos - pos
    rel_vel = entity_v - v

    rel_speed = util.magnitude(rel_vel[0], rel_vel[1])

    rel_dist = util.magnitude(rel_pos[0], rel_pos[1])

    # check for parallel paths
    if rel_speed == 0:
        if rel_dist < margin + entity_radius:
            # this can cause discontinuities in approach_time
            return rel_dist, 0., rel_pos, rel_vel, rel_dist, entity_pos, 0.
        return rel_dist, np.inf, rel_pos, rel_vel, np.inf, ZERO_VECTOR, np.inf

    rel_tangent = rel_vel / rel_speed
    approach_t = -1 * rel_tangent.dot(rel_pos) / rel_speed

    if approach_t <= 0 or approach_t >= max_approach_time:
        if rel_dist < margin + entity_radius:
            # this can cause discontinuities in approach_time
            return rel_dist, 0., rel_pos, rel_vel, rel_dist, entity_pos, 0.
        return rel_dist, np.inf, rel_pos, rel_vel, np.inf, ZERO_VECTOR, np.inf

    # compute the closest approach within max_distance
    collision_distance = speed * approach_t
    if collision_distance > max_distance:
        approach_t = max_distance / speed
        collision_distance = max_distance - VELOCITY_EPS

    sep_vec = (pos + v*approach_t) - (entity_pos + entity_v*approach_t)
    min_sep = util.magnitude(sep_vec[0], sep_vec[1])
    collision_loc = entity_pos + entity_v * approach_t

    return rel_dist, approach_t, rel_pos, rel_vel, min_sep, collision_loc, collision_distance

@jit(cache=True, nopython=True)
def _analyze_neighbors(
        hits_l:npt.NDArray[np.float64],
        hits_v:npt.NDArray[np.float64],
        hits_r:npt.NDArray[np.float64],
        pos:npt.NDArray[np.float64],
        v:npt.NDArray[np.float64],
        max_distance:float,
        ship_radius:float,
        margin:float,
        neighborhood_radius:float,
        ) -> tuple[
            int,
            float,
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            float,
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
    """ Analyzes neighbors and determines collision threat parameters. """

    approach_time = np.inf
    idx = -1
    relative_position = ZERO_VECTOR
    relative_velocity: np.ndarray  = ZERO_VECTOR
    minimum_separation = np.inf
    collision_loc = ZERO_VECTOR
    nearest_neighbor_idx = -1
    nearest_neighbor_dist = np.inf

    collision_threats:list[tuple[int, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float], npt.NDArray[np.float64]]] = []

    neighborhood_size = 0.
    threat_count = 0
    for eidx in range(len(hits_l)):
        entity_pos = hits_l[eidx]
        entity_v = hits_v[eidx]
        entity_radius = hits_r[eidx]

        rel_dist, approach_t, rel_pos, rel_vel, min_sep, c_loc, collision_distance = _analyze_neighbor(
                pos, v, entity_radius, entity_pos, entity_v, max_distance, np.inf, ship_radius + margin)

        if rel_dist < neighborhood_radius:
            neighborhood_size += 1.

        if rel_dist < nearest_neighbor_dist:
            nearest_neighbor_idx = eidx
            nearest_neighbor_dist = rel_dist

        # this neighbor isn't going to collide with us
        if not (min_sep < np.inf):
            continue

        # we need to keep track of all collision threats for coalescing later
        collision_threats.append((eidx, (entity_pos, entity_v, entity_radius), c_loc))
        if min_sep > entity_radius + ship_radius + margin:
            continue

        threat_count += 1

        # most threatening is the soonest, so keep track of that one
        if approach_t < approach_time:
            approach_time = approach_t
            idx = eidx
            relative_position = rel_pos
            relative_velocity = rel_vel
            minimum_separation = float(min_sep)
            collision_loc = c_loc

    # Once we have a single most threatening future collision, coalesce nearby
    # threats so we can avoid all of them at once, instead of avoiding one only
    # to make another inevitable.
    # this isn't exactly going to yield a circumcircle around the points. For
    # one thing it's not clear there's an "optimal" way to choose the points to
    # include.
    # see https://github.com/marmakoide/miniball and https://github.com/weddige/miniball
    coalesced_threats = 0
    ct = []
    if idx >= 0:
        threat_radius = hits_r[idx]
        threat_loc = collision_loc
        threat_velocity = hits_v[idx]

        # coalesce nearby threats
        # this avoids flapping in collision targets
        coalesced_threats = 1
        ct.append(idx)
        if threat_count > 1:
            threat_loc = threat_loc.copy()
            for eidx, (t_loc, t_velocity, t_radius), t_loc in collision_threats:
                if idx == eidx:
                    continue
                t_rel_pos = (threat_loc - t_loc)
                t_dist, t_angle = util.cartesian_to_polar(t_rel_pos[0], t_rel_pos[1])
                if t_dist + t_radius < threat_radius:
                    # the old radius completely covers the new one
                    threat_velocity = (threat_velocity * coalesced_threats + t_velocity)/(coalesced_threats + 1)
                    coalesced_threats += 1
                    ct.append(eidx)
                elif t_dist < threat_radius + t_radius + 2*margin:
                    # new is within coalesce dist, but not already covered
                    # coalesced threat should just cover both
                    # diameter = 2*threat_radius + 2*t_radius + (t_dist - threat_radius - t_radius)
                    # diameter = t_dist + threat_radius + t_radius
                    coalesced_radius = (t_dist + threat_radius + t_radius)/2

                    c_rel_x, c_rel_y = util.polar_to_cartesian(coalesced_radius - threat_radius, t_angle+np.pi)

                    #assert np.linalg.norm((c_rel_x, c_rel_y)) + threat_radius < coalesced_radius + VELOCITY_EPS
                    #assert np.linalg.norm((threat_loc[0]+c_rel_x - t_loc[0], threat_loc[1]+c_rel_y - t_loc[1])) + t_radius < coalesced_radius + VELOCITY_EPS

                    threat_loc[0] += c_rel_x
                    threat_loc[1] += c_rel_y
                    threat_radius = coalesced_radius
                    threat_velocity = (threat_velocity * coalesced_threats + t_velocity)/(coalesced_threats + 1)
                    coalesced_threats += 1
                    ct.append(eidx)
    else:
        # no threat found, return some default values
        threat_radius = 0.
        threat_loc = ZERO_VECTOR
        threat_velocity = ZERO_VECTOR

    return idx, approach_time, relative_position, relative_velocity, minimum_separation, threat_count, coalesced_threats, threat_radius, threat_loc, threat_velocity, nearest_neighbor_idx, nearest_neighbor_dist, neighborhood_size / (np.pi * neighborhood_radius ** 2), np.array(ct)

@jit(cache=True, nopython=True)
def _collision_dv(entity_pos:npt.NDArray[np.float64], entity_vel:npt.NDArray[np.float64], pos:npt.NDArray[np.float64], vel:npt.NDArray[np.float64], margin:float, v_d:npt.NDArray[np.float64], cbdr:bool) -> npt.NDArray[np.float64]:
    """ Computes a divert vector (as in accelerate_to(v + dv)) to avoid a
    collision by at least distance m. This divert will be of minimum size
    relative to the desired velocity.

    entity_pos: location of the threat
    entity_pos: velocity of the threat
    pos: our position
    v: our velocity
    v_d: the desired velocity
    """

    # rel pos
    r = entity_pos - pos
    # rel vel
    v = entity_vel - vel
    # margin, including radii
    m = margin

    # desired diversion from v
    a = v_d + vel

    # check if the desired divert is already viable
    x = a[0]
    y = a[1]

    if util.isclose(v[0], 0.) and util.isclose(v[1], 0.):
        do_nothing_margin_sq = r[0]**2+r[1]**2
    else:
        do_nothing_margin_sq = r[0]**2+r[1]**2 - (r[0]*x+r[1]*y+(2*r[0]*v[0]+2*r[1]*v[1]))**2/((2*v[0]+x)**2+(2*v[1]+y)**2)
    if do_nothing_margin_sq > 0 and do_nothing_margin_sq >= m**2:
        return ZERO_VECTOR

    if util.magnitude(r[0], r[1]) <= margin:
        raise ValueError()

    # given divert (x,y):
    # (r[0]**2+r[1]**2)-(2*(r[0]*v[0]+r[1]*v[1])+(r[0]*x+r[1]*y))**2/((2*v[0]+x)**2 + (2*v[1]+y)**2) > m**2
    # this forms a pair of intersecting lines with viable diverts between them

    # given divert (x,y):
    # cost_from desired = (a[0]-x)**2 +(a[1]-y)**2
    # see https://www.desmos.com/calculator/qvk8fpbw3k

    # to understand the margin, we end up with two circles whose intersection
    # points are points on the tangent lines that form the boundary of our
    # viable diverts
    # (x+2*v[0])**2 + (y+2*v[1])**2 = r[0]**2+r[1]**2-m**2
    # (x+2*v[0]-r[0])**2 + (y+2*v[1]-r[1])**2 = m**2
    # a couple of simlifying substitutions:
    # we can translate the whole system to the origin:
    # let s_x,s_y = (x + 2*v[0], y + 2*v[1])
    # let p = r[0]**2 + r[1]**2 - m**2
    # having done this we can subtract the one from the other, solve for y,
    # plug back into one of the equations
    # solve the resulting quadratic eq for x (two roots)
    # plug back into one of the equations to get y (two sets of two roots)
    # for the y roots, only one will satisfy the other equation, pick that one
    # also, we'll divide by r[1] below. So if that's zero we have an alternate
    # form where there's a single value for x

    p = r[0]**2 + r[1]**2 - m**2

    if util.isclose(r[1]**2, 0, atol=1e-5):
        # this case would cause a divide by zero when computing the
        # coefficients of the quadratic equation below
        s_1x = s_2x = p/r[0]
    else:
        # note that r[0] and r[1] cannot both be zero (assuming m>0)
        q_a = r[0]**2/r[1]**2+1
        q_b = -2*p*r[0]/r[1]**2
        q_c = p**2/r[1]**2 - p
        # quadratic formula
        # note that we get real roots as long as the problem is feasible (i.e.
        # we're not already inside the margin
        s_1x = (-q_b-np.sqrt(q_b**2-4*q_a*q_c)) / (2*q_a)
        s_2x = (-q_b+np.sqrt(q_b**2-4*q_a*q_c)) / (2*q_a)

    # y roots are y_i and -y_i, but only one each for i=0,1 will be on the curve
    s_1y = np.sqrt(p-s_1x**2)
    if not util.isclose((s_1x - r[0])**2 + (s_1y - r[1])**2, m**2):
        s_1y = -s_1y
    s_2y = np.sqrt(p-s_2x**2)
    if not util.isclose((s_2x - r[0])**2 + (s_2y - r[1])**2, m**2):
        s_2y = -s_2y

    # subbing back in our x_hat,y_hat above,
    # these determine the slope of the boundry lines of our viable region
    # (1) y+2*v[1] = s_iy/s_ix * (x+2*v[0]) for i = 0,1 (careful if x_i = 0)
    # with perpendiculars going through the desired_divert point
    # (2) y-a[1] = -s_ix/s_iy * (x-a[0]) (careful if y_i == 0)
    # so find the intersection of each of these pairs of equations

    if util.isclose(s_1x, 0):
        # tangent line is vertical
        # implies perpendicular is horizontal
        y1 = a[1]
        x1 = 0
    elif util.isclose(s_1y, 0):
        # tangent line is horizontal
        # implies perpendicular is vertical
        x1 = a[0]
        y1 = 0
    else:
        # solve (1) for y in terms of x and plug into (2), solve for x
        x1 = (s_1x/s_1y*a[0]+a[1] - s_1y/s_1x*2*v[0] + 2*v[1]) / (s_1y/s_1x + s_1x/s_1y)
        # plug back into (1)
        y1 = s_1y/s_1x * (x1+2*v[0]) - 2*v[1]

    if util.isclose(s_2x, 0):
        y2 = a[1]
        x2 = 0
    elif util.isclose(s_2y, 0):
        x2 = a[0]
        y2 = 0
    else:
        x2 = (s_2x/s_2y*a[0]+a[1] - s_2y/s_2x*2*v[0] + 2*v[1]) / (s_2y/s_2x + s_2x/s_2y)
        y2 = s_2y/s_2x * (x2+2*v[0]) - 2*v[1]

    cost1 = (a[0]-x1)**2 +(a[1]-y1)**2
    cost2 = (a[0]-x2)**2 +(a[1]-y2)**2

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
        assert not math.isnan(cost1) or not math.isnan(cost2)

    if cbdr:
        if util.isclose(s_x, 0):
            dx = 0
            dy = np.sqrt(cost*2)
            y += np.sqrt(cost*2)
        elif util.isclose(s_y, 0):
            dx = np.sqrt(cost*2)
            dy = 0
            x += dx
        else:
            dx = np.sqrt(cost*2 / ((s_y/s_x)**2 + 1))
            dy = s_y/s_x * dx
            x += dx
            y += dy

    # useful assert when testing
    # this asserts that the resulting x,y point matches the the contraint on
    # the margin
    assert util.isclose(
            (r[0]**2+r[1]**2)-(2*(r[0]*v[0]+r[1]*v[1])+(r[0]*x+r[1]*y))**2/((2*v[0]+x)**2 + (2*v[1]+y)**2),
            m**2,
            rtol=1e-3)
    return np.array((x, y))

# numba seems to have trouble with this method and recompiles it with some
# frequency. So we explicitly specify types here to avoid that.
@jit(
        nb.types.Tuple(
            (nb.float64[::1], nb.float64, nb.float64, nb.boolean)
        )(
            nb.float64[::1], nb.float64, nb.float64,
            nb.float64[::1], nb.float64[::1], nb.float64, nb.float64, nb.float64,
            nb.float64, nb.float64, nb.float64, nb.float64
        ), cache=True, nopython=True)
def find_target_v(
        target_location:np.ndarray, arrival_distance:float, min_distance:float,
        current_location:np.ndarray, v:np.ndarray, theta:float, omega:float,
        max_acceleration:float, max_angular_acceleration:float, max_speed:float,
        dt:float, safety_factor:float) -> tuple[np.ndarray, float, float, bool]:
    """ Given goto location params, determine the desired velocity.

    returns a tuple:
        target velocity vector
        distance to the target location
        an estimate of the distance to the target location after dt
        boolean indicator if we cannot stop before reaching location
    """

    course = target_location - current_location
    distance, target_angle = util.cartesian_to_polar(course[0], course[1])

    distance_estimate = distance - max_speed*dt

    # if we were to cancel the velocity component in the direction of the
    # target, will we travel enough so that we cross min_distance?
    d = (np.dot(v, course) / distance)**2 / (2* max_acceleration)

    if d > distance-min_distance:
        cannot_stop = True
    else:
        cannot_stop = False

    if distance < arrival_distance + VELOCITY_EPS:
        #target_v = ZERO_VECTOR
        target_v = np.array((0.,0.))
        desired_speed = 0.
    else:
        rot_time = rotation_time(abs(util.normalize_angle(theta-(target_angle+np.pi), shortest=True)), omega, max_angular_acceleration, safety_factor)
        #rot_time = rotation_time(np.pi, 0, self.ship.max_angular_acceleration(), self.safety_factor)

        # choose a desired speed such that if we were at that speed right
        # now we would have enough time to rotate 180 degrees and
        # decelerate to a stop at full thrust by the time we reach arrival
        # distance
        a = max_acceleration
        s = distance - min_distance
        if s < 0:
            s = 0

        # there are two roots to the quadratic equation
        desired_speed_1 = (-2 * a * rot_time + np.sqrt((2 * a  * rot_time) ** 2 + 8 * a * s))/2
        # we discard the smaller one
        #desired_speed_2 = (-2 * a * rot_time - np.sqrt((2 * a * rot_time) ** 2 + 8 * a * s))/2

        # numba doesn't support np.clip
        #desired_speed = np.clip(desired_speed_1/safety_factor, 0, max_speed)
        desired_speed = util.clip(desired_speed_1/safety_factor, 0, max_speed)


        target_v = course/distance * desired_speed

    return target_v, distance, distance_estimate, cannot_stop

def detect_cbdr(rel_pos_hist:Deque[np.ndarray], min_hist:int) -> bool:
    if len(rel_pos_hist) < min_hist:
        return False

    oldest_rel_pos = rel_pos_hist[0]
    oldest_distance, oldest_bearing = util.cartesian_to_polar(oldest_rel_pos[0], oldest_rel_pos[1])

    latest_rel_pos = rel_pos_hist[-1]
    latest_distance, latest_bearing = util.cartesian_to_polar(latest_rel_pos[0], latest_rel_pos[1])

    return abs(oldest_bearing - latest_bearing) < ANGLE_EPS and oldest_distance - latest_distance > CBDR_DIST_EPS

class AbstractSteeringOrder(core.Order):
    def __init__(self, *args: Any, safety_factor:float=2., **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.safety_factor = safety_factor

        self.collision_margin = 2e2
        self.nearest_neighbor:Optional[core.SectorEntity] = None
        self.nearest_neighbor_dist = np.inf

        self.collision_threat:Optional[core.SectorEntity] = None
        self.collision_dv = ZERO_VECTOR
        self.collision_threat_time = 0.
        self.collision_relative_velocity = ZERO_VECTOR
        self.collision_relative_position = ZERO_VECTOR
        self.collision_approach_time = np.inf
        self.cbdr_ticks=int(CBDR_HIST_SEC/self.gamestate.dt)
        self.collision_rel_pos_hist:Deque[np.ndarray] = collections.deque(maxlen=self.cbdr_ticks)
        self.collision_cbdr = False
        self.collision_cbdr_divert_angle = np.pi/4
        self.collision_threat_count = 0.
        self.collision_coalesced_threats = 0.
        self.collision_coalesced_neighbors:list[core.SectorEntity] = []
        self.collision_threat_loc = ZERO_VECTOR
        self.collision_threat_radius = 0.

        self.cannot_avoid_collision = False

        self.collision_hits:list[core.SectorEntity] = []
        self.collision_hits_age = 0.
        self.collision_hits_max_age = 1./10.

        # how many neighbors per m^2
        self.neighborhood_density = 0.

        self._next_accelerate_compute_ts = 0.
        self._accelerate_force:npt.NDArray[np.float64] = ZERO_VECTOR

    def to_history(self) -> dict:
        history = super().to_history()
        if self.collision_threat:
            history["ct"] = str(self.collision_threat.entity_id)
            history["ct_loc"] = (self.collision_threat.loc[0], self.collision_threat.loc[1])
            history["ct_v"] = (self.collision_threat.velocity[0], self.collision_threat.velocity[1])
            history["ct_ts"] = self.collision_threat_time
            history["ct_dv"] = (self.collision_dv[0], self.collision_dv[1])
            history["ct_tc"] = self.collision_threat_count
            history["ct_ct"] = self.collision_coalesced_threats
            history["ct_cloc"] = (self.collision_threat_loc[0], self.collision_threat_loc[1])
            history["ct_cradius"] = self.collision_threat_radius
            history["ct_cn"] = [(cn.loc[0], cn.loc[1]) for cn in self.collision_coalesced_neighbors]
            history["cac"] = bool(self.cannot_avoid_collision)
            history["cbdr"] = self.collision_cbdr
            history["cbdr_hist"] = [(loc[0], loc[1]) for loc in self.collision_rel_pos_hist]

        history["nd"] = self.neighborhood_density
        history["nnd"] = self.nearest_neighbor_dist
        return history

    def _rotate_to(self, target_angle: float, dt: float) -> None:
        # given current angle and angular_velocity and max torque, choose
        # torque to apply for dt now to hit target angle

        w = self.ship.angular_velocity
        moment = self.ship.moment

        t = torque_for_angle(
                target_angle, self.ship.angle, w,
                self.ship.max_torque, moment, dt,
                self.safety_factor)

        if t == 0 and abs(util.normalize_angle(self.ship.angle - target_angle, shortest=True)) <= ANGLE_EPS:
            self.ship.set_angle(target_angle)
            self.ship.set_angular_velocity(0.)
        else:
            #self.logger.debug(f'apply torque {t:.0f} for desired target_angle {target_angle:.3f} from {target_angle:.3f} at {w:.2f}rad/sec')
            self.ship.apply_torque(t)

    def _accelerate_to(self, target_velocity: np.ndarray, dt: float, force_recompute:bool=False, time_step:float=0.) -> None:
        if target_velocity[0] == self.ship.velocity[0] and target_velocity[1] == self.ship.velocity[1]:
            return

        mass = self.ship.mass
        moment = self.ship.moment
        angle = self.ship.angle
        w = self.ship.angular_velocity
        v = self.ship.velocity
        max_speed = self.ship.max_speed()
        max_torque = self.ship.max_torque
        max_thrust = self.ship.max_thrust
        max_fine_thrust = self.ship.max_fine_thrust

        force, torque, target_velocity, difference_mag, _ = force_torque_for_delta_velocity(
                    target_velocity,
                    mass, moment, angle, w, v,
                    max_speed, max_torque, max_thrust, max_fine_thrust,
                    dt, self.safety_factor
            )

        if util.both_almost_zero(force):
            if difference_mag > 0.:
                self.ship.set_velocity(target_velocity)
        else:
            self.ship.apply_force(force)

        if torque != 0.:
            self.ship.apply_torque(torque)

        #t = difference_mag / (np.linalg.norm(np.array((x,y))) / mass) if (x,y) != (0,0) else 0
        #self.logger.debug(f'force: {(x, y)} {np.linalg.norm(np.array((x,y)))} in {t:.2f}s')

    def _clear_collision_info(self) -> None:
        self.collision_threat = None
        self.collision_dv = ZERO_VECTOR
        self.collision_threat_time = 0.
        self.collision_relative_velocity = ZERO_VECTOR
        self.collision_relative_position = ZERO_VECTOR
        self.collision_approach_time = np.inf
        self.collision_rel_pos_hist.clear()
        self.collision_cbdr = False
        self.cannot_avoid_collision = False

    def _collision_neighbor(
            self,
            sector: core.Sector,
            neighborhood_loc: np.ndarray,
            neighborhood_dist: float,
            margin: float,
            max_distance: float
        ) -> tuple[
            Optional[core.SectorEntity],
            float,
            np.ndarray,
            np.ndarray,
            float,
            float,
            np.ndarray,
            np.ndarray,
            float]:

        pos = self.ship.loc
        v = self.ship.velocity

        last_collision_threat = self.collision_threat

        # cache the neighborhood for a short period. note we cache the entities
        # (not the locations or velocities) since we want to get updated
        # loc/vel data for the neighbors.

        if self.gamestate.timestamp - self.collision_hits_age < self.collision_hits_max_age:
            #TODO: what to do if the neighbor isn't in this sector any more?
            hits = self.collision_hits
        else:
            hits = list(e for e in  sector.spatial_point(neighborhood_loc, neighborhood_dist) if e != self.ship)
            self.collision_hits_age = self.gamestate.timestamp
            self.collision_hits = hits
        self.cannot_avoid_collision = False
        self.nearest_neighbor_dist = np.inf

        if len(hits) > 0:
            #TODO: this is really not ideal: we go into pymunk to get hits via
            # cffi and then come back to python and then do some marshalling
            # there and then back into numba. the perf win from numba here is
            # actually fairly small since we're doing a lot of iteration in
            # python
            hits_l:npt.NDArray[np.float64] = np.ndarray((len(hits),2), np.float64)
            hits_v:npt.NDArray[np.float64] = np.ndarray((len(hits),2), np.float64)
            hits_r:npt.NDArray[np.float64] = np.ndarray((len(hits),), np.float64)

            for i, e in enumerate(hits):
                hits_l[i] = e.loc
                hits_v[i] = e.velocity
                hits_r[i] = e.radius

            (
                    idx,
                    approach_time,
                    relative_position,
                    relative_velocity,
                    minimum_separation,
                    threat_count,
                    coalesced_threats,
                    threat_radius,
                    threat_loc,
                    threat_velocity,
                    nearest_neighbor_idx,
                    nearest_neighbor_dist,
                    neighborhood_density,
                    coalesced_idx,
            ) = _analyze_neighbors(
                    hits_l, hits_v, hits_r,
                    pos, v,
                    max_distance, self.ship.radius, margin, neighborhood_dist)
            coalesced_neighbors = [hits[i] for i in coalesced_idx]

            # we want to avoid nearby, dicontinuous changes to threat loc and
            # radius. this can happen when two threats are near each other.
            # if the new and old threat circles overlap "significantly", we are
            # careful about discontinuouse changes
            new_old_dist = util.distance(self.collision_threat_loc, threat_loc)
            if new_old_dist < threat_radius or new_old_dist < self.collision_threat_radius:
                old_loc = self.collision_threat_loc
                old_radius = self.collision_threat_radius
                if new_old_dist + threat_radius > old_radius + VELOCITY_EPS:
                    # the new circle does not completely eclipse the old
                    # find the smallest enclosing circle for both
                    old_loc, old_radius = util.enclosing_circle(threat_loc, threat_radius, self.collision_threat_loc, self.collision_threat_radius)
                    new_old_dist = util.distance(old_loc, threat_loc)

                new_old_vec = threat_loc - old_loc

                # the new threat is smaller than the old one and completely
                # contained in the old one. let's scale and translate the old
                # one toward the new one so that it still contains it, but
                # asymptotically approaches it. this will avoid
                # discontinuities.
                new_radius = util.clip(
                    old_radius * THREAT_RADIUS_SCALE_FACTOR,
                    threat_radius,
                    old_radius
                )
                if util.isclose(new_old_dist, 0.):
                    new_loc = old_loc
                else:
                    new_loc = (
                        new_old_vec/new_old_dist
                        * (old_radius-new_radius)
                        + old_loc
                    )
                # useful assert during testing
                #assert np.linalg.norm(threat_loc - new_loc) + threat_radius <= new_radius + VELOCITY_EPS
                threat_loc = new_loc
                threat_radius = new_radius

            """
            if self.collision_threat_radius - threat_radius > VELOCITY_EPS and loc_dist < self.collision_threat_radius:
                # find the circle that contains both original circles, scale from this circle towards the smaller one
                # one of the two is contained in the other, exponentially scale
                new_radius = THREAT_LOCATION_ALPHA * threat_radius + (1.0 - THREAT_LOCATION_ALPHA) * self.collision_threat_radius
                new_loc = THREAT_LOCATION_ALPHA * threat_loc + (1.0 - THREAT_LOCATION_ALPHA) * self.collision_threat_loc
                threat_loc = new_loc
                threat_radius = new_radius
            """

        else:
            idx = -1
            approach_time = np.inf
            relative_position = ZERO_VECTOR
            relative_velocity  = ZERO_VECTOR
            minimum_separation = np.inf
            collision_loc = ZERO_VECTOR
            threat_count = 0
            coalesced_threats = 0
            nearest_neighbor_idx = -1
            nearest_neighbor_dist = np.inf
            threat_loc = ZERO_VECTOR
            threat_velocity = ZERO_VECTOR
            threat_radius = 0.
            neighborhood_density = 0.
            coalesced_neighbors = []

        if idx < 0:
            neighbor = None
        else:
            neighbor = hits[idx]

        if neighbor is not None and neighbor == last_collision_threat:
            self.collision_rel_pos_hist.append(relative_position)
        else:
            self.collision_rel_pos_hist.clear()
            self.collision_cbdr = False

        # cache the collision threat
        if neighbor != self.collision_threat:
            self.collision_threat = neighbor
            self.collision_approach_time = approach_time
            self.collision_relative_velocity = relative_velocity
            self.collision_relative_position = relative_position
            self.collision_threat_time = self.gamestate.timestamp

        if nearest_neighbor_idx < 0:
            self.nearest_neighbor = None
        else:
            self.nearest_neighbor = hits[nearest_neighbor_idx]
        self.nearest_neighbor_dist = nearest_neighbor_dist

        self.collision_coalesced_threats = coalesced_threats
        self.collision_coalesced_neighbors = coalesced_neighbors
        self.collision_threat_loc = threat_loc
        self.collision_threat_radius = threat_radius

        return neighbor, approach_time, relative_position, relative_velocity, minimum_separation, threat_radius, threat_loc, threat_velocity, neighborhood_density

    def _avoid_collisions_dv(self, sector: core.Sector, neighborhood_loc: np.ndarray, neighborhood_dist: float, margin: float, max_distance: float=np.inf, margin_histeresis:Optional[float]=None, desired_direction:Optional[np.ndarray]=None) -> tuple[np.ndarray, float, float, float]:
        """ Given current velocity, try to avoid collisions with neighbors

        sector
        neighborhood_loc: centerpoint for looking for threats
        neighborhood_dist: how far away to look for threats
        margin: how far apart (between envelopes) to target
        max_distance: max distance to care about collisions
        v: our velocity
        margin_histeresis: additional distance, on top of margin to target when avoiding collisions
        desired_direction: the desired velocity we're shooting for when not avoiding a collision

        returns tuple of desired delta v to avoid collision, approach time, minimum separation and the target distance
        """

        v = self.ship.velocity

        if margin_histeresis is None:
            # add additional margin of size this factor
            # the margin basically scales by 1 + this amount
            margin_histeresis = margin * 1

        if desired_direction is None:
            desired_direction = v

        # if we already have a threat increase margin to get extra far from it
        neighbor_margin = margin
        if self.collision_threat:
            neighbor_margin += margin_histeresis

        # find neighbor with soonest closest appraoch
        neighbor, approach_time, relative_position, relative_velocity, minimum_separation, threat_radius, threat_loc, threat_velocity, neighborhood_density  = self._collision_neighbor(sector, neighborhood_loc, neighborhood_dist, neighbor_margin, max_distance)

        self.neighborhood_density = neighborhood_density

        if neighbor is None:
            self.cannot_avoid_collision = False
            self.ship.collision_threat = None
            self.collision_dv = ZERO_VECTOR
            return ZERO_VECTOR, np.inf, np.inf, 0

        if self.collision_coalesced_threats == 1 and not util.both_almost_zero(threat_velocity) and detect_cbdr(self.collision_rel_pos_hist, self.cbdr_ticks):
            self.collision_cbdr = True
        else:
            self.collision_cbdr = False

        desired_margin = margin + threat_radius + self.ship.radius
        current_threat_loc = threat_loc-threat_velocity*approach_time
        current_threat_vec = current_threat_loc - self.ship.loc
        distance_to_threat = util.magnitude(current_threat_vec[0], current_threat_vec[1])


        if distance_to_threat <= desired_margin + VELOCITY_EPS:
            delta_velocity = (current_threat_loc - self.ship.loc) / distance_to_threat * self.ship.max_speed() * -1
            self.cannot_avoid_collision = True
        else:
            if minimum_separation < desired_margin:
                if approach_time > 0.:
                    # check if we can avoid collision
                    # s = u*t + 1/2 a * t^2
                    # u = rel_speed
                    # t = approach_time
                    # s = minimum_separation - desired_margin
                    # a = (s - u*t) * 2 / t^2
                    # F = m * a
                    rel_speed = util.magnitude(relative_velocity[0], relative_velocity[1])
                    required_acceleration = (minimum_separation - desired_margin - rel_speed * approach_time) * 2 / (approach_time ** 2)
                    required_thrust = self.ship.mass * required_acceleration
                    self.cannot_avoid_collision = required_thrust > self.ship.max_thrust
                else:
                    self.cannot_avoid_collision = True
            else:
                self.cannot_avoid_collision = False

            desired_margin += util.clip((distance_to_threat - desired_margin)/2, 0, margin_histeresis)
            delta_velocity = -1 * _collision_dv(
                    current_threat_loc, threat_velocity,
                    self.ship.loc, self.ship.velocity,
                    desired_margin, -1 * desired_direction,
                    self.collision_cbdr)

        assert not util.either_nan_or_inf(delta_velocity)
        self.collision_dv = delta_velocity

        return delta_velocity, approach_time, minimum_separation, self.ship.radius + neighbor.radius + margin - minimum_separation
