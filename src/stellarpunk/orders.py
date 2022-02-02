""" Orders that can be given to ships. """

from __future__ import annotations

import logging
import collections
from typing import Optional, Deque

import numpy as np
import numpy.typing as npt
import numba as nb # type: ignore
from numba import jit # type: ignore

from stellarpunk import util, core

ANGLE_EPS = 1e-3
PARALLEL_EPS = 1e-3
VELOCITY_EPS = 1e-2

CBDR_HIST_SEC = 0.5
CBDR_DIST_EPS = 5

# think of this as the gimballing angle (?)
# pi/16 is ~11 degrees
COARSE_ANGLE_MATCH = np.pi/16

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
def force_for_delta_velocity(dv:np.ndarray, max_thrust:float, mass:float, dt:float) -> np.ndarray:
    """ What force to apply to get dv change in velocity. Ignores heading. """

    dv_magnitude, dv_angle = util.cartesian_to_polar(dv[0], dv[1])
    if dv_magnitude < VELOCITY_EPS:
        x,y = (0.,0.)
    else:
        thrust = util.clip(mass * dv_magnitude / dt, 0, max_thrust)
        x, y = util.polar_to_cartesian(thrust, dv_angle)
    return np.array((x,y))

@jit(cache=True, nopython=True)
def force_torque_for_delta_velocity(target_velocity:np.ndarray, mass:float, moment:float, angle:float, w:float, v:np.ndarray, max_speed:float, max_torque:float, max_thrust:float, max_fine_thrust:float, dt:float, safety_factor:float) -> tuple[np.ndarray, float, np.ndarray]:
    target_speed = np.linalg.norm(target_velocity)
    if target_speed > max_speed:
        target_velocity = target_velocity / target_speed * max_speed

    dv = target_velocity - v
    difference_mag, difference_angle = util.cartesian_to_polar(dv[0], dv[1])

    if difference_mag < VELOCITY_EPS and abs(w) < ANGLE_EPS:
        return ZERO_VECTOR, 0., target_velocity

    delta_heading = util.normalize_angle(angle-difference_angle, shortest=True)
    rot_time = rotation_time(delta_heading, w, max_torque/moment, safety_factor)

    # while we've got a lot of thrusting to do, we can tolerate only
    # approximately matching our desired angle
    # this should have something to do with how much we expect this angle
    # to change in dt time, but order of magnitude seems reasonable approx

    coarse_angle_match = COARSE_ANGLE_MATCH

    if (difference_mag * mass / max_fine_thrust > rot_time and np.abs(delta_heading) > coarse_angle_match) or difference_mag < VELOCITY_EPS:
        # we need to rotate in direction of thrust
        torque = torque_for_angle(difference_angle, angle, w, max_torque, moment, dt, safety_factor)

        # also apply thrust depending on where we're pointed
        force = force_for_delta_velocity(dv, max_fine_thrust, mass, dt)
    else:
        torque = torque_for_angle(difference_angle, angle, w, max_torque, moment, dt, safety_factor)

        # we should apply thrust, however we can with the current heading
        # max thrust is main engines if we're pointing in the desired
        # direction, otherwise use fine thrusters
        if np.abs(delta_heading) < coarse_angle_match:
            max_thrust = max_thrust
        else:
            max_thrust = max_fine_thrust

        force = force_for_delta_velocity(dv, max_thrust, mass, dt)

    return force, torque, target_velocity

@jit(cache=True, nopython=True)
def _analyze_neighbor(pos:np.ndarray, v:np.ndarray, entity_radius:float, entity_pos:np.ndarray, entity_v:np.ndarray, max_distance:float, max_approach_time:float, margin:float) -> tuple[float, float, np.ndarray, np.ndarray, float, np.ndarray, float]:
    speed = np.linalg.norm(v)
    rel_pos = entity_pos - pos
    rel_vel = entity_v - v

    rel_speed = np.linalg.norm(rel_vel)

    rel_dist = np.linalg.norm(rel_pos)

    # check for parallel paths
    if rel_speed == 0:
        return rel_dist, np.inf, rel_pos, rel_vel, rel_dist, entity_pos, np.inf

    rel_tangent = rel_vel / rel_speed
    approach_t = -1 * rel_tangent.dot(rel_pos) / rel_speed

    if approach_t <= 0 or approach_t >= max_approach_time:
        return rel_dist, np.inf, rel_pos, rel_vel, np.inf, ZERO_VECTOR, np.inf

    # compute the closest approach within max_distance
    collision_distance = speed * approach_t
    if collision_distance > max_distance:
        approach_t = max_distance / speed
        collision_distance = max_distance - VELOCITY_EPS

    min_sep = np.linalg.norm((pos + v*approach_t) - (entity_pos + entity_v*approach_t))
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
            float,
            float
        ]:
    """ Analyzes neighbors and determines collision threat parameters. """

    approach_time = np.inf
    idx = -1
    relative_position = ZERO_VECTOR
    relative_velocity: np.ndarray  = ZERO_VECTOR
    minimum_separation = np.inf
    collision_loc = ZERO_VECTOR
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

        # this neighbor isn't going to collide with us
        if not (min_sep < np.inf):
            continue

        # we need to keep track of all collision threats for coalescing later
        collision_threats.append((eidx, (entity_pos, entity_v, entity_radius), c_loc))
        if min_sep > entity_radius + ship_radius + margin:
            continue

        threat_count += 1

        if rel_dist < nearest_neighbor_dist:
            nearest_neighbor_dist = rel_dist

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
    if idx >= 0:
        threat_radius = hits_r[idx]
        threat_loc = collision_loc
        threat_velocity = hits_v[idx]

        # coalesce nearby threats
        # this avoids flapping in collision targets
        coalesced_threats = 1
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
                elif t_dist < threat_radius + t_radius + 2*margin:
                    # new is within coalesce dist, but not already covered
                    # coalesced threat should just cover both
                    # diameter = 2*threat_radius + 2*t_radius + (t_dist - threat_radius - t_radius)
                    # diameter = t_dist + threat_radius + t_radius
                    coalesced_radius = (t_dist + threat_radius + t_radius)/2

                    c_rel_x, c_rel_y = util.polar_to_cartesian(coalesced_radius - threat_radius, t_angle+np.pi)

                    #assert np.isclose(np.linalg.norm((c_rel_x, c_rel_y)) + threat_radius, coalesced_radius)
                    #assert np.isclose(np.linalg.norm((threat_loc[0]+c_rel_x - t_loc[0], threat_loc[1]+c_rel_y - t_loc[1])) + t_radius, coalesced_radius)

                    threat_loc[0] += c_rel_x
                    threat_loc[1] += c_rel_y
                    threat_radius = coalesced_radius
                    threat_velocity = (threat_velocity * coalesced_threats + t_velocity)/(coalesced_threats + 1)
                    coalesced_threats += 1
    else:
        # no threat found, return some default values
        threat_radius = 0.
        threat_loc = ZERO_VECTOR
        threat_velocity = ZERO_VECTOR

    return idx, approach_time, relative_position, relative_velocity, minimum_separation, threat_count, coalesced_threats, threat_radius, threat_loc, threat_velocity, nearest_neighbor_dist, neighborhood_size / (np.pi * neighborhood_radius ** 2)

@jit(cache=True, nopython=True)
def estimate_cost(
        delta_v:npt.NDArray[np.float64],
        desired_velocity:npt.NDArray[np.float64],
        v:npt.NDArray[np.float64],
        angle:float,
        w:float, max_acceleration:float, max_fine_acceleration:float,
        max_angular_acceleration:float, safety_factor:float
    ) -> float:
    # model: rotate to delta_v, achieve it, rotate to desired_velocity - delta_v, achieve it

    # first we accelerate to delta_v on top of desired_velocity
    combined_v = (delta_v + desired_velocity) - v
    difference_mag, difference_angle = util.cartesian_to_polar(combined_v[0], combined_v[1])
    delta_heading = util.normalize_angle(difference_angle-angle, shortest=True)
    rot_time = rotation_time(delta_heading, w, max_angular_acceleration, safety_factor)
    accelerate_time = difference_mag / max_acceleration * safety_factor

    # then we undo delta_v
    remainder_v = ((desired_velocity - v) - delta_v)
    remainder_mag, remainder_angle = util.cartesian_to_polar(remainder_v[0], remainder_v[1])
    remainder_delta_heading = util.normalize_angle(remainder_angle-difference_angle, shortest=True)
    remainder_rot_time = rotation_time(remainder_delta_heading, 0, max_angular_acceleration, safety_factor)
    undo_time = np.linalg.norm(((desired_velocity - v) - delta_v)) / max_acceleration * safety_factor

    return rot_time + accelerate_time + remainder_rot_time + undo_time

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
def _find_target_v(
        target_location:np.ndarray, arrival_distance:float, min_distance:float,
        current_location:np.ndarray, v:np.ndarray, theta:float, omega:float,
        max_acceleration:float, max_angular_acceleration:float, max_speed:float,
        dt:float, safety_factor:float) -> tuple[np.ndarray, float, float, bool]:

        course = target_location - current_location
        distance, target_angle = util.cartesian_to_polar(course[0], course[1])

        distance_estimate = distance - max_speed*dt

        # if we were to cancel the velocity component in the direction of the
        # target, will we travel enough so that we cross min_distance?
        d = np.dot(v, course) / distance / (2* max_acceleration)
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

def detect_cbdr(rel_pos_hist:Deque[np.ndarray], min_hist:int):
    if len(rel_pos_hist) < min_hist:
        return False

    oldest_rel_pos = rel_pos_hist[0]
    oldest_distance, oldest_bearing = util.cartesian_to_polar(oldest_rel_pos[0], oldest_rel_pos[1])

    latest_rel_pos = rel_pos_hist[-1]
    latest_distance, latest_bearing = util.cartesian_to_polar(latest_rel_pos[0], latest_rel_pos[1])

    return abs(oldest_bearing - latest_bearing) < ANGLE_EPS and oldest_distance - latest_distance > CBDR_DIST_EPS

class AbstractSteeringOrder(core.Order):
    def __init__(self, *args, safety_factor:float=2., **kwargs):
        super().__init__(*args, **kwargs)
        self.safety_factor = safety_factor

        self.collision_margin = 5e2
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
        self.collision_threat_loc = ZERO_VECTOR
        self.collision_threat_radius = 0.

        self.cannot_avoid_collision = False

        self.collision_threat_max_age = 0.
        self.high_awareness_dist = self.ship.max_speed() * self.collision_threat_max_age * 2 * self.safety_factor

        # how many neighbors per m^2
        self.neighborhood_density = 0.

    def to_history(self) -> dict:
        history = super().to_history()
        if self.collision_threat:
            history["ct"] = str(self.collision_threat.entity_id)
            history["ct_loc"] = self.collision_threat.loc.tolist()
            history["ct_v"] = self.collision_threat.velocity.tolist()
            history["ct_ts"] = self.collision_threat_time
            history["ct_dv"] = self.collision_dv.tolist()
            history["ct_tc"] = self.collision_threat_count
            history["ct_ct"] = self.collision_coalesced_threats
            history["ct_cloc"] = self.collision_threat_loc.tolist()
            history["ct_cradius"] = self.collision_threat_radius
            history["cac"] = self.cannot_avoid_collision
            history["cbdr"] = self.collision_cbdr

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
            self.ship.phys.angle = target_angle
            self.ship.phys.angular_velocity = 0
        else:
            #self.logger.debug(f'apply torque {t:.0f} for desired target_angle {target_angle:.3f} from {target_angle:.3f} at {w:.2f}rad/sec')
            self.ship.phys.torque = t

    def _accelerate_to(self, target_velocity: np.ndarray, dt: float) -> None:
        mass = self.ship.mass
        moment = self.ship.moment
        angle = self.ship.angle
        w = self.ship.angular_velocity
        v = self.ship.velocity
        max_speed = self.ship.max_speed()
        max_torque = self.ship.max_torque
        max_thrust = self.ship.max_thrust
        max_fine_thrust = self.ship.max_fine_thrust

        force, torque, target_velocity = force_torque_for_delta_velocity(
                target_velocity,
                mass, moment, angle, w, v,
                max_speed, max_torque, max_thrust, max_fine_thrust,
                dt, self.safety_factor
        )

        if force[0] == 0. and force[1] == 0.:
            self.ship.phys.velocity = (target_velocity[0], target_velocity[1])
        else:
            self.ship.phys.apply_force_at_world_point(
                    (force[0], force[1]),
                    (self.ship.loc[0], self.ship.loc[1])
            )

        self.ship.phys.torque = torque

        #t = difference_mag / (np.linalg.norm(np.array((x,y))) / mass) if (x,y) != (0,0) else 0
        #self.logger.debug(f'force: {(x, y)} {np.linalg.norm(np.array((x,y)))} in {t:.2f}s')

    def _clear_collision_info(self):
        self.collision_threat:Optional[core.SectorEntity] = None
        self.collision_dv = ZERO_VECTOR
        self.collision_threat_time = 0
        self.collision_relative_velocity = ZERO_VECTOR
        self.collision_relative_position = ZERO_VECTOR
        self.collision_approach_time = np.inf
        self.collision_rel_pos_hist.clear()
        self.collision_cbdr = False
        self.cannot_avoid_collision = False

    def _collision_neighbor(
            self,
            sector: core.Sector,
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
        #TODO: this seems unsatable, so I set max age to zero to avoid it
        # we cache the collision threat to avoid searching space
        if self.nearest_neighbor_dist > self.high_awareness_dist and self.gamestate.timestamp - self.collision_threat_time < self.collision_threat_max_age:
            hits = [self.collision_threat,] if self.collision_threat else []
        else:
            self.collision_threat = None
            self.cannot_avoid_collision = False
            self.collision_threat_time = self.gamestate.timestamp
            self.nearest_neighbor_dist = np.inf
            hits = list(e for e in  sector.spatial_point(self.ship.loc, neighborhood_dist) if e != self.ship)

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
            idx, approach_time, relative_position, relative_velocity, minimum_separation, threat_count, coalesced_threats, threat_radius, threat_loc, threat_velocity, nearest_neighbor_dist, neighborhood_density = _analyze_neighbors(hits_l, hits_v, hits_r, pos, v, max_distance, self.ship.radius, margin, neighborhood_dist)
        else:
            idx = -1
            approach_time = np.inf
            relative_position = ZERO_VECTOR
            relative_velocity  = ZERO_VECTOR
            minimum_separation = np.inf
            collision_loc = ZERO_VECTOR
            threat_count = 0
            coalesced_threats = 0
            nearest_neighbor_dist = np.inf
            threat_loc = ZERO_VECTOR
            threat_velocity = ZERO_VECTOR
            threat_radius = 0.
            neighborhood_density = 0.

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

        self.nearest_neighbor_dist = nearest_neighbor_dist

        self.collision_coalesced_threats = coalesced_threats
        self.collision_threat_loc = threat_loc
        self.collision_threat_radius = threat_radius

        return neighbor, approach_time, relative_position, relative_velocity, minimum_separation, threat_radius, threat_loc, threat_velocity, neighborhood_density

    def _avoid_collisions_dv(self, sector: core.Sector, neighborhood_dist: float, margin: float, max_distance: float=np.inf, margin_histeresis:Optional[float]=None, desired_direction:Optional[np.ndarray]=None) -> tuple[np.ndarray, float, float, float]:
        """ Given current velocity, try to avoid collisions with neighbors

        sector
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
        neighbor, approach_time, relative_position, relative_velocity, minimum_separation, threat_radius, threat_loc, threat_velocity, neighborhood_density  = self._collision_neighbor(sector, neighborhood_dist, neighbor_margin, max_distance)

        self.neighborhood_density = neighborhood_density

        if neighbor is None:
            self.cannot_avoid_collision = False
            self.ship.collision_threat = None
            self.collision_dv = ZERO_VECTOR
            return ZERO_VECTOR, np.inf, np.inf, 0

        distance_to_threat = np.linalg.norm(threat_loc - self.ship.loc)

        relative_position = threat_loc - (self.ship.loc + self.ship.velocity * approach_time)
        distance = np.linalg.norm(relative_position)

        if np.any(threat_velocity != ZERO_VECTOR) and detect_cbdr(self.collision_rel_pos_hist, self.cbdr_ticks):
            if not self.collision_cbdr:
                self.collision_cbdr = True
        else:
            self.collision_cbdr = False

        speed = np.linalg.norm(desired_direction)

        dv:npt.NDArray[np.float64] = relative_position
        if distance_to_threat <= self.ship.radius + threat_radius + margin:
            # if we're already inside the margin, just get away
            self.logger.debug(f'already inside margin: {distance}')
            dv = relative_position
        elif speed < VELOCITY_EPS:
            # if desired speed is (effectively) zero, then just avoid in
            # direction of min sep.
            dv = relative_position
        else:
            # if the angle between relative pos and velocity is very small or
            # close to pi we could have instability
            parallelness = np.dot(relative_position/distance,desired_direction/speed)

            if parallelness < -1+PARALLEL_EPS or parallelness > 1-PARALLEL_EPS:
                #TODO: is this weird if we're already offset by a bunch?

                # prefer diverting clockwise
                dv = np.array((desired_direction[1], -desired_direction[0]))
                dv = dv / np.linalg.norm(dv)
            else:
                # get the component of relative position perpendicular to our
                # desired direction. This will be the direction of our divert
                # which will limit the impact on our desired path.
                dv = relative_position - desired_direction * np.dot(relative_position, desired_direction) / np.dot(desired_direction, desired_direction)

        dv_r, dv_theta = util.cartesian_to_polar(dv[0], dv[1])
        if self.collision_cbdr:
            # when we're in a CBDR situation, we divert from our desired
            # vector by a fixed amount in a fixed direction, this should
            # help avoid two agents following the same algorithm steering
            # into each other.

            dv_theta += self.collision_cbdr_divert_angle
            dv = np.array(util.polar_to_cartesian(dv_r, dv_theta))

        # model is that we will accelerate to avoid the threat by desired
        # margin by the time we reach our closest approach. That is, we will
        # accelerate in direction dv to achieve distance d = radii + margin in
        # time=approach_time
        # in addition, we want to accelerate along  dv, chosen to be for
        # minimal impact to desired course.
        # dv points in the direction of our threat (so we want negative
        # acceleration w.r.t. dv)
        # so by approach_time from now we want to have traveled distance d
        # in direction dv, under constant acceleration, given the current
        # relative velocity

        #TODO: delta_dist assumes we continue on velocity v, but if we start
        # accelerating to desired_direction for approach_time seconds we'll
        delta_dist = dv_r

        d1 = self.ship.radius + threat_radius + margin + margin_histeresis - delta_dist
        d2 = self.ship.radius + threat_radius + margin + margin_histeresis + delta_dist

        # assumes we have constant acceleration between now and approach_time
        # in reality we'll have low acceleration while turning toward the
        # target and then fast acceleration
        v_f1 = (2 * -d1 / approach_time) * self.safety_factor
        v_f2 = (2 * d2 / approach_time) * self.safety_factor

        # these are the candidate final velocities we want to hit by the time
        # we got to the point of collision, under constant acceleration
        # if we do that, we'll avoid the collision
        delta_velocity1 = dv / delta_dist * v_f1
        delta_velocity2 = dv / delta_dist * v_f2

        desired_divert = desired_direction - v
        max_delta_v = self.ship.max_acceleration() * approach_time
        self.cannot_avoid_collision = False

        # we have two choices. we want to choose the least costly one
        # cost here is in terms of how long it'll take to accelerate to
        # delta_velocity plus desired velocity plus the time to make up that
        # difference. A big portion of this will be the time it takes to
        # rotate, this will favor diverting in the direction closest to our
        # current heading, which has the effect of avoiding "flapping" in
        # collision diversions

        cost_1 = estimate_cost(delta_velocity1, desired_divert, v, self.ship.angle, self.ship.angular_velocity, self.ship.max_acceleration(), self.ship.max_fine_acceleration(), self.ship.max_angular_acceleration(), self.safety_factor)
        cost_2 = estimate_cost(delta_velocity2, desired_divert, v, self.ship.angle, self.ship.angular_velocity, self.ship.max_acceleration(), self.ship.max_fine_acceleration(), self.ship.max_angular_acceleration(), self.safety_factor)

        if cost_2 < cost_1:
            delta_velocity = delta_velocity2
            dist_to_avoid = d2
            delta_speed = v_f2
            delta_theta = dv_theta + np.pi
        else:
            delta_velocity = delta_velocity1
            dist_to_avoid = d1
            delta_speed = v_f1
            delta_theta = dv_theta

        self.collision_dv = delta_velocity

        # check if we cannot avoid the collision
        if delta_speed > max_delta_v:
            # estimate if we're going to collide with the neighbor
            # note, we use the neighbor in particular and not the coalesced
            # threat because that might not actually be a collision and running
            # into the neighbor would be. the neighbor is also the target we're
            # threatened by soonest, so if there's a collision, it should be
            # that one
            collision_radius = self.ship.radius + neighbor.radius
            exp_pos_them = neighbor.loc + neighbor.velocity * approach_time

            rot_time = rotation_time(
                    util.normalize_angle(
                        delta_theta - self.ship.angle, shortest=True),
                    self.ship.angular_velocity,
                    self.ship.max_angular_acceleration(),
                    self.safety_factor
            )
            # compute our expected position after approach time, assuming we
            # just try to avoid the collision by accelerating to delta_velocity
            if rot_time > approach_time:
                exp_avg_v = v
            else:
                exp_avg_v = (v * rot_time + delta_velocity/abs(delta_speed) * self.ship.max_acceleration() * (approach_time - rot_time)) / approach_time
            exp_pos_us = (self.ship.loc + approach_time * exp_avg_v)
            expected_rel_pos:npt.NDArray[np.float64] = exp_pos_them - exp_pos_us

            # see if that avoids the collision
            if np.linalg.norm(exp_pos_them - exp_pos_us) < collision_radius:
                self.cannot_avoid_collision = True
                self.logger.debug(f'cannot avoid collision: {abs(v_f1)} and {abs(v_f2)} > {self.ship.max_thrust} / {self.ship.mass} * {approach_time}')

        return delta_velocity, approach_time, minimum_separation, dist_to_avoid

class KillRotationOrder(core.Order):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def is_complete(self) -> bool:
        return self.ship.angular_velocity == 0

    def act(self, dt: float) -> None:
        # apply torque up to max torque to kill angular velocity
        # torque = moment * angular_acceleration
        # the perfect acceleration would be -1 * angular_velocity / timestep
        # implies torque = moment * -1 * angular_velocity / timestep
        t = self.ship.moment * -1 * self.ship.angular_velocity / dt
        if t == 0:
            self.ship.phys.angular_velocity = 0
        else:
            self.ship.phys.torque = np.clip(t, -9000, 9000)

class RotateOrder(AbstractSteeringOrder):
    def __init__(self, target_angle: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.target_angle = util.normalize_angle(target_angle)

    def is_complete(self) -> bool:
        return self.ship.angular_velocity == 0 and util.normalize_angle(self.ship.angle) == self.target_angle

    def act(self, dt: float) -> None:
        self._rotate_to(self.target_angle, dt)

class KillVelocityOrder(AbstractSteeringOrder):
    """ Applies thrust and torque to zero out velocity and angular velocity.

    Rotates to opposite direction of current velocity and applies thrust to
    zero out velocity. """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def is_complete(self) -> bool:
        return self.ship.angular_velocity == 0 and np.allclose(self.ship.velocity, ZERO_VECTOR)

    def act(self, dt: float) -> None:
        self._accelerate_to(ZERO_VECTOR, dt)

class GoToLocation(AbstractSteeringOrder):

    @staticmethod
    def goto_entity(
            entity:core.SectorEntity,
            ship:core.Ship,
            gamestate:core.Gamestate,
            arrival_distance:float=1.5e3,
            collision_margin:float=1e3) -> GoToLocation:

        _, angle = util.cartesian_to_polar(*(ship.loc - entity.loc))
        target_angle = angle + gamestate.random.uniform(-np.pi/2, np.pi/2)
        target_arrival_distance = (arrival_distance - entity.radius - collision_margin)/2
        target_loc = entity.loc + util.polar_to_cartesian(entity.radius + collision_margin + target_arrival_distance, target_angle)

        return GoToLocation(
                target_loc, ship, gamestate,
                arrival_distance=target_arrival_distance,
                min_distance=0.)

    def __init__(self, target_location: npt.NDArray[np.float64], *args, arrival_distance: float=1.5e3, min_distance:Optional[float]=None, **kwargs) -> None:
        """ Creates an order to go to a specific location.

        The order will arrivate at the location approximately and with zero
        velocity.

        target_location the location
        arrival_distance how close to the location to arrive
        """

        super().__init__(*args, **kwargs)
        self.target_location = target_location
        self.target_v = ZERO_VECTOR
        self.arrival_distance = arrival_distance
        if min_distance is None:
            min_distance = self.arrival_distance * 0.9
        self.min_distance = min_distance

        self.cannot_stop = False

        self.distance_estimate = 0.

    def to_history(self) -> dict:
        data = super().to_history()
        data["t_loc"] = self.target_location.tolist()
        data["t_v"] = self.target_v.tolist()
        data["cs"] = self.cannot_stop

        return data

    def __str__(self) -> str:
        return f'GoToLocation: {self.target_location} ad:{self.arrival_distance} sf:{self.safety_factor}'

    def eta(self) -> float:
        course = self.target_location - (self.ship.loc)
        distance, target_angle = util.cartesian_to_polar(course[0], course[1])
        rotate_towards = rotation_time(util.normalize_angle(target_angle-self.ship.angle, shortest=True), self.ship.angular_velocity, self.ship.max_angular_acceleration(), self.safety_factor)

        # we cap at max_speed, so need to account for that by considering a
        # "cruise" period where we travel at max_speed, but only if we have
        # enough distance to make it to cruise speed
        if np.sqrt(2. * self.ship.max_acceleration() * distance/2.) < self.ship.max_speed():
            accelerate_up = np.sqrt( 2. * (distance/2.) / self.ship.max_acceleration()) * self.safety_factor
            cruise = 0.
        else:
            # v_f**2 = 2 * a * d
            # d = v_f**2 / (2*a)
            d_accelerate = self.ship.max_speed()**2 / (2*self.ship.max_acceleration())
            accelerate_up = self.ship.max_speed() / self.ship.max_acceleration() * self.safety_factor

            d_cruise = distance - 2*d_accelerate
            cruise = d_cruise / self.ship.max_speed() * self.safety_factor

        rotate_away = rotation_time(np.pi, 0, self.ship.max_angular_acceleration(), self.safety_factor)
        accelerate_down = accelerate_up
        return rotate_towards + accelerate_up + rotate_away + cruise + accelerate_down
    def is_complete(self) -> bool:
        # computing this is expensive, so don't if we can avoid it
        if self.distance_estimate > self.arrival_distance*5:
            return False
        else:
            return bool(np.linalg.norm(self.target_location - self.ship.loc) < self.arrival_distance + VELOCITY_EPS and np.allclose(self.ship.velocity, ZERO_VECTOR))

    def act(self, dt: float) -> None:
        if self.ship.sector is None:
            raise Exception(f'{self.ship} not in any sector')
        # essentially the arrival steering behavior but with some added
        # obstacle avoidance

        v = self.ship.velocity
        theta = self.ship.angle
        omega = self.ship.angular_velocity

        max_acceleration = self.ship.max_acceleration()
        max_angular_acceleration = self.ship.max_angular_acceleration()
        max_speed = self.ship.max_speed()

        # ramp down speed as nearby density increases
        # ramp down with inverse of the density
        # d_low, s_high is one point we want to hit
        # d_high, s_low is another
        d_low = 1/(np.pi*1e4**2)
        s_high = max_speed
        d_high = 30/(np.pi*1e4**2)
        s_low = 100
        b = (s_low * d_high - s_high * d_low)/(s_high - s_low)
        m = s_high*(d_low + b)
        max_speed = min(max_speed, m / (self.neighborhood_density + b))

        self.target_v, distance, self.distance_estimate, self.cannot_stop = _find_target_v(
                self.target_location, self.arrival_distance, self.min_distance,
                self.ship.loc, v, theta, omega,
                max_acceleration, max_angular_acceleration, max_speed,
                dt, self.safety_factor)

        if self.cannot_stop:
            max_distance = np.inf
            self.logger.debug(f'cannot stop in time distance: {distance} v: {v}')
        else:
            max_distance = distance-self.arrival_distance+VELOCITY_EPS#/self.safety_factor

        #collision avoidance for nearby objects
        #   this includes fixed bodies as well as dynamic ones
        collision_dv, approach_time, minimum_separation, distance_to_avoid_collision = self._avoid_collisions_dv(
                self.ship.sector, 2e4, self.collision_margin,
                max_distance=max_distance,
                desired_direction=self.target_v)

        # combine target_v with collision_dv
        # this is what we hope to accelerate to, but if we can't accomplish
        # both, prioritize collision_db
        #TODO: the theta we need to get to is the difference between the
        # desired velocity and v right?
        #combined_v = (self.target_v + collision_dv) - v
        combined_v = self.target_v + collision_dv
        combined_r, combined_theta = util.cartesian_to_polar(combined_v[0], combined_v[1])
        rot_time = rotation_time(util.normalize_angle(combined_theta - self.ship.angle, shortest=True), self.ship.angular_velocity, self.ship.max_angular_acceleration(), self.safety_factor)
        if max_acceleration * (approach_time-rot_time) / self.safety_factor < distance_to_avoid_collision:
            self._accelerate_to(v + collision_dv, dt)
        else:
            self._accelerate_to(self.target_v + collision_dv, dt)

class WaitOrder(AbstractSteeringOrder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def is_complete(self) -> bool:
        # wait forever
        return False

    def act(self, dt:float) -> None:
        if self.ship.sector is None:
            raise Exception(f'{self.ship} not in any sector')

        # avoid collisions while we're waiting
        # but only if those collisions are really imminent
        # we want to have enough time to get away
        collision_dv, approach_time, min_separation, distance=self._avoid_collisions_dv(self.ship.sector, 5e3, 3e2)
        if distance > 0:
            t = approach_time - rotation_time(2*np.pi, self.ship.angular_velocity, self.ship.max_angular_acceleration())
            if t < 0 or distance > 1/2 * self.ship.max_acceleration()*t**2 / self.safety_factor:
                self._accelerate_to(collision_dv, dt)
                return
        self._accelerate_to(ZERO_VECTOR, dt)

class MineOrder(AbstractSteeringOrder):
    def __init__(self, target: core.Asteroid, amount: float, *args, max_dist=2e3, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.target = target
        self.max_dist = max_dist
        self.amount = 0
        self.harvested = 0

    def is_complete(self) -> bool:
        # we're full or asteroid is empty or we're too far away
        #TODO: actually check that we've harvested enough
        return self.target.amount <= 0 or self.harvested < self.amount

    def act(self, dt: float) -> None:
        # grab resources from the asteroid and add to our cargo
        distance = np.linalg.norm(self.ship.loc - self.target.loc)
        if distance > self.max_dist:
            self.ship.orders.appendleft(GoToLocation.goto_entity(self.target, self.ship, self.gamestate))
            return

        #TODO: actually implement harvesting, taking time, maybe visual fx
        amount = np.clip(self.amount, 0, self.target.amount)
        self.target.amount -= amount
        self.harvested += amount

class TransferCargo(core.Order):
    def __init__(self, target: core.SectorEntity, resource: int, amount: float, *args, max_dist=2e3, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.target = target
        self.resource = resource
        self.amount = amount
        self.transferred = 0.
        self.max_dist = max_dist

    def is_complete(self) -> bool:
        return self.transferred == self.amount

    def act(self, dt:float) -> None:
        # if we're too far away, go to the target
        distance = np.linalg.norm(self.ship.loc - self.target.loc)
        if distance > self.max_dist:
            self.ship.orders.appendleft(GoToLocation.goto_entity(self.target, self.ship, self.gamestate))
            return

        # otherwise, transfer the goods
        #TODO: transfer?
        self.transferred = self.amount

class HarvestOrder(core.Order):
    def __init__(self, base: core.SectorEntity, resource: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.base = base
        self.resource = resource
        self.keep_harvesting = True

    def is_complete(self):
        #TODO: harvest forever?
        return not self.keep_harvesting

    def act(self, dt):
        # if our cargo is full, send it back to home base
        cargo_full = False
        if cargo_full:
            self.logger.debug("cargo full, heading to {self.base} to dump cargo")
            self.ship.orders.appendleft(TransferCargo(self.base, self.resource, 1, self.ship, self.gamestate))
            return

        # choose an asteroid to harvest
        self.logger.debug("searching for next asteroid")
        #TODO: how should we find the nearest asteroid? point_query_nearest with ShipFilter?
        nearest = None
        nearest_dist = np.inf
        for hit in self.ship.sector.spatial_point(self.ship.loc):
            if not isinstance(hit, core.Asteroid):
                continue
            if hit.resource != self.resource:
                continue

            dist = np.linalg.norm(self.ship.loc - hit.loc)
            if dist < nearest_dist:
                nearest = hit
                nearest_dist = dist

        if nearest is None:
            self.logger.info(f'could not find asteroid of type {self.resource} in {self.sector}, stopping harvest')
            self.keep_harvesting = False
            return

        #TODO: worry about other people harvesting asteroids
        #TODO: choose amount to harvest
        # push mining order
        self.ship.orders.appendleft(MineOrder(nearest, 1e3, self.ship, self.gamestate))
