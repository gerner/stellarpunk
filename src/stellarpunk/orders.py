""" Orders that can be given to ships. """

import logging
from typing import Optional

import numpy as np
from numba import jit # type: ignore

from stellarpunk import util, core

ANGLE_EPS = 1e-2
PARALLEL_EPS = 1e-3
VELOCITY_EPS = 1e-2

ZERO_VECTOR = np.array((0.,0.))
ZERO_VECTOR.flags.writeable = False

@jit(cache=True, nopython=True)
def rotation_time(delta_angle: float, max_angular_acceleration: float) -> float:
    # theta_f = theta_0 + omega_0*t + 1/2 * alpha * t^2
    # assume omega_0 = 0 <--- assumes we're not currently rotating!
    # assume we constantly accelerate half way, constantly accelerate the
    # other half
    return 2*np.sqrt(abs(delta_angle)/max_angular_acceleration)


@jit(cache=True, nopython=True)
def torque_for_angle(target_angle: float, angle:float, w:float, max_torque:float, moment:float, dt:float, eps:float=ANGLE_EPS) -> float:
    """ What torque to apply to achieve target angle """

    difference_angle = util.normalize_angle(target_angle - angle, shortest=True)
    braking_angle =  -1 * np.sign(w) * -0.5 * w*w * moment / max_torque

    if abs(w) < eps and abs(difference_angle) < eps:
        # bail if we're basically already there
        # caller can handle this, e.g. set rotation to target and w to 0
        t = 0.0
    elif abs(braking_angle) > abs(difference_angle) or abs(difference_angle) < eps:
        # we can't break in time, so just start breaking and we'll fix it later
        t = moment * -1.0 * w / dt
    else:
        # add torque in the desired direction to get
        # accel = tau / moment
        # dw = accel * dt
        # desired w is w such that braking_angle = difference_angle
        # braking_angle =  -1 * np.sign(w) * -0.5 * w*w * moment / max_torque
        # sqrt(difference_angle * max_torque / (0.5 * moment)) = w
        desired_w = np.sign(difference_angle) * np.sqrt(abs(difference_angle) * max_torque / (0.5 * moment))

        t = (desired_w - w)*moment/dt

        # crude method:
        #t = max_torque * np.sign(difference_angle)

    return util.clip(t, -1.0*max_torque, max_torque)
    #return np.clip(t, -1*max_torque, max_torque)

@jit(cache=True, nopython=True)
def force_for_delta_velocity(dv:np.ndarray, max_thrust:float, mass:float, dt:float, eps:float=VELOCITY_EPS) -> np.ndarray:
    """ What force to apply to get dv change in velocity. Ignores heading. """

    dv_magnitude, dv_angle = util.cartesian_to_polar(dv[0], dv[1])
    if dv_magnitude < eps:
        x,y = (0.,0.)
    else:
        #thrust = np.clip(mass * dv_magnitude / dt, 0, max_thrust)
        thrust = util.clip(mass * dv_magnitude / dt, 0, max_thrust)
        x, y = util.polar_to_cartesian(thrust, dv_angle)
    return np.array((x,y))

@jit(cache=True, nopython=True)
def _analyze_neighbor(pos:np.ndarray, v:np.ndarray, entity_radius:float, entity_pos:np.ndarray, entity_v:np.ndarray, approach_time: float, margin: float) -> tuple[float, float, np.ndarray, np.ndarray, float]:
    rel_pos = entity_pos - pos
    rel_vel = entity_v - v

    rel_speed = np.linalg.norm(rel_vel)

    rel_dist = np.linalg.norm(rel_pos)

    # check for parallel paths
    if rel_speed == 0:
        return rel_dist, np.inf, ZERO_VECTOR, ZERO_VECTOR, np.inf

    rel_tangent = rel_vel / rel_speed
    approach_t = -1 * rel_tangent.dot(rel_pos) / rel_speed

    if approach_t <= 0 or approach_t >= approach_time:
        return rel_dist, np.inf, ZERO_VECTOR, ZERO_VECTOR, np.inf

    min_sep = np.linalg.norm((pos + v*approach_t) - (entity_pos + entity_v*approach_t))

    if min_sep > entity_radius + margin:
        return rel_dist, np.inf, ZERO_VECTOR, ZERO_VECTOR, np.inf

    return rel_dist, approach_t, rel_pos, rel_vel, float(min_sep)

class AbstractSteeringOrder(core.Order):
    def __init__(self, *args, safety_factor:float=1.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.safety_factor = safety_factor

        self.collision_margin = 3e2
        self.collision_threat:Optional[core.SectorEntity] = None
        self.collision_dv = ZERO_VECTOR
        self.nearest_neighbor_dist = np.inf
        self.collision_threat_time = 0
        self.collision_relative_velocity = ZERO_VECTOR
        self.collision_relative_position = ZERO_VECTOR
        self.collision_approach_time = np.inf
        self.cannot_avoid_collision = False

        self.collision_threat_max_age = 0
        self.high_awareness_dist = self.ship.max_speed() * self.collision_threat_max_age * 2 * self.safety_factor

    def to_history(self) -> dict:
        history = super().to_history()
        if self.collision_threat:
            history["ct"] = str(self.collision_threat.entity_id)
            history["ct_loc"] = self.collision_threat.loc.tolist()
            history["ct_ts"] = self.collision_threat_time
            history["cac"] = self.cannot_avoid_collision
        else:
            assert not self.cannot_avoid_collision
        history["nnd"] = self.nearest_neighbor_dist
        return history

    def _rotate_to(self, target_angle: float, dt: float) -> None:
        # given current angle and angular_velocity and max torque, choose
        # torque to apply for dt now to hit target angle

        angle = util.normalize_angle(self.ship.angle)
        w = self.ship.angular_velocity
        moment = self.ship.moment

        t = torque_for_angle(
                target_angle, angle, w,
                self.ship.max_torque, moment, dt)

        if t == 0:
            self.ship.phys.angle = target_angle
            self.ship.phys.angular_velocity = 0
        else:
            #self.logger.debug(f'apply torque {t:.0f} for desired angle {target_angle:.3f} from {angle:.3f} at {w:.2f}rad/sec')
            self.ship.phys.torque = t

    def _accelerate_to(self, target_velocity: np.ndarray, dt: float) -> None:

        mass = self.ship.mass
        moment = self.ship.moment
        angle = self.ship.angle
        w = self.ship.angular_velocity
        v = self.ship.velocity

        #self.logger.debug("accelerate to %s from %s", target_velocity, v)

        max_speed = self.ship.max_speed()
        target_speed = np.linalg.norm(target_velocity)
        if target_speed > max_speed:
            target_velocity = target_velocity / target_speed * max_speed
            #self.logger.debug("limiting target velocity to %s", target_velocity)

        # orient toward the opposite of the direction of travel
        # thrust until zero velocity
        velocity_magnitude, velocity_angle = util.cartesian_to_polar(*v)

        dv = target_velocity - v
        difference_mag, difference_angle = util.cartesian_to_polar(*dv)
        #self.logger.debug("accelerate to dv %s", dv)

        if difference_mag < VELOCITY_EPS and abs(w) < ANGLE_EPS:
            self.ship.phys.angular_velocity = 0
            self.ship.phys.velocity = tuple(target_velocity)
            return

        delta_heading = util.normalize_angle(angle-difference_angle, shortest=True)
        rot_time = rotation_time(delta_heading, self.ship.max_angular_acceleration())

        # while we've got a lot of thrusting to do, we can tolerate only
        # approximately matching our desired angle
        # this should have something to do with how much we expect this angle
        # to change in dt time, but order of magnitude seems reasonable approx
        coarse_angle_match = np.pi/32

        if (difference_mag * mass / self.ship.max_fine_thrust > rot_time and abs(delta_heading) > coarse_angle_match) or difference_mag < VELOCITY_EPS:
            #self.logger.debug(f'rotate {abs(delta_heading)} {abs(w)}')
            #self.logger.debug(f'heading adjust: {delta_heading} at {w}rad/sec')
            # we need to rotate in direction of thrust
            self._rotate_to(difference_angle, dt)

            # also apply thrust depending on where we're pointed
            max_thrust = self.ship.max_fine_thrust
            x,y = force_for_delta_velocity(dv, max_thrust, mass, dt)
            if (x,y) == (0.,0.):
                self.ship.phys.velocity = tuple(target_velocity)
            else:
                #TODO: worry about center of gravity?
                #self.logger.debug("apply force (while rotating) (%f, %f)", x, y)
                self.ship.phys.apply_force_at_world_point(
                        (x,y),
                        (self.ship.x, self.ship.y)
                )
        else:
            #self.logger.debug(f'main engines {abs(delta_heading)} {abs(w)}')
            self._rotate_to(difference_angle, dt)

            # we should apply thrust, however we can with the current heading
            # max thrust is main engines if we're pointing in the desired
            # direction, otherwise use fine thrusters
            if abs(delta_heading) < coarse_angle_match:
                max_thrust = self.ship.max_thrust
            else:
                max_thrust = self.ship.max_fine_thrust

            x,y = force_for_delta_velocity(
                    dv,
                    max_thrust, mass, dt)
            if (x,y) == (0.,0.):
                self.ship.phys.velocity = tuple(target_velocity)
            else:
                #TODO: worry about center of gravity?
                #self.logger.debug("apply force (%f, %f)", x, y)
                self.ship.phys.apply_force_at_world_point(
                        (x,y),
                        (self.ship.x, self.ship.y)
                )

        #t = difference_mag / (np.linalg.norm(np.array((x,y))) / mass) if (x,y) != (0,0) else 0
        #self.logger.debug(f'force: {(x, y)} {np.linalg.norm(np.array((x,y)))} in {t:.2f}s')


    def _collision_neighbor(self, sector: core.Sector, neighborhood_dist: float, margin: float) -> tuple[Optional[core.SectorEntity], float, np.ndarray, np.ndarray, float]:
        approach_time = np.inf
        neighbor: Optional[core.SectorEntity] = None
        relative_position: np.ndarray = ZERO_VECTOR
        relative_velocity: np.ndarray  = ZERO_VECTOR
        minimum_separation = np.inf


        pos = np.array((self.ship.x, self.ship.y))
        v = np.array(self.ship.velocity)

        # we cache the collision threat to avoid searching space
        if self.nearest_neighbor_dist > self.high_awareness_dist and self.gamestate.timestamp - self.collision_threat_time < self.collision_threat_max_age:
            hits = (self.collision_threat,) if self.collision_threat else ()
        else:
            bounds = (
                    self.ship.x-neighborhood_dist, self.ship.y-neighborhood_dist,
                    self.ship.x+neighborhood_dist, self.ship.y+neighborhood_dist
            )

            self.collision_threat = None
            self.collision_threat_time = self.gamestate.timestamp
            self.nearest_neighbor_dist = np.inf
            hits = sector.spatial_query(bounds)

        for entity in hits:

            # the query will include ourselves, so let's skip that
            if entity == self.ship:
                continue

            entity_pos = np.array((entity.x, entity.y))
            entity_v = np.array(entity.velocity)

            #self.logger.debug(f'analyze with {pos, v, entity.radius, entity_pos, entity_v, approach_time, self.ship.radius + margin}')
            rel_dist, approach_t, rel_pos, rel_vel, min_sep = _analyze_neighbor(
                    pos, v, entity.radius, entity_pos, entity_v, approach_time, self.ship.radius + margin)

            if rel_dist < self.nearest_neighbor_dist:
                self.nearest_neighbor_dist = rel_dist

            if approach_t < approach_time:
                approach_time = approach_t
                neighbor = entity
                relative_position = rel_pos
                relative_velocity = rel_vel
                minimum_separation = float(min_sep)

        # cache the collision threat
        if neighbor != self.collision_threat:
            self.collision_threat = neighbor
            self.collision_approach_time = approach_time
            self.collision_relative_velocity = relative_velocity
            self.collision_relative_position = relative_position
            self.collision_threat_time = self.gamestate.timestamp

        return neighbor, approach_time, relative_position, relative_velocity, minimum_separation

    def _avoid_collisions_dv(self, sector: core.Sector, neighborhood_dist: float, margin: float, max_distance: float=np.inf, margin_histeresis:float=30, desired_direction:Optional[np.ndarray]=None) -> tuple[np.ndarray, float, float, float]:
        """ Given current velocity, try to avoid collisions with neighbors

        sector
        neighborhood_dist: how far away to look for threats
        margin: how far apart (between envelopes) to target
        max_distance: max distance to care about collisions
        v: our velocity
        margin_histeresis: margin to avoid rapid switching at the boundary

        returns tuple of desired delta v to avoid collision, approach time, minimum separation and the target distance
        """

        v = np.array(self.ship.velocity)

        if desired_direction is None:
            desired_direction = v

        # if we already have a threat increase margin to get extra far from it
        neighbor_margin = margin
        if self.collision_threat:
            neighbor_margin += margin_histeresis

        # find neighbor with soonest closest appraoch
        neighbor, approach_time, relative_position, relative_velocity, minimum_separation = self._collision_neighbor(sector, neighborhood_dist, neighbor_margin)

        if neighbor is None:
            self.cannot_avoid_collision = False
            self.ship.collision_threat = None
            self.collision_dv = ZERO_VECTOR
            return ZERO_VECTOR, np.inf, np.inf, 0

        # if the collision will happen outside our max time horizon (e.g. we
        # have plenty of time to stop), then don't worry about it.
        #max_time = (np.linalg.norm(self.ship.velocity) / (self.ship.max_thrust / self.ship.phys.mass) + self._rotation_time(2*math.pi)) * 1.2
        #if approach_time > max_time:
        #    self.ship.collision_threat = None
        #    return np.array((0,0))

        # if the collision distance is outside our max distance horizon (e.g.
        # before which we expect to stop), then don't worry about it. this
        # helps ignore collisions with stuff we're trying to arrive at
        collision_distance = np.linalg.norm(self.ship.velocity * approach_time)
        if collision_distance > max_distance:
            self.collision_threat = None
            self.ship.collision_threat = None
            self.collision_dv = ZERO_VECTOR
            return ZERO_VECTOR, np.inf, np.inf, 0
        #elif np.linalg.norm(self.collision_threat.loc - self.target_location) < 10:
        #    raise Exception()

        distance = np.linalg.norm(relative_position)

        #self.logger.debug(f'collision in {approach_time:.2f}s, {collision_distance}m, {relative_position + relative_velocity * approach_time}')

        # if we're going to exactly collide or we're already inside of the
        # desired margin, try to go directly away, other wise (this if) go away
        # from the collision positjion
        if minimum_separation > VELOCITY_EPS and distance > self.ship.radius + neighbor.radius + margin:
            relative_position = relative_position + relative_velocity * approach_time
            distance = np.linalg.norm(relative_position)
        elif distance <= self.ship.radius + neighbor.radius + margin:
            self.logger.debug(f'already inside margin: {distance}')

        speed = np.linalg.norm(desired_direction)
        if speed < VELOCITY_EPS:
            desired_direction = relative_velocity
            speed = np.linalg.norm(relative_velocity)

        # if the angle between relative pos and velocity is very small or close to pi
        # we could have instability
        parallelness = np.dot(relative_position/distance,desired_direction/speed)

        if parallelness < -1+PARALLEL_EPS or parallelness > 1-PARALLEL_EPS:
            #TODO: is this weird if we're already offset by a bunch?

            # prefer diverting clockwise
            dv = np.array((desired_direction[1], -desired_direction[0]))
            dv = dv / np.linalg.norm(dv)
        else:
            # get the component of relative position perpendicular to our velocity
            #   this will be the direction of delta v
            #   we want this to be big enough so that by the time we reach collision, we're radius + radius + margin apart
            dv = relative_position - desired_direction * np.dot(relative_position, desired_direction) / np.dot(desired_direction, desired_direction)

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

        #TODO: need to account for acceleration time
        # really we're going to have a period of time when we're accelerating
        # to the target (relative) speed and a period of time when we're
        # cruising at that (relative) speed
        # triangle: v_i to v_f over t_accelerate
        # rectangle: v_f over t_cruise
        # areas should equal d

        delta_dist = np.linalg.norm(dv)
        d = self.ship.radius + neighbor.radius + margin + margin_histeresis - delta_dist

        # this discounts apporach_time by a factor 1.2 for safety, but see
        # above TODO for a more principled way to do this
        # d = 1/2  * (v_f + v_i) * t
        # 2d/t = v_f + v_i
        # v_f = 2d/t - v_i
        # v_f = 2 * d / t - v_i
        # linear case where v_i is the scalar projection on dv
        # note, dv is TOWARD the threat, so we want to move -d away
        v_i = np.dot(relative_velocity, dv) / delta_dist
        v_f = (2 * -d / approach_time - v_i) * self.safety_factor

        #TODO: this is a temp asserts for testing
        assert abs((v_f/self.safety_factor + v_i)/2 * approach_time + d) < VELOCITY_EPS
        #TODO: we want to get to v_f, but we're already at v_i, right?
        # so if this is delta_velocity, shouldn't we subtract v_i?
        delta_velocity = dv / delta_dist * v_f# - v_i * dv/delta_dist

        self.ship.collision_threat = neighbor
        #self.logger.debug(f'collision imminent in {approach_time}s at {minimum_separation}m in {collision_distance}m < {max_distance}m dist: {distance}m rel vel: {relative_velocity}m/s')
        #self.logger.debug(f'avoid: {v_f}m/s {delta_velocity} vs {v}')

        if minimum_separation < self.ship.radius + neighbor.radius + margin and abs(v_f)/self.safety_factor > self.ship.max_acceleration() * approach_time:
            self.cannot_avoid_collision = True
            self.logger.debug(f'cannot avoid collision: {abs(v_f)} > {self.ship.max_thrust} / {self.ship.mass} * {approach_time}')
        else:
            self.cannot_avoid_collision = False

        #TODO: hack to coordinate betweent two steering agents
        # only one gets to steer to avoid the collision
        #if isinstance(neighbor, core.Ship) and isinstance(neighbor.orders[0], AbstractSteeringOrder) and neighbor.orders[0].collision_threat == self.ship and np.all(neighbor.orders[0].collision_dv != ZERO_VECTOR):
        #    delta_velocity = ZERO_VECTOR

        #TODO: these are temp asserts for testing
        ret_old = _analyze_neighbor(self.ship.loc, self.ship.velocity, neighbor.radius, neighbor.loc, neighbor.velocity, np.inf, self.ship.radius+neighbor_margin)
        ret_new = _analyze_neighbor(self.ship.loc, self.ship.velocity + delta_velocity, neighbor.radius, neighbor.loc, neighbor.velocity, np.inf, self.ship.radius+neighbor_margin)
        assert ret_old[1] < np.inf
        assert not (ret_new[1] < np.inf)

        self.collision_dv = delta_velocity
        return delta_velocity, approach_time, minimum_separation, d

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
    def __init__(self, target_location: np.ndarray, *args, arrival_distance: float=1e3, **kwargs) -> None:
        """ Creates an order to go to a specific location.

        The order will arrivate at the location approximately and with zero
        velocity.

        target_location the location
        arrival_distance how close to the location to arrive
        """

        super().__init__(*args, **kwargs)
        self.target_location = target_location
        self.arrival_distance = arrival_distance

        self.cannot_stop = False

    def to_history(self) -> dict:
        data = super().to_history()
        data["t_loc"] = self.target_location.tolist()
        data["cs"] = self.cannot_stop

        return data

    def __str__(self) -> str:
        return f'GoToLocation: {self.target_location} ad:{self.arrival_distance} sf:{self.safety_factor}'

    def is_complete(self) -> bool:
        return bool(np.linalg.norm(self.target_location - np.array((self.ship.x, self.ship.y))) < self.arrival_distance + VELOCITY_EPS and np.allclose(self.ship.velocity, ZERO_VECTOR))

    def act(self, dt: float) -> None:
        if self.ship.sector is None:
            raise Exception(f'{self.ship} not in any sector')
        # essentially the arrival steering behavior but with some added
        # obstacle avoidance

        v = self.ship.velocity

        # vector toward target
        current_location = self.ship.loc
        course = self.target_location - (current_location)
        distance, target_angle = util.cartesian_to_polar(*course)

        #collision avoidance for nearby objects
        #   this includes fixed bodies as well as dynamic ones
        #TODO: should we pass in where we'd really like to go? (e.g. course to the target?)
        collision_dv, _, _, _ = self._avoid_collisions_dv(self.ship.sector, 5e4, self.collision_margin, max_distance=distance-self.arrival_distance/self.safety_factor, desired_direction=course)

        #TODO: quick hack to test just avoiding collisions
        #   if we do this then we get to our target margin
        if np.linalg.norm(collision_dv) > VELOCITY_EPS:
            self._accelerate_to(np.array(v) + collision_dv, dt)
            return

        if distance < self.arrival_distance + VELOCITY_EPS:
            self._accelerate_to(collision_dv, dt)
        else:
            max_acceleration = self.ship.max_acceleration()
            rot_time = rotation_time(abs(util.normalize_angle(self.ship.angle-(target_angle+np.pi), shortest=True)), self.ship.max_angular_acceleration())

            # choose a desired speed such that if we were at that speed right
            # now we would have enough time to rotate 180 degrees and
            # decelerate to a stop at full thrust by the time we reach arrival
            # distance
            a = -1 * max_acceleration
            s = distance-self.arrival_distance
            desired_speed_1 = (-1 * rot_time + np.sqrt(rot_time ** 2 - 2*s/a))/(-1/a)
            desired_speed_2 = (-1 * rot_time - np.sqrt(rot_time ** 2 - 2*s/a))/(-1/a)

            desired_speed = np.max((desired_speed_1, desired_speed_2))/self.safety_factor
            assert desired_speed >= 0

            #TODO: what happens if we can't stop in time?
            d = (np.linalg.norm(v)**2) / (2* max_acceleration)
            if d > distance:
                self.cannot_stop = True
                self.logger.debug(f'cannot stop in time {d} > {distance}')
            else:
                self.cannot_stop = False

            target_v = course/distance * desired_speed + collision_dv
            #self.logger.debug(f'target_v: {target_v}')
            self._accelerate_to(target_v + collision_dv, dt)

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
            t = approach_time - rotation_time(2*np.pi, self.ship.max_angular_acceleration())
            if t < 0 or distance > 1/2 * self.ship.max_acceleration()*t**2 / self.safety_factor:
                self._accelerate_to(collision_dv, dt)
                return
        self._accelerate_to(ZERO_VECTOR, dt)

class MineOrder(AbstractSteeringOrder):
    def __init__(self, target: core.Asteroid, amount: float, *args, max_dist=1e3, **kwargs) -> None:
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
            self.ship.orders.appendleft(GoToLocation(self.target.loc, self.ship, self.gamestate))
            return

        #TODO: actually implement harvesting, taking time, maybe visual fx
        amount = np.clip(self.amount, 0, self.target.amount)
        self.target.amount -= amount
        self.harvested += amount

class TransferCargo(core.Order):
    def __init__(self, target: core.SectorEntity, resource: int, amount: float, *args, max_dist=1e3, **kwargs) -> None:
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
            self.ship.orders.appendleft(GoToLocation(self.target.loc))
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
