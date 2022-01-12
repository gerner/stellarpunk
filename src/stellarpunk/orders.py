""" Orders that can be given to ships. """

import logging
import math

import numpy as np

from stellarpunk import util, core

ANGLE_EPS = 1e-2
VELOCITY_EPS = 1e-2

def torque_for_angle(target_angle, angle, w, max_torque, moment, dt, eps=ANGLE_EPS):
    """ What torque to apply to achieve target angle """

    difference_angle = util.normalize_angle(target_angle - angle, shortest=True)
    braking_angle =  -1 * np.sign(w) * -0.5 * w*w * moment / max_torque

    if abs(w) < eps and abs(difference_angle) < eps:
        # bail if we're basically already there
        # caller can handle this, e.g. set rotation to target and w to 0
        t = 0
    elif abs(braking_angle) > abs(difference_angle):
        # we can't break in time, so just start breaking and we'll fix it later
        t = moment * -1 * w / dt
    else:
        # add torque in the desired direction to get
        # accel = tau / moment
        # dw = accel * dt
        # desired w is w such that braking_angle = difference_angle
        # braking_angle =  -1 * np.sign(w) * -0.5 * w*w * moment / max_torque
        # sqrt(difference_angle * max_torque / (0.5 * moment)) = w
        desired_w = np.sign(difference_angle) * math.sqrt(abs(difference_angle) * max_torque / (0.5 * moment))

        t = (desired_w - w)*moment/dt

        # crude method:
        #t = max_torque * np.sign(difference_angle)

    return np.clip(t, -1*max_torque, max_torque)

def force_for_delta_velocity(dv, max_thrust, mass, dt, eps=VELOCITY_EPS):
    """ What force to apply to get dv change in velocity. Ignores heading. """

    dv_magnitude, dv_angle = util.cartesian_to_polar(*dv)
    if dv_magnitude < eps:
        x,y = (0,0)
    else:
        thrust = np.clip(mass * dv_magnitude / dt, 0, max_thrust)
        x, y = util.polar_to_cartesian(thrust, dv_angle)
    return (x,y)

class AbstractSteeringOrder(core.Order):
    def _rotation_time(self, delta_angle):
        max_angular_acceleration = self.ship.max_torque / self.ship.phys.moment
        # theta_f = theta_0 + omega_0*t + 1/2 * alpha * t^2
        # assume omega_0 = 0 <--- assumes we're not currently rotating!
        # assume we constantly accelerate half way, constantly accelerate the
        # other half
        return 2*math.sqrt(abs(delta_angle)/max_angular_acceleration)

    def _rotate_to(self, target_angle, dt):
        # given current angle and angular_velocity and max torque, choose
        # torque to apply for dt now to hit target angle

        angle = util.normalize_angle(self.ship.phys.angle)
        w = self.ship.phys.angular_velocity
        moment = self.ship.phys.moment

        t = torque_for_angle(
                target_angle, angle, w,
                self.ship.max_torque, moment, dt)

        if t == 0:
            self.ship.phys.angle = target_angle
            self.ship.phys.angular_velocity = 0
        else:
            self.ship.phys.torque = t

    def _accelerate_to(self, target_velocity, dt):
        mass = self.ship.phys.mass
        moment = self.ship.phys.moment
        angle = self.ship.angle
        w = self.ship.phys.angular_velocity
        v = self.ship.phys.velocity

        # orient toward the opposite of the direction of travel
        # thrust until zero velocity
        velocity_magnitude, velocity_angle = util.cartesian_to_polar(*v)

        dv = np.array(target_velocity) - np.array(v)
        difference_mag, difference_angle = util.cartesian_to_polar(*dv)

        if difference_mag < VELOCITY_EPS and abs(w) < ANGLE_EPS:
            self.ship.phys.angular_velocity = 0
            self.ship.phys.velocity = tuple(target_velocity)
            return

        delta_heading = util.normalize_angle(angle-difference_angle, shortest=True)
        rotation_time = self._rotation_time(delta_heading)

        # while we've got a lot of thrusting to do, we can tolerate only
        # approximately matching our desired angle
        # this should have something to do with how much we expect this angle
        # to change in dt time, but order of magnitude seems reasonable approx
        coarse_angle_match = ANGLE_EPS

        if (difference_mag * mass / self.ship.max_fine_thrust > rotation_time and abs(delta_heading) > coarse_angle_match) or abs(w) > coarse_angle_match or (difference_mag < VELOCITY_EPS and (abs(w) > ANGLE_EPS)):
            # we need to rotate in direction of thrust
            self._rotate_to(difference_angle, dt)

            # also apply thrusters
            x,y = force_for_delta_velocity(
                    dv,
                    self.ship.max_fine_thrust, mass, dt)
            if (x,y) == (0,0):
                self.ship.phys.velocity = tuple(target_velocity)
            else:
                self.ship.phys.apply_force_at_world_point(
                        (x,y),
                        self.ship.phys.position+self.ship.phys.center_of_gravity)
        else:
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
            if (x,y) == (0,0):
                self.ship.phys.velocity = tuple(target_velocity)
            else:
                self.ship.phys.apply_force_at_world_point(
                        (x,y),
                        self.ship.phys.position+self.ship.phys.center_of_gravity)

    def _collision_neighbor(self, sector, neighborhood_dist, margin):
        bounds = (
                self.ship.x-neighborhood_dist, self.ship.y-neighborhood_dist,
                self.ship.x+neighborhood_dist, self.ship.y+neighborhood_dist
        )

        approach_time = math.inf
        neighbor = None
        relative_position = None
        relative_velocity = None
        minimum_separation = None

        pos = np.array((self.ship.x, self.ship.y))
        v = np.array(self.ship.phys.velocity)

        for hit in sector.spatial.intersection(bounds, objects=True):
            entity = sector.entities[hit.object]
            rel_pos = np.array((entity.x, entity.y)) - pos
            rel_vel = np.array(entity.phys.velocity) - v
            rel_speed_sqrd = np.inner(rel_vel, rel_vel)

            # check for parallel paths
            if rel_speed_sqrd == 0:
                continue

            approach_t = -1 * np.dot(rel_pos, rel_vel) / rel_speed_sqrd;

            # check for moving away or farther in future than current soonest
            if approach_t <= 0 or approach_t >= approach_time:
                continue

            # check for collision
            distance = np.linalg.norm(rel_pos)
            min_sep = distance - math.sqrt(rel_speed_sqrd) * approach_t
            if min_sep > self.ship.radius + entity.radius + margin:
                continue

            approach_time = approach_t
            neighbor = entity
            relative_position = rel_pos
            relative_velocity = rel_vel
            minimum_separation = min_sep

        return neighbor, approach_time, relative_position, relative_velocity, minimum_separation

    def _avoid_collisions(self, sector, neighborhood_dist, margin, dt, max_distance=math.inf):
        """ Given current velocity, try to avoid collisions with neighbors """

        # find neighbor with soonest closest appraoch
        neighbor, approach_time, relative_position, relative_velocity, minimum_separation = self._collision_neighbor(sector, neighborhood_dist, margin)

        if neighbor is None:
            return False

        # if the collision distance is outside our max distance horizon (e.g.
        # before which we expect to stop), then don't worry about it. this
        # helps ignore collisions with stuff we're trying to arrive at
        collision_distance = np.linalg.norm(np.linalg.norm(self.ship.phys.velocity * approach_time))
        if collision_distance > max_distance:
            return False

        distance = np.linalg.norm(relative_position)

        # if we're going to exactly collide or we're already inside of the
        # desired margin, try to go directly away
        if minimum_separation > 0 and distance > self.ship.radius + neighbor.radius + margin:
            relative_position += relative_velocity * approach_time

        if neighbor is None:
            return False

        # alter our current velocity such that our closest approach will be at
        # least margin distance away

        # relative_position is the direction we want to move in
        # over approach_time we want to have moved radii + margin - minimum_separation
        # d = 1/2  * v_f * t + 1/2 v_i * t
        # v_i = relative_position
        # v_f = 2 * d / t - v_i
        d = self.ship.radius + neighbor.radius + margin*2 - minimum_separation
        target_relative_speed = -1 * 2 * d / approach_time - np.linalg.norm(relative_velocity)

        target_velocity = self.ship.phys.velocity + relative_position / distance * target_relative_speed

        logging.debug(f'collision imminent in {approach_time}s at {minimum_separation}m in {collision_distance}m < {max_distance}m rel vel: {relative_velocity}m/s')
        logging.debug(f'remediate: {target_relative_speed}m/s {target_velocity}m/s total')

        self._accelerate_to(target_velocity, dt)
        return True

class KillRotationOrder(core.Order):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def is_complete(self):
        return self.ship.phys.angular_velocity == 0

    def act(self, dt):
        # apply torque up to max torque to kill angular velocity
        # torque = moment * angular_acceleration
        # the perfect acceleration would be -1 * angular_velocity / timestep
        # implies torque = moment * -1 * angular_velocity / timestep
        t = self.ship.phys.moment * -1 * self.ship.phys.angular_velocity / dt
        if t == 0:
            self.ship.phys.angular_velocity = 0
        else:
            self.ship.phys.torque = np.clip(t, -9000, 9000)

class RotateOrder(AbstractSteeringOrder):
    def __init__(self, target_angle, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_angle = util.normalize_angle(target_angle)

    def is_complete(self):
        return self.ship.phys.angular_velocity == 0 and util.normalize_angle(self.ship.phys.angle) == self.target_angle

    def act(self, dt):
        self._rotate_to(self.target_angle, dt)

class KillVelocityOrder(AbstractSteeringOrder):
    """ Applies thrust and torque to zero out velocity and angular velocity.

    Rotates to opposite direction of current velocity and applies thrust to
    zero out velocity. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def is_complete(self):
        return self.ship.phys.angular_velocity == 0 and self.ship.phys.velocity == (0,0)

    def act(self, dt):
        self._accelerate_to((0,0), dt)

class GoToLocation(AbstractSteeringOrder):
    def __init__(self, target_location, sector, *args, arrival_distance=1e3, safety_factor=1.2, **kwargs):
        """ Creates an order to go to a specific location.

        The order will arrivate at the location approximately and with zero
        velocity.

        target_location the location
        arrival_distance how close to the location to arrive
        safety_factor how much safety to build into velocity and force calcs
        """

        super().__init__(*args, **kwargs)
        self.target_location = np.array(target_location)
        self.arrival_distance = arrival_distance
        self.safety_factor = safety_factor
        self.sector = sector

    def is_complete(self):
        return np.linalg.norm(self.target_location - np.array((self.ship.x, self.ship.y))) < self.arrival_distance and self.ship.phys.velocity == (0,0)

    def act(self, dt):
        # essentially the arrival steering behavior but with some added
        # obstacle avoidance

        v = self.ship.phys.velocity

        # vector toward target
        current_location = np.array((self.ship.x, self.ship.y))
        course = self.target_location - (current_location + v * dt)
        distance, target_angle = util.cartesian_to_polar(*course)

        #collision avoidance for nearby objects
        #   this includes fixed bodies as well as dynamic ones
        if self._avoid_collisions(self.sector, 1e4, 3e2, dt, max_distance=distance-self.arrival_distance):
            return

        if distance < self.arrival_distance:
            if np.linalg.norm(v) < VELOCITY_EPS:
                self.ship.phys.velocity = (0,0)
            else:
                self._accelerate_to((0,0), dt)
            return

        max_acceleration = self.ship.max_thrust / self.ship.phys.mass
        if distance < self.arrival_distance:
            self._accelerate_to((0,0))
        else:
            # accelerate along a vector toward the target location to a speed
            # as big as possible so we can reach zero velocity at the target
            # build in some saftey margin: desired final position and max speed
            rot_time = self._rotation_time(2*math.pi)
            target_distance = distance - rot_time * v.length - self.arrival_distance * (self.safety_factor - 1)
            desired_speed = math.sqrt(2 * max_acceleration * max(0, target_distance))/self.safety_factor

            #TODO: what happens if we can't stop in time?
            d = (v.length**2) / (2* max_acceleration)
            if d > distance:
                logging.debug(f'{d} > {distance}')

            self._accelerate_to(course/distance * desired_speed, dt)
