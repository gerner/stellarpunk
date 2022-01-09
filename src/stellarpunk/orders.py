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
        # assume omega_0 = 0
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

        if (difference_mag * mass / self.ship.max_fine_thrust > rotation_time and abs(delta_heading) > ANGLE_EPS) or abs(w) > ANGLE_EPS:
            # we need to rotate in direction of thrust
            self._rotate_to(difference_angle, dt)
        else:
            # we should apply thrust, however we can with the current heading
            # max thrust is main engines if we're pointing in the desired
            # direction, otherwise use fine thrusters
            if abs(delta_heading) < ANGLE_EPS:
                max_thrust = self.ship.max_thrust
            else:
                max_thrust = self.ship.max_fine_thrust


            x,y = force_for_delta_velocity(
                    dv,
                    self.ship.max_thrust, mass, dt)
            if (x,y) == (0,0):
                self.ship.phys.velocity = tuple(target_velocity)
            else:
                self.ship.phys.apply_force_at_world_point(
                        (x,y),
                        self.ship.phys.position+self.ship.phys.center_of_gravity)
                d = (v.length**2) / (2* self.ship.phys.force.length / self.ship.phys.mass)

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
    def __init__(self, target_location, *args, arrival_distance=5e2, safety_factor=1.2, **kwargs):
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

    def _deceleration_radius(self):
        # braking distance is how much distance we'll travel at current
        # velocity while thrusting
        # to zero velocity
        #   F = ma
        #   max_accel = max_thrust / m
        #   v_f^2 = 2a*d
        #   d = v_f^2 / 2a
        max_acceleration = self.ship.max_thrust / self.ship.phys.mass
        braking_distance = self.ship.phys.velocity.get_length_sqrd() / (2 * max_acceleration)

        # rotation distance is how much linear distance we'll travel at current
        # velocity while rotating to oppose direction of velocity, worst case
        max_angular_acceleration = self.ship.max_torque / self.ship.phys.moment
        rot_time = 2*math.sqrt(math.pi/max_angular_acceleration)
        rotation_distance = rot_time * self.ship.phys.velocity.length

        dist = braking_distance + rotation_distance + self.arrival_distance
        return max(dist * self.safety_factor, self.arrival_distance)

    def is_complete(self):
        return np.linalg.norm(self.target_location - np.array((self.ship.x, self.ship.y))) < self.arrival_distance and self.ship.phys.velocity == (0,0)

    def act(self, dt):
        # essentially the arrival steering behavior but with some added
        # obstacle avoidance

        v = self.ship.phys.velocity

        # vector toward target
        current_location = np.array((self.ship.x, self.ship.y))
        course = self.target_location - current_location
        distance, target_angle = util.cartesian_to_polar(*course)
        if distance < self.arrival_distance:
            if np.linalg.norm(v) < VELOCITY_EPS:
                self.ship.phys.velocity = (0,0)
            else:
                self._accelerate_to((0,0), dt)
            return


        max_acceleration = self.ship.max_thrust / self.ship.phys.mass
        if distance < self.arrival_distance:
            self._accelerate_to(course/distance * VELOCITY_EPS/2)
        elif distance <= self._deceleration_radius():
            # when we start braking we want our velocity to be pointed at the
            max_angular_acceleration = self.ship.max_torque / self.ship.phys.moment
            rot_time = math.sqrt(2*math.pi/max_angular_acceleration)
            # target with enough speed that we can still slow down
            target_distance = (distance - rot_time * v.length)
            desired_speed = math.sqrt(2 * max_acceleration * max(0, target_distance))/self.safety_factor

            #TODO: what happens if we can't stop in time?
            d = (v.length**2) / (2* max_acceleration)
            if d > distance:
                raise Exception()

            self._accelerate_to(course/distance * desired_speed, dt)
        else:
            # plan is to accelerate constantly to a max velocity and then flip
            # and accelerate constantly to zero velocity
            # the max speed possible is if we accelerate constantly to a max
            # velocity and then rotate 180deg and then decelerate to zero
            #   d_a: distance over which we acelerate
            #   d_r: distance traveled when we rotate 180deg
            #   d_b: distance over which we brake
            #   d_r = v_f * t_r ; velocity times time to rotate
            #   t_r = sqrt(2pi * moment / torque)
            #   d = d_a + d_r + d_b ; how we divide the distance up
            #   v_f^2 = v_i^2 + 2a * d_a ; when we're acelerating
            #   v_f^2 = 2a * d_b ; when we're braking
            #
            #   d_b = v_f^2 / 2a
            #   d_a = d - v_f * t_r - v_f^2/2a
            #   v_f^2 = v_i^2 + 2a * (d - v_f * t_r - v_f^2/2a)
            #   v_f^2 = v_i^2 + 2a*d - 2a*v_f*t_r - v_f^2
            #   v_f^2 + a*t_r * v_f - (v_i^2 + a*d) = 0
            #
            #   (-b +/- sqrt(b^2 -4ac)) / 2a
            #
            #   (-a*t_r +/- sqrt( (a*t_r)^2 - 4(v_i^2 + a*d))) / 2
            speed_to_target_sqrd = v.projection(tuple(course)).get_length_sqrd()
            max_speed = math.sqrt(4*max_acceleration*(distance-self.arrival_distance) - speed_to_target_sqrd) / self.safety_factor
            self._accelerate_to(course/distance * max_speed, dt)
