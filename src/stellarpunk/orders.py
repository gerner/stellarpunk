""" Orders that can be given to ships. """

import math

import numpy as np

from stellarpunk import util, core

def torque_for_angle(target_angle, angle, w, max_torque, moment, dt, eps=1e-3):
    difference_angle = util.normalize_angle(target_angle - angle, shortest=True)
    braking_angle =  -1 * np.sign(w) * -0.5 * w*w * moment / max_torque

    if abs(w) < eps and abs(difference_angle) < eps:
        # bail if we're basically already there
        t = 0
    elif abs(braking_angle) > abs(difference_angle):
        # we can't break in time, so just start breaking and we'll fix it later
        t = moment * -1 * w / dt
        t = np.clip(t, -1*max_torque, max_torque)
    else:
        # add torque in the desired direction to get
        # accel = tau / moment
        # dw = accel * dt
        # desired w is w such that braking_angle = difference_angle

        t = max_torque * np.sign(difference_angle)

    return t

def force_for_zero_velocity(v, max_thrust, mass, dt, eps=1e-2):
    velocity_magnitude, velocity_angle = util.cartesian_to_polar(*v)
    if velocity_magnitude < eps:
        # bail if we're basically already there
        x,y = (0,0)
    else:
        thrust = np.clip(mass * (velocity_magnitude-eps/2) / dt, 0, max_thrust)
        x, y = util.polar_to_cartesian(thrust, velocity_angle + math.pi)
    return (x,y)

class RotateOrder(core.Order):
    def __init__(self, target_angle, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_angle = util.normalize_angle(target_angle)

    def is_complete(self):
        return self.ship.phys.angular_velocity == 0 and util.normalize_angle(self.ship.phys.angle) == self.target_angle

    def act(self, dt):
        # given current angle and angular_velocity and max torque, choose
        # torque to apply for dt now to hit target angle


        angle = util.normalize_angle(self.ship.phys.angle)
        max_torque = 9000
        w = self.ship.phys.angular_velocity
        moment = self.ship.phys.moment

        t = torque_for_angle(self.target_angle, angle, w, max_torque, moment, dt)

        if t == 0:
            self.ship.phys.angle = self.target_angle
            self.ship.phys.angular_velocity = 0
        else:
            self.ship.phys.torque = t
        return

class KillVelocityOrder(core.Order):
    """ Applies thrust and torque to zero out velocity and angular velocity.

    Rotates to opposite direction of current velocity and applies thrust to
    zero out velocity. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_velocity_magnitude = math.inf
        self.expected_next_velocity = self.ship.phys.velocity

    def is_complete(self):
        return self.ship.phys.angular_velocity == 0 and self.ship.phys.velocity == (0,0)

    def act(self, dt):
        velocity_eps = 1e-2
        angle_eps = 1e-3
        max_thrust = 0.5 * 1e6
        max_torque = 90000
        mass = self.ship.phys.mass
        moment = self.ship.phys.moment
        angle = self.ship.angle
        w = self.ship.phys.angular_velocity
        v = self.ship.phys.velocity

        # orient toward the opposite of the direction of travel
        # thrust until zero velocity
        velocity_magnitude, velocity_angle = util.cartesian_to_polar(*v)

        reverse_velocity_angle = util.normalize_angle(velocity_angle + math.pi)
        assert velocity_magnitude <= self.last_velocity_magnitude
        assert abs(v[0] - self.expected_next_velocity[0]) < velocity_eps
        assert abs(v[1] - self.expected_next_velocity[1]) < velocity_eps
        self.last_velocity_magnitude = velocity_magnitude

        if velocity_magnitude < velocity_eps and abs(w) < angle_eps:
            self.ship.phys.angular_velocity = 0
            self.ship.phys.velocity = (0,0)
            return

        if abs(angle - reverse_velocity_angle) > angle_eps or abs(w) > angle_eps:
            # first aim ship opposity velocity
            t = torque_for_angle(reverse_velocity_angle, angle, w, max_torque, moment, dt)
            if t == 0:
                self.ship.phys.angle = reverse_velocity_angle
                self.ship.phys.angular_velocity = 0
            else:
                self.ship.phys.torque = t
        else:
            x,y = force_for_zero_velocity(self.ship.phys.velocity, max_thrust, mass, dt)
            if (x,y) == (0,0):
                self.ship.phys.velocity = (0,0)
            else:
                self.expected_next_velocity = (v[0] + x/mass*dt, v[1] + y/mass*dt)
                self.ship.phys.apply_force_at_world_point(
                        (x,y),
                        self.ship.phys.position+self.ship.phys.center_of_gravity)

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
        self.ship.phys.torque = np.clip(t, -9000, 9000)
        #TODO: do we need a hack for very low angular velocities?

