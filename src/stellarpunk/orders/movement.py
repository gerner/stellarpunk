""" Basic orders, moving around, waiting, etc. """

from __future__ import annotations

from typing import Optional, Any

import numpy as np
import numpy.typing as npt

from stellarpunk import util, core

from .steering import VELOCITY_EPS, ZERO_VECTOR, rotation_time, find_target_v, AbstractSteeringOrder

class KillRotationOrder(core.Order):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
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
    def __init__(self, target_angle: float, *args: Any, **kwargs: Any) -> None:
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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
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
            surface_distance:float=2e3,
            collision_margin:float=1e3) -> GoToLocation:

        # pick a point on this side of the target entity that is midway in the
        # "viable" arrival band: the space between the collision margin and
        # surface distance away from the radius
        _, angle = util.cartesian_to_polar(*(ship.loc - entity.loc))
        target_angle = angle + gamestate.random.uniform(-np.pi/2, np.pi/2)
        target_arrival_distance = (surface_distance - collision_margin)/2
        target_loc = entity.loc + util.polar_to_cartesian(entity.radius + collision_margin + target_arrival_distance, target_angle)

        return GoToLocation(
                target_loc, ship, gamestate,
                arrival_distance=target_arrival_distance,
                min_distance=0.)

    @staticmethod
    def compute_eta(ship:core.Ship, target_location:npt.NDArray[np.float64], safety_factor:float=2.0) -> float:
        course = target_location - (ship.loc)
        distance, target_angle = util.cartesian_to_polar(course[0], course[1])
        rotate_towards = rotation_time(util.normalize_angle(target_angle-ship.angle, shortest=True), ship.angular_velocity, ship.max_angular_acceleration(), safety_factor)

        # we cap at max_speed, so need to account for that by considering a
        # "cruise" period where we travel at max_speed, but only if we have
        # enough distance to make it to cruise speed
        if np.sqrt(2. * ship.max_acceleration() * distance/2.) < ship.max_speed():
            accelerate_up = np.sqrt( 2. * (distance/2.) / ship.max_acceleration()) * safety_factor
            cruise = 0.
        else:
            # v_f**2 = 2 * a * d
            # d = v_f**2 / (2*a)
            d_accelerate = ship.max_speed()**2 / (2*ship.max_acceleration())
            accelerate_up = ship.max_speed() / ship.max_acceleration() * safety_factor

            d_cruise = distance - 2*d_accelerate
            cruise = d_cruise / ship.max_speed()# * safety_factor

        rotate_away = rotation_time(np.pi, 0, ship.max_angular_acceleration(), safety_factor)
        accelerate_down = accelerate_up

        assert rotate_towards >= 0.
        assert accelerate_up >= 0.
        assert cruise >= 0.
        assert accelerate_down >= 0.
        assert rotate_away >= 0.

        return rotate_towards + accelerate_up + rotate_away + cruise + accelerate_down

    def __init__(self, target_location: npt.NDArray[np.float64], *args: Any, arrival_distance: float=1.5e3, min_distance:Optional[float]=None, **kwargs: Any) -> None:
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
        data["t_loc"] = (self.target_location[0], self.target_location[1])
        data["ad"] = self.arrival_distance
        data["md"] = self.min_distance
        data["t_v"] = (self.target_v[0], self.target_v[1])
        data["cs"] = self.cannot_stop

        return data

    def __str__(self) -> str:
        return f'GoToLocation: {self.target_location} ad:{self.arrival_distance} sf:{self.safety_factor}'

    def estimate_eta(self) -> float:
        return GoToLocation.compute_eta(self.ship, self.target_location, self.safety_factor)

    def is_complete(self) -> bool:
        # computing this is expensive, so don't if we can avoid it
        if self.distance_estimate > self.arrival_distance*5:
            return False
        else:
            return bool(np.linalg.norm(self.target_location - self.ship.loc) < self.arrival_distance + VELOCITY_EPS and np.allclose(self.ship.velocity, ZERO_VECTOR))

    def _begin(self) -> None:
        self.init_eta = self.estimate_eta()

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
        # ramp down with inverse of the density: max_speed = m / (density + b)
        # d_low, s_high is one point we want to hit
        # d_high, s_low is another
        d_low = 1/(np.pi*1e4**2)
        s_high = max_speed
        d_high = 30/(np.pi*1e4**2)
        s_low = 100
        b = (s_low * d_high - s_high * d_low)/(s_high - s_low)
        m = s_high*(d_low + b)
        max_speed = min(max_speed, m / (self.neighborhood_density + b))

        self.target_v, distance, self.distance_estimate, self.cannot_stop = find_target_v(
                self.target_location, self.arrival_distance, self.min_distance,
                self.ship.loc, v, theta, omega,
                max_acceleration, max_angular_acceleration, max_speed,
                dt, self.safety_factor)

        if self.cannot_stop:
            max_distance = np.inf
            self.logger.debug(f'cannot stop in time distance: {distance} v: {v}')
        else:
            max_distance = distance-self.min_distance

        #collision avoidance for nearby objects
        #   this includes fixed bodies as well as dynamic ones
        collision_dv, approach_time, minimum_separation, distance_to_avoid_collision = self._avoid_collisions_dv(
                self.ship.sector, 1e4, self.collision_margin,
                max_distance=max_distance,
                desired_direction=self.target_v)

        if abs(collision_dv[0]) < VELOCITY_EPS and abs(collision_dv[1]) < VELOCITY_EPS:
            self._accelerate_to(self.target_v, dt)
        else:
            # if we're over max speed, let's slow down in addition to avoiding
            # collision
            v_mag = util.magnitude(v[0], v[1])
            if v_mag > max_speed:
                v = v / v_mag * max_speed
            self._accelerate_to(v + collision_dv, dt)
        return

class WaitOrder(AbstractSteeringOrder):
    def __init__(self, *args: Any, **kwargs: Any):
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
