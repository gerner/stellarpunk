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
            self.ship.apply_torque(np.clip(t, -9000, 9000))

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
        """ Makes a GoToLocation order to get near a target entity.

        entity: the entity to get near
        ship: the ship that will execute the order
        gamestate: gamestate
        surface_distance: target distance from surface of the entity (max dist)
        collision_margin: min distance to stay away from the entity (min dist)

        returns a GoToLocation order matching those parameters
        """

        if entity.sector is None:
            raise ValueError("entity {entity} is not in any sector")

        # pick a point on this side of the target entity that is midway in the
        # "viable" arrival band: the space between the collision margin and
        # surface distance away from the radius
        tries = 0
        max_tries = 5
        while tries < max_tries:
            _, angle = util.cartesian_to_polar(*(ship.loc - entity.loc))
            target_angle = angle + gamestate.random.uniform(-np.pi/2, np.pi/2)
            target_arrival_distance = (surface_distance - collision_margin)/2
            target_loc = entity.loc + util.polar_to_cartesian(entity.radius + collision_margin + target_arrival_distance, target_angle)

            # check if there's other stuff nearby this point
            # note: this isn't perfect since these entities might be transient,
            # but this is best effort
            if next(entity.sector.spatial_point(target_loc, collision_margin), None) == None:
                break
            tries += 1

        return GoToLocation(
                target_loc, ship, gamestate,
                arrival_distance=target_arrival_distance,
                min_distance=0.,
                target_sector=entity.sector)

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

    def __init__(self,
            target_location: npt.NDArray[np.float64],
            *args: Any,
            arrival_distance: float=1.5e3,
            min_distance:Optional[float]=None,
            target_sector: Optional[core.Sector]=None,
            neighborhood_radius: float = 2.5e4,
            **kwargs: Any) -> None:
        """ Creates an order to go to a specific location.

        The order will arrivate at the location approximately and with zero
        velocity.

        target_sector the sector with the location (only used for error checking and aborting the order)
        target_location the location
        arrival_distance how close to the location to arrive
        """

        super().__init__(*args, **kwargs)
        self.neighborhood_radius = neighborhood_radius
        if target_sector is None:
            if self.ship.sector is None:
                raise ValueError(f'no target sector provided and ship {self.ship} is not in any sector')
            target_sector = self.ship.sector

        self.target_sector = target_sector
        self.target_location = target_location
        self.target_v = ZERO_VECTOR
        self.arrival_distance = arrival_distance
        if min_distance is None:
            min_distance = self.arrival_distance * 0.9
        self.min_distance = min_distance

        self.cannot_stop = False

        self.distance_estimate = 0.

        # after taking into account collision avoidance
        self._desired_velocity = ZERO_VECTOR
        # the next time we should do a costly computation of desired velocity
        self._next_compute_ts = 0.

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
            return util.magnitude(*(self.target_location - self.ship.loc)) < self.arrival_distance + VELOCITY_EPS and util.both_almost_zero(self.ship.velocity)

    def _begin(self) -> None:
        self.init_eta = self.estimate_eta()

    def act(self, dt: float) -> None:
        # make sure we're in the right sector
        if self.ship.sector != self.target_sector:
            raise ValueError(f'{self.ship} in {self.ship.sector} instead of target {self.target_sector}')

        #TODO: check if it's time for us to do a careful calculation or a simple one
        if self.gamestate.timestamp < self._next_compute_ts:
            self._accelerate_to(self._desired_velocity, dt, time_step=self._next_compute_ts - self.gamestate.timestamp)
            return

        # essentially the arrival steering behavior but with some added
        # obstacle avoidance

        v = self.ship.velocity
        speed = util.magnitude(v[0], v[1])
        theta = self.ship.angle
        omega = self.ship.angular_velocity

        max_acceleration = self.ship.max_acceleration()
        max_angular_acceleration = self.ship.max_angular_acceleration()
        max_speed = self.ship.max_speed()

        # ramp down speed as nearby density increases
        # ramp down with inverse of the density: max_speed = m / (density + b)
        # d_low, s_high is one point we want to hit (speed at low density)
        # d_high, s_low is another (speed at high density
        d_low = 1/(np.pi*self.neighborhood_radius**2)
        s_high = max_speed
        d_high = 30/(np.pi*self.neighborhood_radius**2)
        s_low = 100
        density_max_speed = util.clip(util.interpolate(d_low, s_high, d_high, s_low, self.neighborhood_density), s_low, max_speed)

        # also ramp down speed with distance to nearest neighbor
        # nn_d_high, nn_speed_high is one point
        # nn_d_low, nn_speed_low is another
        nn_d_high = 5e3
        nn_s_high = 1000#max_speed
        nn_d_low = 5e2
        nn_s_low = 100
        nn_max_speed = util.clip(util.interpolate(nn_d_high, nn_s_high, nn_d_low, nn_s_low, self.nearest_neighbor_dist), nn_s_low, max_speed)

        max_speed = min(max_speed, density_max_speed, nn_max_speed)

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

        # scale collision margin with speed, more speed = more margin
        cm_low = self.collision_margin
        cm_high = self.collision_margin*10
        cm_speed_low = 100
        cm_speed_high = 1500
        scaled_collision_margin = util.interpolate(cm_speed_low, cm_low, cm_speed_high, cm_high, speed)
        scaled_collision_margin = util.clip(scaled_collision_margin, cm_low, cm_high)

        if speed > 0:
            # offset looking for threats in the direction we're travelling,
            # depending on our speed
            noff_low = 0
            noff_speed_low = 0
            noff_high = 2.5e4
            noff_speed_high = 2500
            neighborhood_offset = util.clip(
                    util.interpolate(noff_speed_low, noff_low, noff_speed_high, noff_high, speed),
                    0, self.neighborhood_radius - self.collision_margin)
            neighborhood_loc = self.ship.loc + neighborhood_offset / speed * v
        else:
            neighborhood_loc = self.ship.loc

        #collision avoidance for nearby objects
        #   this includes fixed bodies as well as dynamic ones
        collision_dv, approach_time, minimum_separation, distance_to_avoid_collision = self._avoid_collisions_dv(
                self.ship.sector, neighborhood_loc, self.neighborhood_radius, scaled_collision_margin,
                max_distance=max_distance,
                desired_direction=self.target_v)

        if util.both_almost_zero(collision_dv):
            self._accelerate_to(self.target_v, dt, force_recompute=True)
            self._desired_velocity = self.target_v

            # compute a time delta for our next desired velocity computation
            nts_low = 1/70.
            nts_high = 1.0
            nts_nnd_low = 2e3
            nts_nnd_high = 1e4
            nts_nnd = util.interpolate(nts_nnd_low, nts_low, nts_nnd_high, nts_high, self.nearest_neighbor_dist)

            nts_dist_low = 1e3
            nts_dist_high = 1e4
            nts_dist = util.interpolate(nts_dist_low, nts_low, nts_dist_high, nts_high, distance)
            nts = min(min(nts_nnd, nts_dist), nts_high)
        else:
            # if we're over max speed, let's slow down in addition to avoiding
            # collision
            v_mag = util.magnitude(v[0], v[1])
            if v_mag > max_speed:
                v = v / v_mag * max_speed
            self._desired_velocity = v + collision_dv
            desired_mag = util.magnitude(*self._desired_velocity)
            if desired_mag > max_speed:
                self._desired_velocity = self._desired_velocity/desired_mag * max_speed
            self._accelerate_to(self._desired_velocity, dt, force_recompute=True)

            nts = 1/70.

        self._next_compute_ts = self.gamestate.timestamp + nts

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
        self._accelerate_to(ZERO_VECTOR, dt)
        return

        # avoid collisions while we're waiting
        # but only if those collisions are really imminent
        # we want to have enough time to get away
        collision_dv, approach_time, min_separation, distance=self._avoid_collisions_dv(self.ship.sector, self.ship.loc, 5e3, 3e2)
        if distance > 0:
            t = approach_time - rotation_time(2*np.pi, self.ship.angular_velocity, self.ship.max_angular_acceleration(), self.safety_factor)
            if t < 0 or distance > 1/2 * self.ship.max_acceleration()*t**2 / self.safety_factor:
                self._accelerate_to(collision_dv, dt)
                return
        #if np.allclose(self.ship.velocity, ZERO_VECTOR):
        if util.both_almost_zero(self.ship.velocity):
            if self.ship.phys.angular_velocity != 0.:
                t = self.ship.moment * -1 * self.ship.angular_velocity / dt
                if util.isclose(t, 0):
                    self.ship.phys.angular_velocity = 0.
                else:
                    self.ship.apply_torque(np.clip(t, -1*self.ship.max_torque, self.ship.max_torque))
        else:
            self._accelerate_to(ZERO_VECTOR, dt)
