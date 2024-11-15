""" Basic orders, moving around, waiting, etc. """

from __future__ import annotations

from typing import Optional, Any, Union, Tuple
import math

import numpy as np
import numpy.typing as npt
import cymunk # type: ignore

from stellarpunk import util, core

from .steering import ANGLE_EPS, VELOCITY_EPS, ZERO_VECTOR, CYZERO_VECTOR, AbstractSteeringOrder
from stellarpunk.orders import collision

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
            # schedule again to get cleaned up on next tick
            self.ship.apply_torque(0., False)
            self.gamestate.schedule_order_immediate(self)
        else:
            self.ship.apply_torque(np.clip(t, -9000, 9000), True)

            self.gamestate.schedule_order_immediate(self)

class RotateOrder(AbstractSteeringOrder):
    def __init__(self, target_angle: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.target_angle = util.normalize_angle(target_angle)

    def is_complete(self) -> bool:
        return self.ship.angular_velocity == 0 and util.isclose(util.normalize_angle(self.ship.angle), self.target_angle)

    def act(self, dt: float) -> None:
        #period = self._rotate_to(self.target_angle, dt)
        period = collision.rotate_to(self.ship.phys, self.target_angle, dt, self.ship.max_torque)
        if period < np.inf:
            # don't need to wake up again until the rotation is complete
            self.gamestate.schedule_order(self.gamestate.timestamp + period/self.safety_factor, self)
        else:
            # schedule ourselves to get cleaned up
            self.gamestate.schedule_order_immediate(self)

class KillVelocityOrder(AbstractSteeringOrder):
    """ Applies thrust and torque to zero out velocity and angular velocity.

    Rotates to opposite direction of current velocity and applies thrust to
    zero out velocity. """

    @staticmethod
    def in_motion(ship: core.Ship) -> bool:
        return (
            ship.angular_velocity != 0. or
            ship.phys.torque != 0. or
            not np.allclose(ship.velocity, ZERO_VECTOR) or
            ship.phys.force != CYZERO_VECTOR
        )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def is_complete(self) -> bool:
        return not KillVelocityOrder.in_motion(self.ship)

    def act(self, dt: float) -> None:
        period = collision.accelerate_to(
                self.ship.phys, cymunk.Vec2d(0,0), dt,
                self.ship.max_speed(), self.ship.max_torque,
                self.ship.max_thrust, self.ship.max_fine_thrust,
                self.ship.sensor_settings)
        # don't need to wake up again until the acceleration is complete
        if period < math.inf:
            self.gamestate.schedule_order(self.gamestate.timestamp + period, self)
        else:
            self.gamestate.schedule_order_immediate(self)
            assert self.is_complete()

class GoToLocation(AbstractSteeringOrder):
    class NoEmptyArrivalError(Exception):
        pass

    @staticmethod
    def goto_entity(
            entity:core.SectorEntity,
            ship:core.Ship,
            gamestate:core.Gamestate,
            surface_distance:float=2e3,
            collision_margin:float=7e2,
            empty_arrival:bool=False,
            observer:Optional[core.OrderObserver]=None) -> GoToLocation:
        """ Makes a GoToLocation order to get near a target entity.

        entity: the entity to get near
        ship: the ship that will execute the order
        gamestate: gamestate
        surface_distance: target distance from surface of the entity (max dist)
        collision_margin: min distance to stay away from the entity (min dist)
        empty_arrival: require an empty arrival area or raise (vs best effort)

        returns a GoToLocation order matching those parameters
        """

        if entity.sector is None:
            raise ValueError("entity {entity} is not in any sector")

        # pick a point on this side of the target entity that is midway in the
        # "viable" arrival band: the space between the collision margin and
        # surface distance away from the radius
        tries = 0
        max_tries = 20
        target_loc = ZERO_VECTOR
        target_arrival_distance = 0.
        while tries < max_tries:
            _, angle = util.cartesian_to_polar(*(ship.loc - entity.loc))
            target_angle = angle + gamestate.random.uniform(-math.pi/1.5, math.pi/1.5)
            target_arrival_distance = (surface_distance - collision_margin)/2
            target_loc = entity.loc + util.polar_to_cartesian(entity.radius + collision_margin + target_arrival_distance, target_angle)

            # check if there's other stuff nearby this point
            # note: this isn't perfect since these entities might be transient,
            # but this is best effort
            if next(entity.sector.spatial_point(target_loc, collision_margin), None) == None:
                break
            tries += 1

        if empty_arrival and tries >= max_tries:
            raise GoToLocation.NoEmptyArrivalError()

        return GoToLocation(
                target_loc, ship, gamestate,
                arrival_distance=target_arrival_distance,
                min_distance=0.,
                target_sector=entity.sector,
                observer=observer)

    @staticmethod
    def compute_eta(ship:core.Ship, target_location:Union[npt.NDArray[np.float64], Tuple[float, float]], safety_factor:float=2.0, starting_loc:Optional[npt.NDArray[np.float64]]=None) -> float:
        if starting_loc is None:
            starting_loc = ship.loc
        course = target_location - starting_loc
        distance, target_angle = util.cartesian_to_polar(course[0], course[1])
        rotate_towards = collision.rotation_time(util.normalize_angle(target_angle-ship.angle, shortest=True), ship.angular_velocity, ship.max_angular_acceleration())

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

        rotate_away = collision.rotation_time(np.pi, 0, ship.max_angular_acceleration())
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
            **kwargs: Any) -> None:
        """ Creates an order to go to a specific location.

        The order will arrivate at the location approximately and with zero
        velocity.

        target_sector the sector with the location (only used for error checking and aborting the order)
        target_location the location
        arrival_distance how close to the location to arrive
        """

        super().__init__(*args, **kwargs)
        if target_sector is None:
            if self.ship.sector is None:
                raise ValueError(f'no target sector provided and ship {self.ship} is not in any sector')
            target_sector = self.ship.sector

        self.target_sector = target_sector
        self._target_location = cymunk.Vec2d(*target_location)

        self.target_v = CYZERO_VECTOR
        self.arrival_distance = arrival_distance
        if min_distance is None:
            min_distance = self.arrival_distance * 0.9
        self.min_distance = min_distance

        self.neighbor_analyzer.set_location_params(
                self._target_location, arrival_distance, min_distance
        )

        self.cannot_stop = False

        self.distance_estimate = 0.

        # after taking into account collision avoidance
        self._desired_velocity = CYZERO_VECTOR
        # the next time we should do a costly computation of desired velocity
        self._next_compute_ts = 0.
        self._nts = 0.

    def set_target_location(self, target_location:cymunk.Vec2d) -> None:
        """ For testing support """
        self._target_location = target_location
        self.neighbor_analyzer.set_location_params(
                self._target_location, self.arrival_distance, self.min_distance
        )

    def to_history(self) -> dict:
        data = super().to_history()
        data["t_loc"] = (self._target_location[0], self._target_location[1])
        data["ad"] = self.arrival_distance
        data["md"] = self.min_distance
        data["t_v"] = (self.target_v[0], self.target_v[1])
        data["cs"] = self.cannot_stop
        data["scm"] = self.neighbor_analyzer.get_margin()
        data["_ncts"] = self._next_compute_ts
        data["_dv"] = [self._desired_velocity[0], self._desired_velocity[1]]

        return data

    def __str__(self) -> str:
        return f'GoToLocation: {self._target_location} nts:{self._nts}s'

    def estimate_eta(self) -> float:
        return GoToLocation.compute_eta(self.ship, np.array(self._target_location), self.safety_factor)

    def is_complete(self) -> bool:
        # computing this is expensive, so don't if we can avoid it
        if self.distance_estimate > self.arrival_distance*5:
            return False
        else:
            if self._target_location.get_distance(self.ship.phys.position) < self.arrival_distance + VELOCITY_EPS and util.isclose(self.ship.phys.speed, 0.):
                #assert not self.ship._persistent_force
                self.ship.apply_force(ZERO_VECTOR, False)
                #assert not self.ship._persistent_torque
                self.ship.apply_torque(0., False)
                return True
            else:
                return False

    def _begin(self) -> None:
        self.init_eta = self.estimate_eta()

    def act(self, dt: float) -> None:
        # make sure we're in the right sector
        if self.ship.sector != self.target_sector:
            raise ValueError(f'{self.ship} in {self.ship.sector} instead of target {self.target_sector}')

        # check if it's time for us to do a careful calculation or a simple one
        if self.gamestate.timestamp < self._next_compute_ts:
            force_recompute = self.distance_estimate < self.arrival_distance * 5
            continue_time = collision.accelerate_to(self.ship.phys, self._desired_velocity, dt, self.ship.max_speed(), self.ship.max_torque, self.ship.max_thrust, self.ship.max_fine_thrust, self.ship.sensor_settings)
            self.gamestate.counters[core.Counters.GOTO_ACT_FAST] += 1
            self.gamestate.counters[core.Counters.GOTO_ACT_FAST_CT] += continue_time

            # don't need to wake up again until the acceleration is complete
            next_ts = min(self._next_compute_ts, self.gamestate.timestamp + continue_time)
            self._nts = next_ts - self.gamestate.timestamp
            self.gamestate.schedule_order(next_ts, self)
            return

        self.gamestate.counters[core.Counters.GOTO_ACT_SLOW] += 1

        # essentially the arrival steering behavior but with some added
        # obstacle avoidance

        self.neighbor_analyzer.prepare_analysis(self.gamestate.timestamp)

        prev_cannot_avoid_collision = self.cannot_avoid_collision

        self.target_v, distance, self.distance_estimate, self.cannot_stop, delta_v = self.neighbor_analyzer.find_target_v(
                dt, self.safety_factor
        )

        if self.cannot_stop:
            max_distance = np.inf
            self.logger.debug(f'cannot stop in time distance: {distance} v: {self.ship.velocity}')
        else:
            max_distance = distance-self.min_distance

        #collision avoidance for nearby objects
        #   this includes fixed bodies as well as dynamic ones
        collision_dv, approach_time = self._avoid_collisions_dv(
                self.ship.sector,
                max_distance=max_distance,
                desired_direction=self.target_v)

        nts_low = 6/70.
        nts_high = 2.3

        # if there's no collision diversion OR we're at the destination and can
        # quickly (1 sec) come to a stop
        if util.both_almost_zero(collision_dv):
            continue_time = collision.accelerate_to(self.ship.phys, self._desired_velocity, dt, self.ship.max_speed(), self.ship.max_torque, self.ship.max_thrust, self.ship.max_fine_thrust, self.ship.sensor_settings)
            self.gamestate.counters[core.Counters.GOTO_THREAT_NO] += 1
            self.gamestate.counters[core.Counters.GOTO_THREAT_NO_CT] += continue_time
            self._desired_velocity = self.target_v

            # if we previously could not avoid collision and now we have no
            # collision avoidance, it's possible we shrunk the margin and
            # cleared margin histeresis, so let's check again ASAP with a
            # normal margin
            if prev_cannot_avoid_collision:
                nts = 1/70
            elif distance < self.arrival_distance * 2.5:
                nts = nts_low
            else:
                # compute a time delta for our next desired velocity computation
                nts_nnd_low = 2e3#5e2
                nts_nnd_high = 1e4#5e3
                nts_nnd = util.interpolate(nts_nnd_low, nts_low, nts_nnd_high, nts_high, self.nearest_neighbor_dist)

                nts_dist_low = 1e3
                nts_dist_high = 1e4
                nts_dist = util.interpolate(nts_dist_low, nts_low, nts_dist_high, nts_high, distance)

                nts = max(nts_low, min(min(nts_nnd, nts_dist), nts_high))
        else:
            # if we're over max speed, let's slow down in addition to avoiding
            # collision
            #TODO: this code has the benefit of trying to slow down in addition
            # to trying to avoid collision. however, it also means we don't
            # apply the result of the collision avoidance. not sure the
            # tradeoff or alternative there...
            #v_mag = util.magnitude(v[0], v[1])
            #if v_mag > max_speed:
            #    v = v / v_mag * max_speed
            self._desired_velocity = self.ship.phys.velocity + collision_dv
            #TODO: see todo above about slowing down
            #desired_mag = util.magnitude(*self._desired_velocity)
            #if desired_mag > max_speed:
            #    self._desired_velocity = self._desired_velocity/desired_mag * max_speed
            continue_time = collision.accelerate_to(self.ship.phys, self._desired_velocity, dt, self.ship.max_speed(), self.ship.max_torque, self.ship.max_thrust, self.ship.max_fine_thrust, self.ship.sensor_settings)
            self.gamestate.counters[core.Counters.GOTO_THREAT_YES] += 1
            self.gamestate.counters[core.Counters.GOTO_THREAT_YES_CT] += continue_time

            if self.neighbor_analyzer.get_cannot_avoid_collision():
                nts = 1/70
            else:
                nts = nts_low

        self._next_compute_ts = self.gamestate.timestamp + nts
        self._nts = min(nts, continue_time)
        next_ts = self.gamestate.timestamp + self._nts
        self.gamestate.schedule_order(next_ts, self)

        return

class EvadeOrder(AbstractSteeringOrder, core.SectorEntityObserver):
    def __init__(self, target_image:core.AbstractSensorImage, *args: Any, escape_distance:float=np.inf, max_thrust:Optional[float]=None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        assert self.ship.sector
        self.target = target_image

        self.intercept_location = np.array((0.0, 0.0))
        self.intercept_time = 0.0

        # volume-weighted exponential average of target velocity
        # "volume" here is time between measurements
        # estimate of target velocity, weighted by amount of time since the
        # last estimate
        #self.est_tv_velocity = np.array((0.0, 0.0))
        #self.est_tv_volume = 0.
        #self.last_est_tv = 0.

        self.escape_distance = escape_distance
        if max_thrust:
            self.max_thrust = max(0.0, min(max_thrust, self.ship.max_thrust))
            self.max_fine_thrust = max(0.0, min(max_thrust, self.ship.max_fine_thrust))
        else:
            self.max_thrust = self.ship.max_thrust
            self.max_fine_thrust = self.ship.max_fine_thrust

    def __str__(self) -> str:
        return f'Evade: {self.target.identity.short_id} dist: {util.human_distance(float(np.linalg.norm(self.target.loc-self.ship.loc)))} escape: {util.human_distance(self.escape_distance)}'

    #def _begin(self) -> None:
    #    self.est_tv_velocity = self.target.velocity
    #    self.est_tv_volume = self.gamestate.dt
    #    self.last_est_tv = self.gamestate.timestamp

    def is_complete(self) -> bool:
        # TODO: when do we stop evading?
        return self.completed_at > 0 or float(np.linalg.norm(self.target.loc - self.ship.loc)) > self.escape_distance or not self.target.is_active()

    def act(self, dt:float) -> None:
        assert self.ship.sector
        self.target.update()

        # have a high max speed, but not crazy since we still need to avoid
        # collisions and recover from the evasion once we're done
        max_speed = self.ship.max_thrust / self.ship.mass * 90

        # OPTION A:
        # plot a course away from intercept location
        #target_velocity, _, self.intercept_time, self.intercept_location = collision.find_intercept_v(
        #        self.ship.phys,
        #        self.target.phys,
        #        self.target.radius/5,
        #        self.ship.max_acceleration(),
        #        self.ship.max_angular_acceleration(),
        #        self.ship.max_speed(),
        #        dt,
        #        self.ship.max_speed())
        # take intercept course and invert it
        #target_velocity = target_velocity * -1

        # OPTION B:
        # estimate our closest approach and flee from there
        rel_dist, self.intercept_time, rel_pos, rel_vel, _, self.intercept_location, _ = self.neighbor_analyzer.analyze_neighbor(cymunk.Vec2d(self.target.loc), cymunk.Vec2d(self.target.velocity), 0.0, 1e6)
        rel_speed = np.linalg.norm(rel_vel)

        if rel_speed < 5 or self.intercept_time > 5:
            # plot a course away from that closest approach
            course = self.ship.loc - self.intercept_location
            course_mag = util.magnitude_sq(*course)
            if course_mag == 0:
                target_velocity = rel_pos / rel_dist * max_speed
                raise Exception()
            else:
                course = course / course_mag
                target_velocity = cymunk.Vec2d(course * max_speed)
        else:
            # OPTION C:
            # want a velocity perpendicular to current relative velocity
            #self.est_tv_velocity, self.est_tv_volume = util.update_vema(self.est_tv_velocity, self.est_tv_volume, 0.05, self.target.velocity, self.gamestate.timestamp - self.last_est_tv)
            #self.last_est_tv = self.gamestate.timestamp
            #rel_vel = np.array(self.est_tv_velocity/self.est_tv_volume - self.ship.velocity)
            # two perpendicular options, pick the one with the least rotation
            a = np.array((-rel_vel[1], rel_vel[0]))
            a = a / rel_speed
            b = np.array((rel_vel[1], -rel_vel[0]))
            b = b / rel_speed
            heading = np.array(util.polar_to_cartesian(1.0, self.ship.angle))
            if np.dot(a, heading) > np.dot(b, heading):
                target_velocity = cymunk.Vec2d(a * max_speed)
            else:
                target_velocity = cymunk.Vec2d(b * max_speed)

        collision_dv, approach_time = self._avoid_collisions_dv(
                self.ship.sector,
                desired_direction=target_velocity)

        if not util.both_almost_zero(collision_dv) or approach_time < self.intercept_time:
            target_velocity = self.ship.phys.velocity + collision_dv

        continue_time = collision.accelerate_to(self.ship.phys, target_velocity, dt, max_speed, self.ship.max_torque, self.max_thrust, self.max_fine_thrust, self.ship.sensor_settings)

        next_ts = self.gamestate.timestamp + min(1/10, continue_time)
        self.gamestate.schedule_order(next_ts, self)

class PursueOrder(AbstractSteeringOrder, core.SectorEntityObserver):
    """ Steer toward a collision with the target """
    def __init__(self, target_image:core.AbstractSensorImage, *args:Any, arrival_distance:float=0., avoid_collisions:bool=True, max_speed:Optional[float]=None, max_thrust:Optional[float]=None, final_speed:Optional[float]=None, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        assert self.ship.sector

        self.target = target_image

        self.intercept_location = np.array((0.0, 0.0))
        self.intercept_time = 0.0

        self.arrival_distance = arrival_distance

        self.avoid_collisions=avoid_collisions
        if max_speed:
            self.max_speed = max_speed
        else:
            self.max_speed = self.ship.max_speed()

        if max_thrust:
            self.max_thrust = max_thrust
            self.max_fine_thrust = min(max_thrust, self.ship.max_fine_thrust)
        else:
            self.max_thrust = self.ship.max_thrust
            self.max_fine_thrust = self.ship.max_fine_thrust

        if final_speed:
            self.final_speed = final_speed
        else:
            self.final_speed = self.max_thrust / self.ship.mass * 0.5

    def __str__(self) -> str:
        return f'PursueOrder: {self.target.identity.short_id} dist: {util.human_distance(float(np.linalg.norm(self.target.loc-self.ship.loc)))} arrival: {util.human_distance(self.arrival_distance)}'

    def estimate_eta(self) -> float:
        return self.intercept_time

    def is_complete(self) -> bool:
        return self.completed_at > 0 or float(np.linalg.norm(self.target.loc - self.ship.loc)) < self.arrival_distance or not self.target.is_active()

    def act(self, dt:float) -> None:
        assert self.ship.sector
        self.target.update()
        # we won't get called if we're complete

        # interception algorithm:
        # assume target velocity is constant
        # solve problem in their frame of reference, so they appear stationary
        # try to "arrive" at their location (using arrival steering behavior)
        # whatever velocity we need in this frame of reference, add in their
        # velocity

        # this is the desired final speed we'll try to achieve at the intercept
        # this is on top of the target's velocity
        # a large value here reduces intercept time
        # a small value here makes it easy to correct at the last minute

        target_velocity, _, self.intercept_time, self.intercept_location = collision.find_intercept_v(
                self.ship.phys,
                cymunk.Vec2d(self.target.loc),
                cymunk.Vec2d(self.target.velocity),
                self.arrival_distance,
                self.ship.max_acceleration(),
                self.ship.max_angular_acceleration(),
                self.max_speed,
                dt,
                self.final_speed)

        if self.avoid_collisions:
            # avoid collisions
            collision_dv, approach_time = self._avoid_collisions_dv(
                    self.ship.sector,
                    desired_direction=target_velocity)

            if not util.both_almost_zero(collision_dv):
                target_velocity = cymunk.Vec2d(self.ship.velocity + collision_dv)

        continue_time = collision.accelerate_to(
                self.ship.phys,
                target_velocity,
                dt,
                self.max_speed,
                self.ship.max_torque,
                self.max_thrust,
                self.max_fine_thrust,
                self.ship.sensor_settings
        )

        next_ts = self.gamestate.timestamp + min(1/10, continue_time)

        self.gamestate.schedule_order(next_ts, self)

class WaitOrder(AbstractSteeringOrder):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.wait_wakeup_period = 10.

    def is_complete(self) -> bool:
        # wait forever
        return False

    def act(self, dt:float) -> None:
        if self.ship.sector is None:
            raise ValueError(f'{self.ship} not in any sector')
        period = collision.accelerate_to(
                self.ship.phys, cymunk.Vec2d(0,0), dt,
                self.ship.max_speed(), self.ship.max_torque,
                self.ship.max_thrust, self.ship.max_fine_thrust,
                self.ship.sensor_settings)

        if period < np.inf:
            # don't need to wake up again until the acceleration is complete
            self.gamestate.schedule_order(self.gamestate.timestamp + period, self)
            return
        else:
            self.gamestate.schedule_order(self.gamestate.timestamp + self.wait_wakeup_period, self, jitter=self.wait_wakeup_period/2)
            return

        """
        # avoid collisions while we're waiting
        # but only if those collisions are really imminent
        # we want to have enough time to get away
        collision_dv, approach_time, min_separation, distance=self._avoid_collisions_dv(self.ship.sector, self.ship.loc, 5e3, 3e2)
        if distance > 0:
            t = approach_time - collision.rotation_time(2*np.pi, self.ship.angular_velocity, self.ship.max_angular_acceleration())
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
        """
