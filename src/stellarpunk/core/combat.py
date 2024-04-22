""" Combat """

import enum
from typing import Optional, Tuple, Any, Set, List, Deque
import collections
import math
import uuid
import numpy as np
import numpy.typing as npt
import cymunk # type: ignore

from stellarpunk import util, core
from stellarpunk.orders import movement, collision
from stellarpunk.core.gamestate import Gamestate, ScheduledTask
from .sector_entity import SectorEntity, ObjectType
from .order import Effect

class TimedOrderTask(ScheduledTask, core.OrderObserver):
    @staticmethod
    def ttl_order(order:core.Order, ttl:float) -> "TimedOrderTask":
        tot = TimedOrderTask(order)
        Gamestate.gamestate.schedule_task(Gamestate.gamestate.timestamp + ttl, tot)
        return tot
    def __init__(self, order:core.Order) -> None:
        self.order = order
        order.observe(self)
    def order_cancel(self, order:core.Order) -> None:
        Gamestate.gamestate.unschedule_task(self)
    def order_complete(self, order:core.Order) -> None:
        Gamestate.gamestate.unschedule_task(self)
    def act(self) -> None:
        self.order.cancel_order()

def is_hostile(craft:core.SectorEntity, threat:core.SectorEntity) -> bool:
    #TODO: improve this logic
    hostile = False
    # is it a weapon (e.g. missile)
    if isinstance(threat, core.Missile):
        hostile = True
    # is it known to be hostile?
    # is it running with its transponder off?
    if not threat.sensor_settings.transponder:
        hostile = True
    # is it otherwise operating in a hostile manner (e.g. firing weapons)
    return hostile

class MissileOrder(movement.PursueOrder, core.CollisionObserver):
    """ Steer toward a collision with the target """

    @staticmethod
    def spawn_missile(ship:core.Ship, gamestate:Gamestate, target:Optional[SectorEntity]=None, target_image:Optional[core.AbstractSensorImage]=None, initial_velocity:float=100, spawn_distance_forward:float=100, spawn_radius:float=100, owner:Optional[SectorEntity]=None) -> core.Missile:
        assert ship.sector
        loc = gamestate.generator.gen_sector_location(ship.sector, center=ship.loc + util.polar_to_cartesian(spawn_distance_forward, ship.angle), occupied_radius=75, radius=spawn_radius, strict=True)
        v = util.polar_to_cartesian(initial_velocity, ship.angle) + ship.velocity
        new_entity = gamestate.generator.spawn_sector_entity(core.Missile, ship.sector, loc[0], loc[1], v=v, w=0.0)
        assert isinstance(new_entity, core.Missile)
        missile:core.Missile = new_entity
        missile.firer = owner
        if target:
            target_image = ship.sector.sensor_manager.target(target, missile)
        elif target_image:
            target_image = target_image.copy(missile)
        else:
            raise ValueError("one of target or target_image must be set")
        missile_order = MissileOrder(missile, gamestate, target_image=target_image)
        missile.prepend_order(missile_order)
        missile.sensor_settings.set_sensors(1.0)

        return missile

    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, avoid_collisions=False, **kwargs)
        self.avoid_collisions=False

        self.ttl = 240
        self.expiration_time = self.gamestate.timestamp + self.ttl

    def _begin(self) -> None:
        assert self.ship.sector
        self.expiration_time = self.gamestate.timestamp + self.ttl
        self.ship.sector.register_collision_observer(self.ship.entity_id, self)

    def _complete(self) -> None:
        self.logger.debug(f'completing missile after {self.gamestate.timestamp - self.started_at}s at distance {util.distance(self.ship.loc, self.target.loc)}m age {self.target.age}s')
        super()._complete()
        # ship might already have been removed from the sector
        # assume if that happens it got unregistered
        if self.ship.sector:
            self.ship.sector.unregister_collision_observer(self.ship.entity_id, self)
        self.gamestate.destroy_entity(self.ship)

    def _cancel(self) -> None:
        self._complete()

    def is_complete(self) -> bool:
        return self.completed_at > 0 or self.gamestate.timestamp > self.expiration_time or not self.target.is_active()

    def collision(self, missile:SectorEntity, target:SectorEntity, impulse:Tuple[float, float], ke:float) -> None:
        #if self.ship.firer == target:
        #    raise Exception()
        assert missile == self.ship
        if target.entity_id == self.target.target_entity_id:
            self.logger.debug(f'missile hit desired target {target} impulse: {impulse} ke: {ke}!')
        else:
            self.logger.debug(f'missile hit collateral target {target} impulse: {impulse} ke: {ke}!')

        self.complete_order()
        self.gamestate.destroy_entity(target)

class AttackOrder(movement.AbstractSteeringOrder):
    """ Objective is to destroy a target. """

    class State(enum.Enum):
        APPROACH = enum.auto()
        WITHDRAW = enum.auto()
        SHADOW = enum.auto()
        FIRE = enum.auto()
        AIM = enum.auto()
        LAST_LOCATION = enum.auto()
        SEARCH = enum.auto()
        GIVEUP = enum.auto()

    def __init__(self, target:core.SectorEntity, *args:Any, distance_min:float=7.5e4, distance_max:float=2e5, max_active_age:float=35, max_passive_age:float=15, search_distance:float=2.5e4, max_fire_rel_bearing:float=np.pi/8, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        assert self.ship.sector
        self.target = self.ship.sector.sensor_manager.target(target, self.ship)
        self.distance_min = distance_min
        self.distance_max = distance_max
        self.max_active_age = max_active_age
        self.max_passive_age = max_passive_age
        self.search_distance = search_distance
        self.max_fire_age = 3.0
        self.min_profile_to_threshold = 4.0
        self.fire_backoff_time = 1.0
        self.max_fire_rel_bearing = max_fire_rel_bearing
        self.ttl_order_time = 2.

        self.state = AttackOrder.State.APPROACH

        self.last_fire_ts = -3600.
        #TODO: fire period is a temporary hack to limit firing
        self.fire_period = 5.
        self.missiles_fired = 0

    def __str__(self) -> str:
        return f'Attack: {self.target.target_short_id()} state: {self.state} age: {self.target.age:.1f}s dist: {util.human_distance(float(np.linalg.norm(self.target.loc-self.ship.loc)))}'

    def _begin(self) -> None:
        super()._begin()
        self.logger.debug(f'beginning attack on {self.target.target_short_id()}')

    def is_complete(self) -> bool:
        return self.completed_at > 0. or not self.target.is_active()

    def _ttl_order(self, order:core.Order, ttl:Optional[float]=None) -> None:
        if ttl is None:
            ttl = self.ttl_order_time
        TimedOrderTask.ttl_order(order, ttl)
        self._add_child(order)

    def _do_last_location(self) -> None:
        self._ttl_order(movement.PursueOrder(self.ship, self.gamestate, target_image=self.target, arrival_distance=self.search_distance*0.8, max_speed=self.ship.max_speed()*5, final_speed=self.ship.max_thrust / self.ship.mass * 5.))

    def _do_search(self) -> None:
        #TODO: search, for now give up
        self.logger.debug(f'giving up search for target {self.target.target_short_id()}')
        self.state = AttackOrder.State.GIVEUP
        self.cancel_order()

    def _do_approach(self) -> None:
        self._ttl_order(movement.PursueOrder(self.ship, self.gamestate, target_image=self.target, arrival_distance=self.distance_max-0.2*(self.distance_max-self.distance_min), max_speed=self.ship.max_speed()*2.))

    def _do_withdraw(self) -> None:
        self._ttl_order(movement.EvadeOrder(self.ship, self.gamestate, target_image=self.target, escape_distance=self.distance_min+0.2*(self.distance_max-self.distance_min)))

    def _do_shadow(self, dt:float) -> None:
        assert self.ship.sector
        # make velocity parallel to target (i.e. want relative velocity to be zero)
        target_velocity = self.target.velocity

        # add velocity component to get to center of standoff range
        #away_vector = self.ship.loc - self.target.loc
        #dist = util.magnitude(*away_vector)
        #away_vector = away_vector / dist
        #dist = (self.distance_max + self.distance_min) / 2 - dist
        #away_vector *= dist / 5
        #mag = util.magnitude(*away_vector)
        #away_vector = away_vector / mag * min(mag, 0.2 * util.magnitude(*target_velocity))
        #target_velocity += away_vector

        # avoid collisions
        collision_dv, approach_time = self._avoid_collisions_dv(
                self.ship.sector,
                desired_direction=cymunk.Vec2d(target_velocity))

        if not util.both_almost_zero(collision_dv):
            target_velocity = self.ship.velocity + collision_dv

        # TODO: choose a max thrust appropriate for desired sensor profile

        shadow_time = collision.accelerate_to(self.ship.phys, cymunk.Vec2d(target_velocity), dt, self.ship.max_speed(), self.ship.max_torque, self.ship.max_thrust, self.ship.max_fine_thrust, self.ship.sensor_settings)
        self.gamestate.schedule_order(self.gamestate.timestamp + min(shadow_time, 1/10), self)

    def _do_fire(self) -> None:
        rel_bearing = util.normalize_angle(util.bearing(self.ship.loc, self.target.loc) - self.ship.angle, shortest=True)
        missile = MissileOrder.spawn_missile(self.ship, self.gamestate, target_image=self.target, owner=self.ship)
        self.logger.debug(f'firing missile {missile} at distance {util.distance(self.ship.loc, self.target.loc)}m {rel_bearing=} with ptr {self.target.profile / self.ship.sensor_settings.max_threshold}')

        assert util.distance(missile.loc, self.target.loc) < util.distance(self.ship.loc, self.target.loc)
        self.missiles_fired += 1
        self.last_fire_ts = self.gamestate.timestamp

        # get away from the missile
        self._ttl_order(core.Order(self.ship, self.gamestate), ttl=self.fire_backoff_time)#movement.EvadeOrder(self.ship, self.gamestate, target=missile, escape_distance=1e5), ttl=self.fire_backoff_time)

    def _do_aim(self, dt:float) -> None:
        # get in close enough that a launched missile will be able to lock on to the target

        if self.target.profile / self.ship.sensor_settings.max_threshold < self.min_profile_to_threshold:
            self._ttl_order(movement.PursueOrder(self.ship, self.gamestate, target_image=self.target, arrival_distance=1e3, max_speed=self.ship.max_speed()*5, final_speed=self.ship.max_thrust / self.ship.mass * 5.))
            return

        bearing = util.bearing(self.ship.loc, self.target.loc)
        rotate_time = collision.rotate_to(self.ship.phys, bearing, dt, self.ship.max_torque)
        self.gamestate.schedule_order(self.gamestate.timestamp + min(rotate_time, 1/10), self)

    def _set_sensors(self) -> None:
        assert self.ship.sector
        if self.target.age > self.max_passive_age:
            self.ship.sensor_settings.set_sensors(1.0)
        else:
            self.ship.sensor_settings.set_sensors(0.0)

    def _choose_state(self) -> "AttackOrder.State":
        if self.gamestate.timestamp - self.last_fire_ts < self.fire_backoff_time:
            # back off immediately following firing a missile to reduce risk of
            # colliding with our own missile
            return AttackOrder.State.WITHDRAW

        distance = np.linalg.norm(self.target.loc - self.ship.loc)
        if self.target.age > self.max_active_age:
            if distance > self.search_distance:
                return AttackOrder.State.LAST_LOCATION
            else:
                return AttackOrder.State.SEARCH

        if distance > self.distance_max:
            return AttackOrder.State.APPROACH
        #TODO: more complex logic around firing weapons, also some limit on
        # weapon use
        # if weapon systems not ready, ready weapons
        # if we've got enough confidence take shot, else gain confidence
        # only fire a missile if we're vaguely pointed toward the target
        if self.target.age < self.max_fire_age and self.gamestate.timestamp - self.last_fire_ts > self.fire_period:
            if abs(util.normalize_angle(util.bearing(self.ship.loc, self.target.loc) - self.ship.angle, shortest=True)) < self.max_fire_rel_bearing and self.target.profile / self.ship.sensor_settings.max_threshold > self.min_profile_to_threshold:
                return AttackOrder.State.FIRE
            else:
                return AttackOrder.State.AIM

        if distance < self.distance_min:
            return AttackOrder.State.WITHDRAW
        else:
            return AttackOrder.State.SHADOW

    def act(self, dt:float) -> None:
        assert self.ship.sector
        self.target.update()
        if util.magnitude(*self.target.velocity) > 15500:
            raise Exception()

        self._set_sensors()

        self.state = self._choose_state()

        if self.state == AttackOrder.State.LAST_LOCATION:
            self._do_last_location()
        elif self.state == AttackOrder.State.SEARCH:
            self._do_search()
        elif self.state == AttackOrder.State.APPROACH:
            self._do_approach()
        elif self.state == AttackOrder.State.WITHDRAW:
            self._do_withdraw()
        elif self.state == AttackOrder.State.SHADOW:
            self._do_shadow(dt)
        elif self.state == AttackOrder.State.FIRE:
            self._do_fire()
        elif self.state == AttackOrder.State.AIM:
            self._do_aim(dt)
        else:
            raise ValueError(f'unknown attack order state {self.state}')

class PointDefenseEffect(core.Effect, core.SectorEntityObserver, core.CollisionObserver):
    def __init__(self, craft:core.SectorEntity, target:core.AbstractSensorImage, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.craft = craft
        self.heading = self.craft.angle
        # phalanx has rof of 4500/min = 75/sec, muzzle velocity of 1100 m/s
        self.rof:float = 100.
        self.muzzle_velocity = 3000.
        self.projectile_ttl = 3.0
        self.target_max_age = 120.
        self.dispersion_angle = 0.0083
        self.target:core.AbstractSensorImage = target

        self.last_shot_ts = 0.
        self.projectiles:Deque[cymunk.shape] = collections.deque()
        self.projectiles_fired = 0
        self.targets_destroyed = 0

    def _expire_projectiles(self, expiration_time:float=math.inf) -> None:
        while len(self.projectiles) > 0 and self.projectiles[0].created_at < expiration_time:
            p = self.projectiles.popleft()
            p.unobserve(self)
            core.Gamestate.gamestate.destroy_entity(p)

    def _begin(self) -> None:
        self.last_shot_ts = core.Gamestate.gamestate.timestamp
        self.craft.observe(self)
        self.gamestate.schedule_effect_immediate(self, jitter=0.1)

    def _complete(self) -> None:
        self._expire_projectiles()

    def _cancel(self) -> None:
        self._complete()

    def is_complete(self) -> bool:
        if self.completed_at > 0.:
            return True
        if not self.target.is_active():
            self.logger.debug(f'threat {self.target.target_short_id()} no longer active')
            return True
        if self.target.age > self.target_max_age:
            self.logger.debug(f'lost threat {self.target.target_short_id()} after {self.target.age}s')
            return True
        return False

    def entity_destroyed(self, entity:core.SectorEntity) -> None:
        if isinstance(entity, core.Projectile):
            self.projectiles.remove(entity)
        elif self.craft == entity:
            self.cancel_effect()
        else:
            raise ValueError(f'got unexpected entity {entity}')

    def entity_migrated(self, entity:core.SectorEntity, from_sector:core.Sector, to_sector:core.Sector) -> None:
        if isinstance(entity, core.Projectile):
            entity.unobserve(self)
            self.projectiles.remove(entity)
        elif self.craft == entity:
            self.craft.unobserve(self)
            self.cancel_effect()
        else:
            raise ValueError(f'got unexpected entity {entity}')

    def collision(self, projectile:core.SectorEntity, target:core.SectorEntity, impulse:Tuple[float, float], ke:float) -> None:
        self.logger.debug(f'point defense from {self.craft} hit {target} with {projectile} at distance {util.distance(self.craft.loc, target.loc)}m age {self.gamestate.timestamp - projectile.created_at}s')

        self.gamestate.destroy_entity(target)
        self.gamestate.destroy_entity(projectile)
        self.targets_destroyed += 1

    def bbox(self) -> Tuple[float, float, float, float]:
        return (0., 0., 0., 0.)

    def act(self, dt:float) -> None:
        self.target.update()
        if self.is_complete():
            self.logger.debug(f'point defense is complete')
            self.cancel_effect()
            return

        # kill old projectiles
        expiration_time = core.Gamestate.gamestate.timestamp - self.projectile_ttl
        self._expire_projectiles(expiration_time)

        # aim point defense
        #TODO: what if there's another target that is likely to be hit by PD?
        intercept_time, intercept_loc, intercept_angle = collision.find_intercept_heading(cymunk.Vec2d(self.craft.loc), cymunk.Vec2d(self.craft.velocity), cymunk.Vec2d(self.target.loc), cymunk.Vec2d(self.target.velocity), self.muzzle_velocity)

        if intercept_time > 0 and intercept_time < self.projectile_ttl:
            # add some projectiles at rof
            # projectiles have trajectory for intercept given current information
            # projectiles should be physically simulated
            # projectiles should last for some fixed lifetime and then be removed
            num_shots = int((core.Gamestate.gamestate.timestamp - self.last_shot_ts) * self.rof)

            next_index = None
            for i in range(num_shots):
                angle = intercept_angle + core.Gamestate.gamestate.random.uniform(-self.dispersion_angle/2., self.dispersion_angle/2.)
                loc, next_index = core.Gamestate.gamestate.generator.gen_projectile_location(self.craft.loc + util.polar_to_cartesian(self.craft.radius+5, angle), next_index)
                v = util.polar_to_cartesian(self.muzzle_velocity, angle) + self.craft.velocity
                new_entity = core.Gamestate.gamestate.generator.spawn_sector_entity(core.Projectile, self.sector, loc[0], loc[1], v=v, w=0.0)
                new_entity.observe(self)
                self.sector.register_collision_observer(new_entity.entity_id, self)

                self.projectiles.append(new_entity)
                self.projectiles_fired += 1
        self.last_shot_ts = core.Gamestate.gamestate.timestamp
        self.gamestate.schedule_effect(core.Gamestate.gamestate.timestamp + 1/10., self)

class HuntOrder(core.Order):
    def __init__(self, target_id:uuid.UUID, *args:Any, start_loc:Optional[npt.NDArray[np.float64]], **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.target_id = target_id
        if start_loc is None:
            self.start_loc = self.ship.loc
        else:
            self.start_loc = start_loc

        self.ttl_order_time = 5.

    def _ttl_order(self, order:core.Order, ttl:Optional[float]=None) -> None:
        if ttl is None:
            ttl = self.ttl_order_time
        TimedOrderTask.ttl_order(order, ttl)
        self._add_child(order)

    def _scan_target(self) -> Optional[core.SectorEntity]:
        assert self.ship.sector
        # note: the model is that this ship doesn't know if the target is in
        # sector or not. we're reaching inside sector.entities for convenience
        if self.target_id in self.ship.sector.entities:
            target = self.ship.sector.entities[self.target_id]
            if self.ship.sector.sensor_manager.detected(target, self.ship):
                return target
        return None

    def act(self, dt:float) -> None:
        # alternate between traveling to a search point and scanning for the
        # target
        target = self._scan_target()
        if target:
            self.attack_order = AttackOrder(target, self.ship, self.gamestate)
            self._add_child(self.attack_order)
            return
        else:
            # choose a search location
            loc = self.start_loc
            # go there for a bit
            self._ttl_order(movement.GoToLocation(loc, self.ship, self.gamestate))

class FleeOrder(core.Order, core.SectorEntityObserver):
    """ Keeps track of threats and flees from them until "safe" """
    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.threats:Set[core.AbstractSensorImage] = set()
        self.closest_threat:Optional[core.AbstractSensorImage] = None
        self.threat_ids:Set[uuid.UUID] = set()
        self.threat_ttl = 120.
        self.ttl_order_time = 5.

        self.point_defense:Optional[PointDefenseEffect] = None
        self.point_defense_count = 0

        self.last_target_ts = 0.
        self.max_thrust = self.ship.max_thrust

    def _ttl_order(self, order:core.Order) -> None:
        TimedOrderTask.ttl_order(order, self.ttl_order_time)
        self._add_child(order)

    def add_threat(self, threat:core.AbstractSensorImage) -> None:
        if threat.target_entity_id in self.threat_ids:
            return
        self.logger.debug(f'adding threat {threat}')
        self.threats.add(threat)
        self.threat_ids.add(threat.target_entity_id)

    def entity_targeted(self, craft:core.SectorEntity, threat:core.SectorEntity) -> None:
        assert craft == self.ship
        assert self.ship.sector
        if threat.entity_id in self.threat_ids:
            self.last_target_ts = self.gamestate.timestamp
            return

        # no need to worry about non-hostile threats
        if not is_hostile(self.ship, threat):
            return

        self.last_target_ts = self.gamestate.timestamp
        self.add_threat(self.ship.sector.sensor_manager.target(threat, self.ship))

    def _begin(self) -> None:
        self.ship.sensor_settings.set_sensors(0.0)
        self.ship.observe(self)
        self.last_target_ts = self.gamestate.timestamp

    def _complete(self) -> None:
        if self.point_defense:
            self.point_defense.cancel_effect()
        self.ship.sensor_settings.set_sensors(1.0)
        self.ship.unobserve(self)

    def _cancel(self) -> None:
        if self.point_defense:
            self.point_defense.cancel_effect()
        self.ship.sensor_settings.set_sensors(1.0)
        self.ship.unobserve(self)

    def is_complete(self) -> bool:
        if self.completed_at > 0:
            return True
        elif len(self.threats) == 0:
            return True
        return False

    def _find_closest_threat(self) -> Optional[core.AbstractSensorImage]:
        # first remove eliminated/expired threats
        closest_dist = np.inf
        self.closest_threat = None
        dead_threats:List[core.AbstractSensorImage] = []
        for t in self.threats:
            t.update()
            if not t.is_active() or t.age > self.threat_ttl:
                self.logger.debug(f'dropping threat {t}')
                dead_threats.append(t)
            else:
                dist = util.distance(t.loc, self.ship.loc)
                if dist < closest_dist:
                    self.closest_threat = t
                    closest_dist = dist
        for t in dead_threats:
            self.threats.remove(t)
            self.threat_ids.remove(t.target_entity_id)

        return self.closest_threat

    def _choose_thrust(self) -> float:
        assert self.ship.sector
        assert self.closest_threat
        # determine thrust we want to use to evade
        # if we've got low pressure to get away fast, we want to minimize
        # thrust in order to minimize our sensor profile

        # ideally we'd choose a thrust that keeps us hidden under active
        # sensors but at least we should choose a thrust that requires active
        # sensors
        max_active_thrust = self.ship.max_thrust
        max_passive_thrust = self.ship.max_thrust
        for threat in self.threats:
            dist_sq = util.distance_sq(self.ship.loc, threat.loc)
            active_threshold_thrust = self.ship.sector.sensor_manager.compute_thrust_for_profile(self.ship, dist_sq, self.ship.sensor_settings.effective_threshold(self.ship.sensor_settings.max_sensor_power))
            passive_threshold_thrust = self.ship.sector.sensor_manager.compute_thrust_for_sensor_power(self.ship, dist_sq, 0.0)
            if active_threshold_thrust < max_active_thrust:
                max_active_thrust = active_threshold_thrust
            if passive_threshold_thrust < max_passive_thrust:
                max_passive_thrust = passive_threshold_thrust

        if max_active_thrust > 0.:
            return max_active_thrust * 0.8
        elif max_passive_thrust > 0.:
            return max_passive_thrust * 0.8
        else:
            return self.ship.max_thrust

    def act(self, dt:float) -> None:
        self.logger.debug(f'act at {self.gamestate.timestamp}')
        self._find_closest_threat()
        if self.closest_threat is None:
            self.logger.debug(f'no more threats, completing flee order')
            self.complete_order()
            return

        # aim point defense at the closest threat
        if not self.point_defense or self.point_defense.is_complete():
            assert self.ship.sector
            self.logger.debug(f'initiating point defense at {self.closest_threat}')
            self.point_defense = PointDefenseEffect(self.ship, self.closest_threat, self.ship.sector, self.gamestate)
            self.point_defense_count += 1
            self.ship.sector.add_effect(self.point_defense)
        else:
            self.point_defense.target = self.closest_threat

        #TODO: better logic on how to evade all the threats simultaneously
        # evade closest threat

        self.max_thrust = self._choose_thrust()
        self._ttl_order(movement.EvadeOrder(self.ship, self.gamestate, target_image=self.closest_threat, max_thrust=self.max_thrust))
