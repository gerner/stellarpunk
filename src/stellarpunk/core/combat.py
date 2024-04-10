""" Combat """

import enum
from typing import Optional, Tuple, Any, Set, List, Deque
import collections
import math
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

class MissileOrder(movement.PursueOrder, core.CollisionObserver):
    """ Steer toward a collision with the target """

    @staticmethod
    def spawn_missile(ship:core.Ship, gamestate:Gamestate, target:Optional[SectorEntity]=None, target_image:Optional[core.AbstractSensorImage]=None, initial_velocity:float=100, spawn_distance_forward:float=100, spawn_radius:float=100) -> core.Missile:
        assert ship.sector
        loc = gamestate.generator.gen_sector_location(ship.sector, center=ship.loc + util.polar_to_cartesian(spawn_distance_forward, ship.angle), occupied_radius=75, radius=spawn_radius)
        v = util.polar_to_cartesian(initial_velocity, ship.angle) + ship.velocity
        new_entity = gamestate.generator.spawn_sector_entity(core.Missile, ship.sector, loc[0], loc[1], v=v, w=0.0)
        assert isinstance(new_entity, core.Missile)
        missile:core.Missile = new_entity
        if target:
            target_image = ship.sector.sensor_manager.target(target, missile)
        elif target_image:
            target_image = target_image.copy(missile)
        else:
            raise ValueError("one of target or target_image must be set")
        missile_order = MissileOrder(missile, gamestate, target_image=target_image)
        missile.prepend_order(missile_order)

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
        super()._complete()
        # ship might already have been removed from the sector
        # assume if that happens it got unregistered
        if self.ship.sector:
            self.ship.sector.unregister_collision_observer(self.ship.entity_id, self)
        self.gamestate.destroy_entity(self.ship)

    def _cancel(self) -> None:
        self._complete()

    def is_complete(self) -> bool:
        return self.completed_at > 0 or self.gamestate.timestamp > self.expiration_time
    def collision(self, missile:SectorEntity, target:SectorEntity, impulse:Tuple[float, float], ke:float) -> None:
        assert missile == self.ship
        self.logger.debug(f'missile {self.ship} hit {target} impulse: {impulse} ke: {ke}!')

        self.cancel_order()
        self.gamestate.destroy_entity(target)

class AttackOrder(movement.AbstractSteeringOrder):
    """ Objective is to destroy a target. """

    class State(enum.Enum):
        APPROACH = enum.auto()
        WITHDRAW = enum.auto()
        SHADOW = enum.auto()
        FIRE = enum.auto()
        LAST_LOCATION = enum.auto()
        SEARCH = enum.auto()

    def __init__(self, target:core.SectorEntity, *args:Any, distance_min:float=2.5e5, distance_max:float=5e5, max_active_age:float=35, max_passive_age:float=30, search_distance:float=5e4, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        assert self.ship.sector
        self.target = self.ship.sector.sensor_manager.target(target, self.ship)
        self.distance_min = distance_min
        self.distance_max = distance_max
        self.max_active_age = max_active_age
        self.max_passive_age = max_passive_age
        self.search_distance = search_distance
        self.ttl_order_time = 5.

        self.state = AttackOrder.State.APPROACH

        #TODO: this is a temporary hack to get something reasonable for firing
        self.last_fire_ts = 0.
        self.fire_period = 15.

    def __str__(self) -> str:
        return f'Attack: {self.target.short_id()} state: {self.state} age: {self.target.age:.1f}s dist: {util.human_distance(float(np.linalg.norm(self.target.loc-self.ship.loc)))}'

    def is_complete(self) -> bool:
        return self.completed_at > 0. or not self.target.is_active()

    def _ttl_order(self, order:core.Order) -> None:
        TimedOrderTask.ttl_order(order, self.ttl_order_time)
        self._add_child(order)

    def _do_last_location(self) -> None:
        self._ttl_order(movement.PursueOrder(self.ship, self.gamestate, target_image=self.target, arrival_distance=self.search_distance*0.8, max_speed=self.ship.max_speed()*5, final_speed=self.ship.max_thrust / self.ship.mass * 5.))

    def _do_search(self) -> None:
        #TODO: search, for now give up
        self.cancel_order()

    def _do_approach(self) -> None:
        self._ttl_order(movement.PursueOrder(self.ship, self.gamestate, target_image=self.target, arrival_distance=self.distance_max-0.2*(self.distance_max-self.distance_min), max_speed=self.ship.max_speed()*2.))

    def _do_withdraw(self) -> None:
        self._ttl_order(movement.EvadeOrder(self.ship, self.gamestate, target_image=self.target, escape_distance=self.distance_min+0.2*(self.distance_max-self.distance_min)))

    def _do_shadow(self, dt:float) -> None:
        assert self.ship.sector
        # make velocity parallel to target (i.e. want relative velocity to be zero)
        target_velocity = self.target.velocity

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
        MissileOrder.spawn_missile(self.ship, self.gamestate, target_image=self.target)
        self.last_fire_ts = self.gamestate.timestamp

        self.gamestate.schedule_order_immediate(self)

    def _set_sensors(self) -> None:
        assert self.ship.sector
        if self.target.age > self.max_passive_age:
            self.ship.sensor_settings.set_sensors(1.0)
        else:
            self.ship.sensor_settings.set_sensors(0.0)

    def _choose_state(self) -> "AttackOrder.State":
        distance = np.linalg.norm(self.target.loc - self.ship.loc)
        if self.target.age > self.max_active_age:
            if distance > self.search_distance:
                return AttackOrder.State.LAST_LOCATION
            else:
                return AttackOrder.State.SEARCH

        if distance > self.distance_max:
            return AttackOrder.State.APPROACH
        elif distance < self.distance_min:
            return AttackOrder.State.WITHDRAW

        # TODO: attacking
        # if weapon systems not ready, ready weapons
        # if we've got enough confidence take shot, else gain confidence

        if self.gamestate.timestamp - self.last_fire_ts > self.fire_period:
            return AttackOrder.State.FIRE

        return AttackOrder.State.SHADOW

    def act(self, dt:float) -> None:
        assert self.ship.sector
        self.target.update()

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
        else:
            raise ValueError(f'unknown attack order state {self.state}')

class PointDefenseEffect(core.Effect, core.SectorEntityObserver, core.CollisionObserver):
    def __init__(self, craft:core.SectorEntity, target:core.AbstractSensorImage, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.craft = craft
        self.heading = self.craft.angle
        # phalanx has rof of 4500/min = 75/sec, muzzle velocity of 1100 m/s
        self.rof:float = 75.
        self.muzzle_velocity = 3000.
        self.projectile_ttl = 5.
        self.target_max_age = 120.
        self.dispersion_angle = 0.0083
        self.target:core.AbstractSensorImage = target

        self.last_shot_ts = 0.
        self.projectiles:Deque[cymunk.shape] = collections.deque()

    def _begin(self) -> None:
        self.last_shot_ts = core.Gamestate.gamestate.timestamp
        self.craft.observe(self)
        self.gamestate.schedule_effect_immediate(self, jitter=0.1)

    def _complete(self) -> None:
        while len(self.projectiles) > 0:
            p = self.projectiles.popleft()
            p.unobserve(self)
            core.Gamestate.gamestate.destroy_entity(p)

    def _cancel(self) -> None:
        self._complete()

    def is_complete(self) -> bool:
        return self.completed_at > 0. or not self.target.is_active() or self.target.age > self.target_max_age

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

        self.gamestate.destroy_entity(target)
        self.gamestate.destroy_entity(projectile)

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
        while len(self.projectiles) > 0 and self.projectiles[0].created_at < expiration_time:
            p = self.projectiles.popleft()
            p.unobserve(self)
            core.Gamestate.gamestate.destroy_entity(p)

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
        self.last_shot_ts = core.Gamestate.gamestate.timestamp
        self.gamestate.schedule_effect(core.Gamestate.gamestate.timestamp + 1/10., self)

class FleeOrder(core.Order):
    """ Keeps track of threats and flees from them until "safe" """
    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.threats:Set[core.AbstractSensorImage] = set()
        self.threat_ttl = 120.
        self.ttl_order_time = 5.

    def _ttl_order(self, order:core.Order) -> None:
        TimedOrderTask.ttl_order(order, self.ttl_order_time)
        self._add_child(order)

    def add_threat(self, threat:core.SectorEntity) -> None:
        pass

    def is_complete(self) -> bool:
        return self.completed_at > 0 or len(self.threats) == 0

    def act(self, dt:float) -> None:

        # first remove eliminated/expired threats
        closest_dist = np.inf
        closest_threat:Optional[core.AbstractSensorImage] = None
        dead_threats:List[core.AbstractSensorImage] = []
        for t in self.threats:
            if not t.is_active() or t.age > self.threat_ttl:
                dead_threats.append(t)
            else:
                dist = util.distance(t.loc, self.ship.loc)
                if dist < closest_dist:
                    closest_threat = t
                    closest_dist = dist
        for t in dead_threats:
            self.threats.remove(t)

        if closest_threat is None:
            self.complete_order()
            return

        #TODO: better logic on how to evade
        # evade closest threat
        self._ttl_order(movement.EvadeOrder(self.ship, self.gamestate, target_image=closest_threat))
