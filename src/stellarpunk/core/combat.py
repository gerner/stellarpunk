""" Combat """

import enum
from typing import Optional, Tuple, Any
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

class Missile(core.Ship):
    id_prefix = "MSL"
    object_type = ObjectType.MISSILE

    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        # missiles don't run transponders
        self.transponder_on = False

class MissileOrder(movement.PursueOrder, core.CollisionObserver):
    """ Steer toward a collision with the target """

    @staticmethod
    def spawn_missile(ship:core.Ship, gamestate:Gamestate, target:Optional[SectorEntity]=None, target_image:Optional[core.AbstractSensorImage]=None) -> Missile:
        assert ship.sector
        loc = gamestate.generator.gen_sector_location(ship.sector, center=ship.loc + util.polar_to_cartesian(100, ship.angle), occupied_radius=75, radius=100)
        v = util.polar_to_cartesian(100, ship.angle) + ship.velocity
        new_entity = gamestate.generator.spawn_sector_entity(Missile, ship.sector, loc[0], loc[1], v=v, w=0.0)
        assert isinstance(new_entity, Missile)
        missile:Missile = new_entity
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
        self.ttl_order_time = 5

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

class FleeOrder(movement.AbstractSteeringOrder):
    pass
