""" Combat """

from typing import Optional, Tuple, Any
import math
import numpy as np
import numpy.typing as npt
import cymunk # type: ignore

from stellarpunk import util, core
from stellarpunk.orders import movement, collision
from stellarpunk.core.gamestate import Gamestate
from .sector_entity import SectorEntity, ObjectType
from .order import Effect

class Missile(core.Ship):
    id_prefix = "MSL"
    object_type = ObjectType.MISSILE

    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        # missiles don't run transponders
        self.transponder_on = False

class MissileOrder(movement.PursueOrder, core.CollisionObserver):
    """ Steer toward a collision with the target """

    @classmethod
    def spawn_missile(cls, ship:core.Ship, target:SectorEntity, gamestate:Gamestate) -> Missile:
        assert ship.sector
        loc = gamestate.generator.gen_sector_location(ship.sector, center=ship.loc + util.polar_to_cartesian(100, ship.angle), occupied_radius=75, radius=100)
        v = util.polar_to_cartesian(100, ship.angle) + ship.velocity
        new_entity = gamestate.generator.spawn_sector_entity(Missile, ship.sector, loc[0], loc[1], v=v, w=0.0)
        assert isinstance(new_entity, Missile)
        missile:Missile = new_entity
        missile_order = MissileOrder(missile, gamestate, target=target)
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
        self.gamestate.destroy_sector_entity(self.ship)

    def _cancel(self) -> None:
        self._complete()

    def is_complete(self) -> bool:
        return self.completed_at > 0 or self.gamestate.timestamp > self.expiration_time
    def collision(self, missile:SectorEntity, target:SectorEntity, impulse:Tuple[float, float], ke:float) -> None:
        assert missile == self.ship
        self.logger.debug(f'missile {self.ship} hit {target} impulse: {impulse} ke: {ke}!')

        self.cancel_order()
        self.gamestate.destroy_sector_entity(target)

class AttackOrder(movement.AbstractSteeringOrder, core.SectorEntityObserver):
    """ Objective is to destroy a target. """
    def __init__(self, target:core.SectorEntity, *args:Any, distance_min:float=2.5e5, distance_max:float=5e5, max_active_age:float=35, max_passive_age:float=30, search_distance:float=5e4, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        assert self.ship.sector
        self.target = self.ship.sector.sensor_manager.target(target, self.ship)
        self.distance_min = distance_min
        self.distance_max = distance_max
        self.max_active_age = max_active_age
        self.max_passive_age = max_passive_age
        self.search_distance = search_distance
        self.standoff_order:Optional[core.Order] = None

    def __str__(self) -> str:
        return f'Attack: age:{self.target.age} dist:{np.linalg.norm(self.target.loc-self.ship.loc)}'

    def _complete(self) -> None:
        if self.standoff_order:
            self.standoff_order = None

    def _cancel(self) -> None:
        self._complete()

    def is_complete(self) -> bool:
        return self.completed_at > 0. or not self.target.is_active()

    def _keep_image_fresh(self, dt:float) -> bool:
        assert self.ship.sector
        # keep sensor image fresh
        if self.target.age > self.max_active_age:
            # actively search for the target
            self.ship.sector.sensor_manager.set_sensors(self.ship, 1.0)
            if self.target.update():
                return True
            if np.linalg.norm(self.target.loc - self.ship.loc) > self.search_distance:
                self.standoff_order = movement.PursueOrder(self.ship, self.gamestate, target_image=self.target, arrival_distance=self.search_distance*0.8)
            else:
                # TODO: search pattern
                pass
            return False
        elif self.target.age > self.max_passive_age:
            # go active, but make no other effort to re-acquire target
            self.ship.sector.sensor_manager.set_sensors(self.ship, 1.0)
            self.target.update()
            return True
        else:
            # image fresh enough make sure our sensors are off
            self.ship.sector.sensor_manager.set_sensors(self.ship, 0.0)
            return True

    def _maintain_standoff(self, dt:float) -> bool:
        # TODO: determine our standoff range

        # get to a standoff distance
        distance = np.linalg.norm(self.target.loc - self.ship.loc)
        if distance > self.distance_max:
            self.standoff_order = movement.PursueOrder(self.ship, self.gamestate, target_image=self.target, arrival_distance=self.distance_max-0.2*(self.distance_max-self.distance_min))
            self._add_child(self.standoff_order)
            return False
        elif distance < self.distance_min:
            self.standoff_order = movement.EvadeOrder(self.ship, self.gamestate, target_image=self.target, escape_distance=self.distance_min+0.2*(self.distance_max-self.distance_min))
            self._add_child(self.standoff_order)
            return False

        if self.standoff_order:
            assert self.standoff_order.is_complete()
            self.standoff_order = None

        return True

    def _move_shadow_target(self, dt:float) -> float:
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

        return collision.accelerate_to(self.ship.phys, cymunk.Vec2d(target_velocity), dt, self.ship.max_speed(), self.ship.max_torque, self.ship.max_thrust, self.ship.max_fine_thrust)

    def act(self, dt:float) -> None:
        assert self.ship.sector
        self.target.update()

        if not self._keep_image_fresh(dt):
            return

        if not self._maintain_standoff(dt):
            return
        # inside standoff zone, move shadow the target
        shadow_time = self._move_shadow_target(dt)

        # TODO: make attacks
        # if weapon systems are ready, else ready weapons
        # if we've got enough confidence in taking a shot, else gain confidence
        # take the shot

        self.gamestate.schedule_order(self.gamestate.timestamp + min(shadow_time, 1/10), self)

class FleeOrder(movement.AbstractSteeringOrder):
    pass
