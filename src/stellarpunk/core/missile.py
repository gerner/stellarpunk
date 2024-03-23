""" Missile

SectorEntity that flies toward a target. """

from typing import Optional, Tuple, Any
import numpy as np
import numpy.typing as npt
import cymunk # type: ignore

from stellarpunk import util, core
from stellarpunk.orders import collision
from stellarpunk.core.gamestate import Gamestate
from .sector_entity import SectorEntity, ObjectType
from .order import Effect

class Missile(SectorEntity):
    id_prefix = "MSL"
    object_type = ObjectType.MISSILE

    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        # SI units (newtons and newton-meters)
        # max thrust along heading vector
        self.max_thrust = 0.
        # max thrust in any direction
        self.max_fine_thrust = 0.
        # max torque for turning (in newton-meters)
        self.max_torque = 0.

        self.missile_effect:Optional[MissileEffect] = None

class MissileEffect(Effect, core.CollisionObserver):
    """ Steer toward a collision with the target """
    def __init__(self, missile:Missile, target:SectorEntity, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        self.ttl = 60
        self.expiration_time = self.gamestate.timestamp + self.ttl

        self.missile = missile
        self.target = target

    def bbox(self) -> Tuple[float, float, float, float]:
        return util.circle_bbox(self.missile.loc, self.missile.radius)

    def _begin(self) -> None:
        if self.missile.sector is None:
            raise Exception("cannot begin missile with missile not in sector")
        self.expiration_time = self.gamestate.timestamp + self.ttl
        self.missile.sector.register_collision_observer(self.missile.entity_id, self)
        self.gamestate.schedule_effect_immediate(self)

    def _complete(self) -> None:
        if self.missile.sector is None:
            raise Exception(f'double complete for {self}')
        self.missile.sector.remove_entity(self.missile)

    def _cancel(self) -> None:
        self._complete()

    def is_complete(self) -> bool:
        return self.completed_at > 0 or self.gamestate.timestamp > self.expiration_time

    def collision(self, missile:SectorEntity, target:SectorEntity, impulse:Tuple[float, float], ke:float) -> None:
        if missile != self.missile:
            raise Exception(f'got unexpected collision between {missile} != {self.missile} and {target}')
        self.logger.info(f'missile {self.missile} hit {target} impulse: {impulse} ke: {ke}!')
        self.complete_effect()

    def act(self, dt:float) -> None:
        if self.is_complete():
            self.complete_effect()
            return

        #TODO: act should be the pursuit steering behavior
        # figure out the desired velocity to intercept the target given their
        # and our velocities
        target_velocity, distance, distance_estimate, cannot_stop, delta_v = collision.find_target_v(
                self.missile.phys,
                self.target.phys.position,
                self.target.radius,
                0,
                self.missile.max_thrust / self.missile.mass,
                self.missile.max_thrust / self.missile.phys.moment,
                self.missile.max_thrust / self.missile.mass * 30,
                dt,
                1.0,
                self.missile.max_thrust / self.missile.mass * 1
        )

        continue_time = collision.accelerate_to(
                self.missile.phys,
                target_velocity,
                dt,
                self.missile.max_thrust / self.missile.mass * 30,
                self.missile.max_torque,
                self.missile.max_thrust,
                self.missile.max_fine_thrust
        )

        next_ts = self.gamestate.timestamp + min(1/10, continue_time)

        self.gamestate.schedule_effect(next_ts, self)

def setup_missile(missile:Missile, target:SectorEntity, gamestate:Gamestate) -> None:
    if missile.sector is None:
        return
    missile_effect = MissileEffect(missile, target, missile.sector, gamestate)
    missile.missile_effect = missile_effect
    missile.sector.add_effect(missile_effect)
