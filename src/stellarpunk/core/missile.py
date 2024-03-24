""" Missile

SectorEntity that flies toward a target. """

from typing import Optional, Tuple, Any
import math
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

class MissileEffect(Effect, core.CollisionObserver, core.SectorEntityObserver):
    """ Steer toward a collision with the target """
    def __init__(self, missile:Missile, target:SectorEntity, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        self.ttl = 240
        self.expiration_time = self.gamestate.timestamp + self.ttl

        self.missile = missile
        self.missile.observe(self)
        self.target = target
        self.target.observe(self)

        self.intercept_location = np.array((0.0, 0.0))
        self.intercept_time = 0.0

    def entity_migrated(self, entity:SectorEntity, from_sector:core.Sector, to_sector:core.Sector) -> None:
        if to_sector == self.missile.sector and to_sector == self.target.sector:
            pass

        self.cancel_effect()

    def entity_destroyed(self, entity:SectorEntity) -> None:
        self.cancel_effect()

    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.intercept_location[0]-1, self.intercept_location[1]-1, self.intercept_location[0]+1, self.intercept_location[1]+1)#util.circle_bbox(self.missile.loc, self.missile.radius)

    def _begin(self) -> None:
        if self.missile.sector is None:
            raise Exception("cannot begin missile with missile not in sector")
        self.expiration_time = self.gamestate.timestamp + self.ttl
        self.missile.sector.register_collision_observer(self.missile.entity_id, self)
        self.gamestate.schedule_effect_immediate(self)

    def _complete(self) -> None:
        if self.missile.sector is None:
            raise Exception(f'double complete for {self}')
        self.missile.unobserve(self)
        self.target.unobserve(self)
        self.gamestate.destroy_sector_entity(self.missile)

    def _cancel(self) -> None:
        self.missile.unobserve(self)
        self.target.unobserve(self)
        self.gamestate.destroy_sector_entity(self.missile)

    def is_complete(self) -> bool:
        return self.completed_at > 0 or self.gamestate.timestamp > self.expiration_time

    def collision(self, missile:SectorEntity, target:SectorEntity, impulse:Tuple[float, float], ke:float) -> None:
        if missile != self.missile:
            raise Exception(f'got unexpected collision between {missile} != {self.missile} and {target}')
        self.logger.info(f'missile {self.missile} hit {target} impulse: {impulse} ke: {ke}!')

        self.complete_effect()
        self.gamestate.destroy_sector_entity(self.target)

    def act(self, dt:float) -> None:
        if self.is_complete():
            self.complete_effect()
            return

        # interception algorithm:
        # assume target velocity is constant
        # solve problem in their frame of reference, so they appear stationary
        # try to "arrive" at their location (using arrival steering behavior)
        # whatever velocity we need in this frame of reference, add in their
        # velocity

        # paramters for the algorithm
        max_speed = self.missile.max_thrust / self.missile.mass * 300
        max_acceleration = self.missile.max_thrust / self.missile.mass
        # this is the desired final speed we'll try to achieve at the intercept
        # this is on top of the target's velocity
        # a large value here reduces intercept time
        # a small value here makes it easy to correct at the last minute
        final_speed = self.missile.max_thrust / self.missile.mass * 0.5


        target_velocity, _, self.intercept_time, self.intercept_location = collision.find_intercept_v(
                self.missile.phys,
                self.target.phys,
                self.target.radius/5,
                max_acceleration,
                self.missile.max_thrust/self.missile.phys.moment,
                max_speed,
                dt,
                final_speed)

        continue_time = collision.accelerate_to(
                self.missile.phys,
                target_velocity,
                dt,
                max_speed,
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
