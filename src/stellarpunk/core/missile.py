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

class Missile(core.Ship):
    id_prefix = "MSL"
    object_type = ObjectType.MISSILE

    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

class MissileOrder(core.Order, core.CollisionObserver, core.SectorEntityObserver):
    """ Steer toward a collision with the target """
    def __init__(self, target:SectorEntity, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        self.ttl = 240
        self.expiration_time = self.gamestate.timestamp + self.ttl

        self.ship.observe(self)
        self.target = target
        self.target.observe(self)

        self.intercept_location = np.array((0.0, 0.0))
        self.intercept_time = 0.0

    def estimate_eta(self) -> float:
        return self.intercept_time

    def entity_migrated(self, entity:SectorEntity, from_sector:core.Sector, to_sector:core.Sector) -> None:
        if to_sector == self.ship.sector and to_sector == self.target.sector:
            pass

        self.cancel_order()

    def entity_destroyed(self, entity:SectorEntity) -> None:
        self.cancel_order()

    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.intercept_location[0]-1, self.intercept_location[1]-1, self.intercept_location[0]+1, self.intercept_location[1]+1)#util.circle_bbox(self.ship.loc, self.ship.radius)

    def _begin(self) -> None:
        assert self.ship.sector
        self.expiration_time = self.gamestate.timestamp + self.ttl
        self.ship.sector.register_collision_observer(self.ship.entity_id, self)

    def _complete(self) -> None:
        assert self.ship.sector
        self.ship.sector.unregister_collision_observer(self.ship.entity_id, self)
        self.ship.unobserve(self)
        self.target.unobserve(self)
        self.gamestate.destroy_sector_entity(self.ship)

    def _cancel(self) -> None:
        self._complete()

    def is_complete(self) -> bool:
        return self.completed_at > 0 or self.gamestate.timestamp > self.expiration_time

    def collision(self, missile:SectorEntity, target:SectorEntity, impulse:Tuple[float, float], ke:float) -> None:
        assert missile == self.ship
        self.logger.debug(f'missile {self.ship} hit {target} impulse: {impulse} ke: {ke}!')

        self.cancel_order()
        self.gamestate.destroy_sector_entity(self.target)

    def act(self, dt:float) -> None:
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
        final_speed = self.ship.max_thrust / self.ship.mass * 0.5


        target_velocity, _, self.intercept_time, self.intercept_location = collision.find_intercept_v(
                self.ship.phys,
                self.target.phys,
                self.target.radius/5,
                self.ship.max_acceleration(),
                self.ship.max_angular_acceleration(),
                self.ship.max_speed(),
                dt,
                final_speed)

        continue_time = collision.accelerate_to(
                self.ship.phys,
                target_velocity,
                dt,
                self.ship.max_speed(),
                self.ship.max_torque,
                self.ship.max_thrust,
                self.ship.max_fine_thrust
        )

        next_ts = self.gamestate.timestamp + min(1/10, continue_time)

        self.gamestate.schedule_order(next_ts, self)

def setup_missile(missile:Missile, target:SectorEntity, gamestate:Gamestate) -> None:
    if missile.sector is None:
        return
    missile_order = MissileOrder(target, missile, gamestate)
    missile.prepend_order(missile_order)
