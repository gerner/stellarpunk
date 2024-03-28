""" Missile

SectorEntity that flies toward a target. """

from typing import Optional, Tuple, Any
import math
import numpy as np
import numpy.typing as npt
import cymunk # type: ignore

from stellarpunk import util, core
from stellarpunk.orders import movement
from stellarpunk.core.gamestate import Gamestate
from .sector_entity import SectorEntity, ObjectType
from .order import Effect

class Missile(core.Ship):
    id_prefix = "MSL"
    object_type = ObjectType.MISSILE

    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

class MissileOrder(movement.PursueOrder, core.CollisionObserver):
    """ Steer toward a collision with the target """

    @classmethod
    def spawn_missile(cls, ship:core.Ship, target:SectorEntity, gamestate:Gamestate) -> Missile:
        assert ship.sector
        loc = gamestate.generator.gen_sector_location(ship.sector, center=ship.loc + util.polar_to_cartesian(100, ship.angle), occupied_radius=75, radius=30000)
        v = util.polar_to_cartesian(100, ship.angle) + ship.velocity
        new_entity = gamestate.generator.spawn_sector_entity(Missile, ship.sector, loc[0], loc[1], v=v, w=0.0)
        assert isinstance(new_entity, Missile)
        missile:Missile = new_entity
        missile_order = MissileOrder(target, missile, gamestate)
        missile.prepend_order(missile_order)

        return missile

    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        self.ttl = 240
        self.expiration_time = self.gamestate.timestamp + self.ttl

    def _begin(self) -> None:
        assert self.ship.sector
        self.expiration_time = self.gamestate.timestamp + self.ttl
        self.ship.sector.register_collision_observer(self.ship.entity_id, self)

    def _complete(self) -> None:
        super()._complete()
        assert self.ship.sector
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

