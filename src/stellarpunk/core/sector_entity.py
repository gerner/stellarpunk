""" Sector Entities that live in a Sector, have physics, etc. """

import enum
import uuid
import collections
from typing import Optional, Dict, Mapping, Any, Deque, Sequence, Union, Iterable, Set

import numpy as np
import numpy.typing as npt
import cymunk # type: ignore

from stellarpunk import util
from . import base, character, sector

class Planet(character.CrewedSectorEntity, character.Asset):
    id_prefix = "HAB"
    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.population = 0.

class Station(character.CrewedSectorEntity, character.Asset):
    id_prefix = "STA"
    def __init__(self, sprite:base.Sprite, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.resource: int = -1
        self.next_batch_time = 0.
        self.next_production_time = 0.
        self.cargo_capacity = 1e5

        self.sprite = sprite

class Asteroid(sector.SectorEntity):
    id_prefix = "AST"
    def __init__(self, resource:int, amount:float, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.resource = resource
        self.cargo[self.resource] = amount


class TravelGate(sector.SectorEntity):
    """ Represents a "gate" to another sector """
    id_prefix = "GAT"
    def __init__(self, direction:float, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.destination:sector.Sector = None # type: ignore
        # radian angle toward the destination
        self.direction:float = direction
        self.direction_vector = np.array(util.polar_to_cartesian(1., direction))

class Projectile(sector.SectorEntity):
    id_prefix = "PJT"
    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        # projectiles don't run transponders

