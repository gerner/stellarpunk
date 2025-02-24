""" Sector Entities that live in a Sector, have physics, etc. """

import enum
import uuid
import collections
from typing import Optional, Dict, Mapping, Any, Deque, Sequence, Union, Iterable, Set

import numpy as np
import numpy.typing as npt
import cymunk # type: ignore

from stellarpunk import util
from . import base, character, sector, gamestate

class Planet(character.Asset, character.CrewedSectorEntity):
    id_prefix = "HAB"
    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.population = 0.

    @property
    def loc(self) -> npt.NDArray[np.float64]: return self._loc
    @property
    def velocity(self) -> npt.NDArray[np.float64]: return self._velocity

class Station(character.Asset, character.CrewedSectorEntity):
    id_prefix = "STA"
    def __init__(self, sprite:base.Sprite, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.resource: int = -1
        self.next_batch_time = 0.
        self.next_production_time = 0.
        self.cargo_capacity = 1e5

        self.sprite = sprite

    @property
    def loc(self) -> npt.NDArray[np.float64]: return self._loc
    @property
    def velocity(self) -> npt.NDArray[np.float64]: return self._velocity

class Asteroid(sector.SectorEntity):
    id_prefix = "AST"
    def __init__(self, resource:int, amount:float, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.resource = resource
        self.cargo[self.resource] = amount

    @property
    def loc(self) -> npt.NDArray[np.float64]: return self._loc
    @property
    def velocity(self) -> npt.NDArray[np.float64]: return self._velocity

class TravelGate(sector.SectorEntity):
    """ Represents a "gate" to another sector """
    id_prefix = "GAT"

    @classmethod
    def compute_sector_network(cls, gamestate:gamestate.Gamestate) -> tuple[dict[uuid.UUID, int], npt.NDArray[np.float64]]:
        sector_idx_lookup:dict[uuid.UUID, int] = {sector_id: sector_idx for sector_idx, sector_id in enumerate(gamestate.sectors.keys())}
        adj_matrix:npt.NDArray[np.float64] = np.ones((len(gamestate.sectors), len(gamestate.sectors))) * np.inf
        for sector_id, sector in gamestate.sectors.items():
            for gate in sector.entities_by_type(TravelGate):
                adj_matrix[sector_idx_lookup[sector_id], sector_idx_lookup[gate.destination.entity_id]] = 1.0

        return sector_idx_lookup, adj_matrix

    def __init__(self, direction:float, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.destination:sector.Sector = None # type: ignore
        # radian angle toward the destination
        self.direction:float = direction
        self.direction_vector = np.array(util.polar_to_cartesian(1., direction))

    @property
    def loc(self) -> npt.NDArray[np.float64]: return self._loc
    @property
    def velocity(self) -> npt.NDArray[np.float64]: return self._velocity

class Projectile(sector.SectorEntity):
    id_prefix = "PJT"
    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        # projectiles don't run transponders

