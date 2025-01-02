import uuid
import collections
import abc
from collections.abc import Collection, MutableMapping
from typing import Type, Any, Optional, TypeVar

import numpy as np
import numpy.typing as npt

from stellarpunk import core
#TODO: sector_entity shouldn't be in core
from stellarpunk.core import sector_entity

class SectorEntityIntel[T:core.SectorEntity](core.EntityIntel[T]):
    @classmethod
    def create_sector_entity_intel[T2:"SectorEntityIntel[T]"](cls:Type[T2], entity:T, gamestate:core.Gamestate, *args:Any, **kwargs:Any) -> T2:
        assert(entity.sector)
        sector_id = entity.sector.entity_id
        loc = entity.loc
        entity_id = entity.entity_id
        entity_class = type(entity)
        intel = cls(*args, sector_id, loc, entity_id, entity_class, gamestate, **kwargs)
        return intel

    def __init__(self, sector_id:uuid.UUID, loc:npt.NDArray[np.float64], *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.sector_id = sector_id
        self.loc = loc

class AsteroidIntel(SectorEntityIntel[sector_entity.Asteroid]):
    @classmethod
    def create_asteroid_intel(cls, asteroid:sector_entity.Asteroid, *args:Any, **kwargs:Any) -> AsteroidIntel:

        return cls.create_sector_entity_intel(asteroid, *args, asteroid.resource, asteroid.cargo[asteroid.resource], **kwargs)

    def __init__(self, resource:int, amount:float, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.resource = resource
        self.amount = amount

class StationIntel(SectorEntityIntel[sector_entity.Station]):
    @classmethod
    def create_station_intel(cls, station:sector_entity.Station, *args:Any, **kwargs:Any) -> StationIntel:
        return cls.create_sector_entity_intel(station, *args, **kwargs)
        return StationIntel(station.entity_id, sector_entity.Station, *args, **kwargs)

    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        # we don't store resource or cargo information.
        # The corresponding EconAgentIntel can store that
        #TODO: is there anything we store on top of SectorEntityIntel?

class EconAgentIntel(core.EntityIntel[core.EconAgent]):
    @classmethod
    def create_econ_agent_intel(cls, econ_agent:core.EconAgent, *args:Any, **kwargs:Any) -> EconAgentIntel:
        return EconAgentIntel(econ_agent.entity_id, core.EconAgent, *args, **kwargs)

    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        #TODO: what resources are bought and sold
        #TODO: what prices for each resouce
        #TODO: what amounts for sale and/or what budget for buying

class IntelMaker[T:core.Intel]:
    def __init__(self, gamestate:core.Gamestate) -> None:
        self.gamestate = gamestate

    @property
    @abc.abstractmethod
    def intel_type(self) -> Type[T]: ...

class EntityIntelMaker[EntityType:core.Entity, T:core.EntityIntel](IntelMaker[T]):
    @abc.abstractmethod
    def create_entity_intel(self, entity:EntityType) -> T: ...

class AsteroidIntelMaker(EntityIntelMaker[sector_entity.Asteroid, AsteroidIntel]):
    @property
    def intel_type(self) -> Type[AsteroidIntel]:
        return AsteroidIntel
    def create_entity_intel(self, asteroid:sector_entity.Asteroid) -> AsteroidIntel:
        return AsteroidIntel.create_asteroid_intel(asteroid, self.gamestate)

class StationIntelMaker(EntityIntelMaker[sector_entity.Station, StationIntel]):
    @property
    def intel_type(self) -> Type[StationIntel]:
        return StationIntel
    def create_entity_intel(self, station:sector_entity.Station) -> StationIntel:
        return StationIntel.create_station_intel(station, self.gamestate)

class EconAgentIntelMaker(EntityIntelMaker[core.EconAgent, EconAgentIntel]):
    @property
    def intel_type(self) -> Type[EconAgentIntel]:
        return EconAgentIntel
    def create_entity_intel(self, agent:core.EconAgent) -> EconAgentIntel:
        return EconAgentIntel.create_econ_agent_intel(agent, self.gamestate)

class IntelFactory:
    def __init__(self, gamestate:core.Gamestate) -> None:
        self.gamestate = gamestate

        # a list of (entity type, entity intel type) representing a mapping
        # this is maintained in sorted order by entity type specificity
        # that is, any entry which represents a superclass of another entry
        # appears before that second entry.
        # this guarantees the most specific intel type is found first
        self._entity_intel_type:collections.deque[tuple[Type[core.Entity], Type[core.EntityIntel]]] = collections.deque()
        self._entity_intel_makers:dict[Type[core.EntityIntel], EntityIntelMaker] = {}

    def add_entity_intel_maker(self, entity_type:Type[core.Entity], intel_maker:EntityIntelMaker) -> None:

        intel_type = intel_maker
        #assert(issubclass(intel_type, core.EntityIntel))
        self._entity_intel_makers[intel_maker.intel_type] = intel_maker

        for i, (other_entity_type, _) in enumerate(self._entity_intel_type):
            if issubclass(entity_type, other_entity_type):
                break

        self._entity_intel_type.insert(i, (entity_type, intel_maker.intel_type))

    def get_entity_intel_type(self, entity:core.Entity) -> Type[core.EntityIntel]:
        for entity_type, intel_type in self._entity_intel_type:
            if isinstance(entity, entity_type):
                return intel_type
        #TODO: what if there's no matching intel?
        raise ValueError(f'do not know how to make intel for {entity}')

    def create_entity_intel(self, entity:core.Entity, intel_type:Optional[Type[core.EntityIntel]]) -> core.EntityIntel:
        if intel_type is None:
            intel_type = self.get_entity_intel_type(entity)

        return self._entity_intel_makers[intel_type].create_entity_intel(entity)

class IntelManager(core.AbstractIntelManager):
    """ Manages known intel and creates intel items for a character. """
    @classmethod
    def create_intel_manager(cls, owner:core.Character, intel_factory:IntelFactory, gamestate:core.Gamestate, *args:Any, **kwargs:Any) -> IntelManager:
        intel_manager = cls(intel_factory, gamestate, *args, **kwargs)
        intel_manager.owner = owner
        return intel_manager

    def __init__(self, intel_factory:IntelFactory, gamestate:core.Gamestate) -> None:
        self.owner:Character = None # type: ignore
        self.gamestate = gamestate
        self.intel_factory = intel_factory
        self._intel:set[uuid.UUID] = set()
        self._intel_by_type:MutableMapping[Type[core.Intel], set[uuid.UUID]] = collections.defaultdict(set)
        self._entity_intel:dict[uuid.UUID, uuid.UUID] = {}

    def witness_entity(self, entity:core.Entity) -> None:
        #TODO: check if we already have intel about this entity
        intel_type = self.intel_factory.get_entity_intel_type(entity)
        entity_intel = self.get_entity_intel(entity.entity_id, intel_type)
        if entity_intel:
            # if ours is just as good, ignore this. note: a fresh observation
            # cannot be worse than old intel
            if entity_intel.is_fresh():
                return
        entity_intel = self.intel_factory.create_entity_intel(entity, intel_type=intel_type)
        self._add_intel(entity_intel)

    def _add_intel(self, intel:core.Intel) -> None:
        self._intel.add(intel.entity_id)
        self._intel_by_type[type(intel)].add(intel.entity_id)
        if isinstance(intel, core.EntityIntel):
            self._entity_intel[intel.intel_entity_id] = intel.entity_id

    def add_intel(self, intel:core.Intel) -> None:
        #TODO: check if we already have a partial match for this intel
        # if ours is better, ignore this

        # if theirs is better, replace ours with theirs
        pass

    def intel[T:core.Intel](self, cls:Type[T]) -> Collection[T]:
        return list(self.gamestate.get_entity(intel_id, cls) for intel_id in self._intel_by_type[cls])

    def get_entity_intel[T:core.EntityIntel](self, entity_id:uuid.UUID, cls:Type[T]) -> Optional[T]:
        # check if we have such intel
        if entity_id not in self._entity_intel:
            return None
        # we assume the intel is of the right type
        #TODO: could you ever have multiple types of intel about the same entity?
        #   e.g. the location of a Character vs a Character's affiliations
        intel_id = self._entity_intel[entity_id]
        intel = self.gamestate.get_entity(intel_id, cls)
        return intel
