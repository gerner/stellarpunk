import uuid
import collections
import abc
from collections.abc import Collection, MutableMapping, Mapping
from typing import Type, Any, Optional, TypeVar, Union

import numpy as np
import numpy.typing as npt

from stellarpunk import core, events, sensors
#TODO: sector_entity shouldn't be in core
from stellarpunk.core import sector_entity

class IntelManager(core.AbstractIntelManager):
    """ Manages known intel and creates intel items for a character. """
    @classmethod
    def create_intel_manager(cls, gamestate:core.Gamestate, *args:Any, **kwargs:Any) -> "IntelManager":
        intel_manager = cls(gamestate, *args, **kwargs)
        return intel_manager

    def __init__(self, gamestate:core.Gamestate) -> None:
        self.owner:Character = None # type: ignore
        self.gamestate = gamestate
        self._intel:set[uuid.UUID] = set()
        self._intel_by_type:MutableMapping[Type[core.Intel], set[uuid.UUID]] = collections.defaultdict(set)
        self._entity_intel:dict[uuid.UUID, uuid.UUID] = {}

    def _add_intel(self, intel:core.Intel) -> None:
        self._intel.add(intel.entity_id)
        self._intel_by_type[type(intel)].add(intel.entity_id)
        if isinstance(intel, core.EntityIntel):
            self._entity_intel[intel.intel_entity_id] = intel.entity_id

    def add_intel(self, intel:core.Intel) -> None:
        #TODO: check if we already have a partial match for this intel
        # if ours is better, ignore this

        # if theirs is better, replace ours with theirs
        self._add_intel(intel)

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
    def create_asteroid_intel(cls, asteroid:sector_entity.Asteroid, *args:Any, **kwargs:Any) -> "AsteroidIntel":

        return cls.create_sector_entity_intel(asteroid, *args, asteroid.resource, asteroid.cargo[asteroid.resource], **kwargs)

    def __init__(self, resource:int, amount:float, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.resource = resource
        self.amount = amount

class StationIntel(SectorEntityIntel[sector_entity.Station]):
    @classmethod
    def create_station_intel(cls, station:sector_entity.Station, *args:Any, **kwargs:Any) -> "StationIntel":
        return cls.create_sector_entity_intel(station, *args, **kwargs)
        return StationIntel(station.entity_id, sector_entity.Station, *args, **kwargs)

    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        # we don't store resource or cargo information.
        # The corresponding EconAgentIntel can store that
        #TODO: is there anything we store on top of SectorEntityIntel?

class EconAgentIntel(core.EntityIntel[core.EconAgent]):
    @classmethod
    def create_econ_agent_intel(cls, econ_agent:core.EconAgent, *args:Any, **kwargs:Any) -> "EconAgentIntel":
        return EconAgentIntel(econ_agent.entity_id, core.EconAgent, *args, **kwargs)

    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        #TODO: what resources are bought and sold
        #TODO: what prices for each resouce
        #TODO: what amounts for sale and/or what budget for buying

class IdentifyAsteroidAction(events.Action):
    def act(self,
            character:core.Character,
            event_type:int,
            event_context:Mapping[int,int],
            event_args: MutableMapping[str, Union[int,float,str,bool]],
            action_args: Mapping[str, Union[int,float,str,bool]]
    ) -> None:
        asteroid = self.gamestate.get_entity_short(event_context[self.gamestate.event_manager.ck(sensors.ContextKeys.TARGET)], sector_entity.Asteroid)
        entity_intel = character.intel_manager.get_entity_intel(asteroid.entity_id, AsteroidIntel)
        if not entity_intel or not entity_intel.is_fresh():
            character.intel_manager.add_intel(AsteroidIntel.create_asteroid_intel(asteroid, self.gamestate))

class IdentifyStationAction(events.Action):
    def act(self,
            character:core.Character,
            event_type:int,
            event_context:Mapping[int,int],
            event_args: MutableMapping[str, Union[int,float,str,bool]],
            action_args: Mapping[str, Union[int,float,str,bool]]
    ) -> None:
        station = self.gamestate.get_entity_short(event_context[self.gamestate.event_manager.ck(sensors.ContextKeys.TARGET)], sector_entity.Station)
        entity_intel = character.intel_manager.get_entity_intel(station.entity_id, StationIntel)
        if not entity_intel or not entity_intel.is_fresh():
            character.intel_manager.add_intel(StationIntel.create_station_intel(station, self.gamestate))

class DockingAction(events.Action):
    def act(self,
            character:core.Character,
            event_type:int,
            event_context:Mapping[int,int],
            event_args: MutableMapping[str, Union[int,float,str,bool]],
            action_args: Mapping[str, Union[int,float,str,bool]]
    ) -> None:
        station = self.gamestate.get_entity_short(event_context[self.gamestate.event_manager.ck(sensors.ContextKeys.TARGET)], sector_entity.Station)
        agent = self.gamestate.econ_agents[station.entity_id]
        entity_intel = character.intel_manager.get_entity_intel(agent.entity_id, EconAgentIntel)
        if not entity_intel or not entity_intel.is_fresh():
            character.intel_manager.add_intel(EconAgentIntel.create_econ_agent_intel(agent, self.gamestate))
        #TODO: what other intel do we want to create now that we're docked?

def pre_initialize(event_manager:events.EventManager) -> None:
    event_manager.register_action(IdentifyAsteroidAction(), "identify_asteroid", "intel")
    #TODO: what about planets?
    event_manager.register_action(IdentifyStationAction(), "identify_station", "intel")
    #TODO: what about planets?
    event_manager.register_action(DockingAction(), "witness_docking", "intel")
