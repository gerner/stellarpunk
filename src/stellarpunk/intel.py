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

class IntelManager(core.IntelObserver, core.AbstractIntelManager):
    """ Manages known intel and creates intel items for a character. """
    @classmethod
    def create_intel_manager(cls, gamestate:core.Gamestate, *args:Any, **kwargs:Any) -> "IntelManager":
        intel_manager = cls(gamestate, *args, **kwargs)
        return intel_manager

    def __init__(self, gamestate:core.Gamestate, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.character:Character = None # type: ignore
        self.gamestate = gamestate
        self._intel:set[uuid.UUID] = set()
        self._intel_by_type:MutableMapping[Type[core.Intel], set[uuid.UUID]] = collections.defaultdict(set)
        self._entity_intel:dict[uuid.UUID, uuid.UUID] = {}

    def sanity_check(self) -> None:
        intel_count = 0
        entity_intel_count = 0
        for intel_type, intels in self._intel_by_type.items():
            for intel_id in intels:
                intel_count += 1
                assert intel_id in self._intel
                intel = self.gamestate.get_entity(intel_id, intel_type)
                assert intel.entity_id == intel_id
                if isinstance(intel, core.EntityIntel):
                    entity_intel_count += 1
                    assert intel_id == self._entity_intel[intel.intel_entity_id]

        assert intel_count == len(self._intel)
        assert entity_intel_count == len(self._entity_intel)

    @property
    def observer_id(self) -> uuid.UUID:
        return self.character.entity_id

    def intel_expired(self, intel:core.Intel) -> None:
        self._remove_intel(intel)

    def _remove_intel(self, old_intel:core.Intel) -> None:
        self._intel.remove(old_intel.entity_id)
        self._intel_by_type[type(old_intel)].remove(old_intel.entity_id)
        if isinstance(old_intel, core.EntityIntel):
            del self._entity_intel[old_intel.intel_entity_id]
        old_intel.unobserve(self)

    def _add_intel(self, intel:core.Intel) -> None:
        intel.observe(self)
        self._intel.add(intel.entity_id)
        self._intel_by_type[type(intel)].add(intel.entity_id)
        if isinstance(intel, core.EntityIntel):
            self._entity_intel[intel.intel_entity_id] = intel.entity_id

    def add_intel(self, intel:core.Intel) -> None:
        old_intel:Optional[core.Intel] = None
        if isinstance(intel, core.EntityIntel):
            old_intel = self.get_entity_intel(intel.entity_id, type(intel))
        else:
            for candidate_id in self._intel:
                intel_candidate = self.gamestate.get_entity(candidate_id, core.Intel)
                if intel_candidate.matches(intel):
                    old_intel = intel_candidate
                    break;

        # if we already have fresh matching intel
        if old_intel and old_intel.created_at > intel.created_at:
            # ours is better, ignore theirs
            return
        elif old_intel:
            # theirs is better, drop ours
            self._remove_intel(old_intel)

        self._add_intel(intel)

    def intel[T:core.Intel](self, cls:Optional[Type[T]]=None) -> Collection[T]:
        if cls:
            return list(self.gamestate.get_entity(intel_id, cls) for intel_id in self._intel_by_type[cls])
        else:
            return list(self.gamestate.get_entity(intel_id, core.Intel) for intel_id in self._intel) # type: ignore

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

class ExpireIntelTask(core.ScheduledTask):
    @classmethod
    def expire_intel(cls, intel:core.Intel) -> "ExpireIntelTask":
        task = ExpireIntelTask()
        task.intel = intel
        return task

    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.intel:core.Intel = None # type: ignore

    def act(self) -> None:
        self.intel.expire()

class SectorEntityIntel[T:core.SectorEntity](core.EntityIntel[T]):
    @classmethod
    def create_sector_entity_intel[T2:"SectorEntityIntel[T]"](cls:Type[T2], entity:T, gamestate:core.Gamestate, *args:Any, **kwargs:Any) -> T2:
        assert(entity.sector)
        sector_id = entity.sector.entity_id
        loc = entity.loc
        entity_id = entity.entity_id
        entity_short_id = entity.short_id()
        entity_class = type(entity)
        intel = cls(*args, sector_id, loc, entity_id, entity_short_id, entity_class, gamestate, **kwargs)

        # schedule this intel to expire only once per shared intel
        if intel.expires_at < np.inf:
            gamestate.schedule_task(intel.expires_at, ExpireIntelTask.expire_intel(intel))

        return intel

    def __init__(self, sector_id:uuid.UUID, loc:npt.NDArray[np.float64], *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.sector_id = sector_id
        self.loc = loc

    def sanity_check(self) -> None:
        super().sanity_check()
        #TODO: is it possible sectors go away?
        sector = core.Gamestate.gamestate.get_entity(self.sector_id, core.Sector)

class AsteroidIntel(SectorEntityIntel[sector_entity.Asteroid]):
    @classmethod
    def create_asteroid_intel(cls, asteroid:sector_entity.Asteroid, *args:Any, **kwargs:Any) -> "AsteroidIntel":

        return cls.create_sector_entity_intel(asteroid, *args, asteroid.resource, asteroid.cargo[asteroid.resource], **kwargs)

    def __init__(self, resource:int, amount:float, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.resource = resource
        self.amount = amount

    def sanity_check(self) -> None:
        super().sanity_check()
        #TODO: can asteroids go away?
        asteroid = core.Gamestate.gamestate.get_entity(self.intel_entity_id, sector_entity.Asteroid)
        assert(asteroid.resource == self.resource)

class EconAgentIntel(core.EntityIntel[core.EconAgent]):
    @classmethod
    def create_econ_agent_intel(cls, econ_agent:core.EconAgent, gamestate:core.Gamestate, *args:Any, **kwargs:Any) -> "EconAgentIntel":
        entity_id = econ_agent.entity_id
        entity_short_id = econ_agent.short_id()
        entity_class = type(econ_agent)
        agent_intel = EconAgentIntel(entity_id, entity_short_id, entity_class, gamestate, **kwargs)
        agent_intel.sector_entity_id = gamestate.agent_to_entity[entity_id].entity_id

        for resource in econ_agent.sell_resources():
            agent_intel.sell_offers[resource] = (
                econ_agent.sell_price(resource),
                econ_agent.inventory(resource),
            )

        for resource in econ_agent.buy_resources():
            agent_intel.buy_offers[resource] = (
                econ_agent.buy_price(resource),
                econ_agent.budget(resource) / econ_agent.buy_price(resource),
            )

        return agent_intel

    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.sector_entity_id:uuid.UUID = None # type: ignore

        # resource id -> price, amount
        self.sell_offers:dict[int, tuple[float, float]] = {}
        self.buy_offers:dict[int, tuple[float, float]] = {}

class IdentifyAsteroidAction(events.Action):
    def act(self,
            character:core.Character,
            event_type:int,
            event_context:Mapping[int,int],
            event_args: MutableMapping[str, Union[int,float,str,bool]],
            action_args: Mapping[str, Union[int,float,str,bool]]
    ) -> None:
        asteroid = self.gamestate.get_entity_short(self.ck(event_context, sensors.ContextKeys.TARGET), sector_entity.Asteroid)
        entity_intel = character.intel_manager.get_entity_intel(asteroid.entity_id, AsteroidIntel)
        if not entity_intel or not entity_intel.is_fresh():
            character.intel_manager.add_intel(AsteroidIntel.create_asteroid_intel(asteroid, self.gamestate))

class IdentifySectorEntityAction(events.Action):
    def __init__(self, *args:Any, intel_ttl:float=300, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.intel_ttl = intel_ttl

    def act(self,
            character:core.Character,
            event_type:int,
            event_context:Mapping[int,int],
            event_args: MutableMapping[str, Union[int,float,str,bool]],
            action_args: Mapping[str, Union[int,float,str,bool]]
    ) -> None:
        sentity = self.gamestate.get_entity_short(self.ck(event_context, sensors.ContextKeys.TARGET), core.SectorEntity)
        entity_intel = character.intel_manager.get_entity_intel(sentity.entity_id, SectorEntityIntel)
        if not entity_intel or not entity_intel.is_fresh():
            # intel about these select objects
            #TODO: TravelGate needs its own intel to include where the travel gate goes
            intel:SectorEntityIntel
            if isinstance(sentity, (sector_entity.Station, sector_entity.Planet, sector_entity.TravelGate)):
                intel = SectorEntityIntel.create_sector_entity_intel(sentity, self.gamestate)
            # otherwise we'll give it some ttl
            else:
                fresh_until = self.gamestate.timestamp + self.intel_ttl*0.2
                expires_at = self.gamestate.timestamp + self.intel_ttl
                intel = SectorEntityIntel.create_sector_entity_intel(sentity, self.gamestate, expires_at=expires_at, fresh_until=fresh_until)
            character.intel_manager.add_intel(intel)

class DockingAction(events.Action):
    def __init__(self, *args:Any, econ_intel_ttl:float=300.0, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.econ_intel_ttl=econ_intel_ttl

    def act(self,
            character:core.Character,
            event_type:int,
            event_context:Mapping[int,int],
            event_args: MutableMapping[str, Union[int,float,str,bool]],
            action_args: Mapping[str, Union[int,float,str,bool]]
    ) -> None:
        station = self.gamestate.get_entity_short(self.ck(event_context, events.ContextKeys.TARGET), sector_entity.Station)

        # first make some econ agent intel to record resources and prices at
        # this station
        agent = self.gamestate.econ_agents[station.entity_id]
        econ_agent_intel = character.intel_manager.get_entity_intel(agent.entity_id, EconAgentIntel)
        if not econ_agent_intel or not econ_agent_intel.is_fresh():
            fresh_until = self.gamestate.timestamp + self.econ_intel_ttl*0.2
            expires_at = self.gamestate.timestamp + self.econ_intel_ttl
            character.intel_manager.add_intel(EconAgentIntel.create_econ_agent_intel(agent, self.gamestate, expires_at=expires_at, fresh_until=fresh_until))
        #TODO: what other intel do we want to create now that we're docked?

def pre_initialize(event_manager:events.EventManager) -> None:
    event_manager.register_action(IdentifyAsteroidAction(), "identify_asteroid", "intel")
    event_manager.register_action(IdentifySectorEntityAction(), "identify_sector_entity", "intel")
    event_manager.register_action(DockingAction(), "witness_docking", "intel")
