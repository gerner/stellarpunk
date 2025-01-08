import uuid
import collections
import abc
from collections.abc import Collection, MutableMapping, Mapping
from typing import Type, Any, Optional, TypeVar, Union

import numpy as np
import numpy.typing as npt

from stellarpunk import core, events, sensors, util
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
        self._intel_map:dict[core.IntelMatchCriteria, uuid.UUID] = {}

    def sanity_check(self) -> None:
        intel_count = 0
        entity_intel_count = 0
        for intel_type, intels in self._intel_by_type.items():
            for intel_id in intels:
                intel_count += 1
                assert intel_id in self._intel
                intel = self.gamestate.get_entity(intel_id, intel_type)
                assert intel.entity_id == intel_id
                match_criteria = intel.match_criteria()
                assert intel.entity_id == self._intel_map[match_criteria]

        assert intel_count == len(self._intel)

    @property
    def observer_id(self) -> uuid.UUID:
        return self.character.entity_id

    def intel_expired(self, intel:core.Intel) -> None:
        self._remove_intel(intel)

    def _remove_intel(self, old_intel:core.Intel) -> None:
        self._intel.remove(old_intel.entity_id)
        self._intel_by_type[type(old_intel)].remove(old_intel.entity_id)
        del self._intel_map[old_intel.match_criteria()]
        old_intel.unobserve(self)

    def _add_intel(self, intel:core.Intel) -> None:
        intel.observe(self)
        self._intel.add(intel.entity_id)
        self._intel_by_type[type(intel)].add(intel.entity_id)
        self._intel_map[intel.match_criteria()] = intel.entity_id

    def add_intel(self, intel:core.Intel) -> None:
        old_intel:Optional[core.Intel] = self.get_intel(intel.match_criteria(), type(intel))

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

    def get_intel[T:core.Intel](self, match_criteria:core.IntelMatchCriteria, cls:Type[T]) -> Optional[T]:
        # check if we have such intel
        if match_criteria not in self._intel_map:
            return None
        # we assume the intel is of the right type
        #TODO: could you ever have multiple types of intel about the same entity?
        #   e.g. the location of a Character vs a Character's affiliations
        intel_id = self._intel_map[match_criteria]
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

class EntityIntelMatchCriteria(core.IntelMatchCriteria):
    def __init__(self, entity_id:uuid.UUID) -> None:
        self.entity_id = entity_id

    def __hash__(self) -> int:
        return hash(self.entity_id)

    def __eq__(self, other:Any) -> bool:
        if not isinstance(other, EntityIntelMatchCriteria):
            return False
        return self.entity_id == other.entity_id

    def matches(self, other:core.Intel) -> bool:
        if not isinstance(other, EntityIntel):
            return False
        return other.intel_entity_id == self.entity_id

class EntityIntel[T:core.Entity](core.Intel):
    def __init__(self, intel_entity_id:uuid.UUID, id_prefix:str, intel_entity_short_id:str, intel_entity_type:Type[T], *args:Any, **kwargs:Any) -> None:
        # we need to set these fields before the super constructor because we
        # override __str__ which might be called in a super constructor
        # we specifically do not retain a reference to the original entity
        self.intel_entity_id = intel_entity_id
        self.intel_entity_id_prefix = id_prefix
        self.intel_entity_short_id = intel_entity_short_id
        self.intel_entity_type:Type[T] = intel_entity_type
        super().__init__(*args, **kwargs)

    def __str__(self) -> str:
        return f'{self.short_id()} {type(self)} on {self.intel_entity_short_id} valid:{self.is_valid()} fresh:{self.is_fresh()}'

    def match_criteria(self) -> core.IntelMatchCriteria:
        return EntityIntelMatchCriteria(self.intel_entity_id)

    def matches(self, other:core.Intel) -> bool:
        if not isinstance(other, EntityIntel):
            return False
        return other.intel_entity_id == self.intel_entity_id

    def sanity_check(self) -> None:
        super().sanity_check()
        assert(issubclass(self.intel_entity_type, core.Entity))
        if core.Gamestate.gamestate.contains_entity(self.intel_entity_id):
            entity = core.Gamestate.gamestate.get_entity(self.intel_entity_id, self.intel_entity_type)
            assert(entity.short_id() == self.intel_entity_short_id)
            assert(entity.id_prefix == self.intel_entity_id_prefix)

class SectorEntityIntel[T:core.SectorEntity](EntityIntel[T]):
    @classmethod
    def create_sector_entity_intel[T2:"SectorEntityIntel[T]"](cls:Type[T2], entity:T, gamestate:core.Gamestate, *args:Any, **kwargs:Any) -> T2:
        assert(entity.sector)
        sector_id = entity.sector.entity_id
        loc = entity.loc
        radius = entity.radius
        is_static = entity.is_static
        entity_id = entity.entity_id
        id_prefix = entity.id_prefix
        entity_short_id = entity.short_id()
        entity_class = type(entity)
        intel = cls(*args, sector_id, loc, radius, is_static, entity_id, id_prefix, entity_short_id, entity_class, gamestate, **kwargs)

        # schedule this intel to expire only once per shared intel
        if intel.expires_at < np.inf:
            gamestate.schedule_task(intel.expires_at, ExpireIntelTask.expire_intel(intel))

        return intel

    def __init__(self, sector_id:uuid.UUID, loc:npt.NDArray[np.float64], radius:float, is_static:bool, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.sector_id = sector_id
        self.loc = loc
        self.radius = radius
        self.is_static = is_static

    def sanity_check(self) -> None:
        super().sanity_check()
        #TODO: is it possible sectors go away?
        assert(issubclass(self.intel_entity_type, core.SectorEntity))
        if core.Gamestate.gamestate.contains_entity(self.intel_entity_id):
            sector = core.Gamestate.gamestate.get_entity(self.sector_id, core.Sector)
            sector_entity = core.Gamestate.gamestate.get_entity(self.intel_entity_id, self.intel_entity_type)
            assert(sector_entity.radius == self.radius)
            assert(sector_entity.is_static == self.is_static)
            if self.is_static:
                assert(util.both_isclose(sector_entity.loc, self.loc))
        else:
            assert(not self.is_static)

    def create_sensor_identity(self) -> core.SensorIdentity:
        """ creates a sensor identity out of this intel.

        useful for creating sensor images based on historical intel. """
        return core.SensorIdentity(
                object_type=self.intel_entity_type,
                id_prefix=self.intel_entity_id_prefix,
                entity_id=self.intel_entity_id,
                short_id=self.intel_entity_short_id,
                radius=self.radius,
                is_static=self.is_static
        )

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

class EconAgentIntel(EntityIntel[core.EconAgent]):
    @classmethod
    def create_econ_agent_intel(cls, econ_agent:core.EconAgent, gamestate:core.Gamestate, *args:Any, **kwargs:Any) -> "EconAgentIntel":
        entity_id = econ_agent.entity_id
        entity_id_prefix = econ_agent.id_prefix
        entity_short_id = econ_agent.short_id()
        entity_class = type(econ_agent)
        agent_intel = EconAgentIntel(entity_id, entity_id_prefix, entity_short_id, entity_class, gamestate, **kwargs)
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

class SectorHexMatchCriteria(core.IntelMatchCriteria):
    def __init__(self, sector_id:uuid.UUID, hex_loc:npt.NDArray[np.float64], is_static:bool) -> None:
        self.sector_id = sector_id
        self.hex_loc = hex_loc
        self.is_static = is_static

    def __hash__(self) -> int:
        #TODO: is it bad that we're hashing a float here but using isclose
        # below for comparison? (hint: yes, but does it matter?)
        return hash((self.sector_id, util.int_coords(self.hex_loc)))

    def __eq__(self, other:Any) -> bool:
        if not isinstance(other, type(self)):
            return False
        return self.is_static == other.is_static and self.sector_id == other.sector_id and util.both_isclose(self.hex_loc, other.hex_loc)

    def matches(self, intel:"core.Intel") -> bool:
        if not isinstance(intel, SectorHexIntel):
            return False
        return self.is_static == intel.is_static and self.sector_id == intel.sector_id and util.both_isclose(self.hex_loc, intel.hex_loc)

class SectorHexIntel(core.Intel):
    def __init__(self, sector_id:uuid.UUID, hex_loc:npt.NDArray[np.float64], is_static:bool, entity_count:int, type_counts:dict[str,int], *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.sector_id = sector_id
        self.hex_loc = hex_loc
        self.is_static = is_static
        self.entity_count = entity_count
        self.type_counts = type_counts

    def match_criteria(self) -> core.IntelMatchCriteria:
        return SectorHexMatchCriteria(self.sector_id, self.hex_loc, self.is_static)

    def matches(self, other:core.Intel) -> bool:
        if not isinstance(other, SectorHexIntel):
            return False
        return self.is_static == other.is_static and self.sector_id == other.sector_id and util.both_isclose(self.hex_loc, other.hex_loc)


# Intel Witness Actions

class IdentifyAsteroidAction(events.Action):
    def act(self,
            character:core.Character,
            event_type:int,
            event_context:Mapping[int,int],
            event_args: MutableMapping[str, Union[int,float,str,bool]],
            action_args: Mapping[str, Union[int,float,str,bool]]
    ) -> None:
        asteroid = self.gamestate.get_entity_short(self.ck(event_context, sensors.ContextKeys.TARGET), sector_entity.Asteroid)
        entity_intel = character.intel_manager.get_intel(EntityIntelMatchCriteria(asteroid.entity_id), AsteroidIntel)
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
        entity_intel = character.intel_manager.get_intel(EntityIntelMatchCriteria(sentity.entity_id), SectorEntityIntel)
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
        econ_agent_intel = character.intel_manager.get_intel(EntityIntelMatchCriteria(agent.entity_id), EconAgentIntel)
        if not econ_agent_intel or not econ_agent_intel.is_fresh():
            fresh_until = self.gamestate.timestamp + self.econ_intel_ttl*0.2
            expires_at = self.gamestate.timestamp + self.econ_intel_ttl
            character.intel_manager.add_intel(EconAgentIntel.create_econ_agent_intel(agent, self.gamestate, expires_at=expires_at, fresh_until=fresh_until))
        #TODO: what other intel do we want to create now that we're docked?

class ScanAction(events.Action):
    def __init__(self, *args:Any, static_intel_ttl:float=3600, dynamic_intel_ttl:float=900, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.static_intel_ttl = static_intel_ttl
        self.dynamic_intel_ttl = dynamic_intel_ttl

    def act(self,
            character:core.Character,
            event_type:int,
            event_context:Mapping[int,int],
            event_args: MutableMapping[str, Union[int,float,str,bool]],
            action_args: Mapping[str, Union[int,float,str,bool]]
    ) -> None:
        sector = core.Gamestate.gamestate.get_entity_short(self.ck(event_context, sensors.ContextKeys.SECTOR), core.Sector)
        detector = core.Gamestate.gamestate.get_entity_short(self.ck(event_context, sensors.ContextKeys.DETECTOR), core.CrewedSectorEntity)
        assert(character == detector.captain)

        #TODO: figure out the set of hexes that are relevant
        passive_range, thrust_range, active_range = sector.sensor_manager.sensor_ranges(detector)
        sector_hexes = util.hexes_within_pixel_dist(detector.loc, passive_range, sector.hex_size)

        # eliminate hexes we already have good intel for
        # and create a dict from hex coord to info about the hex
        static_intel:dict[tuple[int, int], SectorHexIntel] = {}
        dynamic_intel:dict[tuple[int, int], SectorHexIntel] = {}
        static_fresh_until = self.gamestate.timestamp + self.static_intel_ttl*0.2
        static_expires_at = self.gamestate.timestamp + self.static_intel_ttl
        dynamic_fresh_until = self.gamestate.timestamp + self.dynamic_intel_ttl*0.2
        dynamic_expires_at = self.gamestate.timestamp + self.dynamic_intel_ttl

        for hex_coords in sector_hexes:
            static_criteria = SectorHexMatchCriteria(sector.entity_id, hex_coords, True)
            s_intel = character.intel_manager.get_intel(static_criteria, SectorHexIntel)
            if not s_intel or not s_intel.is_fresh():
                static_intel[util.int_coords(hex_coords)] = SectorHexIntel(sector.entity_id, hex_coords, True, 0, {}, self.gamestate, expires_at=static_expires_at, fresh_until=static_fresh_until)

            dynamic_criteria = SectorHexMatchCriteria(sector.entity_id, hex_coords, False)
            d_intel = character.intel_manager.get_intel(dynamic_criteria, SectorHexIntel)
            if not d_intel or not d_intel.is_fresh():
                dynamic_intel[util.int_coords(hex_coords)] = SectorHexIntel(sector.entity_id, hex_coords, False, 0, {}, self.gamestate, expires_at=dynamic_expires_at, fresh_until=dynamic_fresh_until)

        # iterate over all the sensor images we've got accumulating info for
        # the hex they lie in, if it's within range
        for image in detector.sensor_settings.images:
            if not image.identified:
                continue

            h_coords = util.int_coords(util.axial_round(util.pixel_to_pointy_hex(image.loc, sector.hex_size)))

            if image.identity.is_static:
                intels = static_intel
            else:
                intels = dynamic_intel

            if h_coords in static_intel:
                intels[h_coords].entity_count += 1
                type_name = util.fullname(image.identity.object_type)
                if type_name in intels[h_coords].type_counts:
                    intels[h_coords].type_counts[type_name] += 1
                else:
                    intels[h_coords].type_counts[type_name] = 1

        for s_intel in static_intel.values():
            character.intel_manager.add_intel(s_intel)
        for d_intel in dynamic_intel.values():
            character.intel_manager.add_intel(d_intel)


def pre_initialize(event_manager:events.EventManager) -> None:
    event_manager.register_action(IdentifyAsteroidAction(), "identify_asteroid", "intel")
    event_manager.register_action(IdentifySectorEntityAction(), "identify_sector_entity", "intel")
    event_manager.register_action(DockingAction(), "witness_docking", "intel")
    event_manager.register_action(ScanAction(), "witness_scan", "intel")
