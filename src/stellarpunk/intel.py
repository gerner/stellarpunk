import logging
import uuid
import collections
import functools
import abc
from collections.abc import Collection, MutableMapping, Mapping, Iterator
from typing import Type, Any, Optional, TypeVar, Union

import numpy as np
import numpy.typing as npt

from stellarpunk import core, events, sensors, util, config
#TODO: sector_entity shouldn't be in core
from stellarpunk.core import sector_entity

class Intel(core.AbstractIntel):
    @classmethod
    def create_intel[T:Intel](cls:Type[T], *args:Any, **kwargs:Any) -> T:
        obj = cls(*args, _check_flag=True, **kwargs)
        if obj.expires_at < np.inf:
            obj.expire_task = ExpireIntelTask.expire_intel(obj)
        return obj

    def __init__(self, *args:Any, _check_flag:bool=False, **kwargs:Any) -> None:
        assert(_check_flag)
        super().__init__(*args, **kwargs)
        self.expire_task:Optional[ExpireIntelTask] = None

    def _unobserve(self, observer:core.IntelObserver) -> None:
        super()._unobserve(observer)
        if len(self.observers) == 0 and self.expire_task:
            core.Gamestate.gamestate.unschedule_task(self.expire_task)
            self.expire_task = None

    def sanity_check(self) -> None:
        super().sanity_check()
        assert len(self.observers) > 0
        if self.expires_at < np.inf:
            # we should have an expiration task and it should be scheduled
            # if we have already expired, or all observers dropped us, we
            # should be gone from the entity store and no one should be sanity
            # checking us.
            assert self.expire_task
            assert core.Gamestate.gamestate.is_task_scheduled(self.expire_task)
        if not self.expires_at < np.inf:
            assert self.expire_task is None

class ExpireIntelTask(core.ScheduledTask):
    @classmethod
    def expire_intel(cls, intel:core.AbstractIntel) -> "ExpireIntelTask":
        task = ExpireIntelTask()
        task.intel = intel
        core.Gamestate.gamestate.schedule_task(intel.expires_at, task)
        return task

    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.intel:core.AbstractIntel = None # type: ignore

    def act(self) -> None:
        self.intel.expire()

    def sanity_check(self, ts:float) -> None:
        assert self.intel.expires_at < np.inf
        assert len(self.intel.observers) > 0
        assert util.isclose(ts, self.intel.expires_at)
        assert core.Gamestate.gamestate.contains_entity(self.intel.entity_id)

class IntelManager(core.IntelObserver, core.AbstractIntelManager):
    """ Manages known intel and creates intel items for a character. """
    @classmethod
    def create_intel_manager(cls, gamestate:core.Gamestate, *args:Any, **kwargs:Any) -> "IntelManager":
        intel_manager = cls(gamestate, *args, **kwargs)
        return intel_manager

    def __init__(self, gamestate:core.Gamestate, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(util.fullname(self))
        self.character:Character = None # type: ignore
        self.gamestate = gamestate
        self._intel_map:dict[core.IntelMatchCriteria, uuid.UUID] = {}
        self._intel_tree = util.TypeTree(core.AbstractIntel)

    def sanity_check(self) -> None:
        assert self.character.intel_manager == self

        intel_count = 0
        entity_intel_count = 0
        for intel in self._intel_tree:
            intel_count += 1
            match_criteria = intel.match_criteria()
            assert intel.entity_id == self._intel_map[match_criteria]

    # core.Observable

    @property
    def observable_id(self) -> uuid.UUID:
        # we strongly assume there's a 1:1 relationship between intel managers
        # and characters. this intel manager can be retrieved from gamestate by
        # finding the corresponding character and getting its intel manager,
        # which is this object
        return self.character.entity_id

    # core.IntelObserver

    @property
    def observer_id(self) -> uuid.UUID:
        return self.character.entity_id

    def intel_expired(self, intel:core.AbstractIntel) -> None:
        self.logger.debug(f'{self.character} expired intel {intel}')
        self._remove_intel(intel)

        #TODO: are there other ways intel might be removed?
        for observer in self.observers:
            observer.intel_removed(self, intel)

    def _remove_intel(self, old_intel:core.AbstractIntel) -> None:
        self.logger.debug(f'{self.character} removing intel {old_intel}')
        del self._intel_map[old_intel.match_criteria()]
        old_intel.unobserve(self)
        self._intel_tree.remove(old_intel)

    def _add_intel(self, intel:core.AbstractIntel) -> None:
        self.logger.debug(f'{self.character} adding intel {intel}')
        intel.observe(self)
        self._intel_map[intel.match_criteria()] = intel.entity_id
        self._intel_tree.add(intel)

    def add_intel(self, intel:core.AbstractIntel) -> bool:
        old_intel:Optional[core.AbstractIntel] = self.get_intel(intel.match_criteria(), type(intel))

        # if we already have fresh matching intel
        if old_intel and old_intel.created_at > intel.created_at:
            # ours is better, ignore theirs
            return False
        elif old_intel:
            # theirs is better, drop ours

            for observer in self.observers:
                observer.intel_removed(self, old_intel)
            self._remove_intel(old_intel)

        self._add_intel(intel)

        for observer in self.observers:
            observer.intel_added(self, intel)

        return True

    def intel[T:core.AbstractIntel](self, match_criteria:core.IntelMatchCriteria, cls:Optional[Type[T]]=None) -> list[T]:
        if cls is None:
            cls = core.AbstractIntel # type: ignore
        assert(cls is not None)
        if match_criteria in self._intel_map:
            intel = self._intel_map[match_criteria]
            assert(isinstance(intel, cls))
            return [intel]

        return list(x for x in self._intel_tree.get(cls) if match_criteria.matches(x))

    def get_intel[T:core.AbstractIntel](self, match_criteria:core.IntelMatchCriteria, cls:Type[T]) -> Optional[T]:
        # we assume the intel is of the right type
        # check if we have such intel via exact match

        if match_criteria.is_exact():
            if match_criteria in self._intel_map:
                intel_id = self._intel_map[match_criteria]
                intel = self.gamestate.get_entity(intel_id, cls)
                return intel
        else:
            for intel in self.intel(match_criteria, cls):
                return intel
        return None

    def register_intel_interest(self, interest:core.IntelMatchCriteria, source:Optional[core.IntelMatchCriteria]=None) -> None:
        self.logger.debug(f'{self.character} registered interest {interest} {source=}')
        for observer in self.observers:
            observer.intel_desired(self, interest, source=source)

    def unregister_intel_interest(self, interest:core.IntelMatchCriteria) -> None:
        self.logger.debug(f'{self.character} unregistered interest {interest}')
        for observer in self.observers:
            observer.intel_undesired(self, interest)


class TrivialMatchCriteria(core.IntelMatchCriteria):
    def __init__(self, *args:Any, cls:Type[core.AbstractIntel]=core.AbstractIntel, **kwargs:Any):
        self.cls = cls
    def is_exact(self) -> bool:
        return False
    def matches(self, other:core.AbstractIntel) -> bool:
        if not isinstance(other, self.cls):
            return False
        return True

class EntityIntelMatchCriteria(core.IntelMatchCriteria):
    def __init__(self, entity_id:uuid.UUID) -> None:
        self.entity_id = entity_id

    def __hash__(self) -> int:
        return hash(self.entity_id)

    def __eq__(self, other:Any) -> bool:
        if not isinstance(other, EntityIntelMatchCriteria):
            return False
        return self.entity_id == other.entity_id

    def matches(self, other:core.AbstractIntel) -> bool:
        if not isinstance(other, EntityIntel):
            return False
        return other.intel_entity_id == self.entity_id

class EntityIntel[T:core.Entity](Intel):
    def __init__(self, *args:Any, intel_entity_id:uuid.UUID, intel_entity_type:Type[T], intel_entity_name:str, intel_entity_description:str, **kwargs:Any) -> None:
        # we need to set these fields before the super constructor because we
        # override __str__ which might be called in a super constructor
        # we specifically do not retain a reference to the original entity
        self.intel_entity_id = intel_entity_id
        self.intel_entity_type:Type[T] = intel_entity_type
        self.intel_entity_name = intel_entity_name
        self.intel_entity_description = intel_entity_description
        super().__init__(*args, **kwargs)
        if not isinstance(self.name, str):
             raise Exception()

    @property
    def intel_entity_short_id(self) -> str:
        return self.intel_entity_type.create_short_id(self.intel_entity_id)

    @property
    def intel_entity_id_prefix(self) -> str:
        return self.intel_entity_type.id_prefix

    def __str__(self) -> str:
        return f'{self.short_id()} {type(self)} on {self.intel_entity_short_id} valid:{self.is_valid()} fresh:{self.is_fresh()}'

    def match_criteria(self) -> core.IntelMatchCriteria:
        return EntityIntelMatchCriteria(self.intel_entity_id)

    def matches(self, other:core.AbstractIntel) -> bool:
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

class UniverseView:
    """ Short-lived object representing a character's view of the universe.

    This includes all sectors, all travel gates and edges between sectors.
    """
    @classmethod
    def create(cls, character:core.Character) -> "UniverseView":
        #TODO: should this be intel itself?

        # construct known sector graph in weighted adjacency matrix form
        sector_intels = character.intel_manager.intel(TrivialMatchCriteria(cls=SectorIntel), SectorIntel)
        sector_idx_lookup = {x.intel_entity_id: i for i,x in enumerate(sector_intels)}
        sector_intel_lookup = {x.intel_entity_id: x for x in sector_intels}

        travel_gate_intels = character.intel_manager.intel(TrivialMatchCriteria(cls=TravelGateIntel), TravelGateIntel)

        # augment idx lookup with sectors we know of from travel gate intel, but don't 
        i = len(sector_idx_lookup)
        for gate_intel in travel_gate_intels:
            if gate_intel.destination_id in sector_idx_lookup:
                continue
            sector_idx_lookup[gate_intel.destination_id] = i
            i += 1

        adj_matrix = np.ones((len(sector_idx_lookup), len(sector_idx_lookup))) * np.inf
        for gate_intel in travel_gate_intels:
            # model the cost the edge as the cost of travelling from one sector
            # center to edge plus other sector edge back to center

            # we could be more precise by modelling sector links as nodes and
            # sector paths between gates as edges
            # since the cost to travel from sector to sector is actually
            # incurred while travelling from gate to gate within a sector
            # but that sounds complicated and this is an easy approximation

            # we must have intel on the source
            a_idx = sector_idx_lookup[gate_intel.sector_id]
            b_idx = sector_idx_lookup[gate_intel.destination_id]
            cost_estimate = sector_intel_lookup[gate_intel.sector_id].radius
            # we might not have intel on the destination
            if gate_intel.destination_id not in sector_intel_lookup:
                cost_estimate *= 2.0
            else:
                cost_estimate += sector_intel_lookup[gate_intel.destination_id].radius
            adj_matrix[a_idx, b_idx] = cost_estimate

        return cls(sector_intels, travel_gate_intels, sector_idx_lookup, sector_intel_lookup, adj_matrix)

    def __init__(self, sector_intels:list["SectorIntel"], travel_gate_intels:list["TravelGateIntel"], sector_idx_lookup:dict[uuid.UUID, int], sector_intel_lookup:dict[uuid.UUID, "SectorIntel"], adj_matrix:npt.NDArray[np.float64]):
        self.sector_intels = sector_intels
        self.travel_gate_intels = travel_gate_intels
        self.sector_idx_lookup = sector_idx_lookup
        self.sector_id_lookup = {v:k for k,v in sector_idx_lookup.items()}
        self.sector_intel_lookup = sector_intel_lookup
        self.gate_intel_lookup = {(x.sector_id, x.destination_id):x for x in travel_gate_intels}

        def group_by_source(acc:collections.defaultdict[uuid.UUID, list[TravelGateIntel]], x:TravelGateIntel) -> collections.defaultdict[uuid.UUID, list[TravelGateIntel]]:
            acc[x.sector_id].append(x)
            return acc
        def group_by_destination(acc:collections.defaultdict[uuid.UUID, list[TravelGateIntel]], x:TravelGateIntel) -> collections.defaultdict[uuid.UUID, list[TravelGateIntel]]:
            acc[x.destination_id].append(x)
            return acc
        self.gates_by_source:collections.defaultdict[uuid.UUID, list[TravelGateIntel]] = collections.defaultdict(list)
        functools.reduce(group_by_source, travel_gate_intels, self.gates_by_source)
        self.gates_by_destination:collections.defaultdict[uuid.UUID, list[TravelGateIntel]] = collections.defaultdict(list)
        functools.reduce(group_by_destination, travel_gate_intels, self.gates_by_destination)
        self.adj_matrix = adj_matrix

        if len(self.travel_gate_intels) > 0:
            self.max_edge_length = max(
                util.distance(
                    self.sector_intel_lookup[x.sector_id].loc,
                    self.sector_intel_lookup[x.destination_id].loc if x.destination_id in sector_intel_lookup else self.sector_intel_lookup[x.sector_id].radius * 10.0
                ) for x in self.travel_gate_intels
            )
        else:
            self.max_edge_length = 0.0

    @functools.cache
    def _sector_dfs(self, source_idx:int) -> tuple[Mapping[int, int], Mapping[int, float]]:
        return util.dijkstra(self.adj_matrix, source_idx, len(self.sector_idx_lookup))

    def sector_ids_by_distance(self, source_id:uuid.UUID) -> Iterator[uuid.UUID]:
        source_idx = self.sector_idx_lookup[source_id]
        path_tree, distance_map = self._sector_dfs(source_idx)
        for sector_idx, distance in sorted(distance_map.items(), key=lambda x: x[1]):
            yield self.sector_id_lookup[sector_idx]

    def sectors_by_distance(self, source_id:uuid.UUID) -> Iterator["SectorIntel"]:
        source_idx = self.sector_idx_lookup[source_id]
        path_tree, distance_map = self._sector_dfs(source_idx)
        for sector_idx, distance in sorted(distance_map.items(), key=lambda x: x[1]):
            # skip sectors we don't have intel for
            if sector_idx < len(self.sector_intels):
                yield self.sector_intels[sector_idx]

    def compute_path(self, source_id:uuid.UUID, target_id:uuid.UUID) -> Optional[list[tuple["SectorIntel", "TravelGateIntel", Optional["TravelGateIntel"], float]]]:
        """ Computes the shortest known path from character's sector to target

        returns: list of edges in the form tuple:
            source_sector_intel,
            outbound_gate_intel in source sector,
            inbound_gate_intel in target sector,
            edge_distance from source to target

            or None if no known path exists
        """

        if source_id not in self.sector_idx_lookup:
            # we don't know about source sector
            return None

        if target_id not in self.sector_idx_lookup:
            # we don't know about target sector
            return None

        source_idx = self.sector_idx_lookup[source_id]
        target_idx = self.sector_idx_lookup[target_id]

        path_tree, distance_map = self._sector_dfs(source_idx)

        if target_idx not in path_tree:
            # no path to target found
            return None

        # translate route implied by path_tree
        idx_path:list[int] = []
        cum_distances:list[float] = []
        u = target_idx
        idx_path.append(u)
        cum_distances.append(distance_map[u])
        while u != source_idx:
            u = path_tree[u]
            idx_path.append(u)
            cum_distances.append(distance_map[u])

        # compute per-edge distances as cum_dist[i] - cum_dist[i+1]
        distances = list(a-b for a,b in zip(cum_distances[0:-1], cum_distances[1:]))

        # make from,to pair path and distances in source -> target order
        idx_path.reverse()
        idx_path_pairs = zip(idx_path[0:-1], idx_path[1:])
        distances.reverse()

        # construct return. note that some destination gates might not exist
        sector_path = list(
            (
                self.sector_intels[u],
                self.gate_intel_lookup[self.sector_id_lookup[u], self.sector_id_lookup[v]],
                self.gate_intel_lookup[self.sector_id_lookup[v], self.sector_id_lookup[u]] if (self.sector_id_lookup[v], self.sector_id_lookup[u]) in self.gate_intel_lookup else None,
                dist
            ) for (u,v), dist in zip(idx_path_pairs, distances)
        )

        return sector_path

class SectorIntel(EntityIntel[core.Sector]):
    @property
    def loc(self) -> npt.NDArray[np.float64]:
        return core.Gamestate.gamestate.get_entity(self.intel_entity_id, core.Sector).loc
    @property
    def radius(self) -> float:
        return core.Gamestate.gamestate.get_entity(self.intel_entity_id, core.Sector).radius
    @property
    def culture(self) -> str:
        return core.Gamestate.gamestate.get_entity(self.intel_entity_id, core.Sector).culture

    @property
    def hex_size(self) -> float:
        return core.Gamestate.gamestate.get_entity(self.intel_entity_id, core.Sector).hex_size

    def get_hex_coords(self, coords:npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return core.Gamestate.gamestate.get_entity(self.intel_entity_id, core.Sector).get_hex_coords(coords)

    def get_coords_from_hex(self, hex_coords:npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return core.Gamestate.gamestate.get_entity(self.intel_entity_id, core.Sector).get_coords_from_hex(hex_coords)

    def compute_path(self, source_id:uuid.UUID, character:core.Character) -> Optional[list[tuple["SectorIntel", "TravelGateIntel", "TravelGateIntel", float]]]:
        """ Computes the shortest known path to this sector

        source_id: starting sector id
        character: the character whose intel we'll use

        returns: list of edges in the form tuple:
            source_sector_intel,
            outbound_gate_intel in source sector,
            inbound_gate_intel in target sector,
            edge_distance from source to target

            or None if no known path exists
        """

        # construct known sector graph in weighted adjacency matrix form
        universe_view = UniverseView.create(character)
        source_idx = universe_view.sector_idx_lookup.get(source_id)
        if source_idx is None:
            raise ValueError(f'{character} has no intel for source sector {source_id}')
        target_idx = universe_view.sector_idx_lookup.get(self.intel_entity_id)
        if target_idx is None:
            raise ValueError(f'{character} has no intel for target sector {self.intel_entity_id}')

        if source_idx == target_idx:
            # trivial path to stay here
            return []


        # compute shortest path from source to target, if one
        path_tree, distance_map = util.dijkstra(universe_view.adj_matrix, source_idx, target_idx)

        if source_idx not in path_tree:
            # no path found
            return None

        # translate route implied by path_tree
        idx_path:list[int] = []
        cum_distances:list[float] = []
        u = target_idx
        idx_path.append(u)
        cum_distances.append(distance_map[u])
        while u != source_idx:
            u = path_tree[u]
            idx_path.append(u)
            cum_distances.append(distance_map[u])

        # compute per-edge distances as cum_dist[i] - cum_dist[i+1]
        distances = list(a-b for a,b in zip(cum_distances[0:-1], cum_distances[1:]))

        # make from,to pair path and distances in source -> target order
        idx_path.reverse()
        idx_path_pairs = zip(idx_path[0:-1], idx_path[1:])
        distances.reverse()

        # assume travel gate intel in both directions for every step in path
        sector_path = list(
            (
                universe_view.sector_intels[u],
                universe_view.gate_intel_lookup[universe_view.sector_intels[u].intel_entity_id, universe_view.sector_intels[v].intel_entity_id],
                universe_view.gate_intel_lookup[universe_view.sector_intels[v].intel_entity_id, universe_view.sector_intels[u].intel_entity_id],
                dist
            ) for (u,v), dist in zip(idx_path_pairs, distances)
        )

        return sector_path

class SectorEntityIntel[T:core.SectorEntity](EntityIntel[T]):
    @classmethod
    def create_sector_entity_intel[T2:"SectorEntityIntel[T]"](cls:Type[T2], entity:T, gamestate:core.Gamestate, *args:Any, **kwargs:Any) -> T2:
        assert(entity.sector)
        sector_id = entity.sector.entity_id
        loc = entity.loc
        mass = entity.mass
        radius = entity.radius
        is_static = entity.is_static
        entity_id = entity.entity_id
        entity_class = type(entity)
        intel = cls.create_intel(*args, sector_id, loc, mass, radius, is_static, gamestate, intel_entity_id=entity_id, intel_entity_type=entity_class, intel_entity_name=entity.name, intel_entity_description=entity.description, **kwargs)

        return intel

    def __init__(self, sector_id:uuid.UUID, loc:npt.NDArray[np.float64], mass:float, radius:float, is_static:bool, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.sector_id = sector_id
        self.loc = loc
        self.mass = mass
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
                mass=self.mass,
                radius=self.radius,
                is_static=self.is_static,
                sector_id=self.sector_id
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

class StationIntel(SectorEntityIntel[sector_entity.Station]):
    @classmethod
    def create_station_intel(cls, station:sector_entity.Station, gamestate:core.Gamestate, *args:Any, **kwargs:Any) -> "StationIntel":
        inputs = set(gamestate.production_chain.inputs_of(station.resource))
        return cls.create_sector_entity_intel(station, gamestate, *args, station.resource, inputs, **kwargs)

    def __init__(self, resource:int, inputs:set[int], *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.resource = resource
        self.inputs = inputs

class TravelGateIntel(SectorEntityIntel[sector_entity.TravelGate]):
    def __init__(self, *args:Any, destination_id:uuid.UUID, direction:float, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.destination_id = destination_id
        self.direction = direction

class EconAgentIntel(EntityIntel[core.EconAgent]):
    @classmethod
    def create_econ_agent_intel(cls, econ_agent:core.EconAgent, gamestate:core.Gamestate, *args:Any, **kwargs:Any) -> "EconAgentIntel":
        entity_id = econ_agent.entity_id
        entity_class = type(econ_agent)
        agent_intel = cls.create_intel(gamestate, intel_entity_id=entity_id, intel_entity_type=entity_class, intel_entity_name=econ_agent.name, intel_entity_description=econ_agent.description, **kwargs)
        #TODO: econ agents are not always associated with sector entities!
        underlying_entity = gamestate.agent_to_entity[entity_id]
        agent_intel.underlying_entity_type = type(underlying_entity)
        agent_intel.underlying_entity_id = underlying_entity.entity_id
        for resource in econ_agent.sell_resources():
            # they might want to sell and just not have any inventory at the
            # moment
            #assert econ_agent.inventory(resource) > 0.0
            assert econ_agent.sell_price(resource) < np.inf
            agent_intel.sell_offers[resource] = (
                econ_agent.sell_price(resource),
                econ_agent.inventory(resource),
            )

        for resource in econ_agent.buy_resources():
            # they might want to buy and just not have any inventory at the
            # moment
            #assert econ_agent.budget(resource) > 0.0
            assert econ_agent.buy_price(resource) > 0.0
            agent_intel.buy_offers[resource] = (
                econ_agent.buy_price(resource),
                econ_agent.budget(resource) / econ_agent.buy_price(resource),
            )

        return agent_intel

    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.underlying_entity_id:uuid.UUID = None # type: ignore
        self.underlying_entity_type:Type[core.Entity] = None # type: ignore

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

    def matches(self, intel:"core.AbstractIntel") -> bool:
        if not isinstance(intel, SectorHexIntel):
            return False
        return self.is_static == intel.is_static and self.sector_id == intel.sector_id and util.both_isclose(self.hex_loc, intel.hex_loc)

class SectorHexIntel(Intel):
    def __init__(self, sector_id:uuid.UUID, hex_loc:npt.NDArray[np.float64], is_static:bool, entity_count:int, type_counts:dict[Type[core.SectorEntity],int], *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.sector_id = sector_id
        self.hex_loc = hex_loc
        self.is_static = is_static
        self.entity_count = entity_count
        self.type_counts = type_counts

    def match_criteria(self) -> core.IntelMatchCriteria:
        return SectorHexMatchCriteria(self.sector_id, self.hex_loc, self.is_static)

    def matches(self, other:core.AbstractIntel) -> bool:
        if not isinstance(other, SectorHexIntel):
            return False
        return self.is_static == other.is_static and self.sector_id == other.sector_id and util.both_isclose(self.hex_loc, other.hex_loc)


# Intel Interest Criteria

class IntelPartialCriteria(core.IntelMatchCriteria):
    """ A type of match criteria that matches many intel """
    def is_exact(self) -> bool:
        return False

class SectorEntityPartialCriteria(IntelPartialCriteria):
    def __init__(self, cls:Type[core.SectorEntity]=core.SectorEntity, is_static:Optional[bool]=None, sector_id:Optional[uuid.UUID]=None, jump_distance:int=0):
        self.cls = cls
        self.is_static = is_static
        self.sector_id = sector_id
        self.jump_distance = jump_distance

    def __str__(self) -> str:
        items = []
        if self.cls:
            items.append(f'cls={self.cls}')
        if self.is_static is not None:
            items.append(f'is_static={self.is_static}')
        if self.sector_id:
            items.append(f'sector_id={self.sector_id}')
        return f'{util.fullname(self)} {", ".join(items)}'

    def matches(self, intel:core.AbstractIntel) -> bool:
        if not isinstance(intel, SectorEntityIntel):
            return False
        if not issubclass(intel.intel_entity_type, self.cls):
            return False
        if self.is_static is not None and intel.is_static != self.is_static:
            return False
        if self.sector_id is not None:
            if self.jump_distance == 0 and intel.sector_id != self.sector_id:
                return False
            else:
                jumps = core.Gamestate.gamestate.jump_distance(intel.sector_id, self.sector_id)
                if jumps is None or jumps > self.jump_distance:
                    return False
        return True

    def __hash__(self) -> int:
        return hash((self.cls, self.is_static, self.sector_id))

    def __eq__(self, other:Any) -> bool:
        if not isinstance(other, SectorEntityPartialCriteria):
            return False
        if self.cls != other.cls:
            return False
        if self.is_static != other.is_static:
            return False
        if self.sector_id != other.sector_id:
            return False
        if self.jump_distance != other.jump_distance:
            return False
        return True

class AsteroidIntelPartialCriteria(SectorEntityPartialCriteria):
    def __init__(self, *args:Any, resources:Optional[frozenset[int]]=None, **kwargs:Any) -> None:
        kwargs["cls"] = sector_entity.Asteroid
        kwargs["is_static"] = True
        super().__init__(*args, **kwargs)
        if resources is not None and len(resources) == 0:
            raise ValueError(f'resources must be non-empty if present')
        self.resources = resources

    def __str__(self) -> str:
        items = []
        if self.cls:
            items.append(f'cls={self.cls}')
        if self.is_static is not None:
            items.append(f'is_static={self.is_static}')
        if self.sector_id:
            items.append(f'sector_id={self.sector_id}')
        if self.resources:
            items.append(f'resources={self.resources}')
        return f'{util.fullname(self)} {", ".join(items)}'

    def matches(self, intel:core.AbstractIntel) -> bool:
        if not super().matches(intel):
            return False
        assert(isinstance(intel, AsteroidIntel))
        if self.resources is not None and intel.resource not in self.resources:
            return False
        return True

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.resources))

    def __eq__(self, other:Any) -> bool:
        if not isinstance(other, AsteroidIntelPartialCriteria):
            return False
        if not super().__eq__(other):
            return False
        if self.resources != other.resources:
            return False
        return True

class StationIntelPartialCriteria(SectorEntityPartialCriteria):
    def __init__(self, *args:Any, resources:Optional[frozenset[int]]=None, inputs:Optional[frozenset[int]]=None, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        if resources is not None and len(resources) == 0:
            raise ValueError(f'resources must be non-empty if present')
        if inputs is not None and len(inputs) == 0:
            raise ValueError(f'inputs must be non-empty if present')
        self.resources = resources
        self.inputs = inputs

    def __str__(self) -> str:
        items = []
        if self.cls:
            items.append(f'cls={self.cls}')
        if self.is_static is not None:
            items.append(f'is_static={self.is_static}')
        if self.sector_id:
            items.append(f'sector_id={self.sector_id}')
        if self.resources:
            items.append(f'resources={self.resources}')
        if self.inputs:
            items.append(f'inputs={self.inputs}')
        return f'{util.fullname(self)} {", ".join(items)}'

    def matches(self, intel:core.AbstractIntel) -> bool:
        if not isinstance(intel, StationIntel):
            return False
        if not super().matches(intel):
            return False
        if self.resources is not None and intel.resource not in self.resources:
            return False
        if self.inputs is not None and len(intel.inputs.intersection(self.inputs)) == 0:
            return False
        return True

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.resources, self.inputs))

    def __eq__(self, other:Any) -> bool:
        if not isinstance(other, StationIntelPartialCriteria):
            return False
        if not super().__eq__(other):
            return False
        if self.resources != other.resources:
            return False
        if self.inputs != other.inputs:
            return False
        return True

class SectorHexPartialCriteria(IntelPartialCriteria):
    def __init__(self, sector_id:Optional[uuid.UUID]=None, is_static:Optional[bool]=None, hex_loc:Optional[npt.NDArray[np.float64]]=None, hex_dist:Optional[float]=None, jump_distance:int=0):
        if hex_loc is not None:
            if sector_id is None:
                raise ValueError(f'cannot specify a location without a sector')
            if hex_dist is None:
                hex_dist = 0.0
        else:
            if hex_dist is not None:
                raise ValueError(f'cannot specify a dist without a location')

        self.sector_id = sector_id
        self.jump_distance = jump_distance
        self.hex_loc = hex_loc
        self.hex_dist = hex_dist
        self.is_static = is_static

    def __str__(self) -> str:
        items = []
        if self.sector_id:
            items.append(f'sector_id={self.sector_id}')
        if self.hex_loc is not None:
            items.append(f'hex_loc={self.hex_loc}')
        if self.hex_dist is not None:
            items.append(f'hex_dist={self.hex_dist}')
        if self.is_static is not None:
            items.append(f'is_static={self.is_static}')
        return f'{util.fullname(self)} {", ".join(items)}'

    def matches(self, intel:core.AbstractIntel) -> bool:
        if not isinstance(intel, SectorHexIntel):
            return False
        if self.sector_id:
            if self.jump_distance == 0 and intel.sector_id != self.sector_id:
                return False
            else:
                jumps = core.Gamestate.gamestate.jump_distance(intel.sector_id, self.sector_id)
                if jumps is None or jumps > self.jump_distance:
                    return False
        if self.is_static is not None and intel.is_static != self.is_static:
            return False
        if self.hex_loc is not None and util.axial_distance(self.hex_loc, intel.hex_loc) > self.hex_dist:
            return False
        return True

    def __hash__(self) -> int:
        if self.hex_loc is not None:
            return hash((self.sector_id, self.is_static, (int(self.hex_loc[0]), int(self.hex_loc[1]))))
        else:
            return hash((self.sector_id, self.is_static))

    def __eq__(self, other:Any) -> bool:
        if not isinstance(other, SectorHexPartialCriteria):
            return False
        if self.sector_id != other.sector_id:
            return False
        if self.jump_distance != other.jump_distance:
            return False
        if self.is_static != other.is_static:
            return False
        if self.hex_loc is None != other.hex_loc is None:
            return False
        elif self.hex_loc is not None and util.both_isclose(self.hex_loc, other.hex_loc):
            return False
        if self.hex_dist is None != other.hex_dist is None:
            return False
        elif self.hex_dist is not None and not util.isclose(self.hex_dist, other.hex_dist):
            return False
        return True

class SectorPartialCriteria(IntelPartialCriteria):
    def __init__(self, sector_id:Optional[uuid.UUID]=None, jump_distance:int=0):
        self.sector_id = sector_id
        self.jump_distance = jump_distance

    def __str__(self) -> str:
        if self.sector_id:
            return f'{util.fullname(self)} sector_id={self.sector_id}'
        else:
            return f'{util.fullname(self)}'

    def matches(self, intel:core.AbstractIntel) -> bool:
        if not isinstance(intel, SectorIntel):
            return False
        if self.sector_id:
            if self.jump_distance == 0 and intel.intel_entity_id != self.sector_id:
                return False
            else:
                jumps = core.Gamestate.gamestate.jump_distance(intel.intel_entity_id, self.sector_id)
                if jumps is None or jumps > self.jump_distance:
                    return False
        return True

    def __hash__(self) -> int:
        return hash((self.sector_id, self.jump_distance))

    def __eq__(self, other:Any) -> bool:
        if not isinstance(other, SectorPartialCriteria):
            return False
        if other.sector_id != self.sector_id:
            return False
        if other.jump_distance != self.jump_distance:
            return False
        return True

class EconAgentSectorEntityPartialCriteria(IntelPartialCriteria):
    """ Partial criteria matching econ agents for static sector entities """
    def __init__(self,
            sector_id:Optional[uuid.UUID]=None,
            jump_distance:int=0,
            underlying_entity_id:Optional[uuid.UUID]=None,
            underlying_entity_type:Type[core.SectorEntity]=core.SectorEntity,
            buy_resources:Optional[frozenset[int]]=None,
            sell_resources:Optional[frozenset[int]]=None,
    ) -> None:
        self.sector_id = sector_id
        self.jump_distance = jump_distance
        assert(isinstance(underlying_entity_type, type))
        if not issubclass(underlying_entity_type, core.SectorEntity):
            raise ValueError(f'{underlying_entity_type=} must be a subclass of core.SectorEntity')
        self.underlying_entity_id = underlying_entity_id
        self.underlying_entity_type = underlying_entity_type
        if buy_resources is not None and len(buy_resources) == 0:
            raise ValueError(f'buy_resources must be non-empty if present')
        if sell_resources is not None and len(sell_resources) == 0:
            raise ValueError(f'sell_resources must be non-empty if present')
        self.buy_resources = buy_resources
        self.sell_resources = sell_resources

    def __str__(self) -> str:
        items = []
        if self.sector_id:
            items.append(f'sector_id={self.sector_id}')
        if self.underlying_entity_id:
            items.append(f'underlying_entity_id={self.underlying_entity_id}')
        items.append(f'underlying_entity_type={self.underlying_entity_type}')
        if self.buy_resources:
            items.append(f'buy_resources={self.buy_resources}')
        if self.sell_resources:
            items.append(f'sell_resources={self.sell_resources}')
        return f'{util.fullname(self)} {",".join(items)}'


    def matches(self, intel:core.AbstractIntel) -> bool:
        # check stuff about the intel itself
        if not isinstance(intel, EconAgentIntel):
            return False
        if self.underlying_entity_id is not None and self.underlying_entity_id != intel.underlying_entity_id:
            return False
        if not issubclass(intel.underlying_entity_type, self.underlying_entity_type):
            return False
        if self.buy_resources is not None and len(self.buy_resources.intersection(intel.buy_offers.keys())) == 0:
            return False
        if self.sell_resources is not None and len(self.sell_resources.intersection(intel.sell_offers.keys())) == 0:
            return False

        # check stuff about the underlying entity
        assert(issubclass(self.underlying_entity_type, core.SectorEntity))
        entity = core.Gamestate.gamestate.get_entity(intel.underlying_entity_id, self.underlying_entity_type)
        # this partial criteria is only used for static sector entities (i.e.
        # stations and planets)
        if not entity.is_static:
            return False
        # static entities should always be in a sector
        assert(entity.sector)
        if self.sector_id:
            if self.jump_distance == 0 and entity.sector.entity_id != self.sector_id:
                return False
            else:
                jumps = core.Gamestate.gamestate.jump_distance(entity.sector.entity_id, self.sector_id)
                if jumps is None or jumps > self.jump_distance:
                    return False
        return True

    def __hash__(self) -> int:
        return hash((self.sector_id, self.jump_distance, self.underlying_entity_id, self.underlying_entity_type, self.buy_resources, self.sell_resources))

    def __eq__(self, other:Any) -> bool:
        if not isinstance(other, EconAgentSectorEntityPartialCriteria):
            return False
        if self.sector_id != other.sector_id:
            return False
        if self.jump_distance != other.jump_distance:
            return False
        if self.underlying_entity_id != other.underlying_entity_id:
            return False
        if self.underlying_entity_type != other.underlying_entity_type:
            return False
        if self.buy_resources != other.buy_resources:
            return False
        if self.sell_resources != other.sell_resources:
            return False
        return True


# helpers to create intel
def add_asteroid_intel(asteroid:sector_entity.Asteroid, character:core.Character, gamestate:core.Gamestate, fresh_until:Optional[float]=None, expires_at:Optional[float]=None) -> bool:
    entity_intel = character.intel_manager.get_intel(EntityIntelMatchCriteria(asteroid.entity_id), AsteroidIntel)
    if not entity_intel or not entity_intel.is_fresh():
        ret = character.intel_manager.add_intel(AsteroidIntel.create_asteroid_intel(asteroid, gamestate, author_id=character.entity_id, fresh_until=fresh_until, expires_at=expires_at))
        assert ret
        return True
    else:
        return False

def add_station_intel(station:sector_entity.Station, character:core.Character, gamestate:core.Gamestate, fresh_until:Optional[float]=None, expires_at:Optional[float]=None) -> bool:
    entity_intel = character.intel_manager.get_intel(EntityIntelMatchCriteria(station.entity_id), StationIntel)
    if not entity_intel or not entity_intel.is_fresh():
        ret = character.intel_manager.add_intel(StationIntel.create_station_intel(station, gamestate, author_id=character.entity_id, fresh_until=fresh_until, expires_at=expires_at))
        assert ret
        return True
    else:
        return False

def add_travel_gate_intel(gate:sector_entity.TravelGate, character:core.Character, gamestate:core.Gamestate, fresh_until:Optional[float]=None, expires_at:Optional[float]=None) -> bool:
    entity_intel = character.intel_manager.get_intel(EntityIntelMatchCriteria(gate.entity_id), TravelGateIntel)
    if not entity_intel or not entity_intel.is_fresh():
        new_intel = TravelGateIntel.create_sector_entity_intel(gate, gamestate, destination_id=gate.destination.entity_id, direction=gate.direction, fresh_until=fresh_until, expires_at=expires_at, author_id=character.entity_id)

        ret = character.intel_manager.add_intel(new_intel)
        assert ret
        return True
    else:
        return False

def add_sector_entity_intel(sentity:core.SectorEntity, character:core.Character, gamestate:core.Gamestate, static_fresh_until:Optional[float]=None, static_expires_at:Optional[float]=None, dynamic_fresh_until:Optional[float]=None, dynamic_expires_at:Optional[float]=None) -> bool:
    entity_intel = character.intel_manager.get_intel(EntityIntelMatchCriteria(sentity.entity_id), SectorEntityIntel)
    if not entity_intel or not entity_intel.is_fresh():
        # intel about these select objects
        #TODO: TravelGate needs its own intel to include where the travel gate goes
        intel:SectorEntityIntel
        if sentity.is_static:
            intel = SectorEntityIntel.create_sector_entity_intel(sentity, gamestate, author_id=character.entity_id, fresh_until=static_fresh_until, expires_at=static_expires_at)
        # otherwise we'll give it some ttl
        else:
            intel = SectorEntityIntel.create_sector_entity_intel(sentity, gamestate, author_id=character.entity_id, expires_at=dynamic_expires_at, fresh_until=dynamic_fresh_until)
        ret = character.intel_manager.add_intel(intel)
        assert ret
        return True
    else:
        return False

def add_econ_agent_intel(agent:core.EconAgent, character:core.Character, gamestate:core.Gamestate, fresh_until:Optional[float]=None, expires_at:Optional[float]=None) -> bool:
    econ_agent_intel = character.intel_manager.get_intel(EntityIntelMatchCriteria(agent.entity_id), EconAgentIntel)
    if not econ_agent_intel or not econ_agent_intel.is_fresh():
        ret = character.intel_manager.add_intel(EconAgentIntel.create_econ_agent_intel(agent, gamestate, author_id=character.entity_id, expires_at=expires_at, fresh_until=fresh_until))
        assert ret
        return True
    else:
        return False

def add_sector_scan_intel(detector:core.CrewedSectorEntity, sector:core.Sector, character:core.Character, gamestate:core.Gamestate, static_fresh_until:Optional[float]=None, static_expires_at:Optional[float]=None, dynamic_fresh_until:Optional[float]=None, dynamic_expires_at:Optional[float]=None, loc:Optional[npt.NDArray[np.float64]]=None, images:Optional[Collection[core.AbstractSensorImage]]=None) -> None:
    # figure out the set of hexes that are relevant
    scan_range = sector.sensor_manager.range_to_detect(detector)
    identification_range = sector.sensor_manager.range_to_identify(detector)
    if loc is None:
        loc = detector.loc

    # find hexes that have unidentified sensor images
    # we'll use these to eliminate hexes within detection range below
    unidentified_hexes:set[tuple[int, int]] = set()
    for image in images or detector.sensor_settings.images:
        if image.identified:
            continue
        h_coords = util.int_coords(util.axial_round(util.pixel_to_pointy_hex(image.loc, sector.hex_size)))
        unidentified_hexes.add(h_coords)

    sector_hexes = util.hexes_within_pixel_dist(loc, scan_range, sector.hex_size)

    # eliminate hexes we already have good intel for
    # and create a dict from hex coord to info about the hex
    static_intel:dict[tuple[int, int], SectorHexIntel] = {}
    dynamic_intel:dict[tuple[int, int], SectorHexIntel] = {}
    for hex_coords in sector_hexes:
        # because these hexes are within detection range, we're confident that
        # any craft (of some certain parameters) in it will at least show up as
        # a sensor image. If we aren't able to identify the image, we cannot
        # mark it as explored.
        if util.int_coords(hex_coords) in unidentified_hexes:
            continue

        # check if weather makes this hex unscannable at this range
        hex_sensor_factor = sector.hex_weather(hex_coords).sensor_factor
        # check that all six points of the hex lie within modified sensor range
        pixel_coords = util.pointy_hex_to_pixel(hex_coords, sector.hex_size)
        in_range = True
        for corner_coords in util.hex_corners(pixel_coords, sector.hex_size):
            if util.distance(corner_coords, loc) > scan_range * hex_sensor_factor:
                in_range = False
                break
        if not in_range:
            continue

        # at this point the hex is fully detected and any images in it are
        # identified, so we have fully explored it

        # make a static intel for it if we don't already have one
        static_criteria = SectorHexMatchCriteria(sector.entity_id, hex_coords, True)
        s_intel = character.intel_manager.get_intel(static_criteria, SectorHexIntel)
        if not s_intel or not s_intel.is_fresh():
            static_intel[util.int_coords(hex_coords)] = SectorHexIntel.create_intel(sector.entity_id, hex_coords, True, 0, {}, gamestate, author_id=character.entity_id, expires_at=static_expires_at, fresh_until=static_fresh_until)

        # make a dynamic intel for it if we don't already have one
        dynamic_criteria = SectorHexMatchCriteria(sector.entity_id, hex_coords, False)
        d_intel = character.intel_manager.get_intel(dynamic_criteria, SectorHexIntel)
        if not d_intel or not d_intel.is_fresh():
            dynamic_intel[util.int_coords(hex_coords)] = SectorHexIntel.create_intel(sector.entity_id, hex_coords, False, 0, {}, gamestate, author_id=character.entity_id, expires_at=dynamic_expires_at, fresh_until=dynamic_fresh_until)

    # iterate over all the sensor images we've got accumulating info for
    # the hex they lie in, if it's within range
    for image in images or detector.sensor_settings.images:
        if not image.identified:
            continue

        h_coords = util.int_coords(util.axial_round(util.pixel_to_pointy_hex(image.loc, sector.hex_size)))

        if image.identity.is_static:
            intels = static_intel
        else:
            intels = dynamic_intel

        if h_coords in intels:
            intels[h_coords].entity_count += 1
            type_name = image.identity.object_type
            if type_name in intels[h_coords].type_counts:
                intels[h_coords].type_counts[type_name] += 1
            else:
                intels[h_coords].type_counts[type_name] = 1

    for s_intel in static_intel.values():
        ret = character.intel_manager.add_intel(s_intel)
        assert ret
    for d_intel in dynamic_intel.values():
        ret = character.intel_manager.add_intel(d_intel)
        assert ret

def add_sector_intel(sector:core.Sector, character:core.Character, gamestate:core.Gamestate, fresh_until:Optional[float]=None, expires_at:Optional[float]=None) -> None:
    sector_intel = character.intel_manager.get_intel(EntityIntelMatchCriteria(sector.entity_id), SectorIntel)
    if sector_intel and sector_intel.is_fresh():
        return

    sector_intel = SectorIntel.create_intel(gamestate, intel_entity_id=sector.entity_id, intel_entity_type=type(sector), intel_entity_name=sector.name, intel_entity_description=sector.description, author_id=character.entity_id, expires_at=expires_at, fresh_until=fresh_until)
    ret = character.intel_manager.add_intel(sector_intel)
    assert(ret)


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
        #TODO: expiration?
        add_asteroid_intel(asteroid, character, self.gamestate)

class IdentifyStationAction(events.Action):
    def act(self,
            character:core.Character,
            event_type:int,
            event_context:Mapping[int,int],
            event_args: MutableMapping[str, Union[int,float,str,bool]],
            action_args: Mapping[str, Union[int,float,str,bool]]
    ) -> None:
        station = self.gamestate.get_entity_short(self.ck(event_context, sensors.ContextKeys.TARGET), sector_entity.Station)
        #TODO: expiration?
        add_station_intel(station, character, self.gamestate)

class IdentifyTravelGateAction(events.Action):
    def act(self,
            character:core.Character,
            event_type:int,
            event_context:Mapping[int,int],
            event_args: MutableMapping[str, Union[int,float,str,bool]],
            action_args: Mapping[str, Union[int,float,str,bool]]
    ) -> None:
        gate = self.gamestate.get_entity_short(self.ck(event_context, sensors.ContextKeys.TARGET), sector_entity.TravelGate)
        #TODO: expiration?
        add_travel_gate_intel(gate, character, self.gamestate)


class IdentifySectorEntityAction(events.Action):
    def __init__(self, *args:Any, intel_ttl:float=config.Settings.intel.sector_entity.TTL, **kwargs:Any) -> None:
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
        assert(not isinstance(sentity, sector_entity.Asteroid))
        assert(not isinstance(sentity, sector_entity.Station))
        assert(not isinstance(sentity, sector_entity.TravelGate))
        fresh_until = self.gamestate.timestamp + self.intel_ttl*0.2
        expires_at = self.gamestate.timestamp + self.intel_ttl
        #TODO: should "static" sector entities have expiration?
        add_sector_entity_intel(sentity, character, self.gamestate, dynamic_fresh_until=fresh_until, dynamic_expires_at=expires_at)

class DockingAction(events.Action):
    def __init__(self, *args:Any, econ_intel_ttl:float=config.Settings.intel.econ_agent.TTL, **kwargs:Any) -> None:
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
        fresh_until = self.gamestate.timestamp + self.econ_intel_ttl*0.2
        expires_at = self.gamestate.timestamp + self.econ_intel_ttl
        add_econ_agent_intel(agent, character, self.gamestate, fresh_until=fresh_until, expires_at=expires_at)
        #TODO: what other intel do we want to create now that we're docked?

class ScanAction(events.Action):
    def __init__(self, *args:Any, static_intel_ttl:float=config.Settings.intel.static_hex.TTL, dynamic_intel_ttl:float=config.Settings.intel.dynamic_hex.TTL, **kwargs:Any) -> None:
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

        static_fresh_until = self.gamestate.timestamp + self.static_intel_ttl*0.2
        static_expires_at = self.gamestate.timestamp + self.static_intel_ttl
        dynamic_fresh_until = self.gamestate.timestamp + self.dynamic_intel_ttl*0.2
        dynamic_expires_at = self.gamestate.timestamp + self.dynamic_intel_ttl

        add_sector_scan_intel(detector, sector, character, self.gamestate, static_fresh_until=static_fresh_until, static_expires_at=static_expires_at, dynamic_fresh_until=dynamic_fresh_until, dynamic_expires_at=dynamic_expires_at)

class EnterSectorAction(events.Action):
    def __init__(self, *args:Any, fresh_ttl:float=config.Settings.intel.sector.TTL, expires_ttl:float=config.Settings.intel.sector.TTL, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.fresh_ttl = fresh_ttl
        self.expires_ttl = expires_ttl

    def act(self,
            character:core.Character,
            event_type:int,
            event_context:Mapping[int,int],
            event_args: MutableMapping[str, Union[int,float,str,bool]],
            action_args: Mapping[str, Union[int,float,str,bool]]
    ) -> None:
        sector = core.Gamestate.gamestate.get_entity_short(self.ck(event_context, events.ContextKeys.TARGET), core.Sector)

        fresh_until = self.gamestate.timestamp + self.fresh_ttl
        expires_at = self.gamestate.timestamp + self.expires_ttl

        add_sector_intel(sector, character, self.gamestate, fresh_until, expires_at)

def pre_initialize(event_manager:events.EventManager) -> None:
    event_manager.register_action(IdentifyAsteroidAction(), "identify_asteroid", "intel")
    event_manager.register_action(IdentifyStationAction(), "identify_station", "intel")
    event_manager.register_action(IdentifyTravelGateAction(), "identify_travel_gate", "intel")
    event_manager.register_action(IdentifySectorEntityAction(), "identify_sector_entity", "intel")
    event_manager.register_action(DockingAction(), "witness_docking", "intel")
    event_manager.register_action(ScanAction(), "witness_scan", "intel")
    event_manager.register_action(EnterSectorAction(), "witness_enter_sector", "intel")
