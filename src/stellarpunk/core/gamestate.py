""" Stellarpunk gamestate, a central repository for all gamestate """

import abc
import logging
import enum
import uuid
import collections
import datetime
import itertools
from dataclasses import dataclass
from typing import Optional, Any, Union, Type
from collections.abc import Collection, Mapping, MutableMapping, Sequence, MutableSequence, Iterator, Iterable

import numpy as np
import numpy.typing as npt
import rtree.index # type: ignore

from stellarpunk import util, task_schedule, narrative
from .base import EntityRegistry, Entity, EconAgent, AbstractEconDataLogger, StarfieldLayer, AbstractEffect, AbstractOrder, Observable, stellarpunk_version
from .production_chain import ProductionChain
from .sector import Sector, SectorEntity
from .character import Character, Player, AbstractAgendum, Message, AbstractEventManager, CrewedSectorEntity

DT_EPSILON = 1.0/120.0

class Counters(enum.IntEnum):
    def _generate_next_value_(name, start, count, last_values): # type: ignore
        """generate consecutive automatic numbers starting from zero"""
        return count
    GOTO_ACT_FAST = enum.auto()
    GOTO_ACT_FAST_CT = enum.auto()
    GOTO_ACT_SLOW = enum.auto()
    GOTO_THREAT_YES = enum.auto()
    GOTO_THREAT_YES_CT = enum.auto()
    GOTO_THREAT_NO = enum.auto()
    GOTO_THREAT_NO_CT = enum.auto()
    ACCELERATE_FAST = enum.auto()
    ACCELERATE_SLOW = enum.auto()
    ACCELERATE_FAST_FORCE = enum.auto()
    ACCELERATE_FAST_TORQUE = enum.auto()
    ACCELERATE_SLOW_FORCE = enum.auto()
    ACCELERATE_SLOW_TORQUE = enum.auto()
    COLLISION_HITS_HIT = enum.auto()
    COLLISION_HITS_MISS = enum.auto()
    NON_FRONT_ORDER_ACTION = enum.auto()
    ORDERS_PROCESSED = enum.auto()
    ORDER_SCHEDULE_DELAY = enum.auto()
    ORDER_SCHEDULE_IMMEDIATE = enum.auto()
    COLLISION_NEIGHBOR_NO_NEIGHBORS = enum.auto()
    COLLISION_NEIGHBOR_HAS_NEIGHBORS = enum.auto()
    COLLISION_NEIGHBOR_NUM_NEIGHBORS = enum.auto()
    COLLISION_NEIGHBOR_NONE = enum.auto()
    COLLISION_THREATS_C = enum.auto()
    COLLISION_THREATS_NC = enum.auto()
    COLLISIONS = enum.auto()
    BEHIND_TICKS = enum.auto()
    EVENTS_PROCESSED = enum.auto()
    EVENT_ACTIONS_PROCESSED = enum.auto()
    EVENTS_PROCESSED_OOB = enum.auto()
    EVENT_ACTIONS_PROCESSED_OOB = enum.auto()
    EVENT_CANDIDATES_THROTTLED = enum.auto()
    EVENTS_TOTAL_THROTTLED = enum.auto()


class TickHandler:
    def tick(self) -> None:
        pass

class AbstractGameRuntime:
    """ The game runtime that actually runs the simulation. """

    def get_missed_ticks(self) -> int:
        return 0

    def get_ticktime(self) -> float:
        return 0.

    def get_time_acceleration(self) -> tuple[float, bool]:
        """ Get time acceleration parameters. """
        return (1.0, False)

    def time_acceleration(self, accel_rate:float, fast_mode:bool) -> None:
        """ Request time acceleration. """
        pass

    def exit_startup(self) -> None:
        pass

    def start_game(self) -> None:
        pass

    def game_running(self) -> bool:
        return False

    def quit(self) -> None:
        pass

    def get_desired_dt(self) -> float:
        return 0.0

    def get_dt(self) -> float:
        return 0.0

    def raise_exception(self) -> None:
        pass

    def raise_breakpoint(self) -> None:
        pass

    def should_breakpoint(self) -> bool:
        return False

    def get_breakpoint_sentinel(self) -> Optional[str]:
        return None

    def set_breakpoint_sentinel(self, value:Optional[str]) -> None:
        pass

    def register_tick_handler(self, tick_handler:TickHandler) -> None:
        pass

class AbstractGenerator:
    @abc.abstractmethod
    def gen_sector_location(self, sector:Sector, occupied_radius:float=2e3, center:Union[tuple[float, float],npt.NDArray[np.float64]]=(0.,0.), radius:Optional[float]=None, strict:bool=False)->npt.NDArray[np.float64]: ...
    @abc.abstractmethod
    def gen_projectile_location(self, center:Union[tuple[float, float],npt.NDArray[np.float64]]=(0.,0.), index:Optional[int]=None) -> tuple[npt.NDArray[np.float64],int]: ...
    @abc.abstractmethod
    def spawn_sector_entity(self, klass:Type, sector:Sector, ship_x:float, ship_y:float, v:Optional[npt.NDArray[np.float64]]=None, w:Optional[float]=None, theta:Optional[float]=None, entity_id:Optional[uuid.UUID]=None) -> SectorEntity: ...

class ScheduledTask(abc.ABC):
    def __init__(self, task_id:Optional[uuid.UUID]=None) -> None:
        if task_id is not None:
            self.task_id = task_id
        else:
            self.task_id = uuid.uuid4()

    def is_valid(self) -> bool:
        """ Is this task still valid? helps with book keeping. """
        return True

    @abc.abstractmethod
    def act(self) -> None: ...

    def sanity_check(self, ts:float) -> None:
        pass

class Gamestate(EntityRegistry):
    gamestate:"Gamestate" = None # type: ignore
    def __init__(self) -> None:
        self.logger = logging.getLogger(util.fullname(self))

        # a fingerprint for this game, set at universe generation and constant
        # for this game
        self.fingerprint:bytes = b''
        self.game_version:str = stellarpunk_version()
        self.game_start_version:str = stellarpunk_version()
        self.save_count:int = 0

        self.generator:AbstractGenerator = None #type: ignore
        self.game_runtime:AbstractGameRuntime = AbstractGameRuntime()
        self.event_manager:AbstractEventManager = AbstractEventManager()#DeferredEventManager()

        # this will get replaced by the generator's random generator
        self.random = np.random.default_rng()

        self.entities: dict[uuid.UUID, Entity] = {}
        self.entities_short: dict[int, Entity] = {}
        self.entity_context_store = narrative.EntityStore()

        # global registry of all orders, effects, agenda
        self.orders: dict[uuid.UUID, AbstractOrder] = {}
        self.effects: dict[uuid.UUID, AbstractEffect] = {}
        self.agenda: dict[uuid.UUID, AbstractAgendum] = {}

        # the production chain of resources (ingredients
        self.production_chain = ProductionChain()

        # Universe State
        # the universe is a set of sectors, indexed by their entity id
        self.sectors:dict[uuid.UUID, Sector] = {}
        self.sector_idx_lookup:dict[uuid.UUID, int] = {}
        self.sector_jumps:dict[uuid.UUID, Mapping[int, float]] = {}

        # a spatial index of sectors in the universe
        self.sector_spatial = rtree.index.Index()

        self.starfield:list[StarfieldLayer] = []
        self.sector_starfield:list[StarfieldLayer] = []
        self.portrait_starfield:list[StarfieldLayer] = []


        #TODO: this feels janky, but I do need a way to find the EconAgent
        # representing a station if I want to trade with it.
        #TODO: how do we keep this up to date?
        # collection of EconAgents, by uuid of the entity they represent
        self.econ_agents:dict[uuid.UUID, EconAgent] = {}
        self.agent_to_entity:dict[uuid.UUID, Entity] = {}
        self.econ_logger:AbstractEconDataLogger = AbstractEconDataLogger()

        # convenience access to characters, dupicates stuff in self.entities
        self.characters:dict[uuid.UUID, Character] = {}
        self.characters_by_location: MutableMapping[uuid.UUID, MutableSequence[Character]] = collections.defaultdict(list)

        # priority queue of various behaviors (scheduled timestamp, item)
        # ship orders, sector effects, character agenda and generic tasks
        self._order_schedule:task_schedule.TaskSchedule[AbstractOrder] = task_schedule.TaskSchedule()
        self._effect_schedule:task_schedule.TaskSchedule[AbstractEffect] = task_schedule.TaskSchedule()
        self._agenda_schedule:task_schedule.TaskSchedule[AbstractAgendum] = task_schedule.TaskSchedule()
        self._task_schedule:task_schedule.TaskSchedule[ScheduledTask] = task_schedule.TaskSchedule()

        # Time keeping
        self.base_date = datetime.datetime(2234, 4, 3)
        # this is so 40 hours of gameplay => 4 years of gametime
        # 4 years is long enough to accomplish a lot and 40 hours seems like a
        # long play session.
        #TODO: put this in configuration and save/load it
        self.game_secs_per_sec = 876.
        self.timestamp = 0.
        self.ticks = 0

        self.one_tick = False
        self.paused = False
        self.force_pause_holders:set[object] = set()

        self.player:Player = None # type: ignore[assignment]

        self.counters = [0.] * len(Counters)

        # House keeping state for cleanup, event deduping
        # list allows in iterator appends
        # set allows destroying exactly once
        self.entity_destroy_list:list[Entity] = []
        self.entity_destroy_set:set[Entity] = set()
        self.last_colliders:set[str] = set()

        # We maintain gamestate as a singleton for convenient access to the
        # "current" global gamestate
        #TODO: maintain this class field externally and not in the constructor?
        if Gamestate.gamestate is not None:
            self.logger.info(f'replacing existing gamestate current: {Gamestate.gamestate.ticks} ticks and {Gamestate.gamestate.timestamp} game secs with {len(Gamestate.gamestate.entities)} entities')
        #    raise ValueError()
        #import traceback
        #Gamestate.bt = traceback.format_stack()
        Gamestate.gamestate = self

    # base.EntityRegistry
    def register_entity(self, entity: Entity) -> narrative.EventContext:
        self.logger.debug(f'registering {entity}')
        if entity.entity_id in self.entities:
            raise ValueError(f'entity {entity.entity_id} already registered!')
        if entity.short_id_int() in self.entities_short:
            raise ValueError(f'entity short id collision {entity} and {self.entities_short[entity.short_id_int()]}!')

        self.entities[entity.entity_id] = entity
        self.entities_short[entity.short_id_int()] = entity
        entity.created_at = self.timestamp
        return self.entity_context_store.register_entity(entity.short_id_int())

    def unregister_entity(self, entity: Entity) -> None:
        self.logger.debug(f'unregistering {entity}')
        self.entity_context_store.unregister_entity(entity.short_id_int())
        del self.entities[entity.entity_id]
        del self.entities_short[entity.short_id_int()]

    def now(self) -> float:
        return self.timestamp

    def register_order(self, order: AbstractOrder) -> None:
        self.orders[order.order_id] = order

    def unregister_order(self, order: AbstractOrder) -> None:
        del self.orders[order.order_id]

    def sanity_check_orders(self) -> None:
        for ts, order in self._order_schedule:
            assert order.order_id in self.orders
        for k, order in self.orders.items():
            order.sanity_check(k)
            if isinstance(order, Observable):
                for observer in order.observers:
                    assert order in observer.observings

    def register_effect(self, effect: AbstractEffect) -> None:
        self.effects[effect.effect_id] = effect

    def unregister_effect(self, effect: AbstractEffect) -> None:
        del self.effects[effect.effect_id]

    def sanity_check_effects(self) -> None:
        for ts, effect in self._effect_schedule:
            assert effect.effect_id in self.effects
        for k, effect in self.effects.items():
            effect.sanity_check(k)
            if isinstance(effect, Observable):
                for observer in effect.observers:
                    assert effect in observer.observings

    def register_agendum(self, agendum: AbstractAgendum) -> None:
        self.agenda[agendum.agenda_id] = agendum

    def unregister_agendum(self, agendum: AbstractAgendum) -> None:
        del self.agenda[agendum.agenda_id]

    def sanity_check_agenda(self) -> None:
        for ts, agenda in self._agenda_schedule:
            assert agenda.agenda_id in self.agenda
        for k, agendum in self.agenda.items():
            assert(k == agendum.agenda_id)
            agendum.sanity_check()
            if isinstance(agendum, Observable):
                for observer in agendum.observers:
                    assert agendum in observer.observings

    def contains_entity(self, entity_id:uuid.UUID) -> bool:
        return entity_id in self.entities

    def get_entity[T:Entity](self, entity_id:uuid.UUID, klass:Type[T]) -> T:
        entity = self.entities[entity_id]
        assert(isinstance(entity, klass))
        return entity

    def get_entity_short[T:Entity](self, entity_short_id:int, klass:Type[T]) -> T:
        entity = self.entities_short[entity_short_id]
        assert(isinstance(entity, klass))
        return entity

    def sanity_check_entities(self) -> None:
        for k, entity in self.entities.items():
            assert(k == entity.entity_id)
            if isinstance(entity, Observable):
                for observer in entity.observers:
                    assert entity in observer.observings
            entity.sanity_check()

    def recover_objects[T:tuple](self, objects:T) -> T:
        ret = []
        for o in objects:
            if isinstance(o, Entity):
                o = self.get_entity(o.entity_id, type(o))
            elif isinstance(o, AbstractOrder):
                o = self.get_order(o.order_id, type(o))
            elif isinstance(o, AbstractEffect):
                o = self.get_effect(o.effect_id, type(o))
            elif isinstance(o, AbstractAgendum):
                o = self.get_agendum(o.agenda_id, type(o))
            ret.append(o)

        # mypy isn't quite powerful enough to understand what's going on
        # we want the return type of this method to preserve all the types for
        # the caller
        return tuple(ret) # type: ignore

    def get_effect[T:AbstractEffect](self, effect_id:uuid.UUID, klass:Type[T]) -> T:
        effect = self.effects[effect_id]
        assert(isinstance(effect, klass))
        return effect

    def get_order[T:AbstractOrder](self, order_id:uuid.UUID, klass:Type[T]) -> T:
        order = self.orders[order_id]
        assert(isinstance(order, klass))
        return order

    def get_agendum[T:AbstractAgendum](self, agenda_id:uuid.UUID, klass:Type[T]) -> T:
        agendum = self.agenda[agenda_id]
        assert(isinstance(agendum, klass))
        return agendum

    def _pause(self, paused:Optional[bool]=None, reset_time_accel:bool=False) -> None:
        if reset_time_accel:
            self.game_runtime.time_acceleration(1.0, False)
        if paused is None:
            self.paused = not self.paused
        else:
            self.paused = paused

    def pause(self, paused:Optional[bool]=None) -> None:
        if len(self.force_pause_holders) > 0:
            assert(self.paused)
            return
        self._pause(paused, reset_time_accel=True)

    def force_pause(self, requesting_object:object) -> None:
        #if self.force_pause_holder is not None and self.force_pause_holder != requesting_object:
        #    raise ValueError(f'already paused by {self.force_pause}')
        self._pause(True)
        self.force_pause_holders.add(requesting_object)

    def is_force_paused(self, requesting_object:Optional[object]=None) -> bool:
        if requesting_object is None:
            return len(self.force_pause_holders) > 0
        else:
            return requesting_object in self.force_pause_holders

    def force_unpause(self, requesting_object:object) -> None:
        if requesting_object not in self.force_pause_holders:
            raise ValueError(f'pause requested by {self.force_pause}')
        else:
            self.force_pause_holders.remove(requesting_object)
            if len(self.force_pause_holders) == 0:
                self._pause(False)
            else:
                assert(self.paused)

    def breakpoint(self) -> None:
        if self.game_runtime.should_breakpoint():
            raise Exception("debug breakpoint immediate")

    def conditional_breakpoint(self, value:str) -> None:
        if self.game_runtime.should_breakpoint() and value == self.game_runtime.get_breakpoint_sentinel():
            raise Exception("conditional debug breakpoint")

    def representing_agent(self, entity_id:uuid.UUID, agent:EconAgent) -> None:
        self.econ_agents[entity_id] = agent
        self.agent_to_entity[agent.entity_id] = self.entities[entity_id]

    def withdraw_agent(self, entity_id:uuid.UUID) -> None:
        try:
            agent = self.econ_agents.pop(entity_id)
            assert agent.entity_id in self.agent_to_entity
            del self.agent_to_entity[agent.entity_id]
            self.destroy_entity(agent)
        except KeyError:
            pass

    def add_character(self, character:Character) -> None:
        if character.location is None:
            raise ValueError(f'tried to add character {character} with no location')
        self.characters[character.entity_id] = character
        self.characters_by_location[character.location.entity_id].append(character)

    def move_character(self, character:Character, location:SectorEntity) -> None:
        if character.location is not None:
            self.characters_by_location[character.location.entity_id].remove(character)
        self.characters_by_location[location.entity_id].append(character)
        character.migrate(location)

    def handle_destroy_entities(self) -> None:
        for entity in self.gamestate.entity_destroy_list:
            self._handle_destroy(entity)
        self.entity_destroy_list.clear()
        self.entity_destroy_set.clear()

    def _handle_destroy(self, entity:Entity) -> None:
        self.logger.debug(f'destroying {entity}')
        if isinstance(entity, SectorEntity):
            for character in self.characters_by_location[entity.entity_id]:
                self.destroy_entity(character)
            if entity.sector is not None:
                entity.sector.remove_entity(entity)
            entity.destroy()
        else:
            entity.destroy()

        if entity.entity_id in self.econ_agents:
            raise ValueError(f'{entity} still represented by econ agent {self.econ_agents[entity.entity_id]} after entity destroyed!')

    def destroy_entity(self, entity:Entity) -> None:
        if entity not in self.entity_destroy_set:
            self.entity_destroy_list.append(entity)
            self.entity_destroy_set.add(entity)

    def is_order_scheduled(self, order:AbstractOrder) -> bool:
        return self._order_schedule.is_task_scheduled(order)

    def schedule_order_immediate(self, order:AbstractOrder, jitter:float=0.) -> None:
        self.counters[Counters.ORDER_SCHEDULE_IMMEDIATE] += 1
        self.schedule_order(self.timestamp + DT_EPSILON, order, jitter)

    def schedule_order(self, timestamp:float, order:AbstractOrder, jitter:float=0.) -> None:
        assert timestamp > self.timestamp
        assert timestamp < np.inf

        if jitter > 0.:
            timestamp += self.random.uniform(high=jitter)

        self._order_schedule.push_task(timestamp, order)
        self.counters[Counters.ORDER_SCHEDULE_DELAY] += timestamp - self.timestamp

    def unschedule_order(self, order:AbstractOrder) -> None:
        self._order_schedule.cancel_task(order)

    def pop_current_orders(self) -> Sequence[AbstractOrder]:
        return self._order_schedule.pop_current_tasks(self.timestamp)

    def schedule_effect_immediate(self, effect:AbstractEffect, jitter:float=0.) -> None:
        self.schedule_effect(self.timestamp + DT_EPSILON, effect, jitter)

    def schedule_effect(self, timestamp: float, effect:AbstractEffect, jitter:float=0.) -> None:
        assert timestamp > self.timestamp
        assert timestamp < np.inf

        if jitter > 0.:
            timestamp += self.random.uniform(high=jitter)

        self._effect_schedule.push_task(timestamp, effect)

    def unschedule_effect(self, effect:AbstractEffect) -> None:
        self._effect_schedule.cancel_task(effect)

    def pop_current_effects(self) -> Sequence[AbstractEffect]:
        return self._effect_schedule.pop_current_tasks(self.timestamp)

    def is_agendum_scheduled(self, agendum:AbstractAgendum) -> bool:
        return self._agenda_schedule.is_task_scheduled(agendum)

    def schedule_agendum_immediate(self, agendum:AbstractAgendum, jitter:float=0.) -> None:
        self.schedule_agendum(self.timestamp + DT_EPSILON, agendum, jitter)

    def schedule_agendum(self, timestamp:float, agendum:AbstractAgendum, jitter:float=0.) -> None:
        assert timestamp > self.timestamp
        assert timestamp < np.inf

        if jitter > 0.:
            timestamp += self.random.uniform(high=jitter)

        self._agenda_schedule.push_task(timestamp, agendum)

    def unschedule_agendum(self, agendum:AbstractAgendum) -> None:
        self._agenda_schedule.cancel_task(agendum)

    def pop_current_agenda(self) -> Sequence[AbstractAgendum]:
        return self._agenda_schedule.pop_current_tasks(self.timestamp)

    def is_task_scheduled(self, task:ScheduledTask) -> bool:
        return self._task_schedule.is_task_scheduled(task)

    def schedule_task_immediate(self, task:ScheduledTask, jitter:float=0.) -> None:
        self.schedule_task(self.timestamp + DT_EPSILON, task, jitter)

    def schedule_task(self, timestamp:float, task:ScheduledTask, jitter:float=0.) -> None:
        assert timestamp > self.timestamp
        assert timestamp < np.inf

        if jitter > 0.:
            timestamp += self.random.uniform(high=jitter)

        self._task_schedule.push_task(timestamp, task)

    def unschedule_task(self, task:ScheduledTask) -> None:
        self._task_schedule.cancel_task(task)

    def sanity_check_tasks(self) -> None:
        for ts, task in self._task_schedule:
            task.sanity_check(ts)

    def pop_current_task(self) -> Sequence[ScheduledTask]:
        return self._task_schedule.pop_current_tasks(self.timestamp)

    def transact(self, product_id:int, buyer:EconAgent, seller:EconAgent, price:float, amount:float) -> None:
        self.logger.info(f'transaction: {product_id} from {buyer.agent_id} to {seller.agent_id} at ${price} x {amount} = ${price * amount}')
        seller.sell(product_id, price, amount)
        buyer.buy(product_id, price, amount)
        self.econ_logger.transact(0., product_id, buyer.agent_id, seller.agent_id, price, amount, ticks=self.timestamp)

    def _construct_econ_state(self) -> tuple[
            npt.NDArray[np.float64], # inventory
            npt.NDArray[np.float64], # balance
            npt.NDArray[np.float64], # buy_prices
            npt.NDArray[np.float64], # buy_budget
            npt.NDArray[np.float64], # sell_prices
            npt.NDArray[np.float64], # max_buy_prices
            npt.NDArray[np.float64], # min_sell_prices
            npt.NDArray[np.int64], # cannot_buy_ticks
            npt.NDArray[np.int64], # cannot_sell_ticks
    ]:

        #TODO: given how logging works, this assumes we never lose or gain
        # agents so they always have a consistent id and ordering
        num_agents = EconAgent.num_agents()
        num_products = self.production_chain.num_products

        inventory = np.zeros((num_agents, num_products))
        balance = np.zeros((num_agents, ))
        buy_prices = np.zeros((num_agents, num_products))
        buy_budget = np.zeros((num_agents, num_products))
        sell_prices = np.zeros((num_agents, num_products))

        for agent in self.econ_agents.values():
            i = agent.agent_id
            balance[i] = agent.balance()
            for j in range(num_products):
                inventory[i,j] = agent.inventory(j)
                buy_prices[i,j] = agent.buy_price(j)
                buy_budget[i,j] = agent.budget(j)
                sell_prices[i,j] = agent.sell_price(j)

        return (
            inventory,
            balance,
            buy_prices,
            buy_budget,
            sell_prices,
            buy_prices,
            sell_prices,
            np.zeros((num_agents, num_products), dtype=np.int64),
            np.zeros((num_agents, num_products), dtype=np.int64),
        )


    def log_econ(self) -> None:
        self.econ_logger.log_econ(self.timestamp, *self._construct_econ_state())
        self.econ_logger.flush()

    def add_sector(self, sector:Sector, idx:int) -> None:
        self.sectors[sector.entity_id] = sector
        self.sector_spatial.insert(idx, (sector.loc[0]-sector.radius, sector.loc[1]-sector.radius, sector.loc[0]+sector.radius, sector.loc[1]+sector.radius), sector.entity_id)

    def recompute_jumps(self, sector_idx_lookup:dict[uuid.UUID, int], adj_matrix:npt.NDArray[np.float64]) -> None:
        """ recomputes jump lengths between all sector pairs. """

        self.sector_idx_lookup = sector_idx_lookup
        self.sector_jumps = dict()

        for sector_id, sector_idx in sector_idx_lookup.items():
            path_tree, distance_map = util.dijkstra(adj_matrix, sector_idx, -1)
            self.sector_jumps[sector_id] = distance_map

    def jump_distance(self, a_id:uuid.UUID, b_id:uuid.UUID) -> Optional[int]:
        b_idx = self.sector_idx_lookup[b_id]
        if b_idx not in self.sector_jumps[a_id]:
            return None
        else:
            return int(np.round(self.sector_jumps[a_id][b_idx]))

    def spatial_query(self, bounds:tuple[float, float, float, float]) -> Iterator[uuid.UUID]:
        hits = self.sector_spatial.intersection(bounds, objects="raw")
        return hits # type: ignore

    def sanity_check_sectors(self) -> None:
        pass

    def sanity_check(self) -> None:
        self.sanity_check_entities()
        self.sanity_check_orders()
        self.sanity_check_effects()
        self.sanity_check_agenda()
        self.sanity_check_sectors()
        self.sanity_check_tasks()

    def timestamp_to_datetime(self, timestamp:float) -> datetime.datetime:
        return datetime.datetime.fromtimestamp(self.base_date.timestamp() + timestamp*self.game_secs_per_sec)

    def current_time(self) -> datetime.datetime:
        return self.timestamp_to_datetime(self.timestamp)

    def trigger_event(
        self,
        characters: Collection[Character],
        event_type: int,
        context: dict[int, int],
        event_args: dict[str, Union[int,float,str,bool]] = {},
        merge_key: Optional[uuid.UUID]=None,
    ) -> None:
        self.event_manager.trigger_event(characters, event_type, context, event_args, merge_key=merge_key)

    def trigger_event_immediate(
        self,
        characters: Collection[Character],
        event_type: int,
        context: dict[int, int],
        event_args: dict[str, Union[int,float,str,bool]] = {},
    ) -> None:
        self.event_manager.trigger_event_immediate(characters, event_type, context, event_args)

def captain(craft:SectorEntity) -> Optional[Character]:
    if isinstance(craft, CrewedSectorEntity) and craft.captain:
        return craft.captain
    return None

EMPTY_TUPLE = ()
def crew(craft:SectorEntity) -> Collection[Character]:
    if not isinstance(craft, CrewedSectorEntity):
        return EMPTY_TUPLE
    return Gamestate.gamestate.characters_by_location[craft.entity_id]
