""" Stellarpunk gamestate, a central repository for all gamestate """

import abc
import logging
import enum
import uuid
import collections
import datetime
import itertools
from dataclasses import dataclass
from typing import Dict, Mapping, MutableMapping, Optional, Any, Iterable, Sequence, MutableSequence, Deque, Tuple, Iterator, Union, List, Type

import numpy as np
import numpy.typing as npt
import rtree.index # type: ignore

from stellarpunk import util, task_schedule, narrative
from .base import EntityRegistry, Entity, EconAgent, AbstractEconDataLogger, StarfieldLayer
from .production_chain import ProductionChain
from .sector import Sector
from .sector_entity import SectorEntity
from .order import Order, Effect
from .character import Character, Player, Agendum, Message


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


class AbstractGameRuntime:
    """ The game runtime that actually runs the simulation. """

    def get_time_acceleration(self) -> Tuple[float, bool]:
        """ Get time acceleration parameters. """
        return (1.0, False)

    def time_acceleration(self, accel_rate:float, fast_mode:bool) -> None:
        """ Request time acceleration. """
        pass

    def send_message(
        self,
        recipient: Character,
        message: Message,
    ) -> None:
        pass

    def trigger_event(
        self,
        characters: Iterable[Character],
        event_type: int,
        context: Mapping[int, int],
        event_args: MutableMapping[str, Any] = {},
    ) -> None:
        pass

    def trigger_event_immediate(
        self,
        characters: Iterable[Character],
        event_type: int,
        context: Mapping[int, int],
        event_args: MutableMapping[str, Any] = {},
    ) -> None:
        pass

class AbstractGenerator:
    @abc.abstractmethod
    def gen_sector_location(self, sector:Sector, occupied_radius:float=2e3, center:Union[Tuple[float, float],npt.NDArray[np.float64]]=(0.,0.), radius:Optional[float]=None)->npt.NDArray[np.float64]: ...

    @abc.abstractmethod
    def spawn_sector_entity(self, klass:Type, sector:Sector, ship_x:float, ship_y:float, v:Optional[npt.NDArray[np.float64]]=None, w:Optional[float]=None, theta:Optional[float]=None, entity_id:Optional[uuid.UUID]=None) -> SectorEntity: ...


class Gamestate(EntityRegistry):
    def __init__(self) -> None:
        self.logger = logging.getLogger(util.fullname(self))
        self.generator:AbstractGenerator = None #type: ignore
        self.game_runtime:AbstractGameRuntime = AbstractGameRuntime()

        self.random = np.random.default_rng()

        self.entities: Dict[uuid.UUID, Entity] = {}
        self.entities_short: Dict[int, Entity] = {}
        self.entity_context_store = narrative.EntityStore()

        # the production chain of resources (ingredients
        self.production_chain = ProductionChain()

        # the universe is a set of sectors, indexed by their entity id
        self.sectors:Dict[uuid.UUID, Sector] = {}
        self.sector_ids:npt.NDArray = np.ndarray((0,), uuid.UUID) #indexed same as edges
        self.sector_idx:Mapping[uuid.UUID, int] = {} #inverse of sector_ids
        self.sector_edges:npt.NDArray[np.float64] = np.ndarray((0,0))

        # a spatial index of sectors in the universe
        self.sector_spatial = rtree.index.Index()

        #TODO: this feels janky, but I do need a way to find the EconAgent
        # representing a station if I want to trade with it.
        #TODO: how do we keep this up to date?
        # collection of EconAgents, by uuid of the entity they represent
        self.econ_agents:Dict[uuid.UUID, EconAgent] = {}

        self.econ_logger:AbstractEconDataLogger = AbstractEconDataLogger()

        self.characters:Dict[uuid.UUID, Character] = {}

        # priority queue of order items in form (scheduled timestamp, agendum)
        self._order_schedule:task_schedule.TaskSchedule[Order] = task_schedule.TaskSchedule()

        # priority queue of effects
        self._effect_schedule:task_schedule.TaskSchedule[Effect] = task_schedule.TaskSchedule()

        # priority queue of agenda items in form (scheduled timestamp, agendum)
        self._agenda_schedule:task_schedule.TaskSchedule[Agendum] = task_schedule.TaskSchedule()
        #self.scheduled_agenda:Set[Agendum] = set()

        self.characters_by_location: MutableMapping[uuid.UUID, MutableSequence[Character]] = collections.defaultdict(list)

        self.keep_running = True

        self.base_date = datetime.datetime(2234, 4, 3)
        self.timestamp = 0.

        self.desired_dt = 1/30
        self.dt = self.desired_dt
        self.min_tick_sleep = self.desired_dt/5
        self.ticks = 0
        self.ticktime = 0.
        self.timeout = 0.
        self.missed_ticks = 0

        self.keep_running = True
        self.paused = False
        self.force_pause_holder:Optional[object] = None
        self.should_raise= False

        self.player:Player = None # type: ignore[assignment]

        self.counters = [0.] * len(Counters)

        self.starfield:Sequence[StarfieldLayer] = []
        self.sector_starfield:Sequence[StarfieldLayer] = []
        self.portrait_starfield:Sequence[StarfieldLayer] = []

    def register_entity(self, entity: Entity) -> narrative.EventContext:
        if entity.entity_id in self.entities:
            raise ValueError(f'entity {entity.entity_id} already registered!')
        if entity.short_id_int() in self.entities_short:
            raise ValueError(f'entity short id collision {entity} and {self.entities_short[entity.short_id_int()]}!')

        self.entities[entity.entity_id] = entity
        self.entities_short[entity.short_id_int()] = entity
        return self.entity_context_store.register_entity(entity.short_id_int())

    def unregister_entity(self, entity: Entity) -> None:
        self.entity_context_store.unregister_entity(entity.short_id_int())
        del self.entities[entity.entity_id]
        del self.entities_short[entity.short_id_int()]

    def get_time_acceleration(self) -> Tuple[float, bool]:
        return self.game_runtime.get_time_acceleration()

    def time_acceleration(self, accel_rate:float, fast_mode:bool) -> None:
        self.game_runtime.time_acceleration(accel_rate, fast_mode)

    def _pause(self, paused:Optional[bool]=None) -> None:
        self.time_acceleration(1.0, False)
        if paused is None:
            self.paused = not self.paused
        else:
            self.paused = paused

    def pause(self, paused:Optional[bool]=None) -> None:
        if self.force_pause_holder is not None:
            return
        self._pause(paused)

    def force_pause(self, requesting_object:object) -> None:
        if self.force_pause_holder is not None and self.force_pause_holder != requesting_object:
            raise ValueError(f'already paused by {self.force_pause}')
        else:
            self._pause(True)
            self.force_pause_holder = requesting_object

    def force_unpause(self, requesting_object:object) -> None:
        if self.force_pause_holder != requesting_object:
            raise ValueError(f'pause requested by {self.force_pause}')
        else:
            self.force_pause_holder = None
            self._pause(False)

    def representing_agent(self, entity_id:uuid.UUID, agent:EconAgent) -> None:
        self.econ_agents[entity_id] = agent

    def withdraw_agent(self, entity_id:uuid.UUID) -> None:
        try:
            self.econ_agents.pop(entity_id)
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
        character.location = location

    def destroy_character(self, character:Character) -> None:
        character.destroy()

    def destroy_sector_entity(self, entity:SectorEntity) -> None:
        for character in self.characters_by_location[entity.entity_id]:
            self.destroy_character(character)
        if entity.sector is not None:
            entity.sector.remove_entity(entity)
        entity.destroy()

    def is_order_scheduled(self, order:Order) -> bool:
        return self._order_schedule.is_task_scheduled(order)

    def schedule_order_immediate(self, order:Order, jitter:float=0.) -> None:
        self.counters[Counters.ORDER_SCHEDULE_IMMEDIATE] += 1
        self.schedule_order(self.timestamp + self.desired_dt, order, jitter)

    def schedule_order(self, timestamp:float, order:Order, jitter:float=0.) -> None:
        assert timestamp > self.timestamp
        assert timestamp < np.inf

        if jitter > 0.:
            timestamp += self.random.uniform(high=jitter)

        self._order_schedule.push_task(timestamp, order)
        self.counters[Counters.ORDER_SCHEDULE_DELAY] += timestamp - self.timestamp

    def unschedule_order(self, order:Order) -> None:
        self._order_schedule.cancel_task(order)

    def pop_current_orders(self) -> Sequence[Order]:
        return self._order_schedule.pop_current_tasks(self.timestamp)

    def schedule_effect_immediate(self, effect:Effect, jitter:float=0.) -> None:
        self.schedule_effect(self.timestamp + self.desired_dt, effect, jitter)

    def schedule_effect(self, timestamp: float, effect:Effect, jitter:float=0.) -> None:
        assert timestamp > self.timestamp
        assert timestamp < np.inf

        if jitter > 0.:
            timestamp += self.random.uniform(high=jitter)

        self._effect_schedule.push_task(timestamp, effect)

    def unschedule_effect(self, effect:Effect) -> None:
        self._effect_schedule.cancel_task(effect)

    def pop_current_effects(self) -> Sequence[Effect]:
        return self._effect_schedule.pop_current_tasks(self.timestamp)

    def schedule_agendum_immediate(self, agendum:Agendum, jitter:float=0.) -> None:
        self.schedule_agendum(self.timestamp + self.desired_dt, agendum, jitter)

    def schedule_agendum(self, timestamp:float, agendum:Agendum, jitter:float=0.) -> None:
        assert timestamp > self.timestamp
        assert timestamp < np.inf

        if jitter > 0.:
            timestamp += self.random.uniform(high=jitter)

        self._agenda_schedule.push_task(timestamp, agendum)

    def unschedule_agendum(self, agendum:Agendum) -> None:
        self._agenda_schedule.cancel_task(agendum)

    def pop_current_agenda(self) -> Sequence[Agendum]:
        return self._agenda_schedule.pop_current_tasks(self.timestamp)

    def transact(self, product_id:int, buyer:EconAgent, seller:EconAgent, price:float, amount:float) -> None:
        self.logger.info(f'transaction: {product_id} from {buyer.agent_id} to {seller.agent_id} at ${price} x {amount} = ${price * amount}')
        seller.sell(product_id, price, amount)
        buyer.buy(product_id, price, amount)
        self.econ_logger.transact(0., product_id, buyer.agent_id, seller.agent_id, price, amount, ticks=self.timestamp)

    def _construct_econ_state(self) -> Tuple[
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

    def update_edges(self, sector_edges:npt.NDArray[np.float64], sector_ids:npt.NDArray) -> None:
        self.sector_edges = sector_edges
        self.sector_ids = sector_ids
        self.sector_idx = {v:k for (k,v) in enumerate(sector_ids)}
        self.max_edge_length = max(
            util.distance(self.sectors[a].loc, self.sectors[b].loc) for (i,a),(j,b) in itertools.product(enumerate(sector_ids), enumerate(sector_ids)) if sector_edges[i, j] == 1
        )

    def spatial_query(self, bounds:Tuple[float, float, float, float]) -> Iterator[uuid.UUID]:
        hits = self.sector_spatial.intersection(bounds, objects="raw")
        return hits # type: ignore

    def timestamp_to_datetime(self, timestamp:float) -> datetime.datetime:
        return datetime.datetime.fromtimestamp(self.base_date.timestamp() + timestamp)
    def current_time(self) -> datetime.datetime:
        #TODO: probably want to decouple telling time from ticks processed
        # we want missed ticks to slow time, but if we skip time will we
        # increment the ticks even though we don't process them?
        return self.timestamp_to_datetime(self.timestamp)

    def quit(self) -> None:
        self.keep_running = False

    def send_message(self, recipient: Character, message: Message) -> None:
        recipient.send_message(message)
        self.game_runtime.send_message(recipient, message)

    def trigger_event(
        self,
        characters: Iterable[Character],
        event_type: int,
        context: Mapping[int, int],
        event_args: MutableMapping[str, Any] = {},
    ) -> None:
        self.game_runtime.trigger_event(characters, event_type, context, event_args)

    def trigger_event_immediate(
        self,
        characters: Iterable[Character],
        event_type: int,
        context: Mapping[int, int],
        event_args: MutableMapping[str, Any] = {},
    ) -> None:
        self.game_runtime.trigger_event_immediate(characters, event_type, context, event_args)
