""" Core stuff for Stellarpunk """

from __future__ import annotations

import uuid
import datetime
import enum
import logging
import collections
import gzip
import json
import itertools
from typing import Optional, Deque, Callable, Iterable, Dict, List, Any, Union, TextIO, Tuple, Iterator, Mapping, Sequence, TypeAlias, Iterator, MutableSequence, MutableMapping, TypeVar, Generic, Set, Collection, Generator, Set
import abc
import heapq
from dataclasses import dataclass
import abc

import graphviz # type: ignore
import numpy as np
import numpy.typing as npt
import cymunk # type: ignore
from rtree import index # type: ignore
import drawille # type: ignore

from stellarpunk import util, task_schedule, dialog

RESOURCE_REL_SHIP = 0
RESOURCE_REL_STATION = 1
RESOURCE_REL_CONSUMER = 2

class ProductionChain:
    """ A production chain of resources/products interconnected in a DAG.

    The first rank are raw resources. The second rank are processed forms for
    each of these called "first products." The last rank are final products."""

    def __init__(self) -> None:
        self.num_products = 0
        # how many nodes per rank
        self.ranks = np.zeros((self.num_products,), dtype=np.int64)
        # adjacency matrix for the production chain
        self.adj_matrix = np.zeros((self.num_products,self.num_products))
        # how much each product is marked up over base input price
        self.markup = np.zeros((self.num_products,))
        # how much each product is priced (sum_inputs(input cost * input amount) * markup)
        self.prices = np.zeros((self.num_products,))

        self.production_times = np.zeros((self.num_products,))
        self.production_coolingoff_time = 5.
        self.batch_sizes = np.zeros((self.num_products,))

        self.product_names:Sequence[str] = []

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.adj_matrix.shape

    def initialize(self) -> None:
        self.num_products = self.shape[0]

        assert sum(self.ranks) == self.num_products
        assert self.adj_matrix.shape == (self.num_products, self.num_products)
        assert self.markup.shape == (self.num_products, )
        assert self.prices.shape == (self.num_products, )
        assert self.production_times.shape == (self.num_products, )
        assert self.batch_sizes.shape == (self.num_products, )
        assert len(self.product_names) == self.num_products

    def inputs_of(self, product_id:int) -> npt.NDArray[np.int64]:
        return np.nonzero(self.adj_matrix[:,product_id])[0]

    def first_product_ids(self) -> npt.NDArray[np.int64]:
        return np.arange(self.ranks[0], self.ranks[1]+self.ranks[0])

    def final_product_ids(self) -> npt.NDArray[np.int64]:
        return np.arange(self.num_products-self.ranks[-1], self.num_products)

    def viz(self) -> graphviz.Graph:
        g = graphviz.Digraph("production_chain", graph_attr={"rankdir": "TB"})
        g.attr(compound="true", ranksep="1.5")

        for s in range(self.num_products):
            g.node(f'{s}', label=f'{self.product_names[s]} ({s}):\n${self.prices[s]:,.0f}')
            for t in range(self.num_products):
                if self.adj_matrix[s, t] > 0:
                    g.edge(f'{s}', f'{t}', label=f'{self.adj_matrix[s, t]:.0f}')

        return g

class Entity(abc.ABC):
    id_prefix = "ENT"

    def __init__(self, name:Optional[str]=None, entity_id:Optional[uuid.UUID]=None, description:Optional[str]=None)->None:
        self.entity_id = entity_id or uuid.uuid4()
        self._entity_id_short_int = int.from_bytes(self.entity_id.bytes[0:4], byteorder='big')

        if name is None:
            name = f'{self.__class__} {str(self.entity_id)}'
        self.name = name

        self.description = description or name

        self.flags:Dict[int, int] = {}

    def short_id(self) -> str:
        """ first 32 bits as hex """
        return f'{self.id_prefix}-{self.entity_id.hex[:8]}'

    def short_id_int(self) -> int:
        return self._entity_id_short_int

    def set_flag(self, flag:int, value:int) -> None:
        self.flags[flag] = value

    def to_context(self) -> Dict[int, int]:
        return self.flags

    def __str__(self) -> str:
        return f'{self.short_id()}'

class Asset(Entity):
    """ An abc for classes that are assets ownable by characters. """
    def __init__(self, *args:Any, owner:Optional["Character"]=None, **kwargs:Any) -> None:
        # forward arguments onward, so implementing classes should inherit us
        # first
        super().__init__(*args, **kwargs)
        self.owner = owner

class Sector(Entity):
    """ A region of space containing resources, stations, ships. """

    id_prefix = "SEC"

    def __init__(self, loc:npt.NDArray[np.float64], radius:float, space:cymunk.Space, *args: Any, **kwargs: Any)->None:
        super().__init__(*args, **kwargs)

        # sector's position in the universe
        self.loc = loc

        # one standard deviation
        self.radius = radius

        self.planets:List[Planet] = []
        self.stations:List[Station] = []
        self.ships:List[Ship] = []
        self.asteroids:Dict[int, List[Asteroid]] = collections.defaultdict(list)

        # id -> entity for all entities in the sector
        self.entities:Dict[uuid.UUID, SectorEntity] = {}

        # physics space for this sector
        # we don't manage this, just have a pointer to it
        # we do rely on this to provide a spatial index of the sector
        self.space:cymunk.Space = space

        self._effects: Deque[Effect] = collections.deque()

    def spatial_query(self, bbox:Tuple[float, float, float, float]) -> Iterator[SectorEntity]:
        for hit in self.space.bb_query(cymunk.BB(*bbox)):
            yield hit.body.data

    def spatial_point(self, point:Union[Tuple[float, float], npt.NDArray[np.float64]], max_dist:Optional[float]=None, mask:Optional[ObjectFlag]=None) -> Iterator[SectorEntity]:
        #TODO: honor mask
        if not max_dist:
            max_dist = np.inf
        for hit in self.space.nearest_point_query(cymunk.vec2d.Vec2d(point[0], point[1]), max_dist):
            yield hit.body.data

    def is_occupied(self, x:float, y:float, eps:float=1e1) -> bool:
        return any(True for _ in self.spatial_query((x-eps, y-eps, x+eps, y+eps)))

    def add_effect(self, effect:Effect) -> None:
        self._effects.append(effect)
        effect.begin_effect()

    def remove_effect(self, effect:Effect) -> None:
        self._effects.remove(effect)

    def add_entity(self, entity:SectorEntity) -> None:
        #TODO: worry about collisions at location?

        if isinstance(entity, Planet):
            self.planets.append(entity)
        elif isinstance(entity, Station):
            self.stations.append(entity)
        elif isinstance(entity, Ship):
            self.ships.append(entity)
        elif isinstance(entity, Asteroid):
            self.asteroids[entity.resource].append(entity)
        elif isinstance(entity, TravelGate):
            pass
        else:
            raise ValueError(f'unknown entity type {entity.__class__}')

        if entity.phys.is_static:
            self.space.add(entity.phys_shape)
        else:
            self.space.add(entity.phys, entity.phys_shape)
        entity.sector = self
        self.entities[entity.entity_id] = entity

    def remove_entity(self, entity:SectorEntity) -> None:

        if entity.entity_id not in self.entities:
            raise ValueError(f'entity {entity.entity_id} not in this sector')

        if isinstance(entity, Planet):
            self.planets.remove(entity)
        elif isinstance(entity, Station):
            self.stations.remove(entity)
        elif isinstance(entity, Ship):
            self.ships.remove(entity)
        elif isinstance(entity, Asteroid):
            self.asteroids[entity.resource].remove(entity)
        elif isinstance(entity, TravelGate):
            pass
        else:
            raise ValueError(f'unknown entity type {entity.__class__}')

        if entity.phys.is_static:
            self.space.remove(entity.phys_shape)
        else:
            self.space.remove(entity.phys, entity.phys_shape)
        entity.sector = None
        del self.entities[entity.entity_id]

class ObjectType(enum.IntEnum):
    OTHER = enum.auto()
    SHIP = enum.auto()
    STATION = enum.auto()
    PLANET = enum.auto()
    ASTEROID = enum.auto()
    TRAVEL_GATE = enum.auto()

class ObjectFlag(enum.IntFlag):
    # note: with pymunk we get up to 32 of these (depending on the c-type?)
    SHIP = enum.auto()
    STATION = enum.auto()
    PLANET = enum.auto()
    ASTEROID = enum.auto()
    GATE = enum.auto()

class HistoryEntry:
    def __init__(
            self,
            prefix:str,
            entity_id:uuid.UUID,
            ts:float,
            loc:tuple,
            radius:float,
            angle:float,
            velocity:tuple,
            angular_velocity:float,
            force:tuple,
            torque:float,
            order_hist:Optional[dict]=None
    ) -> None:
        self.entity_id = entity_id
        self.ts = ts

        self.order_hist = order_hist

        self.loc = loc
        self.radius = radius
        self.angle = angle
        self.prefix = prefix

        self.velocity = velocity
        self.angular_velocity = angular_velocity

        self.force = force
        self.torque = torque

    def to_json(self) -> Mapping[str, Any]:
        return {
            "p": self.prefix,
            "eid": str(self.entity_id),
            "ts": self.ts,
            "loc": self.loc,
            "r": self.radius,
            "a": self.angle,
            "v": self.velocity,
            "av": self.angular_velocity,
            "f": self.force,
            "t": self.torque,
            "o": self.order_hist,
        }

class SectorEntity(Entity):
    """ An entity in space in a sector. """

    object_type = ObjectType.OTHER

    def __init__(self, loc:npt.NDArray[np.float64], phys: cymunk.Body, num_products:int, *args:Any, history_length:int=60*60, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        self.sector:Optional[Sector] = None

        # some physical properties (in SI units)
        self.mass = 0.
        self.moment = 0.

        phys.position = (loc[0], loc[1])

        self.cargo_capacity = 5e2

        # physics simulation entity (we don't manage this, just have a pointer to it)
        self.phys = phys
        self.phys_shape:Any = None
        #TODO: are all entities just circles?
        self.radius = 0.

        self.history: Deque[HistoryEntry] = collections.deque(maxlen=history_length)

        self.cargo:npt.NDArray[np.float64] = np.zeros((num_products,))

        # who is responsible for this entity?
        self.captain: Optional[Character] = None

    @property
    def loc(self) -> npt.NDArray[np.float64]: return np.array(self.phys.position)
    @property
    def velocity(self) -> npt.NDArray[np.float64]: return np.array(self.phys.velocity)
    @property
    def speed(self) -> float: return self.phys.velocity.length
    @property
    def angle(self) -> float: return self.phys.angle
    @property
    def angular_velocity(self) -> float: return self.phys.angular_velocity

    def distance_to(self, other:SectorEntity) -> float:
        return util.distance(self.loc, other.loc) - self.radius - other.radius

    def cargo_full(self) -> bool:
        return np.sum(self.cargo) == self.cargo_capacity

    def get_history(self) -> Sequence[HistoryEntry]:

        return (HistoryEntry(
                self.id_prefix,
                self.entity_id, 0,
                tuple(self.phys.position), self.radius, self.angle,
                tuple(self.phys.velocity), self.angular_velocity,
                (0.,0.), 0,
        ),)
        return self.history

    def to_history(self, timestamp:float) -> HistoryEntry:
        return HistoryEntry(
                self.id_prefix,
                self.entity_id, timestamp,
                tuple(self.phys.position), self.radius, self.angle,
                tuple(self.phys.velocity), self.angular_velocity,
                (0.,0.), 0,
        )
    def address_str(self) -> str:
        if self.sector:
            return f'{self.short_id()}@{self.sector.short_id()}'
        else:
            return f'{self.short_id()}@None'

class Planet(SectorEntity, Asset):

    id_prefix = "HAB"
    object_type = ObjectType.PLANET

    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.population = 0.

class Station(SectorEntity, Asset):

    id_prefix = "STA"
    object_type = ObjectType.STATION

    def __init__(self, sprite:Sprite, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.resource: Optional[int] = None
        self.next_batch_time = 0.
        self.next_production_time = 0.
        self.cargo_capacity = 1e5

        self.sprite = sprite

class Ship(SectorEntity, Asset):
    DefaultOrderSig:TypeAlias = "Callable[[Ship, Gamestate], Order]"

    id_prefix = "SHP"
    object_type = ObjectType.SHIP

    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)


        # SI units (newtons and newton-meters)
        # max thrust along heading vector
        self.max_thrust = 0.
        # max thrust in any direction
        self.max_fine_thrust = 0.
        # max torque for turning (in newton-meters)
        self.max_torque = 0.

        self._orders: Deque[Order] = collections.deque()
        self.default_order_fn:Ship.DefaultOrderSig = lambda ship, gamestate: Order(ship, gamestate)

        self.collision_threat: Optional[SectorEntity] = None

    def get_history(self) -> Sequence[HistoryEntry]:
        return self.history

    def to_history(self, timestamp:float) -> HistoryEntry:
        order_hist = self._orders[0].to_history() if self._orders else None
        return HistoryEntry(
                self.id_prefix,
                self.entity_id, timestamp,
                tuple(self.phys.position), self.radius, self.angle,
                tuple(self.phys.velocity), self.angular_velocity,
                tuple(self.phys.force), self.phys.torque,
                order_hist,
        )

    def max_speed(self) -> float:
        return self.max_thrust / self.mass * 30

    def max_acceleration(self) -> float:
        return self.max_thrust / self.mass

    def max_fine_acceleration(self) -> float:
        return self.max_fine_thrust / self.mass

    def max_angular_acceleration(self) -> float:
        return self.max_torque / self.moment

    def apply_force(self, force: Union[Sequence[float], npt.NDArray[np.float64]], persistent:bool) -> None:
        self.phys.force = cymunk.vec2d.Vec2d(*force)

    def apply_torque(self, torque: float, persistent:bool) -> None:
        self.phys.torque = torque

    def set_loc(self, loc: Union[Sequence[float], npt.NDArray[np.float64]]) -> None:
        self.phys.position = (loc[0], loc[1])

    def set_velocity(self, velocity: Union[Sequence[float], npt.NDArray[np.float64]]) -> None:
        self.phys.velocity = (velocity[0], velocity[1])

    def set_angle(self, angle: float) -> None:
        self.phys.angle = angle

    def set_angular_velocity(self, angular_velocity:float) -> None:
        self.phys.angular_velocity = angular_velocity

    def default_order(self, gamestate: Gamestate) -> Order:
        return self.default_order_fn(self, gamestate)

    def prepend_order(self, order:Order, begin:bool=True) -> None:
        co = self.current_order()
        if co is not None:
            #TODO: should we do anything else to suspend the current order?
            if co.gamestate.is_order_scheduled(co):
                co.gamestate.unschedule_order(co)

        self._orders.appendleft(order)
        if begin:
            order.begin_order()

    def append_order(self, order:Order, begin:bool=False) -> None:
        self._orders.append(order)
        if begin:
            order.begin_order()

    def remove_order(self, order:Order) -> None:
        if order in self._orders:
            self._orders.remove(order)

        co = self.current_order()
        if co is not None and not co.gamestate.is_order_scheduled(co):
            co.gamestate.schedule_order_immediate(co)

    def clear_orders(self, gamestate:Gamestate) -> None:
        while self._orders:
            self._orders[0].cancel_order()
        self.prepend_order(self.default_order(gamestate))

    def pop_current_order(self) -> None:
        self._orders.popleft()

    def complete_current_order(self) -> None:
        order = self._orders.popleft()
        order.complete_order()

    def current_order(self) -> Optional[Order]:
        if len(self._orders) > 0:
            return self._orders[0]
        else:
            return None

class Asteroid(SectorEntity):

    id_prefix = "AST"
    object_type = ObjectType.ASTEROID

    def __init__(self, resource:int, amount:float, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.resource = resource
        self.cargo[self.resource] = amount

class TravelGate(SectorEntity):
    """ Represents a "gate" to another sector """

    id_prefix = "GAT"
    object_type = ObjectType.TRAVEL_GATE

    def __init__(self, destination:Sector, direction:float, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.destination = destination
        # radian angle toward the destination
        self.direction:float = direction
        self.direction_vector = np.array(util.polar_to_cartesian(1., direction))

class AgendumLoggerAdapter(logging.LoggerAdapter):
    def __init__(self, character:Character, *args:Any, **kwargs:Any):
        super().__init__(*args, **kwargs)
        self.character = character

    def process(self, msg:str, kwargs:Any) -> tuple[str, Any]:
        return f'{self.character.address_str()} {msg}', kwargs

class Agendum:
    """ Represents an activity a Character is engaged in and how they can
    interact with the world. """

    def __init__(self, character:Character, gamestate:Gamestate) -> None:
        self.character = character
        self.gamestate = gamestate
        self.logger = AgendumLoggerAdapter(
                self.character,
                logging.getLogger(util.fullname(self)),
        )

        logging.getLogger(util.fullname(self))

    def _start(self) -> None:
        pass

    def _stop(self) -> None:
        pass

    def start(self) -> None:
        self._start()

    def stop(self) -> None:
        self._stop()
        self.gamestate.unschedule_agendum(self)

    def is_complete(self) -> bool:
        return False

    def act(self) -> None:
        """ Lets the character interact. Called when scheduled. """
        pass

class Sprite:
    """ A "sprite" from a text file that can be drawn in text """

    @staticmethod
    def load_sprites(data:str, sprite_size:Tuple[int, int]) -> List[Sprite]:
        sprites = []
        sheet = data.split("\n")
        offset_limit = (len(sheet[0])//sprite_size[0], len(sheet)//sprite_size[1])
        for offset_x, offset_y in itertools.product(range(offset_limit[0]), range(offset_limit[1])):
            sprites.append(Sprite(
                [
                    x[offset_x*sprite_size[0]:offset_x*sprite_size[0]+sprite_size[0]] for x in sheet[offset_y*sprite_size[1]:offset_y*sprite_size[1]+sprite_size[1]]
                ]
            ))

        return sprites

    @staticmethod
    def composite_sprites(sprites:Sequence[Sprite]) -> Sprite:
        if len(sprites) == 0:
            raise ValueError("no sprites to composite")

        text = [[" "]*sprites[0].width for _ in range(sprites[0].height)]
        attr:Dict[Tuple[int,int], Tuple[int, int]] = {}

        for sprite in sprites:
            for y, row in enumerate(sprite.text):
                for x, c in enumerate(sprite.text[y]):
                    if c != " " and c != chr(drawille.braille_char_offset):
                        text[y][x] = c
                        if (x,y) in sprite.attr:
                            attr[x,y] = sprite.attr[x,y]
                        else:
                            attr[x,y] = (0,0)

        return Sprite(["".join(t) for t in text], attr)

    def __init__(self, text:Sequence[str], attr:Optional[Dict[Tuple[int,int], Tuple[int, int]]]=None) -> None:
        self.text = text
        self.attr = attr or {}
        self.height = len(text)
        if len(text) > 0:
            self.width = len(text[0])
        else:
            self.width = 0

class Character(Entity):
    id_prefix = "CHR"
    def __init__(self, sprite:Sprite, location:SectorEntity, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        self.portrait:Sprite = sprite
        #TODO: other character background stuff

        #TODO: does location matter?
        self.location:SectorEntity = location

        # how much money
        self.balance:float = 0.

        # owned assets (ships, stations)
        #TODO: are these actually SectorEntity instances? maybe a new co-class (Asset)
        self.assets:MutableSequence[Asset] = []
        # activites this character is enaged in (how they interact)
        self.agenda:MutableSequence[Agendum] = []

    def address_str(self) -> str:
        return f'{self.short_id()}:{self.location.address_str()}'

    def take_ownership(self, asset:Asset) -> None:
        self.assets.append(asset)
        asset.owner = self

    def add_agendum(self, agendum:Agendum, start:bool=True) -> None:
        self.agenda.append(agendum)
        if start:
            agendum.start()

class EffectObserver:
    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

    def effect_begin(self, effect:"Effect") -> None:
        pass

    def effect_complete(self, effect:"Effect") -> None:
        pass

    def effect_cancel(self, effect:"Effect") -> None:
        pass

class Effect(abc.ABC):
    def __init__(self, sector:Sector, gamestate:Gamestate, observer:Optional[EffectObserver]=None) -> None:
        self.sector = sector
        self.gamestate = gamestate
        self.started_at = -1.
        self.completed_at = -1.
        self.observers:List[EffectObserver] = []

        self.logger = logging.getLogger(util.fullname(self))

        if observer is not None:
            self.observe(observer)

    def observe(self, observer:EffectObserver) -> None:
        self.observers.append(observer)

    def _begin(self) -> None:
        pass

    def _complete(self) -> None:
        pass

    def _cancel(self) -> None:
        pass

    @abc.abstractmethod
    def bbox(self) -> Tuple[float, float, float, float]:
        """ returns a 4-tuple bounding box ul_x, ul_y, lr_x, lr_y """
        pass

    def is_complete(self) -> bool:
        return True

    def begin_effect(self) -> None:
        self.started_at = self.gamestate.timestamp
        self._begin()

        for observer in self.observers:
            observer.effect_begin(self)

    def complete_effect(self) -> None:
        self.completed_at = self.gamestate.timestamp
        self._complete()

        self.logger.debug(f'effect {self} in {self.sector.short_id()} complete in {self.gamestate.timestamp - self.started_at:.2f}')

        for observer in self.observers:
            observer.effect_complete(self)

        self.sector.remove_effect(self)

    def cancel_effect(self) -> None:
        self.gamestate.unschedule_effect(self)
        try:
            self.sector.remove_effect(self)
        except ValueError:
            # effect might already have been removed from the queue
            pass

        self._cancel()

        for observer in self.observers:
            observer.effect_cancel(self)

    def act(self, dt:float) -> None:
        # by default we'll just complete the effect if it's done
        if self.is_complete():
            self.complete_effect()

class OrderLoggerAdapter(logging.LoggerAdapter):
    def __init__(self, ship:Ship, *args:Any, **kwargs:Any):
        super().__init__(*args, **kwargs)
        self.ship = ship

    def process(self, msg:str, kwargs:Any) -> tuple[str, Any]:
        assert self.ship.sector is not None
        return f'{self.ship.short_id()}@{self.ship.sector.short_id()} {msg}', kwargs

class OrderObserver:
    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

    def order_begin(self, order:"Order") -> None:
        pass

    def order_complete(self, order:"Order") -> None:
        pass

    def order_cancel(self, order:"Order") -> None:
        pass

class Order:
    def __init__(self, ship: Ship, gamestate: Gamestate, observer:Optional[OrderObserver]=None) -> None:
        self.gamestate = gamestate
        self.ship = ship
        self.logger = OrderLoggerAdapter(
                ship,
                logging.getLogger(util.fullname(self)),
        )
        self.o_name = util.fullname(self)
        self.started_at = -1.
        self.completed_at = -1.
        self.init_eta = np.inf
        self.child_orders:Deque[Order] = collections.deque()

        self.observers:List[OrderObserver] = []
        if observer is not None:
            self.observe(observer)

    def __str__(self) -> str:
        return f'{self.__class__} for {self.ship}'

    def observe(self, observer:OrderObserver) -> None:
        self.observers.append(observer)

    def _add_child(self, order:Order, begin:bool=True) -> None:
        self.child_orders.appendleft(order)
        self.ship.prepend_order(order, begin=begin)

    def to_history(self) -> dict:
        return {"o": self.o_name}

    def is_complete(self) -> bool:
        """ Indicates that this Order is ready to complete and be removed from
        the order queue. """
        return False

    def _begin(self) -> None:
        pass

    def _complete(self) -> None:
        pass

    def _cancel(self) -> None:
        pass

    def begin_order(self) -> None:
        """ Called when an order is ready to start acting, at the front of the
        order queue, before the first call to act. This is a good time to
        compute the eta for the order."""
        self.started_at = self.gamestate.timestamp
        self._begin()

        for observer in self.observers:
            observer.order_begin(self)

        self.gamestate.schedule_order_immediate(self)

    def complete_order(self) -> None:
        """ Called when an order is_complete and about to be removed from the
        order queue. """
        self.completed_at = self.gamestate.timestamp
        self._complete()

        for observer in self.observers:
            observer.order_complete(self)

    def cancel_order(self) -> None:
        """ Called when an order is removed from the order queue, but not
        because it's complete. Note the order _might_ be complete in this case.
        """
        self.gamestate.unschedule_order(self)
        for order in self.child_orders:
            order.cancel_order()
            try:
                self.ship.remove_order(order)
            except ValueError:
                # order might already have been removed from the queue
                pass

        try:
            self.ship.remove_order(self)
        except ValueError:
            # order might already have been removed from the queue
            pass

        self._cancel()

        for observer in self.observers:
            observer.order_cancel(self)

    def act(self, dt:float) -> None:
        """ Performs one immediate tick's worth of action for this order """
        pass

class EconAgent(abc.ABC):
    _next_id = 0

    @classmethod
    def num_agents(cls) -> int:
        return EconAgent._next_id

    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.agent_id = EconAgent._next_id
        EconAgent._next_id += 1

    #@abc.abstractmethod
    #def get_owner(self) -> Character: ...

    #def get_character(self) -> Character:
    #    return self.get_owner()

    @abc.abstractmethod
    def buy_resources(self) -> Collection[int]: ...

    @abc.abstractmethod
    def sell_resources(self) -> Collection[int]: ...

    @abc.abstractmethod
    def buy_price(self, resource:int) -> float: ...

    @abc.abstractmethod
    def sell_price(self, resource:int) -> float: ...

    @abc.abstractmethod
    def balance(self) -> float: ...

    @abc.abstractmethod
    def budget(self, resource:int) -> float: ...

    @abc.abstractmethod
    def inventory(self, resource:int) -> float: ...

    @abc.abstractmethod
    def buy(self, resource:int, price:float, amount:float) -> None: ...

    @abc.abstractmethod
    def sell(self, resource:int, price:float, amount:float) -> None: ...

class AbstractEconDataLogger:
    def transact(self, diff:float, product_id:int, buyer:int, seller:int, price:float, sale_amount:float, ticks:Optional[Union[int,float]]) -> None:
        pass

    def log_econ(self,
            ticks:float,
            inventory:npt.NDArray[np.float64],
            balance:npt.NDArray[np.float64],
            buy_prices:npt.NDArray[np.float64],
            buy_budget:npt.NDArray[np.float64],
            sell_prices:npt.NDArray[np.float64],
            max_buy_prices:npt.NDArray[np.float64],
            min_sell_prices:npt.NDArray[np.float64],
            cannot_buy_ticks:npt.NDArray[np.int64],
            cannot_sell_ticks:npt.NDArray[np.int64],
    ) -> None:
        pass

    def flush(self) -> None:
        pass

class AbstractGameRuntime:
    """ The game runtime that actually runs the simulation. """

    def get_time_acceleration(self) -> Tuple[float, bool]:
        """ Get time acceleration parameters. """
        return (1.0, False)

    def time_acceleration(self, accel_rate:float, fast_mode:bool) -> None:
        """ Request time acceleration. """
        pass

class StarfieldLayer:
    def __init__(self, bbox:Tuple[float, float, float, float], zoom:float) -> None:
        self.bbox:Tuple[float, float, float, float] = bbox

        # list of stars: loc=(x,y), size, spectral_class
        self._star_list:List[Tuple[Tuple[float, float], float, int]] = []
        self.num_stars = 0

        # density in stars per m^2
        self.density:float = 0.

        # zoom level in meters per character
        self.zoom:float = zoom

    def add_star(self, loc:Tuple[float, float], size:float, spectral_class:int) -> None:
        self._star_list.append((loc, size, spectral_class))
        self.num_stars += 1
        self.density = self.num_stars / ((self.bbox[2]-self.bbox[0])*(self.bbox[3]-self.bbox[1]))

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


class EventType(enum.IntEnum):
    BROADCAST = enum.auto()
    APPROACH_DESTINATION = enum.auto()


class ContextKey(enum.IntEnum):
    IS_PLAYER = enum.auto()
    SHIP = enum.auto()
    DESTINATION = enum.auto()
    ENTITY_SHORT_ID_INT = enum.auto()
    MESSAGE_SENDER = enum.auto()
    MESSAGE_ID = enum.auto()


@dataclass
class Event:
    character: Character
    event_type: EventType
    context: Dict[int, int]
    entity_context: Dict[int, Dict[int, int]]
    entities: Dict[int, Entity]
    args: Dict[str, Any]

    def get_entity(self, key: int) -> Entity:
        return self.entities[self.context[key]]


class Gamestate:
    def __init__(self) -> None:
        self.logger = logging.getLogger(util.fullname(self))
        self.game_runtime:AbstractGameRuntime = AbstractGameRuntime()

        self.random = np.random.default_rng()

        # the production chain of resources (ingredients
        self.production_chain = ProductionChain()

        # the universe is a set of sectors, indexed by their entity id
        self.sectors:Dict[uuid.UUID, Sector] = {}
        self.sector_ids:npt.NDArray = np.ndarray((0,), uuid.UUID) #indexed same as edges
        self.sector_idx:Mapping[uuid.UUID, int] = {} #inverse of sector_ids
        self.sector_edges:npt.NDArray[np.float64] = np.ndarray((0,0))

        # a spatial index of sectors in the universe
        self.sector_spatial = index.Index()

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

        self.events: Deque[Event] = collections.deque()

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

    def pause(self) -> None:
        if self.force_pause_holder is not None:
            return
        self._pause()

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

    def withdraw_agent(self, entity_id:uuid.UUID) -> EconAgent:
        return self.econ_agents.pop(entity_id)

    def add_character(self, character:Character) -> None:
        self.characters[character.entity_id] = character
        self.characters_by_location[character.location.entity_id].append(character)

    def move_character(self, character:Character, location:SectorEntity) -> None:
        self.characters_by_location[character.location.entity_id].remove(character)
        self.characters_by_location[location.entity_id].append(character)
        character.location = location

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
        return hits

    def timestamp_to_datetime(self, timestamp:float) -> datetime.datetime:
        return datetime.datetime.fromtimestamp(self.base_date.timestamp() + timestamp)
    def current_time(self) -> datetime.datetime:
        #TODO: probably want to decouple telling time from ticks processed
        # we want missed ticks to slow time, but if we skip time will we
        # increment the ticks even though we don't process them?
        return self.timestamp_to_datetime(self.timestamp)

    def quit(self) -> None:
        self.keep_running = False

    def trigger_event(self, character: Character, event_type: EventType, context: Dict[int, int], *entities: Entity, **kwargs: Any) -> None:
        entity_context: Dict[int, Dict[int, int]] = {}
        entity_dict: Dict[int, Entity] = {}
        for entity in entities:
            entity_context[entity.short_id_int()] = entity.to_context()
            entity_dict[entity.short_id_int()] = entity

        self.events.append(Event(character, event_type, context, entity_context, entity_dict, kwargs))


def write_history_to_file(entity:Union[Sector, SectorEntity], f:Union[str, TextIO], mode:str="w", now:float=-np.inf) -> None:
    fout:TextIO
    if isinstance(f, str):
        needs_close = True
        if f.endswith(".gz"):
            fout = gzip.open(f, mode+"t") # type: ignore[assignment]
        else:
            fout = open(f, mode) # type: ignore[assignment]
    else:
        needs_close = False
        fout = f

    entities:Iterable[SectorEntity]
    if isinstance(entity, Sector):
        entities = entity.entities.values()
    else:
        entities = [entity]

    for ent in entities:
        history = ent.get_history()
        for entry in history:
            fout.write(json.dumps(entry.to_json()))
            fout.write("\n")
        if len(history) == 0 or history[-1].ts < now:
            fout.write(json.dumps(ent.to_history(now).to_json()))
            fout.write("\n")
    if needs_close:
        fout.close()

class Message(Entity):
    id_prefix = "MSG"

    def __init__(self, message:str, timestamp:float, *args:Any, reply_to:Optional[Character]=None, reply_dialog:Optional[dialog.DialogGraph]=None, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        self.message = message
        self.timestamp = timestamp

        self.reply_to = reply_to
        self.reply_dialog = reply_dialog
        self.replied_at:Optional[float] = None

class PlayerObserver(abc.ABC):
    def notification_received(self, player:Player, notification:str) -> None:
        pass
    def message_received(self, player:Player, message:Message) -> None:
        pass
    def flag_set(self, player:Player, flag:int) -> None:
        pass

class Player(Entity):
    id_prefix = "PLR"

    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        self.observers:Set[PlayerObserver] = set()

        # which character the player controls
        self.character:Character = None # type: ignore[assignment]
        self.agent:EconAgent = None # type: ignore[assignment]

        self.notifications:List[str] = []
        self.messages:Dict[uuid.UUID, Message] = {}

    def observe(self, observer:PlayerObserver) -> None:
        self.observers.add(observer)

    def send_notification(self, notification:str) -> None:
        self.notifications.append(notification)
        for observer in self.observers:
            observer.notification_received(self, notification)

    def send_message(self, message:Message) -> None:
        self.messages[message.entity_id] = message
        for observer in self.observers:
            observer.message_received(self, message)

    def set_flag(self, flag: int, value: int) -> None:
        super().set_flag(flag, value)

        for observer in self.observers:
            observer.flag_set(self, flag)

