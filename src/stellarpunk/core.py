""" Core stuff for Stellarpunk """

from __future__ import annotations

import uuid
import datetime
import enum
import logging
import collections
import gzip
import json
from typing import Optional, Deque, Callable, Iterable, Dict, List, Any, Union, TextIO, Tuple, Iterator, Mapping, Sequence, TypeAlias
import abc

import graphviz # type: ignore
import numpy as np
import numpy.typing as npt
import pymunk

from stellarpunk import util

class ProductionChain:
    """ A production chain of resources/products interconnected in a DAG.

    The first rank are raw resources. The second rank are processed forms for
    each of these called "first products." The last rank are final products."""

    def __init__(self) -> None:
        # how many nodes per rank
        self.ranks = np.zeros((0,), dtype=np.int64)
        # adjacency matrix for the production chain
        self.adj_matrix = np.zeros((0,0))
        # how much each product is marked up over base input price
        self.markup = np.zeros((0,))
        # how much each product is priced (sum_inputs(input cost * input amount) * markup)
        self.prices = np.zeros((0,))

        self.production_times = np.zeros((0,))
        self.production_coolingoff_time = 5.
        self.batch_sizes = np.zeros((0,))

        self.sink_names:Sequence[str] = []

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.adj_matrix.shape

    def inputs_of(self, product_id:int) -> npt.NDArray[np.float64]:
        return np.nonzero(self.adj_matrix[:,product_id])[0]

    def first_product_ids(self) -> npt.NDArray[np.int64]:
        return np.arange(self.ranks[0], self.ranks[1]+self.ranks[0])

    def final_product_ids(self) -> npt.NDArray[np.int64]:
        return np.arange(self.adj_matrix.shape[0]-self.ranks[-1], self.adj_matrix.shape[0])

    def viz(self) -> graphviz.Graph:
        g = graphviz.Digraph("production_chain", graph_attr={"rankdir": "TB"})
        g.attr(compound="true", ranksep="1.5")

        for s in range(self.adj_matrix.shape[0]):

            sink_start = self.adj_matrix.shape[0] - len(self.sink_names)
            if s < sink_start:
                node_name = f'{s}'
            else:
                assert np.count_nonzero(self.adj_matrix[:,s]) >= 3
                node_name = f'{self.sink_names[s-sink_start]}'

            g.node(f'{s}', label=f'{node_name}:\n${self.prices[s]:,.0f}')
            for t in range(self.adj_matrix.shape[1]):
                if self.adj_matrix[s, t] > 0:
                    g.edge(f'{s}', f'{t}', label=f'{self.adj_matrix[s, t]:.0f}')

        return g

class Entity:
    id_prefix = "ENT"

    def __init__(self, name:str, entity_id:Optional[uuid.UUID]=None)->None:
        self.entity_id = entity_id or uuid.uuid4()
        self._entity_id_short_int = int.from_bytes(self.entity_id.bytes[0:4], byteorder='big')
        self.name = name

    def short_id(self) -> str:
        """ first 32 bits as hex """
        return f'{self.id_prefix}-{self.entity_id.hex[:8]}'

    def short_id_int(self) -> int:
        return self._entity_id_short_int

    def __str__(self) -> str:
        return f'{self.short_id()}'

class Sector(Entity):
    """ A region of space containing resources, stations, ships. """

    id_prefix = "SEC"

    def __init__(self, x:int, y:int, radius:float, space:pymunk.Space, *args: Any, **kwargs: Any)->None:
        super().__init__(*args, **kwargs)
        # sector's position in the universe
        self.x = x
        self.y = y

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
        self.space:pymunk.Space = space

        self.effects: Deque[Effect] = collections.deque()

    def spatial_query(self, bbox:Tuple[float, float, float, float]) -> Iterator[SectorEntity]:
        for hit in self.space.bb_query(pymunk.BB(*bbox), pymunk.ShapeFilter(categories=pymunk.ShapeFilter.ALL_CATEGORIES())):
            yield hit.body.entity

    def spatial_point(self, point:npt.NDArray[np.float64], max_dist:Optional[float]=None) -> Iterator[SectorEntity]:
        if not max_dist:
            max_dist = self.radius*3
        for hit in self.space.point_query((point[0], point[1]), max_dist, pymunk.ShapeFilter(categories=pymunk.ShapeFilter.ALL_CATEGORIES())):
            yield hit.shape.body.entity # type: ignore[union-attr]

    def is_occupied(self, x:float, y:float, eps:float=1e1) -> bool:
        return any(True for _ in self.spatial_query((x-eps, y-eps, x+eps, y+eps)))

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
        else:
            raise ValueError(f'unknown entity type {entity.__class__}')

        self.space.add(entity.phys, *(entity.phys.shapes))
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
        else:
            raise ValueError(f'unknown entity type {entity.__class__}')

        self.space.remove(entity.phys, *entity.phys.shapes)
        entity.sector = None
        del self.entities[entity.entity_id]

class ObjectType(enum.IntEnum):
    OTHER = enum.auto()
    SHIP = enum.auto()
    STATION = enum.auto()
    PLANET = enum.auto()
    ASTEROID = enum.auto()

class ObjectFlag(enum.IntFlag):
    # note: with pymunk we get up to 32 of these (depending on the c-type?)
    SHIP = enum.auto()
    STATION = enum.auto()
    PLANET = enum.auto()
    ASTEROID = enum.auto()

class HistoryEntry:
    def __init__(
            self,
            prefix:str,
            entity_id:uuid.UUID,
            ts:float,
            loc:np.ndarray,
            radius:float,
            angle:float,
            velocity:np.ndarray,
            angular_velocity:float,
            force:tuple[float,float],
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
            "loc": self.loc.tolist(),
            "r": self.radius,
            "a": self.angle,
            "v": self.velocity.tolist(),
            "av": self.angular_velocity,
            "f": self.force,
            "t": self.torque,
            "o": self.order_hist,
        }

class SectorEntity(Entity):
    """ An entity in space in a sector. """

    object_type = ObjectType.OTHER

    def __init__(self, loc:npt.NDArray[np.float64], phys: pymunk.Body, num_products:int, *args:Any, history_length:int=60*60, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        self.sector: Optional[Sector] = None

        # some physical properties (in SI units)
        self.mass = 0.
        self.moment = 0.
        self.loc = loc
        self.velocity:npt.NDArray[np.float64] = np.array((0.,0.), dtype=np.float64)
        self.angle = 0.
        self.angular_velocity = 0.

        self.cargo_capacity = 1e3

        # physics simulation entity (we don't manage this, just have a pointer to it)
        self.phys = phys
        #TODO: are all entities just circles?
        self.radius = 0.

        self.history: Deque[HistoryEntry] = collections.deque(maxlen=history_length)

        self.cargo:npt.NDArray[np.float64] = np.zeros((num_products,))

    def __str__(self) -> str:
        return f'{self.short_id()} at {self.loc} v:{self.velocity} theta:{self.angle:.1f} w:{self.angular_velocity:.1f}'

    def distance_to(self, other:SectorEntity) -> float:
        return util.distance(self.loc, other.loc) - self.radius - other.radius

    def cargo_full(self) -> bool:
        return np.sum(self.cargo) == self.cargo_capacity

    def get_history(self) -> Iterable[HistoryEntry]:

        return (HistoryEntry(
                self.id_prefix,
                self.entity_id, 0,
                self.loc, self.radius, self.angle,
                self.velocity, self.angular_velocity,
                (0.,0.), 0,
        ),)
        return self.history

    def speed(self) -> float:
        return util.magnitude(self.velocity[0], self.velocity[1])

    def to_history(self, timestamp:float) -> HistoryEntry:
        return HistoryEntry(
                self.id_prefix,
                self.entity_id, timestamp,
                self.loc.copy(), self.radius, self.angle,
                self.velocity.copy(), self.angular_velocity,
                (0.,0.), 0,
        )
    def address_str(self) -> str:
        if self.sector:
            return f'{self.short_id()}@{self.sector.short_id()}'
        else:
            return f'{self.short_id()}@None'

class Planet(SectorEntity):

    id_prefix = "PLT"
    object_type = ObjectType.PLANET

    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.population = 0.

class Station(SectorEntity):

    id_prefix = "STA"
    object_type = ObjectType.STATION

    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.resource: Optional[int] = None
        self.next_batch_time = 0.
        self.next_production_time = 0.
        self.cargo_capacity = 1e5

class Ship(SectorEntity):
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

        self.orders: Deque[Order] = collections.deque()
        self.default_order_fn:Ship.DefaultOrderSig = lambda ship, gamestate: Order(ship, gamestate)

        self.collision_threat: Optional[SectorEntity] = None

    def get_history(self) -> Iterable[HistoryEntry]:
        return self.history

    def to_history(self, timestamp:float) -> HistoryEntry:
        order_hist = self.orders[0].to_history() if self.orders else None
        return HistoryEntry(
                self.id_prefix,
                self.entity_id, timestamp,
                self.loc.copy(), self.radius, self.angle,
                self.velocity.copy(), self.angular_velocity,
                self.phys.force, self.phys.torque,
                order_hist,
        )

    def max_speed(self) -> float:
        return self.max_thrust / self.mass * 10

    def max_acceleration(self) -> float:
        return self.max_thrust / self.mass

    def max_fine_acceleration(self) -> float:
        return self.max_fine_thrust / self.mass

    def max_angular_acceleration(self) -> float:
        return self.max_torque / self.moment

    def apply_force(self, force: npt.NDArray[np.float64]) -> None:
        self.phys.apply_force_at_world_point(
                (force[0], force[1]),
                (self.loc[0], self.loc[1])
        )

    def apply_torque(self, torque: float) -> None:
        self.phys.torque = torque

    def default_order(self, gamestate: Gamestate) -> Order:
        return self.default_order_fn(self, gamestate)

    def clear_orders(self) -> None:
        while self.orders:
            self.orders[0].cancel_order()

class Asteroid(SectorEntity):

    id_prefix = "AST"
    object_type = ObjectType.ASTEROID

    def __init__(self, resource:int, amount:float, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.resource = resource
        self.cargo[self.resource] = amount

    def __str__(self) -> str:
        return f'{self.short_id()} at {self.loc} r:{self.resource} a:{self.cargo[self.resource]}'

class Character(Entity):
    def __init__(self, *args:Any, **kwargs:Any):
        super().__init__(*args, **kwargs)

class Effect(abc.ABC):
    def __init__(self, sector:Sector, gamestate:Gamestate) -> None:
        self.sector = sector
        self.gamestate = gamestate
        self.started_at = -1.
        self.completed_at = -1.

        self.logger = logging.getLogger(util.fullname(self))

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

    def complete_effect(self) -> None:
        self.completed_at = self.gamestate.timestamp
        self._complete()

    def cancel_effect(self) -> None:
        try:
            self.sector.effects.remove(self)
        except ValueError:
            # effect might already have been removed from the queue
            pass

        self._cancel()

    def act(self, dt:float) -> None:
        pass

class OrderLoggerAdapter(logging.LoggerAdapter):
    def __init__(self, ship:Ship, *args:Any, **kwargs:Any):
        super().__init__(*args, **kwargs)
        self.ship = ship

    def process(self, msg:str, kwargs:Any) -> tuple[str, Any]:
        assert self.ship.sector is not None
        return f'{self.ship.short_id()}@{self.ship.sector.short_id()} {msg}', kwargs

class Order:
    def __init__(self, ship: Ship, gamestate: Gamestate) -> None:
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

    def __str__(self) -> str:
        return f'{self.__class__} for {self.ship}'

    def add_child(self, order:Order) -> None:
        self.child_orders.appendleft(order)
        self.ship.orders.appendleft(order)

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

    def complete_order(self) -> None:
        """ Called when an order is_complete and about to be removed from the
        order queue. """
        self.completed_at = self.gamestate.timestamp
        self._complete()

    def cancel_order(self) -> None:
        """ Called when an order is removed from the order queue, but not
        because it's complete. Note the order _might_ be complete in this case.
        """
        for order in self.child_orders:
            order.cancel_order()
            try:
                self.ship.orders.remove(order)
            except ValueError:
                # order might already have been removed from the queue
                pass

        try:
            self.ship.orders.remove(self)
        except ValueError:
            # order might already have been removed from the queue
            pass

        self._cancel()

    def act(self, dt:float) -> None:
        """ Performs one immediate tick's worth of action for this order """
        pass

class Gamestate:
    def __init__(self) -> None:

        self.random = np.random.default_rng()

        # the production chain of resources (ingredients
        self.production_chain = ProductionChain()

        # the universe is a set of sectors, indexed by coordinate
        self.sectors:Dict[tuple[int,int], Sector] = {}
        self.entities:Dict[uuid.UUID, Entity] = {}

        #self.characters = []

        self.keep_running = True

        self.base_date = datetime.datetime(2234, 4, 3)
        self.timestamp = 0.

        self.dt = 1/60
        # how many seconds of simulation (as in dt) should elapse per second
        self.time_accel_rate = 1.0
        self.ticks = 0
        self.ticktime = 0.
        self.timeout = 0.
        self.missed_ticks = 0

        self.keep_running = True
        self.paused = False
        self.should_raise= False

        self.player = Player()

    def current_time(self) -> datetime.datetime:
        #TODO: probably want to decouple telling time from ticks processed
        # we want missed ticks to slow time, but if we skip time will we
        # increment the ticks even though we don't process them?
        return datetime.datetime.fromtimestamp(self.base_date.timestamp() + self.timestamp)

    def quit(self) -> None:
        self.keep_running = False

def write_history_to_file(entity:Union[Sector, SectorEntity], f:Union[str, TextIO], mode:str="w") -> None:
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
        for entry in ent.get_history():
            fout.write(json.dumps(entry.to_json()))
            fout.write("\n")
    if needs_close:
        fout.close()

class Player:
    def __init__(self) -> None:
        # which ship the player is in command of, if any
        self.ship: Optional[Ship] = None
