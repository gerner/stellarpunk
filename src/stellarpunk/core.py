""" Core stuff for Stellarpunk """

from __future__ import annotations

import uuid
import datetime
import enum
import logging
import collections
import gzip
import json
from typing import Optional, Deque, Callable, Iterable

import graphviz # type: ignore
import numpy as np
import numpy.typing as npt
import pymunk

from stellarpunk import util

class ProductionChain:
    """ A production chain of resources/products interconnected in a DAG. """

    def __init__(self):
        # how many nodes per rank
        self.ranks = None
        # adjacency matrix for the production chain
        self.adj_matrix = None
        # how much each product is marked up over base input price
        self.markup = None
        # how much each product is priced (sum_inputs(input cost * input amount) * markup)
        self.prices = None

        self.sink_names = []

    def viz(self):
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

    def __init__(self, name, entity_id=None):
        self.entity_id = entity_id or uuid.uuid4()
        self.name = name

    def short_id(self):
        """ first 32 bits as hex """
        return f'{self.id_prefix}-{self.entity_id.hex[:8]}'

    def short_id_int(self):
        return int.from_bytes(self.entity_id.bytes[0:4], byteorder='big')

    def __str__(self):
        return f'{self.short_id()}'

class Sector(Entity):
    """ A region of space containing resources, stations, ships. """

    id_prefix = "SEC"

    def __init__(self, x, y, radius, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # sector's position in the universe
        self.x = x
        self.y = y

        # one standard deviation
        self.radius = radius

        self.planets = []
        self.stations = []
        self.ships = []
        self.asteroids = collections.defaultdict(list)
        self.resources = []

        # id -> entity for all entities in the sector
        self.entities = {}

        # physics space for this sector
        # we don't manage this, just have a pointer to it
        # we do rely on this to provide a spatial index of the sector
        self.space = None

    def spatial_query(self, bbox):
        for hit in self.space.bb_query(bbox, pymunk.ShapeFilter(categories=pymunk.ShapeFilter.ALL_CATEGORIES())):
            yield hit.body.entity

    def spatial_point(self, point, max_dist=None):
        if not max_dist:
            max_dist = self.radius*3
        for hit in self.space.point_query((point[0], point[1]), max_dist, pymunk.ShapeFilter(categories=pymunk.ShapeFilter.ALL_CATEGORIES())):
            yield hit.shape.body.entity

    def is_occupied(self, x, y, eps=1e1):
        return any(True for _ in self.spatial_query((x-eps, y-eps, x+eps, y+eps)))

    def add_entity(self, entity):
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

    def remove_entity(self, entity):

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
            ts:int,
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

    def to_json(self):
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

    def __init__(self, loc:npt.NDArray[np.float64], phys: pymunk.Body, *args, history_length=60*60, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.sector: Optional[Sector] = None

        # some physical properties
        self.mass = 0.
        self.moment = 0.
        self.loc = loc
        self.velocity:npt.NDArray[np.float64] = np.array((0.,0.), dtype=np.float64)
        self.angle = 0.
        self.angular_velocity = 0.

        # physics simulation entity (we don't manage this, just have a pointer to it)
        self.phys = phys
        #TODO: are all entities just circles?
        self.radius = 0.

        self.history: Deque[HistoryEntry] = collections.deque(maxlen=history_length)

    def __str__(self) -> str:
        return f'{self.short_id()} at {self.loc} v:{self.velocity} theta:{self.angle:.1f} w:{self.angular_velocity:.1f}'

    def get_history(self) -> Iterable[HistoryEntry]:

        return (HistoryEntry(
                self.id_prefix,
                self.entity_id, 0,
                self.loc, self.radius, self.angle,
                self.velocity, self.angular_velocity,
                (0.,0.), 0,
        ),)
        return self.history

    def to_history(self, timestamp) -> HistoryEntry:
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

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.population = 0.

class Station(SectorEntity):

    id_prefix = "STA"
    object_type = ObjectType.STATION

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.resource: Optional[int] = None

class Ship(SectorEntity):

    id_prefix = "SHP"
    object_type = ObjectType.SHIP

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


        # max thrust along heading vector
        self.max_thrust = 0.
        # max thrust in any direction
        self.max_fine_thrust = 0.
        # max torque for turning
        self.max_torque = 0.

        self.orders: Deque[Order] = collections.deque()
        self.default_order_fn: Callable[[Ship, Gamestate], Order] = lambda ship, gamestate: Order(ship, gamestate)

        self.collision_threat: Optional[SectorEntity] = None

    def get_history(self) -> Iterable[HistoryEntry]:
        return self.history

    def to_history(self, timestamp) -> HistoryEntry:
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
        return self.max_thrust / self.mass * 5

    def max_acceleration(self) -> float:
        return self.max_thrust / self.mass

    def max_fine_acceleration(self) -> float:
        return self.max_fine_thrust / self.mass

    def max_angular_acceleration(self) -> float:
        return self.max_torque / self.moment

    def default_order(self, gamestate: Gamestate) -> Order:
        return self.default_order_fn(self, gamestate)

class Asteroid(SectorEntity):

    id_prefix = "AST"
    object_type = ObjectType.ASTEROID

    def __init__(self, resource, amount, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resource = resource
        self.amount = amount

    def __str__(self):
        return f'{self.short_id()} at {self.loc} r:{self.resource} a:{self.amount}'

class Character(Entity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def choose_action(self, game_state):
        pass

class OrderLoggerAdapter(logging.LoggerAdapter):
    def __init__(self, ship, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ship = ship

    def process(self, msg, kwargs):
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
        self.started_at = gamestate.timestamp
        self.init_eta = np.inf

    def __str__(self) -> str:
        return f'{self.__class__} for {self.ship}'

    def to_history(self) -> dict:
        return {"o": self.o_name}

    def is_complete(self) -> bool:
        return False

    def act(self, dt:float) -> None:
        """ Performs one immediate tick's of action for this order """
        pass

class Gamestate:
    def __init__(self):

        self.random = None

        # the production chain of resources (ingredients
        self.production_chain = None

        # the universe is a set of sectors, indexed by coordinate
        self.sectors = {}
        self.entities = {}

        self.characters = []

        self.keep_running = True

        self.base_date = datetime.datetime(2234, 4, 3)
        self.timestamp = 0

        self.dt = 1/60
        self.ticks = 0
        self.ticktime = 0
        self.timeout = 0
        self.missed_ticks = 0

        self.keep_running = True
        self.paused = False
        self.should_raise= False

    def current_time(self):
        #TODO: probably want to decouple telling time from ticks processed
        # we want missed ticks to slow time, but if we skip time will we
        # increment the ticks even though we don't process them?
        return datetime.datetime.fromtimestamp(self.base_date.timestamp() + self.timestamp)

    def quit(self):
        self.keep_running = False

def write_history_to_file(entity, f, mode="w"):
    if isinstance(f, str):
        needs_close = True
        if f.endswith(".gz"):
            f = gzip.open(f, mode+"t")
        else:
            f = open(f, mode)
    else:
        needs_close = False

    if isinstance(entity, Ship):
        entities = [entity]
    elif isinstance(entity, Sector):
        entities = entity.entities.values()

    for ent in entities:
        for entry in ent.get_history():
            f.write(json.dumps(entry.to_json()))
            f.write("\n")
    if needs_close:
        f.close()
