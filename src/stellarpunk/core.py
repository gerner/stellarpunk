""" Core stuff for Stellarpunk """

import uuid
import datetime
import enum
import logging

import graphviz
import rtree
import numpy as np

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
        self.resources = []

        # id -> entity for all entities in the sector
        self.entities = {}

        # spatial index of entities in the sector
        self.spatial = rtree.index.Index()

        # physics space for this sector (we don't manage this, just have a pointer to it)
        self.space = None

    def add_entity(self, entity):
        #TODO: worry about collisions at location?

        if isinstance(entity, Planet):
            self.planets.append(entity)
        elif isinstance(entity, Station):
            self.stations.append(entity)
        elif isinstance(entity, Ship):
            self.ships.append(entity)
        else:
            raise ValueError("unknown entity type {entity.__class__}")

        entity.sector = self
        self.entities[entity.entity_id] = entity
        #TODO: entity bounding box?
        self.spatial.insert(entity.short_id_int(), (entity.x, entity.y, entity.x, entity.y), obj=entity.entity_id)

    def reindex_locations(self):
        # only need to do work if we have any entities
        if self.entities:
            self.spatial = rtree.index.Index(
                    (entity.short_id_int(), (entity.x, entity.y, entity.x, entity.y), entity.entity_id) for entity in self.entities.values()
            )

class ObjectType(enum.IntEnum):
    OTHER = enum.auto()
    SHIP = enum.auto()
    STATION = enum.auto()
    PLANET = enum.auto()

class SectorEntity(Entity):
    """ An entity in space in a sector. """

    object_type = ObjectType.OTHER

    def __init__(self, x, y, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sector = None

        # some physical properties
        self.mass = 0
        self.moment = 0
        self.x = x
        self.y = y
        self.velocity = (0,0)
        self.angle = 0
        self.angular_velocity = 0

        # physics simulation entity (we don't manage this, just have a pointer to it)
        self.phys = None
        #TODO: are all entities just circles?
        self.radius = 0

    def __str__(self):
        return f'{self.short_id()} at {(self.x, self.y)} v:{self.velocity} theta:{self.angle:.1f} w:{self.angular_velocity:.1f}'

class Planet(SectorEntity):

    id_prefix = "PLT"
    object_type = ObjectType.PLANET

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.population = 0

class Station(SectorEntity):

    id_prefix = "STA"
    object_type = ObjectType.STATION

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resource = None

class Ship(SectorEntity):

    id_prefix = "SHP"
    objec_type = ObjectType.SHIP

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # max thrust along heading vector
        self.max_thrust = 0
        # max thrust in any direction
        self.max_fine_thrust = 0
        # max torque for turning
        self.max_torque = 0

        self.order = None
        self.default_order_fn = lambda x: None

        self.collision_threat = None

    def max_speed(self):
        return self.max_thrust / self.mass * 5

    def max_acceleration(self):
        return self.max_thrust / self.mass

    def max_angular_acceleration(self):
        return self.max_torque / self.moment

    def default_order(self):
        return self.default_order_fn(self)


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
    def __init__(self, ship, gamestate):
        self.gamestate = gamestate
        self.ship = ship
        self.eta = 0
        self.logger = OrderLoggerAdapter(
                ship,
                logging.getLogger(util.fullname(self)),
        )

    def __str__(self):
        return f'{self.__class__} for {self.ship.short_id()}@{self.ship.sector.short_id()}'

    def is_complete(self):
        return True

    def act(self, dt):
        """ Performs one immediate tick's of action for this order """
        pass

class StellarPunk:
    def __init__(self):

        self.random = None

        # the production chain of resources (ingredients
        self.production_chain = None

        # the universe is a set of sectors, indexed by coordinate
        self.sectors = {}

        self.characters = []

        self.keep_running = True

        self.base_date = datetime.datetime(2234, 4, 3)

        self.dt = 1/60
        self.ticks = 0
        self.ticktime = 0
        self.timeout = 0
        self.missed_ticks = 0

        self.paused = False

    def current_time(self):
        #TODO: probably want to decouple telling time from ticks processed
        # we want missed ticks to slow time, but if we skip time will we
        # increment the ticks even though we don't process them?
        return datetime.datetime.fromtimestamp(self.base_date.timestamp() + self.ticks*self.dt)

    def tick(self):
        # iterate through characters
        # set up choices/actions (and make instantaneous changes to state)
        # execute actions, run simulation, etc.
        # random events

        for character in self.characters:
            character.choose_action(self)

    def run(self):
        while self.keep_running:
            self.tick()
