""" Core stuff for Stellarpunk """

import logging
import uuid
import math

import graphviz
import rtree
import numpy as np
import pymunk.vec2d

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
                import numpy as np
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
        """ Least significant 32 bits as hex """
        return f'{self.id_prefix}-{self.entity_id.hex[-8:]}'

    def short_id_int(self):
        return self.entity_id.int & (1<<32)-1

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

class SectorEntity(Entity):
    """ An entity in space in a sector. """

    def __init__(self, x, y, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = x
        self.y = y
        self.velocity = (0,0)
        self.angle = 0

        # physics simulation entity (we don't manage this, just have a pointer to it)
        self.phys = None

    #TODO: do we just want all these position things to be properties?
    @property
    def angular_velocity(self):
        if not self.phys:
            return 0
        else:
            return self.phys.angular_velocity

class Planet(SectorEntity):

    id_prefix = "PLT"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.population = 0

class Station(SectorEntity):

    id_prefix = "STA"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resource = None

class Ship(SectorEntity):

    id_prefix = "SHP"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.order = None

class Character(Entity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def choose_action(self, game_state):
        pass

class Action:
    def __init__(self):
        pass

    def act(self, game_state):
        pass

class Order:
    def __init__(self, ship):
        self.ship = ship

    def is_complete(self):
        return True

    def act(self, dt):
        """ Performs one immediate tick's of action for this order """
        pass

class RotateOrder(Order):
    def __init__(self, target_angle, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_angle = util.normalize_angle(target_angle)

    def is_complete(self):
        return self.ship.phys.angular_velocity == 0 and util.normalize_angle(self.ship.phys.angle) == self.target_angle

    def act(self, dt):
        # given current angle and angular_velocity and max torque, choose
        # torque to apply for dt now to hit target angle


        angle = util.normalize_angle(self.ship.phys.angle)
        max_torque = 9000
        w = self.ship.phys.angular_velocity
        moment = self.ship.phys.moment

        t = util.torque_for_angle(self.target_angle, angle, w, max_torque, moment, dt)

        if t == 0:
            self.ship.phys.angle = self.target_angle
            self.ship.phys.angular_velocity = 0
        else:
            self.ship.phys.torque = t
        return

class KillVelocityOrder(Order):
    """ Applies thrust and torque to zero out velocity and angular velocity.

    Rotates to opposite direction of current velocity and applies thrust to
    zero out velocity. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_velocity_magnitude = math.inf
        self.expected_next_velocity = self.ship.phys.velocity

    def is_complete(self):
        return self.ship.phys.angular_velocity == 0 and self.ship.phys.velocity == (0,0)

    def act(self, dt):
        velocity_eps = 1e-2
        angle_eps = 1e-3
        max_thrust = 0.5 * 1e6
        max_torque = 90000
        mass = self.ship.phys.mass
        moment = self.ship.phys.moment
        angle = self.ship.angle
        w = self.ship.phys.angular_velocity
        v = self.ship.phys.velocity

        # orient toward the opposite of the direction of travel
        # thrust until zero velocity
        velocity_magnitude, velocity_angle = util.cartesian_to_polar(*v)

        reverse_velocity_angle = util.normalize_angle(velocity_angle + math.pi)
        assert velocity_magnitude <= self.last_velocity_magnitude
        assert abs(v[0] - self.expected_next_velocity[0]) < velocity_eps
        assert abs(v[1] - self.expected_next_velocity[1]) < velocity_eps
        self.last_velocity_magnitude = velocity_magnitude

        if velocity_magnitude < velocity_eps:
            self.ship.phys.velocity = (0,0)
            return

        if abs(angle - reverse_velocity_angle) > angle_eps or abs(w) > angle_eps:
            # first aim ship opposity velocity
            t = util.torque_for_angle(reverse_velocity_angle, angle, w, max_torque, moment, dt)
            if t == 0:
                self.ship.phys.angle = reverse_velocity_angle
                self.ship.phys.angular_velocity = 0
            else:
                self.ship.phys.torque = t
        else:
            x,y = util.force_for_zero_velocity(self.ship.phys.velocity, max_thrust, mass, dt)
            if (x,y) == (0,0):
                self.ship.phys.velocity = (0,0)
            else:
                self.expected_next_velocity = (v[0] + x/mass*dt, v[1] + y/mass*dt)
                self.ship.phys.apply_force_at_world_point(
                        (x,y),
                        self.ship.phys.position+self.ship.phys.center_of_gravity)

class KillRotationOrder(Order):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def is_complete(self):
        return self.ship.phys.angular_velocity == 0

    def act(self, dt):
        # apply torque up to max torque to kill angular velocity
        # torque = moment * angular_acceleration
        # the perfect acceleration would be -1 * angular_velocity / timestep
        # implies torque = moment * -1 * angular_velocity / timestep
        t = self.ship.phys.moment * -1 * self.ship.phys.angular_velocity / dt
        self.ship.phys.torque = np.clip(t, -9000, 9000)
        #TODO: do we need a hack for very low angular velocities?

class StellarPunk:
    def __init__(self):

        self.random = None

        # the production chain of resources (ingredients
        self.production_chain = None

        # the universe is a set of sectors, indexed by coordinate
        self.sectors = {}

        self.characters = []
        self.actions = []

        self.keep_running = True

        self.ticks = 0
        self.ticktime = 0
        self.timeout = 0

    def tick(self):
        # iterate through characters
        # set up choices/actions (and make instantaneous changes to state)
        # execute actions, run simulation, etc.
        # random events

        for character in self.characters:
            character.choose_action(self)

        for action in self.actions:
            action.act(self)

    def run(self):
        while self.keep_running:
            self.tick()

