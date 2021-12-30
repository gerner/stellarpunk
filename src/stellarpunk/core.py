""" Core stuff for Stellarpunk """

import uuid

import graphviz

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
    def __init__(self, name, entity_id=None):
        self.entity_id = entity_id or str(uuid.uuid4())
        self.name = name

class Sector(Entity):
    """ A region of space containing resources, stations, ships. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.planets = []
        self.stations = []
        self.ships = []
        self.resources = []

class Planet(Entity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.population = 0

class Station(Entity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resource = None

class Ship(Entity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

