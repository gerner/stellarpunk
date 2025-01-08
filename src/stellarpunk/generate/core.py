import logging
import itertools
import uuid
import math
import importlib.resources
import itertools
import enum
import collections
import heapq
import time
import gzip
import os
import weakref
import hashlib
import abc
from typing import Optional, List, Dict, MutableMapping, Mapping, Tuple, Sequence, Union, overload, Any, Collection, Type

import numpy as np
import numpy.typing as npt
import cymunk # type: ignore
import rtree.index # type: ignore
import graphviz # type: ignore

from stellarpunk import util, core, orders, agenda, econ, config, events, sensors, intel
from stellarpunk.core import combat, sector_entity
from . import markov

RESOURCE_REL_SHIP = 0
RESOURCE_REL_STATION = 1
RESOURCE_REL_CONSUMER = 2

def dijkstra(adj:npt.NDArray[np.float64], start:int, target:int) -> Tuple[Mapping[int, int], Mapping[int, float]]:
    """ given adjacency weight matrix, start index, end index, compute
    distances from start to every node up to end.

    returns tuple:
        path encoded as node -> parent node mapping
        distances node -> shortest distance to start
    """
    # inspired by: https://towardsdatascience.com/a-self-learners-guide-to-shortest-path-algorithms-with-implementations-in-python-a084f60f43dc
    d = {start: 0}
    parent = {start: start}
    pq = [(0, start)]
    visited = set()
    while pq:
        du, u = heapq.heappop(pq)
        if u in visited: continue
        if u == target:
            break
        visited.add(u)
        for v, weight in enumerate(adj[u]):
            if not weight < math.inf:
                # inf weight means no edge
                continue
            if v not in d or d[v] > du + weight:
                d[v] = du + weight
                parent[v] = u
                heapq.heappush(pq, (d[v], v))


    return parent, d

def prims_mst(distances:npt.NDArray[np.float64], root_idx:int) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    # prim's algorithm to construct a minimum spanning tree
    # https://en.wikipedia.org/wiki/Prim%27s_algorithm
    # choose starting vertex arbitrarily
    V = np.zeros(len(distances), bool)
    E = np.zeros((len(distances), len(distances)))
    edge_distances = np.full((len(distances), len(distances)), math.inf)
    # while some nodes not connected
    # invariant(s):
    # V is a mask indicating elements in the tree
    # E is adjacency matrix representing the tree
    # distances has distances to nodes in the tree
    #   with inf distance between nodes already in the tree and self edges
    V[root_idx] = True
    while not np.all(V):
        # choose edge from nodes in tree to node not yet in tree with min dist
        d = np.copy(distances)
        # don't choose edges from outside the tree
        d[~V,:] = np.inf
        # don't choose edges into the tree
        d[:,V] = np.inf
        edge = np.unravel_index(np.argmin(d, axis=None), d.shape)
        E[edge] = 1.
        E[edge[1], edge[0]] = 1.
        V[edge[1]] = True
        edge_distances[edge] = distances[edge]
        edge_distances[edge[1], edge[0]] = distances[edge]
    return E, edge_distances

def generate_starfield_layer(random:np.random.Generator, radius:float, num_stars:int, zoom:float, mu:float, sigma:float) -> core.StarfieldLayer:

    if num_stars > 5e6:
        raise ValueError(f'too many {num_stars=}')

    bbox = (
        -radius, -radius,
        radius, radius
    )
    starfield_layer = core.StarfieldLayer(bbox, zoom)

    # stars have location, size
    # location is uniform random in bbox
    # size between 0,1, closer to 0

    x = random.uniform(bbox[0], bbox[2], size=num_stars)
    y = random.uniform(bbox[1], bbox[3], size=num_stars)
    sizes = util.peaked_bounded_random(random, mu, sigma, size=num_stars)
    spectral_classes = random.integers(7, size=num_stars)
    for i in range(num_stars):
        starfield_layer.add_star((x[i], y[i]), sizes[i], spectral_classes[i]) # type: ignore

    return starfield_layer

def generate_starfield(random:np.random.Generator, radius:float, desired_stars_per_char:float, min_zoom:float, max_zoom:float, layer_zoom_step:float, mu:float=0.3, sigma:float=0.15) -> list[core.StarfieldLayer]:
    """ Generates a sequence of starfield layers according to parameters.

    random: npumpy.random.Generator to use for generation of the field
    radius: float radius to generate stars over
    desired_stars_per_character: float the desired star density to display in
        stars per character. E.g. 4/80 would be ~4 stars in a 80 character
        line.
    min_zoom: float minimum (most zoomed out) zoom level in meters per char
    max_zoom: float max (most zoomed in) zoom level in meters per char
    layer_zoom_step: desired step factor between layers (between 0 and 1)

    returns: sequence of core.StarfieldLayer, sorted by density
    """

    if min_zoom < max_zoom:
        raise ValueError(f'{min_zoom=} should be numerically greater than {max_zoom=} (min refers to most zoomed out, or the largest number of meters per character')
    if layer_zoom_step >= 1. or layer_zoom_step <= 0.:
        raise ValueError(f'{layer_zoom_step=} must be strictly between 0 and 1')

    starfield:list[core.StarfieldLayer] = []

    for zoom in [min_zoom, min_zoom*layer_zoom_step]:
        # generate layers of constant density in stars per character
        density = desired_stars_per_char / (zoom**2)
        num_stars = int(density * (2*radius)**2)
        starfield.append(generate_starfield_layer(random, radius, num_stars, zoom, mu, sigma))


    return starfield

def is_occupied(loc:npt.NDArray[np.float64], prev_locs:Sequence[npt.NDArray[np.float64]], occupied_radius:float) -> bool:
    for prev_loc in prev_locs:
        if util.distance(loc, prev_loc) < occupied_radius:
            return True
    return False

class ContextKeys(enum.IntEnum):
    ETYPE_ASTEROID = enum.auto()
    ETYPE_STATION = enum.auto()
    ETYPE_PLANET = enum.auto()
    ETYPE_SHIP = enum.auto()
    ETYPE_MISSILE = enum.auto()

class GenerationErrorCase(enum.Enum):
    DISTINCT_INPUTS = enum.auto()
    INPUT_CONSTRAINTS = enum.auto()
    DISTINCT_INUTS = enum.auto()
    ONE_TO_ONE = enum.auto()
    SINGLE_INPUT = enum.auto()
    SINGLE_OUTPUT = enum.auto()
    NO_OUTPUTS = enum.auto()
    NO_CHAIN = enum.auto()
    ABORT = enum.auto()

class GenerationError(Exception):
    def __init__(self, case:GenerationErrorCase, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.case = case

def order_fn_null( gamestate:core.Gamestate) -> core.Ship.DefaultOrderSig:
    def null_order(ship:core.Ship, gamestate:core.Gamestate=gamestate) -> core.Order:
        return core.NullOrder.create_order(ship, gamestate)
    return null_order

def order_fn_wait(gamestate:core.Gamestate) -> core.Ship.DefaultOrderSig:
    def wait_order(ship:core.Ship, gamestate:core.Gamestate=gamestate) -> core.Order:
        return orders.WaitOrder.create_wait_order(ship, gamestate)
    return wait_order

def order_fn_goto_random_station(gamestate:core.Gamestate) -> core.Ship.DefaultOrderSig:
    def goto_random_station(ship:core.Ship, gamestate:core.Gamestate=gamestate) -> core.Order:
        if ship.sector is None:
            raise Exception("cannot go to location if ship isn't in a sector")
        station = gamestate.random.choice(np.array(ship.sector.entities_by_type(sector_entity.Station)))
        return orders.GoToLocation.goto_entity(station, ship, gamestate)
    return goto_random_station

def order_fn_disembark_to_random_station(gamestate:core.Gamestate) -> core.Ship.DefaultOrderSig:
    def disembark_to_random_station(ship:core.Ship, gamestate:core.Gamestate=gamestate) -> core.Order:
        if ship.sector is None:
            raise Exception("cannot disembark to if ship isn't in a sector")
        station = gamestate.random.choice(np.array(ship.sector.entities_by_type(sector_entity.Station)))
        return orders.DisembarkToEntity.disembark_to(station, ship, gamestate)
    return disembark_to_random_station

class GenerationStep(enum.Enum):
    NONE = enum.auto()
    PRODUCTION_CHAIN = enum.auto()
    STARFIELDS = enum.auto()
    CULTURES = enum.auto()
    INHABITED_SECTORS = enum.auto()
    UNINHABITED_SECTORS = enum.auto()
    PLAYER = enum.auto()

class UniverseGeneratorObserver(abc.ABC):
    def estimated_generation_ticks(self, ticks:int) -> None:
        """ provides an estimate of the number of ticks during generation. """
        pass

    def generation_tick(self) -> None:
        """ indicates progress during generation. """
        pass

    def generation_step(self, step:GenerationStep) -> None:
        """ indicates a generation step has begun. """
        pass

    def player_spawned(self, player:core.Player) -> None:
        pass

    def universe_generated(self, gamestate:core.Gamestate) -> None:
        pass

    def universe_loaded(self, gamestate:core.Gamestate) -> None:
        pass

class UniverseConfig:
    def __init__(self) -> None:

        # names and cultures
        self.empty_name_model_culture = ""
        self.name_model_location = config.Settings.generate.names.NAME_MODEL_LOCATION
        self.num_cultures = config.Settings.generate.Universe.NUM_CULTURES
        self.cultures = config.Settings.generate.Universe.CULTURES
        self.max_sector_name_words = config.Settings.generate.names.MAX_SECTOR_NAME_WORDS
        self.max_station_name_words = config.Settings.generate.names.MAX_STATION_NAME_WORDS
        self.max_ship_name_words = config.Settings.generate.names.MAX_SHIP_NAME_WORDS

        # production chain stuff
        self.ore_names = config.Settings.generate.ProductionChain.ORE_NAMES
        self.intermediate_names = config.Settings.generate.ProductionChain.INTERMEDIATE_NAMES
        self.hightech_names = config.Settings.generate.ProductionChain.HIGHTECH_NAMES
        self.sink_names = config.Settings.generate.ProductionChain.SINK_NAMES

        self.sink_inputs = config.Settings.generate.ProductionChain.SINK_INPUTS
        self.hightech_inputs = config.Settings.generate.ProductionChain.HIGHTECH_INPUTS
        self.intermediate_inputs = config.Settings.generate.ProductionChain.INTERMEDIATE_INPUTS

        # universe setup
        self.universe_radius = config.Settings.generate.Universe.UNIVERSE_RADIUS
        self.sector_radius_mean = config.Settings.generate.Universe.SECTOR_RADIUS_MEAN
        self.sector_radius_std = config.Settings.generate.Universe.SECTOR_RADIUS_STD
        self.max_sector_edge_length = config.Settings.generate.Universe.MAX_SECTOR_EDGE_LENGTH

        self.num_sectors = config.Settings.generate.Universe.NUM_SECTORS
        self.num_habitable_sectors = config.Settings.generate.Universe.NUM_HABITABLE_SECTORS
        self.mean_habitable_resources = config.Settings.generate.Universe.MEAN_HABITABLE_RESOURCES
        self.mean_uninhabitable_resources = config.Settings.generate.Universe.MEAN_UNINHABITABLE_RESOURCES


        self.station_factor = config.Settings.generate.Universe.STATION_FACTOR
        self.mining_ship_factor = config.Settings.generate.Universe.MINING_SHIP_FACTOR
        self.trading_ship_factor = config.Settings.generate.Universe.TRADING_SHIP_FACTOR


class UniverseGenerator(core.AbstractGenerator):
    @staticmethod
    def viz_product_name_graph(names:List[List[str]], edges:List[List[List[int]]]) -> graphviz.Graph:
        g = graphviz.Digraph("product_name_graph", graph_attr={"rankdir": "TB"})
        g.attr(compound="true", ranksep="1.5")


        for rank in range(len(names)-1, -1, -1):
            for node in range(len(edges[rank])):
                g.node(f'{rank}_{node}', label=f'{names[rank][node]} ({rank}_{node})')
                for e in edges[rank][node]:
                    g.edge(f'{rank-1}_{e}', f'{rank}_{node}')

        return g

    def __init__(self, seed:Optional[int]=None) -> None:
        self.logger = logging.getLogger(util.fullname(self))

        self.event_manager:events.EventManager = None # type: ignore
        self.gamestate:Optional[core.Gamestate] = None

        #TODO: should we just use the one on gamestate? They should always be
        # the same thing anyway
        # random generator
        self.r = np.random.default_rng(seed)

        self.production_chain_max_tries = config.Settings.generate.ProductionChain.max_tries

        self.parallel_max_edges_tries = 10000

        # keep track of all sprites by global id (except ephemeral ones)
        self.sprite_store:dict[str, core.Sprite] = {}
        self.portraits:List[core.Sprite] = []
        self.station_sprites:List[core.Sprite] = []

        self.num_projectile_spawn_locs = 100
        self.projectile_spawn_pattern:List[npt.NDArray[np.float64]] = []

        self._sector_name_models:MutableMapping[str, markov.MarkovModel] = {}
        self._station_name_models:MutableMapping[str, markov.MarkovModel] = {}
        self._first_name_models:MutableMapping[str, markov.MarkovModel] = {}
        self._last_name_models:MutableMapping[str, markov.MarkovModel] = {}
        self._ship_name_model:markov.MarkovModel = markov.MarkovModel(roman_numerals=True)

        self.universe_config = UniverseConfig()
        self._culture_map:Mapping[uuid.UUID, str]
        self._empty_name_model_culture = "_ignore_culture_"

        self._observers:weakref.WeakSet[UniverseGeneratorObserver] = weakref.WeakSet()
        self._production_chain_ticks:set[int] = set()

    def observe(self, observer:UniverseGeneratorObserver) -> None:
        self._observers.add(observer)

    def unobserve(self, observer:UniverseGeneratorObserver) -> None:
        try:
            self._observers.remove(observer)
        except KeyError:
            pass

    def _random_bipartite_graph(
            self,
            n:int, m:int, k:int,
            max_out:int, max_in:int,
            total_w:float, min_w:float, max_w:float,
            min_out:int=1, min_in:int=1) -> npt.NDArray[np.float64]:
        """ Creates a bipartite, weighted graph according to model parameters.

        n: number of top nodes
        m: number of bottom nodes
        k: number of edges
        max_out: max out degree on top nodes
        max_in: max in degree on bottom nodes
        total_w: sum of weights on edges from top to bottom
        min_w: min weight on edges
        max_w: max weight on edges

        returns an ndarray edge weight matrix (from, to)
        """

        # check that a random bipartite graph is possible
        if k < n or k < m or k > n*m:
            raise ValueError("k must be >= max(n,m) and <= n*m")
        if k < n*min_out or k > n*max_out:
            raise ValueError("k must be >= n*min_in and <= n*max_out")
        if k < m*min_in or k > m*max_in:
            raise ValueError("k must be >= m*min_in and <= m*max_in")
        if max_out > m or max_in > n:
            raise ValueError("max_out must be less then m and max_in must be less than n")

        if total_w < k*min_w or total_w > k*max_w:
            raise ValueError("total_w must be >= k*min_w and <= k*max_w")

        def choose_seq(n:int, k:int, min_deg:int, max_deg:int) -> npt.NDArray[np.int64]:
            """ Generates a sequence of integers from [0,n) of length k where
            each number occurs at least min_deg and at most max_deg """

            nseq = np.zeros(n, dtype=int) + min_deg

            k_left = k-nseq.sum()
            while k_left > 0:
                v = self.r.choice(np.where(nseq < max_deg)[0])
                nseq[v] += 1
                k_left -= 1

            return nseq

        # prepare edge assignments between top and bottom
        tries = 0
        has_duplicate = True
        nstubs:List[int] = []
        mstubs:List[int] = []
        while has_duplicate:
            if tries > self.parallel_max_edges_tries:
                raise Exception(f'failed to generate a bipartite graph with no parallel edges after {tries} attempts')
            nseq = choose_seq(n, k, min_out, max_out)
            mseq = choose_seq(m, k, min_in, max_in)

            stubs = [[v] * nseq[v] for v in range(n)]
            nstubs = [x for subseq in stubs for x in subseq]
            stubs = [[v] * mseq[v] for v in range(m)]
            mstubs = [x for subseq in stubs for x in subseq]

            assert len(nstubs) == len(mstubs)
            assert len(nstubs) == k

            # shuffle nstubs and mstubs to get source/target pairs
            #   we repeatedly shuffle mstubs to avoid parallel (duplicate) edges
            self.r.shuffle(nstubs)
            self.r.shuffle(mstubs)

            # are there any duplicate edges?
            has_duplicate = len(set(zip(nstubs, mstubs))) < len(mstubs)
            tries += 1

        # prepare weights for those edges
        w = np.ones(k)*min_w
        w_left = total_w - k*min_w

        w_unormalized = self.r.uniform(0, 1, k)
        w += w_unormalized / w_unormalized.sum() * w_left

        assert np.isclose(w.sum(), total_w)

        # fill in the adjacency matrix
        adj_matrix = np.zeros((n, m))

        for (s,t, edge_w) in zip(nstubs, mstubs, w):
            adj_matrix[s,t] = edge_w

        return adj_matrix

    def gen_sector_location(self, sector:core.Sector, occupied_radius:float=2e3, center:Union[Tuple[float, float],npt.NDArray[np.float64]]=(0.,0.), radius:Optional[float]=None, strict:bool=False, min_dist:float=0.)->npt.NDArray[np.float64]:
        if radius is None:
            radius = sector.radius
        if min_dist >= radius:
            raise ValueError(f'{min_dist=} must be < {radius=}')
        center = np.array(center)

        if strict:
            r:npt.NDArray[np.float64] = util.peaked_bounded_random(self.r, 0.5, 0.15, 2) * 2. - 1. # type: ignore
        else:
            r = self.r.normal(0, 1, 2)
        loc = r * radius + center
        while (occupied_radius >= 0. and sector.is_occupied(loc[0], loc[1], eps=occupied_radius)) or util.distance(loc, center) < min_dist:
            if strict:
                r = util.peaked_bounded_random(self.r, 0.5, 0.15, 2) * 2. - 1. # type:ignore
            else:
                r = self.r.normal(0, 1, 2)
            loc = r * radius + center

        return loc

    def gen_projectile_location(self, center:Union[Tuple[float, float],npt.NDArray[np.float64]]=(0.,0.), index:Optional[int]=None) -> Tuple[npt.NDArray[np.float64], int]:
        if index is None:
            index = self.r.integers(self.num_projectile_spawn_locs-1)
        return self.projectile_spawn_pattern[index%self.num_projectile_spawn_locs] + center, (index+1) % self.num_projectile_spawn_locs

    def _gen_sector_culture(self, x:float, y:float, sector_id:uuid.UUID) -> str:
        #TODO: this means we can't generate sectors on the fly!
        return self._culture_map[sector_id]

    def _gen_sector_name(self, culture:str) -> str:
        name = ""
        tries = 0
        max_tries = 10
        while (not name or len(name.split()) > self.universe_config.max_sector_name_words) and tries < max_tries:
            name = self._sector_name_models[culture].generate(self.r)
            assert(culture == self._empty_name_model_culture or name != "")
            tries+=1
        return name

    def _gen_planet_name(self, culture:str) -> str:
        name = ""
        tries = 0
        max_tries = 10
        while (not name or len(name.split()) > self.universe_config.max_station_name_words) and tries < max_tries:
            name = self._station_name_models[culture].generate(self.r)
            assert(culture == self._empty_name_model_culture or name != "")
            tries+=1
        return name

    def _gen_station_name(self, culture:str) -> str:
        name = ""
        tries = 0
        max_tries = 10
        while (not name or len(name.split()) > self.universe_config.max_station_name_words) and tries < max_tries:
            name = self._station_name_models[culture].generate(self.r)
            assert(culture == self._empty_name_model_culture or name != "")
            tries+=1
        return name

    def _gen_ship_name(self, culture:str) -> str:
        name = ""
        tries = 0
        max_tries = 10
        while (not name or len(name.split()) > self.universe_config.max_ship_name_words) and tries < max_tries:
            name = self._ship_name_model.generate(self.r)
            assert(culture == self._empty_name_model_culture or name != "")
            tries+=1
        return name

    def _gen_character_name(self, culture:str) -> str:
        first = self._first_name_models[culture].generate(self.r)
        last = self._last_name_models[culture].generate(self.r)
        return f'{first} {last}'

    def _gen_asteroid_name(self, culture:str) -> str:
        #TODO: sector name?
        return self._sector_name_models[culture].generate(self.r)

    def _gen_gate_name(self, destination:core.Sector, culture:str) -> str:
        return f"Gate to {destination.short_id()}"

    def _assign_names(self, adj_matrix:npt.NDArray[np.float64], allowed_options:List[List[int]]) -> List[int]:
        """ Assigns names from in_options following constraints.

        in_options list of strings to draw from
        adj_matrix pairing of inputs to outputs
        allowed_options allowed options for inputs for each output
        returns indexes into in_options of chosen options
        """

        assert adj_matrix.shape[1] == len(allowed_options)

        # initialize each input with all options
        input_options = np.ones((adj_matrix.shape[0], max(max(x) for x in allowed_options)+1), dtype=int)

        # for each output
        for j in range(adj_matrix.shape[1]):
            # zero out illegal options for each input
            for i in range(adj_matrix.shape[0]):
                if adj_matrix[i,j] > 0:
                    mask = np.ones(input_options.shape[1], dtype=bool)
                    mask[allowed_options[j]] = 0
                    input_options[i][mask] = 0

        # now we have for each input the intersection of the allowed inputs

        # if the intersection of any is empty, this is unsatisfiable
        if np.any(np.sum(input_options, axis=1) == 0):
            raise GenerationError(GenerationErrorCase.INPUT_CONSTRAINTS, "unsatisfiable constraints")

        # now we can choose inputs
        satisified = False
        tries = 0
        max_tries = 32
        while not satisified and tries < 32:
            # choose from most constrained to least
            choice_order = input_options.sum(axis=1).argsort()
            running_input_options = input_options.copy()
            chosen_inputs = np.full((len(input_options), ), -1, dtype=int)
            for i in choice_order:
                if running_input_options[i].sum() == 0:
                    break

                valid_choices = np.where(running_input_options[i] > 0)[0]
                p = 1. / running_input_options[:,valid_choices].sum(axis=0)

                choice = self.r.choice(valid_choices, p=p/np.sum(p))
                chosen_inputs[i] = choice
                # don't pick the same option twice!
                running_input_options[:,choice] = 0
            satisified = not np.any(chosen_inputs == -1)
            tries += 1

        if not satisified:
            raise GenerationError(GenerationErrorCase.DISTINCT_INPUTS, "cannot assign distinct inputs")

        assert len(np.unique(chosen_inputs)) == len(chosen_inputs)
        return list(chosen_inputs)

    def _generate_product_names(self, ranks:npt.NDArray[np.int64], adj_matrix:npt.NDArray[np.float64]) -> List[str]:

        assert 3 <= len(ranks) <= 5

        assert ranks[0] <= len(self.universe_config.ore_names)
        assert ranks[1] == ranks[0], "rank 0 (ores) must have same size as rank 1 (refined ores)"
        if len(ranks) == 5:
            assert ranks[2] <= len(self.universe_config.intermediate_names)

        if len(ranks) >= 4:
            assert ranks[-2] <= len(self.universe_config.hightech_names)
        assert ranks[-1] == len(self.universe_config.sink_names)

        # set up product names in reverse order, respecting allowed inputs
        product_names = list(self.universe_config.sink_names)

        if len(ranks) == 3:
            # ore names don't matter, just assign names
            ore_ids = self.r.choice(np.arange(len(self.universe_config.ore_names)), size=ranks[0], replace=False)
            product_names = [self.universe_config.ore_names[x] for x in ore_ids] + [f'Refined {self.universe_config.ore_names[x]}' for x in ore_ids] + product_names
        else:
            # high tech names matter
            hightech_ids = self._assign_names(adj_matrix[-(ranks[-2]+ranks[-1]):-ranks[-1], -ranks[-1]:], self.universe_config.sink_inputs)
            product_names = [self.universe_config.hightech_names[x] for x in hightech_ids] + product_names

            if len(ranks) == 4:
                # ore names don't matter, just assign names
                ore_ids = self.r.choice(np.arange(len(self.universe_config.ore_names)), size=ranks[0], replace=False)
                product_names = [self.universe_config.ore_names[x] for x in ore_ids] + [f'Refined {self.universe_config.ore_names[x]}' for x in ore_ids] + product_names
            else:
                assert len(ranks) == 5
                # intermediate and ore names matter
                intermediate_ids = self._assign_names(adj_matrix[sum(ranks[:2]):sum(ranks[:3]), sum(ranks[:3]):sum(ranks[:4])], [self.universe_config.hightech_inputs[x] for x in hightech_ids])
                product_names = [self.universe_config.intermediate_names[x] for x in intermediate_ids] + product_names

                ore_ids = self._assign_names(adj_matrix[ranks[0]:sum(ranks[0:2]), sum(ranks[0:2]):sum(ranks[0:3])], [self.universe_config.intermediate_inputs[x] for x in intermediate_ids])
                product_names = [self.universe_config.ore_names[x] for x in ore_ids] + [f'Refined {self.universe_config.ore_names[x]}' for x in ore_ids] + product_names

        assert len(product_names) == sum(ranks)
        return product_names

    def _choose_portrait(self) -> core.Sprite:
        return self.portraits[self.r.integers(0, len(self.portraits))]

    def _choose_station_sprite(self) -> core.Sprite:
        return self.station_sprites[self.r.integers(0, len(self.station_sprites))]

    def phys_body(self, mass:Optional[float]=None, radius:Optional[float]=None) -> cymunk.Body:
        if mass is None:
            body = cymunk.Body()
        else:
            assert radius is not None
            moment = cymunk.moment_for_circle(mass, 0, radius)
            body = cymunk.Body(mass, moment)
        return body

    def phys_shape(self, body:cymunk.Body, entity:core.SectorEntity, radius:float) -> cymunk.Shape:
        assert(not np.isnan(entity.loc[0]))
        assert(not np.isnan(entity.loc[1]))
        shape = cymunk.Circle(body, radius)
        shape.friction=0.1
        shape.collision_type = core.SECTOR_ENTITY_COLLISION_TYPE
        shape.group = id(body)
        body.position = (entity.loc[0], entity.loc[1])
        body.data = entity
        entity.radius = radius
        entity.phys_shape = shape
        return shape

    def _prepare_sprites(self) -> None:
        self.logger.info(f'loading sprites...')
        # load character portraits
        self.portraits = core.Sprite.load_sprites(
                importlib.resources.files("stellarpunk.data").joinpath("portraits.txt").read_text(),
                (32//2, 32//4),
                "portrait",
        )
        for sprite in self.portraits:
            self.sprite_store[sprite.sprite_id] = sprite

        # load station sprites
        self.station_sprites = core.Sprite.load_sprites(
                importlib.resources.files("stellarpunk.data").joinpath("stations.txt").read_text(),
                (96//2, 96//4),
                "station",
        )
        for sprite in self.station_sprites:
            self.sprite_store[sprite.sprite_id] = sprite

    def _prepare_projectile_spawn_pattern(self) -> None:
        radius = 5
        for i in range(self.num_projectile_spawn_locs):
            loc:npt.NDArray[np.float64] = (util.peaked_bounded_random(self.r, 0.5, 0.15, 2) * 2 - 1) * radius # type: ignore
            while is_occupied(loc, self.projectile_spawn_pattern, 0.5):
                loc:npt.NDArray[np.float64] = (util.peaked_bounded_random(self.r, 0.5, 0.15, 2) * 2 - 1) * radius # type: ignore
            self.projectile_spawn_pattern.append(loc)

    def _load_name_models(self, culture_filter:Optional[List[str]]=None) -> None:
        self.logger.info(f'loading name model for ships')
        self._ship_name_model.load(os.path.join(self.universe_config.name_model_location, "shipnames.mmodel.gz"))

        self.logger.info(f'loading name models')

        loaded_cultures = 0
        for culture in culture_filter if culture_filter is not None else self.universe_config.cultures:
            if culture == self._empty_name_model_culture:
                continue
            assert(culture in self.universe_config.cultures)
            self.logger.info(f'loading name models for culture {culture}')
            self._sector_name_models[culture] = markov.MarkovModel(romanize=True).load(os.path.join(self.universe_config.name_model_location, f'sectors.{culture}.mmodel.gz'))
            self._station_name_models[culture] = markov.MarkovModel(romanize=True).load(os.path.join(self.universe_config.name_model_location, f'stations.{culture}.mmodel.gz'))
            self._first_name_models[culture] = markov.MarkovModel(romanize=True).load(os.path.join(self.universe_config.name_model_location, f'firstnames.{culture}.mmodel.gz'))
            self._last_name_models[culture] = markov.MarkovModel(romanize=True).load(os.path.join(self.universe_config.name_model_location, f'lastnames.{culture}.mmodel.gz'))
            for observer in self._observers:
                observer.generation_tick()
            loaded_cultures += 1

        for _ in range(loaded_cultures, self.universe_config.num_cultures[1]):
            for observer in self._observers:
                observer.generation_tick()

    def _load_empty_name_models(self, culture:str) -> None:
        self._ship_name_model = markov.MarkovModel(romanize=False)
        self._sector_name_models[culture] = markov.MarkovModel(romanize=False)
        self._station_name_models[culture] = markov.MarkovModel(romanize=False)
        self._first_name_models[culture] = markov.MarkovModel(romanize=False)
        self._last_name_models[culture] = markov.MarkovModel(romanize=False)


    def pre_initialize(self, event_manager:events.EventManager, empty_name_model_culture:Optional[str]=None) -> None:
        self.event_manager = event_manager
        self.event_manager.register_context_keys(ContextKeys, "generate")
        self._prepare_sprites()
        self._prepare_projectile_spawn_pattern()

        if empty_name_model_culture:
            self._empty_name_model_culture = empty_name_model_culture
            self._load_empty_name_models(empty_name_model_culture)
            self.universe_config.cultures = []
        #else: load culture models during universe generation

    def spawn_station(self, sector:core.Sector, x:float, y:float, resource:Optional[int]=None, entity_id:Optional[uuid.UUID]=None, batches_on_hand:int=0) -> sector_entity.Station:
        assert(self.gamestate)

        if resource is None:
            resource = self.r.integers(0, len(self.gamestate.production_chain.prices)-self.gamestate.production_chain.ranks[-1])

        assert resource < self.gamestate.production_chain.num_products

        station_radius = config.Settings.generate.SectorEntities.station.RADIUS

        sensor_settings = sensors.SensorSettings(max_sensor_power=config.Settings.generate.SectorEntities.station.MAX_SENSOR_POWER, sensor_intercept=config.Settings.generate.SectorEntities.station.SENSOR_INTERCEPT)
        #station_moment = pymunk.moment_for_circle(station_mass, 0, station_radius)
        station_body = self.phys_body()
        station = sector_entity.Station(
            self._choose_station_sprite(),
            np.array((x, y), dtype=np.float64),
            station_body,
            self.gamestate.production_chain.shape[0],
            sensor_settings,
            self.gamestate,
            self._gen_station_name(sector.culture),
            entity_id=entity_id,
            description="A glittering haven among the void at first glance. In reality just as dirty and run down as the habs. Moreso, in fact, since this station was slapped together out of repurposed parts and maintained with whatever cheap replacement parts the crew of unfortunates can get their hands on. Still, it's better than sleeping in your cockpit."
        )
        station.resource = resource
        station.context.set_flag(self.gamestate.event_manager.ck(ContextKeys.ETYPE_STATION), 1)

        station.cargo[resource] += min(self.gamestate.production_chain.batch_sizes[resource] * batches_on_hand, station.cargo_capacity)
        assert station.cargo.sum() <= station.cargo_capacity

        self.phys_shape(station_body, station, station_radius)
        station.mass = config.Settings.generate.SectorEntities.station.MASS

        sector.add_entity(station)

        return station

    def spawn_planet(self, sector:core.Sector, x:float, y:float, entity_id:Optional[uuid.UUID]=None) -> sector_entity.Planet:
        assert(self.gamestate)

        planet_radius = config.Settings.generate.SectorEntities.planet.RADIUS

        sensor_settings = sensors.SensorSettings(max_sensor_power=config.Settings.generate.SectorEntities.planet.MAX_SENSOR_POWER, sensor_intercept=config.Settings.generate.SectorEntities.planet.SENSOR_INTERCEPT)
        planet_body = self.phys_body()
        planet = sector_entity.Planet(
            np.array((x, y), dtype=np.float64),
            planet_body,
            self.gamestate.production_chain.shape[0],
            sensor_settings,
            self.gamestate,
            self._gen_planet_name(sector.culture),
            entity_id=entity_id
        )
        planet.population = self.r.uniform(1e10*5, 1e10*15)
        planet.context.set_flag(self.gamestate.event_manager.ck(ContextKeys.ETYPE_PLANET), 1)

        self.phys_shape(planet_body, planet, planet_radius)
        planet.mass = config.Settings.generate.SectorEntities.planet.MASS

        sector.add_entity(planet)

        return planet

    def spawn_ship(self, sector:core.Sector, ship_x:float, ship_y:float, v:Optional[npt.NDArray[np.float64]]=None, w:Optional[float]=None, theta:Optional[float]=None, default_order_fn:Optional[core.Ship.DefaultOrderSig]=None, entity_id:Optional[uuid.UUID]=None) -> core.Ship:
        assert(self.gamestate)

        if default_order_fn is None:
            default_order_fn = order_fn_null(self.gamestate)

        sensor_settings = sensors.SensorSettings(max_sensor_power=config.Settings.generate.SectorEntities.ship.MAX_SENSOR_POWER, sensor_intercept=config.Settings.generate.SectorEntities.ship.SENSOR_INTERCEPT)
        ship_mass = config.Settings.generate.SectorEntities.ship.MASS
        ship_radius = config.Settings.generate.SectorEntities.ship.RADIUS
        max_thrust = config.Settings.generate.SectorEntities.ship.MAX_THRUST
        max_fine_thrust = config.Settings.generate.SectorEntities.ship.MAX_FINE_THRUST
        max_torque = config.Settings.generate.SectorEntities.ship.MAX_TORQUE

        ship_body = self.phys_body(ship_mass, ship_radius)
        ship = core.Ship(
            np.array((ship_x, ship_y), dtype=np.float64),
            ship_body,
            self.gamestate.production_chain.shape[0],
            sensor_settings,
            self.gamestate,
            self._gen_ship_name(sector.culture),
            is_static=False,
            entity_id=entity_id
        )
        ship.context.set_flag(self.gamestate.event_manager.ck(ContextKeys.ETYPE_SHIP), 1)

        self.phys_shape(ship_body, ship, ship_radius)

        ship.mass = ship_mass
        ship.moment = ship_body.moment
        ship.radius = ship_radius
        ship.max_base_thrust = ship.max_thrust = max_thrust
        ship.max_fine_thrust = max_fine_thrust
        ship.max_torque = max_torque

        if v is None:
            v = (self.r.normal(0, 50, 2))
        ship_body.velocity = cymunk.vec2d.Vec2d(*v)
        ship_body.angle = ship_body.velocity.angle

        if theta is not None:
            ship_body.angle = theta

        if w is None:
            ship_body.angular_velocity = self.r.normal(0, 0.08)
        else:
            ship_body.angular_velocity = w

        sector.add_entity(ship)

        ship.default_order_fn = default_order_fn
        ship.clear_orders()

        return ship

    def spawn_missile(self, sector:core.Sector, ship_x:float, ship_y:float, v:Optional[npt.NDArray[np.float64]]=None, w:Optional[float]=None, theta:Optional[float]=None, default_order_fn:Optional[core.Ship.DefaultOrderSig]=None, entity_id:Optional[uuid.UUID]=None) -> combat.Missile:
        assert(self.gamestate)

        if default_order_fn is None:
            default_order_fn = order_fn_null(self.gamestate)

        sensor_settings = sensors.SensorSettings(max_sensor_power=config.Settings.generate.SectorEntities.missile.MAX_SENSOR_POWER, sensor_intercept=config.Settings.generate.SectorEntities.missile.SENSOR_INTERCEPT)
        ship_mass = config.Settings.generate.SectorEntities.missile.MASS
        ship_radius = config.Settings.generate.SectorEntities.missile.RADIUS
        max_thrust = config.Settings.generate.SectorEntities.missile.MAX_THRUST
        max_fine_thrust = config.Settings.generate.SectorEntities.missile.MAX_FINE_THRUST
        max_torque = config.Settings.generate.SectorEntities.missile.MAX_TORQUE

        ship_body = self.phys_body(ship_mass, ship_radius)
        ship = combat.Missile(
            np.array((ship_x, ship_y), dtype=np.float64),
            ship_body,
            self.gamestate.production_chain.shape[0],
            sensor_settings,
            self.gamestate,
            self._gen_ship_name(sector.culture),
            is_static=False,
            entity_id=entity_id,
        )
        ship.context.set_flag(self.gamestate.event_manager.ck(ContextKeys.ETYPE_MISSILE), 1)
        ship.default_order_fn=default_order_fn

        self.phys_shape(ship_body, ship, ship_radius)

        ship.mass = ship_mass
        ship.moment = ship_body.moment
        ship.radius = ship_radius
        ship.max_base_thrust = ship.max_thrust = max_thrust
        ship.max_fine_thrust = max_fine_thrust
        ship.max_torque = max_torque

        if v is None:
            v = (self.r.normal(0, 50, 2))
        ship_body.velocity = cymunk.vec2d.Vec2d(*v)
        ship_body.angle = ship_body.velocity.angle

        if theta is not None:
            ship_body.angle = theta

        if w is None:
            ship_body.angular_velocity = self.r.normal(0, 0.08)
        else:
            ship_body.angular_velocity = w

        sector.add_entity(ship)

        return ship

    def spawn_projectile(self, sector:core.Sector, ship_x:float, ship_y:float, v:Optional[npt.NDArray[np.float64]]=None, w:Optional[float]=None, theta:Optional[float]=None, entity_id:Optional[uuid.UUID]=None) -> sector_entity.Projectile:
        assert(self.gamestate)

        sensor_settings = sensors.SensorSettings(max_sensor_power=config.Settings.generate.SectorEntities.projectile.MAX_SENSOR_POWER, sensor_intercept=config.Settings.generate.SectorEntities.projectile.SENSOR_INTERCEPT)
        ship_mass = config.Settings.generate.SectorEntities.projectile.MASS
        ship_radius = config.Settings.generate.SectorEntities.projectile.RADIUS
        max_thrust = config.Settings.generate.SectorEntities.projectile.MAX_THRUST
        max_fine_thrust = config.Settings.generate.SectorEntities.projectile.MAX_FINE_THRUST
        max_torque = config.Settings.generate.SectorEntities.projectile.MAX_TORQUE

        ship_body = self.phys_body(ship_mass, ship_radius)
        ship = sector_entity.Projectile(
            np.array((ship_x, ship_y), dtype=np.float64),
            ship_body,
            self.gamestate.production_chain.shape[0],
            sensor_settings,
            self.gamestate,
            self._gen_ship_name(sector.culture),
            is_static=False,
            entity_id=entity_id
        )

        shape = self.phys_shape(ship_body, ship, ship_radius)
        shape.group = 1

        ship.mass = ship_mass
        ship.moment = ship_body.moment
        ship.radius = ship_radius
        #ship.max_base_thrust = ship.max_thrust = max_thrust
        #ship.max_fine_thrust = max_fine_thrust
        #ship.max_torque = max_torque

        if v is None:
            v = (self.r.normal(0, 50, 2))
        ship_body.velocity = cymunk.vec2d.Vec2d(*v)
        ship_body.angle = ship_body.velocity.angle

        if theta is not None:
            ship_body.angle = theta

        if w is None:
            ship_body.angular_velocity = self.r.normal(0, 0.08)
        else:
            ship_body.angular_velocity = w

        sector.add_entity(ship)

        return ship

    def spawn_gate(self, sector: core.Sector, destination: core.Sector, entity_id:Optional[uuid.UUID]=None) -> sector_entity.TravelGate:
        assert(self.gamestate)

        gate_radius = 50

        direction_vec = destination.loc - sector.loc
        _, direction = util.cartesian_to_polar(*direction_vec)

        # choose a location for the gate far away from the center of the sector
        # also make sure the "lane", a path in direction from gate loc to
        # destination is clear out to far away

        #TODO: is it an issue that this isn't (probably) uniform over the space?
        #TODO: make sure there's nothing blocking the lane
        min_r = sector.radius * 2
        max_r = sector.radius * 2.5
        min_theta = direction - math.radians(5)
        max_theta = direction + math.radians(5)

        r = self.r.uniform(min_r, max_r)
        theta = self.r.uniform(min_theta, max_theta)
        x,y = util.polar_to_cartesian(r, theta)

        sensor_settings = sensors.SensorSettings()
        sensor_settings.set_transponder(True)
        body = self.phys_body()
        gate = sector_entity.TravelGate(
            direction,
            np.array((x,y), dtype=np.float64),
            body,
            self.gamestate.production_chain.shape[0],
            sensor_settings,
            self.gamestate,
            self._gen_gate_name(destination, sector.culture),
            entity_id=entity_id
        )
        gate.destination = destination

        self.phys_shape(body, gate, gate_radius)

        sector.add_entity(gate)

        return gate

    def spawn_asteroid(self, sector: core.Sector, x:float, y:float, resource:int, amount:float, entity_id:Optional[uuid.UUID]=None) -> sector_entity.Asteroid:
        assert(self.gamestate)

        asteroid_radius = config.Settings.generate.SectorEntities.asteroid.RADIUS
        sensor_settings = sensors.SensorSettings(max_sensor_power=config.Settings.generate.SectorEntities.asteroid.MAX_SENSOR_POWER, sensor_intercept=config.Settings.generate.SectorEntities.asteroid.SENSOR_INTERCEPT)

        #TODO: stations are static?
        #station_moment = pymunk.moment_for_circle(station_mass, 0, station_radius)
        body = self.phys_body()
        asteroid = sector_entity.Asteroid(
            resource,
            amount,
            np.array((x,y), dtype=np.float64),
            body,
            self.gamestate.production_chain.shape[0],
            sensor_settings,
            self.gamestate,
            self._gen_asteroid_name(sector.culture),
            entity_id=entity_id
        )
        asteroid.context.set_flag(self.gamestate.event_manager.ck(ContextKeys.ETYPE_ASTEROID), 1)

        self.phys_shape(body, asteroid, asteroid_radius)
        asteroid.mass = config.Settings.generate.SectorEntities.asteroid.MASS

        sector.add_entity(asteroid)

        return asteroid

    def spawn_sector_entity(self, klass:Type, sector:core.Sector, ship_x:float, ship_y:float, v:Optional[npt.NDArray[np.float64]]=None, w:Optional[float]=None, theta:Optional[float]=None, entity_id:Optional[uuid.UUID]=None) -> core.SectorEntity:
        if klass == combat.Missile:
            return self.spawn_missile(sector, ship_x, ship_y, v, w, theta, entity_id=entity_id)
        elif klass == sector_entity.Projectile:
            return self.spawn_projectile(sector, ship_x, ship_y, v, w, theta, entity_id=entity_id)
        else:
            raise ValueError(f'do not know how to spawn {klass}')

    def spawn_resource_field(self, sector: core.Sector, x: float, y: float, resource: int, total_amount: float, radius: float=0., mean_per_asteroid: float=5e5, variance_per_asteroid: float=3e4) -> List[sector_entity.Asteroid]:
        """ Spawns a resource field centered on x,y.

        resource : the type of resource
        total_amount : the total amount of that resource in the field
        radius : radius over which asteriods can appear,
            default sector radius / 5
        mean_per_asteroid : mean of resources per asteroid, default 1e5
        stdev_per_asteroid : stdev of resources per asteroid, default 1e4
        """

        if total_amount < 0:
            return []

        field_center = np.array((x,y), dtype=np.float64)
        if radius == 0.:
            radius = sector.radius / 5

        # generate asteroids until we run out of resources
        asteroids = []
        while total_amount > 0:
            loc = self.r.normal(0, radius, 2) + field_center
            if sector.is_occupied(loc[0], loc[1], eps=config.Settings.generate.SectorEntities.asteroid.RADIUS):
                continue
            amount = self.r.normal(mean_per_asteroid, variance_per_asteroid)
            if amount > total_amount:
                amount = total_amount

            asteroid = self.spawn_asteroid(sector, loc[0], loc[1], resource, amount)

            asteroids.append(asteroid)

            total_amount -= amount

        # add in some weather related to this resource field
        region_outer = core.SectorWeatherRegion(field_center, radius*2, 0.5)
        region_inner = core.SectorWeatherRegion(field_center, radius*2*0.8, 0.5)
        sector.add_region(region_outer)
        sector.add_region(region_inner)

        return asteroids

    def spawn_character(self, location:core.SectorEntity, balance:float=10e3) -> core.Character:
        assert(self.gamestate)
        assert location.sector
        intel_manager = intel.IntelManager.create_intel_manager(self.gamestate)
        character = core.Character.create_character(
            self._choose_portrait(),
            intel_manager,
            self.gamestate,
            name=self._gen_character_name(location.sector.culture),
            home_sector_id=location.sector.entity_id,
        )
        character.location = location
        character.balance = balance
        intel_manager.character = character
        self.gamestate.add_character(character)
        return character

    def spawn_player(self, location:core.SectorEntity, balance:float) -> core.Player:
        assert(self.gamestate)

        player_character = self.spawn_character(location, balance=balance)
        player_character.context.set_flag(self.gamestate.event_manager.ck(events.ContextKeys.IS_PLAYER), 1)
        player = core.Player(self.gamestate)
        player.character = player_character
        player.agent = econ.PlayerAgent.create_player_agent(player, self.gamestate)

        for observer in self._observers:
            observer.player_spawned(player)

        return player

    def setup_captain(self, character:core.Character, asset:core.SectorEntity, mining_ships:Collection[core.Ship], trading_ships:Collection[core.Ship]) -> None:
        assert(self.gamestate)

        if isinstance(asset, core.Ship):
            if asset in mining_ships:
                character.add_agendum(agenda.CaptainAgendum.create_eoa(
                    asset,
                    character,
                    self.gamestate
                ))
                character.add_agendum(agenda.MiningAgendum.create_mining_agendum(
                    asset,
                    character,
                    self.gamestate
                ))
            elif asset in trading_ships:
                character.add_agendum(agenda.CaptainAgendum.create_eoa(
                    asset,
                    character,
                    self.gamestate
                ))
                character.add_agendum(agenda.TradingAgendum.create_trading_agendum(
                    asset,
                    character,
                    self.gamestate
                ))
                character.balance += 5e3
            else:
                raise ValueError("got a ship that wasn't in mining_ships or trading_ships")
        elif isinstance(asset, sector_entity.Station):
            character.add_agendum(agenda.StationManager.create_station_manager(
                asset,
                character,
                self.gamestate
            ))
            # give enough money to buy several batches worth of goods
            resource_price:float = self.gamestate.production_chain.prices[asset.resource] # type: ignore
            batch_size:float = self.gamestate.production_chain.batch_sizes[asset.resource] # type: ignore
            character.balance += resource_price * batch_size * 5
        elif isinstance(asset, sector_entity.Planet):
            character.add_agendum(agenda.PlanetManager.create_planet_manager(
                asset,
                character,
                self.gamestate
            ))
            # give enough money to buy some of all the final goods
            character.balance += self.gamestate.production_chain.prices[-self.gamestate.production_chain.ranks[-1]:].max() * 5
        else:
            raise ValueError(f'got an asset of unknown type {asset}')

    def spawn_habitable_sector(self, x:float, y:float, entity_id:uuid.UUID, radius:float, sector_idx:int) -> core.Sector:
        assert(self.gamestate)

        culture = self._gen_sector_culture(x, y, entity_id)

        pchain = self.gamestate.production_chain

        # compute how many raw resources are needed, transitively via
        # production chain, to build each good
        slices = [np.s_[pchain.ranks[0:i].sum():pchain.ranks[0:i+1].sum()] for i in range(len(pchain.ranks))]
        raw_needs = pchain.adj_matrix[slices[0], slices[0+1]]
        for i in range(1, len(slices)-1):
            raw_needs = raw_needs @ pchain.adj_matrix[slices[i], slices[i+1]]

        hex_size = config.Settings.generate.Universe.SECTOR_HEX_SIZE

        sector = core.Sector(
            np.array([x, y]),
            radius,
            hex_size,
            cymunk.Space(),
            self.gamestate,
            self._gen_sector_name(culture),
            entity_id=entity_id,
            culture=culture,
        )
        sector.sensor_manager = sensors.SensorManager(sector)
        self.logger.info(f'generating habitable sector {sector.name} at ({x}, {y})')
        # habitable planet
        # plenty of resources
        # plenty of production

        assets:List[core.Asset] = []

        #TODO: resources we get should be enough to build out a full
        #production chain, plus some more in neighboring sectors, plus
        #fleet of ships, plus support population for a long time (from
        #expansion through the fall)

        #TODO: set up resource fields
        # random number of fields per resource
        # random sizes
        # random allocation to each that sums to desired total
        station_factor = self.universe_config.station_factor
        num_stations = int((self.gamestate.production_chain.ranks.sum() - self.gamestate.production_chain.ranks[0])*station_factor)
        resources_to_generate = raw_needs[:,RESOURCE_REL_STATION] *  self.r.uniform(num_stations, 2*num_stations)
        #resources_to_generate += raw_needs[:,RESOURCE_REL_SHIP] * 100
        #resources_to_generate += raw_needs[:,RESOURCE_REL_CONSUMER] * 100*100
        asteroids:Dict[int, List[sector_entity.Asteroid]] = {}
        for resource, amount in enumerate(resources_to_generate):
            self.logger.info(f'spawning resource fields for {amount} units of {resource} in sector {sector.short_id()}')
            # choose a number of fields to start off with
            # divide resource amount across them
            field_amounts = [amount]
            # generate a field for each one
            asteroids[resource] = []
            for field_amount in field_amounts:
                loc = self.gen_sector_location(sector)
                asteroids[resource].extend(self.spawn_resource_field(sector, loc[0], loc[1], resource, amount))
            self.logger.info(f'generated {len(asteroids[resource])} asteroids for resource {resource} in sector {sector.short_id()}')

        self.logger.info(f'beginning entities: {len(sector.entities)}')

        # set up production stations according to resources
        # every inhabited sector should have a complete production chain
        # exclude raw resources and final products, they are not produced
        # at stations

        # assign "agents" to each production resource
        num_agents = num_stations
        agent_goods = econ.assign_agents_to_products(
                self.gamestate, num_agents,
                self.gamestate.production_chain.ranks[0])

        # each of those agents gets a station producing its good
        for i in range(num_agents):
            # find the one resource this agent produces
            resource = agent_goods[i].argmax()
            entity_loc = self.gen_sector_location(sector)
            for build_resource, amount in enumerate(raw_needs[:,RESOURCE_REL_STATION]):
                self.harvest_resources(sector, entity_loc[0], entity_loc[1], build_resource, amount)
            station = self.spawn_station(
                    sector, entity_loc[0], entity_loc[1], resource=resource, batches_on_hand=25)
            assets.append(station)

        # spend resources to build additional stations
        # consume resources to establish and support population

        # set up population according to production capacity
        entity_loc = self.gen_sector_location(sector)
        planet = self.spawn_planet(sector, entity_loc[0], entity_loc[1])
        assets.append(planet)

        # some factor mining ships for every refinery
        mining_ship_factor = self.universe_config.mining_ship_factor
        num_mining_ships = int(agent_goods[:,self.gamestate.production_chain.ranks.cumsum()[0]:self.gamestate.production_chain.ranks.cumsum()[1]].sum() * mining_ship_factor)

        self.logger.debug(f'adding {num_mining_ships} mining ships to sector {sector.short_id()}')
        mining_ships = set()
        for i in range(num_mining_ships):
            ship_x, ship_y = self.gen_sector_location(sector)
            ship = self.spawn_ship(sector, ship_x, ship_y, default_order_fn=order_fn_wait(self.gamestate))
            assets.append(ship)
            mining_ships.add(ship)

        # some factor trading ships as there are station -> station trade routes
        trade_ship_factor = self.universe_config.trading_ship_factor
        trade_routes_by_good = ((self.gamestate.production_chain.adj_matrix > 0).sum(axis=1))
        num_trading_ships = int((trade_routes_by_good[np.newaxis,:] * agent_goods).sum() * trade_ship_factor)
        self.logger.debug(f'adding {num_trading_ships} trading ships to sector {sector.short_id()}')

        trading_ships = set()
        for i in range(num_trading_ships):
            ship_x, ship_y = self.gen_sector_location(sector)
            ship = self.spawn_ship(sector, ship_x, ship_y, default_order_fn=order_fn_wait(self.gamestate))
            assets.append(ship)
            trading_ships.add(ship)

        sum_eta = 0.
        num_eta = 0
        for ship in sector.entities_by_type(core.Ship):
            for station in sector.entities_by_type(sector_entity.Station):
                sum_eta += orders.movement.GoToLocation.compute_eta(ship, station.loc)
                num_eta += 1
        self.logger.info(f'mean eta: {sum_eta / num_eta}')

        self.logger.info(f'ending entities: {len(sector.entities)}')

        # generate characters to own all the assets
        # each asset owned by exactly 1 character, each character owns 1 to 3
        # assets with mean < 2 1+2*beta
        mu_ownership = 1.35
        min_ownership = 1
        max_ownership = 3
        sigma_ownership = 0.12 * (max_ownership-min_ownership) + min_ownership
        mean_ownership = util.peaked_bounded_random(self.r, mu_ownership, sigma_ownership, lb=min_ownership, ub=max_ownership)
        num_assets = len(assets)
        num_owners = int(np.ceil(num_assets/mean_ownership))
        assert num_assets/num_owners >= min_ownership
        assert num_assets/num_owners <= max_ownership
        ownership_matrix = self._random_bipartite_graph(
                num_owners, num_assets, num_assets,
                max_ownership, min_ownership, num_assets, 1, 1)

        # sanity check on the ownership matrix: every asset owned by exactly 1
        # character and every character owns between min and max
        assert ownership_matrix.shape == (num_owners, num_assets)
        assert ownership_matrix.max() == 1
        assert ownership_matrix.sum() == num_assets
        assert (ownership_matrix == 1).sum() == num_assets
        assert ownership_matrix.sum(axis=1).min() >= min_ownership
        assert ownership_matrix.sum(axis=1).max() <= max_ownership
        assert ownership_matrix.sum(axis=0).min() == 1
        assert ownership_matrix.sum(axis=0).max() == 1

        # set up characters according to ownership_matrix
        for i in range(num_owners):
            owned_asset_ids = np.where(ownership_matrix[i] == 1)[0]
            #TODO: choose a location for the character from among their assets
            location = assets[self.r.choice(owned_asset_ids)]
            assert isinstance(location, core.SectorEntity)
            character = self.spawn_character(location)

            for asset in map(lambda j: assets[j], owned_asset_ids):
                character.take_ownership(asset)
                if character.location == asset:
                    captain = character
                else:
                    captain = self.spawn_character(asset)
                self.setup_captain(captain, asset, mining_ships, trading_ships)

        for asset in assets:
            assert asset.captain.location == asset

        self.gamestate.add_sector(sector, sector_idx)

        return sector

    def spawn_uninhabited_sector(self, x:float, y:float, entity_id:uuid.UUID, radius:float, sector_idx:int) -> core.Sector:
        assert(self.gamestate)

        culture = self._gen_sector_culture(x, y, entity_id)
        hex_size = config.Settings.generate.Universe.SECTOR_HEX_SIZE
        sector = core.Sector(
            np.array([x, y]),
            radius,
            hex_size,
            cymunk.Space(),
            self.gamestate,
            self._gen_sector_name(culture),
            entity_id=entity_id,
            culture=culture
        )
        sector.sensor_manager = sensors.SensorManager(sector)

        #TODO: put interesting stuff in the sector, e.g.
        # lots of asteroids of one or a few different resource types
        # a few asteroids of many resource types
        # lots of sensor affecting  weather
        # pirates (that ask for your cargo if you have some)
        # other interesting encounters

        self.gamestate.add_sector(sector, sector_idx)
        return sector

    def harvest_resources(self, sector: core.Sector, x: float, y: float, resource: int, amount: float) -> None:
        if amount <= 0:
            return

        asteroids = list(x for x in sector.entities_by_type(sector_entity.Asteroid) if x.resource == resource)
        if not asteroids:
            raise ValueError(f'no asteroids of type {resource} in sector {sector.short_id()}')

        center_loc = np.array((x, y), dtype=np.float64)
        # probability of harvest falls off inverse square
        dists = np.sqrt(
            np.sum(
                 np.array(list(center_loc - asteroid.loc for asteroid in asteroids))**2,
                 axis=1
            )
        )

        dists = 1/dists**2
        asteroid_probs = dists / dists.sum()

        # repeatedly mine asteroids, using them up and removing
        # them from the sector until we've paid the resource cost
        # for the station
        while amount > 0:
            i = self.r.choice(len(asteroids), p=asteroid_probs)
            asteroid = asteroids[i]

            amount_to_mine = min(amount, asteroid.cargo[asteroid.resource])
            asteroid.cargo[asteroid.resource] -= amount_to_mine
            amount -= amount_to_mine

            # if we've used up this one, remove it from the sector
            # set its prob to zero and renormalize
            if asteroid.cargo[asteroid.resource] == 0:
                sector.remove_entity(asteroid)
                asteroid_probs[i] = 0
                asteroid_probs = asteroid_probs/asteroid_probs.sum()

    def generate_chain(
            self,
            n_ranks:Optional[int]=None,
            min_per_rank:Optional[Sequence[int]]=None,
            max_per_rank:Optional[Sequence[int]]=None,
            max_outputs:Optional[int]=None,
            max_inputs:Optional[int]=None,
            min_input_per_output:Optional[int]=None,
            max_input_per_output:Optional[int]=None,
            min_raw_price:Optional[float]=None,
            max_raw_price:Optional[float]=None,
            min_markup:Optional[float]=None,
            max_markup:Optional[float]=None,
            min_final_inputs:Optional[int]=None,
            max_final_inputs:Optional[int]=None,
            min_final_prices:Optional[Sequence[float]]=None,
            max_final_prices:Optional[Sequence[float]]=None,
            min_raw_per_processed:Optional[int]=None,
            max_raw_per_processed:Optional[int]=None,
            max_fraction_one_to_one:Optional[float]=None,
            max_fraction_single_input:Optional[float]=None,
            max_fraction_single_output:Optional[float]=None,
            max_tries:Optional[int]=None,
            assign_names:bool=True,
            ) -> core.ProductionChain:
        """ Generates a random production chain.

        Products are divided into ranks with links only between consecutive
        ranks. The first rank represents raw (or simply refined) resources. The
        last rank represents final products.

        An additional rank of sink products is added to the production chain
        representing sinks for the economy. These final sinks have per unit
        target prices. The rest of the chain flows into these targets. The
        entire production chain (units, prices) is balanced to support these
        final per-unit prices.

        n_ranks: int number of products produced and traded in the economy
        min_per_rank: array of ints min number of products in each rank
        max_per_rank: array of ints max number of products in each rank
        min_input_per_output: float min number of units needed from one rank to
            produce a unit of output in the next rank
        max_input_per_output: float max number of units needed from one rank to
            produce a unit of output in the next rank
        min_raw_price: float min price for items in the first rank
        max_raw_price: float max proice for items in the first rank
        min_markup: float min markup factor on input cost when pricing outputs (1.0 = no markup)
        max_markup: float max markup factor
        min_final_inputs: int min number of inputs to produce final outputs
        max_final_inputs: int max number of inputs to produce final outputs
        min_final_prices: array of floats min target prices for final outputs
        max_final_prices: array of floats max target prices for final outputs
        min_raw_per_processed: int min number of raw inputs per processed
        max_raw_per_processed: int max number of raw inputs per processed
        max_fraction_one_to_one: float for internal ranks what fraction of nodes can lead to a  one-to-one edge
        max_fraction_single_input: float for internal ranks what fraction of nodes can have a single input
        max_fraction_single_output: float for internal ranks what fraction of nodes can have a single output
        max_tries: int how many tries should we make to generate a production chain meeting all criteria
        """

        if n_ranks is None:
            n_ranks = config.Settings.generate.ProductionChain.n_ranks
        if min_per_rank is None:
            min_per_rank = config.Settings.generate.ProductionChain.min_per_rank
        if max_per_rank is None:
            max_per_rank = config.Settings.generate.ProductionChain.max_per_rank
        if max_outputs is None:
            max_outputs = config.Settings.generate.ProductionChain.max_outputs
        if max_inputs is None:
            max_inputs = config.Settings.generate.ProductionChain.max_inputs
        if min_input_per_output is None:
            min_input_per_output = config.Settings.generate.ProductionChain.min_input_per_output
        if max_input_per_output is None:
            max_input_per_output = config.Settings.generate.ProductionChain.max_input_per_output
        if min_raw_price is None:
            min_raw_price = config.Settings.generate.ProductionChain.min_raw_price
        if max_raw_price is None:
            max_raw_price = config.Settings.generate.ProductionChain.max_raw_price
        if min_markup is None:
            min_markup = config.Settings.generate.ProductionChain.min_markup
        if max_markup is None:
            max_markup = config.Settings.generate.ProductionChain.max_markup
        if min_final_inputs is None:
            min_final_inputs = config.Settings.generate.ProductionChain.min_final_inputs
        if max_final_inputs is None:
            max_final_inputs = config.Settings.generate.ProductionChain.max_final_inputs
        if min_final_prices is None:
            min_final_prices = config.Settings.generate.ProductionChain.min_final_prices
        if max_final_prices is None:
            max_final_prices = config.Settings.generate.ProductionChain.max_final_prices
        if min_raw_per_processed is None:
            min_raw_per_processed = config.Settings.generate.ProductionChain.min_raw_per_processed
        if max_raw_per_processed is None:
            max_raw_per_processed = config.Settings.generate.ProductionChain.max_raw_per_processed
        if max_fraction_one_to_one is None:
            max_fraction_one_to_one = config.Settings.generate.ProductionChain.max_fraction_one_to_one
        if max_fraction_single_input is None:
            max_fraction_single_input = config.Settings.generate.ProductionChain.max_fraction_single_input
        if max_fraction_single_output is None:
            max_fraction_single_output = config.Settings.generate.ProductionChain.max_fraction_single_output
        if max_tries is None:
            max_tries = self.production_chain_max_tries

        production_chain:Optional[core.ProductionChain] = None
        tries = 0
        generation_error_cases:Dict[GenerationErrorCase, int] = collections.defaultdict(int)
        while production_chain is None and tries < max_tries:
            try:
                production_chain = self._generate_chain(
                    n_ranks, min_per_rank, max_per_rank,
                    max_outputs, max_inputs,
                    min_input_per_output, max_input_per_output,
                    min_raw_price, max_raw_price,
                    min_markup, max_markup,
                    min_final_inputs, max_final_inputs,
                    min_final_prices,
                    max_final_prices,
                    min_raw_per_processed, max_raw_per_processed,
                    max_fraction_one_to_one,
                    max_fraction_single_input,
                    max_fraction_single_output,
                    assign_names,
                )
            except GenerationError as e:
                generation_error_cases[e.case] += 1
                pass
            tries += 1
            if tries in self._production_chain_ticks:
                for observer in self._observers:
                    observer.generation_tick()
        self.logger.info(f'took {tries} tries to generate a production chain {generation_error_cases}')
        if not production_chain:
            raise GenerationError(GenerationErrorCase.NO_CHAIN)

        # spit out all remaining ticks on production chain generation
        while tries <= max_tries:
            if tries in self._production_chain_ticks:
                for observer in self._observers:
                    observer.generation_tick()
            tries += 1
        self.logger.info(f'{tries=} {max_tries=} {self._production_chain_ticks}')

        return production_chain

    def _generate_chain(
            self,
            n_ranks:int,
            min_per_rank:Sequence[int],
            max_per_rank:Sequence[int],
            max_outputs:int,
            max_inputs:int,
            min_input_per_output:int,
            max_input_per_output:int,
            min_raw_price:float,
            max_raw_price:float,
            min_markup:float,
            max_markup:float,
            min_final_inputs:int,
            max_final_inputs:int,
            min_final_prices:Sequence[float],
            max_final_prices:Sequence[float],
            min_raw_per_processed:int,
            max_raw_per_processed:int,
            max_fraction_one_to_one:float,
            max_fraction_single_input:float,
            max_fraction_single_output:float,
            assign_names:bool,
            ) -> core.ProductionChain:
        """ Generates a random production chain.

        Products are divided into ranks with links only between consecutive
        ranks. The first rank represents raw (or simply refined) resources. The
        last rank represents final products.

        An additional rank of sink products is added to the production chain
        representing sinks for the economy. These final sinks have per unit
        target prices. The rest of the chain flows into these targets. The
        entire production chain (units, prices) is balanced to support these
        final per-unit prices.

        n_ranks: int number of products produced and traded in the economy
        min_per_rank: array of ints min number of products in each rank
        max_per_rank: array of ints max number of products in each rank
        min_input_per_output: float min number of units needed from one rank to
            produce a unit of output in the next rank
        max_input_per_output: float max number of units needed from one rank to
            produce a unit of output in the next rank
        min_raw_price: float min price for items in the first rank
        max_raw_price: float max proice for items in the first rank
        min_markup: float min markup factor on input cost when pricing outputs (1.0 = no markup)
        max_markup: float max markup factor
        min_final_inputs: int min number of inputs to produce final outputs
        max_final_inputs: int max number of inputs to produce final outputs
        min_final_prices: array of floats min target prices for final outputs
        max_final_prices: array of floats max target prices for final outputs
        min_raw_per_processed: int min number of raw inputs per processed
        max_raw_per_processed: int max number of raw inputs per processed
        """

        if isinstance(min_per_rank, int):
            if not isinstance(max_per_rank, int):
                raise ValueError("min_per_rank and max_per_rank must both be ints or sequences")
            ranks = self.r.integers(min_per_rank, max_per_rank+1, n_ranks)
        elif isinstance(min_per_rank, (list, tuple, np.ndarray)):
            if not isinstance(max_per_rank, (list, tuple, np.ndarray)):
                raise ValueError("min_per_rank and max_per_rank must both be ints or sequences")
            if len(min_per_rank) != len(max_per_rank):
                raise ValueError("min_per_rank and max_per_rank must be the same length")
            ranks = self.r.integers(min_per_rank, np.asarray(max_per_rank)+1)
        else:
            raise ValueError("min_per_rank and max_per_rank must both be ints or sequences")

        if len(min_final_prices) != len(max_final_prices):
            raise ValueError("min and max final prices must be same length")

        # set up a rank of raw resources that mirrors the first product rank
        # these will feed 1:1 from raw resources to the first products
        ranks = np.pad(ranks, (1,0), mode="edge")

        total_nodes = np.sum(ranks)

        # generate production chain subject to target total value for each
        adj_matrix = np.zeros((total_nodes+len(min_final_prices),total_nodes+len(min_final_prices)))

        # set up 1:1 production from raw resources to first products
        # then scale up inputs as a raw -> processed refinement factor
        refinement_factors = self.r.integers(min_raw_per_processed, max_raw_per_processed, ranks[0])
        adj_matrix[0:ranks[0], ranks[0]:ranks[0]+ranks[1]] = np.eye(ranks[0]) * refinement_factors

        # set up production for rest of products
        so_far = ranks[0]
        for rank, (nodes_from, nodes_to) in enumerate(zip(ranks[1:], ranks[2:]), 1):
            target_edges = self.r.integers(
                np.max((nodes_from, nodes_to)),
                np.min((
                    nodes_from*nodes_to,
                    nodes_from*max_outputs,
                    nodes_to*max_inputs
                ))+1
            )
            target_weight = np.mean((min_input_per_output, max_input_per_output)) * target_edges
            rank_production = np.ceil(self._random_bipartite_graph(
                    nodes_from, nodes_to, target_edges,
                    np.min((nodes_to, max_outputs)),
                    np.min((nodes_from, max_inputs)),
                    target_weight,
                    min_input_per_output, max_input_per_output))

            # check for 1:1 connections (node has exactly 1 output and it's to
            # a node with exactly one input)
            num_one_to_one = 0
            for i in np.where((rank_production > 0).astype(int).sum(axis=1))[0]:
                single_output = rank_production[i].argmax()
                if (rank_production[:, single_output] > 0).astype(int).sum() == 1:
                    num_one_to_one += 1
            if num_one_to_one/nodes_from > max_fraction_one_to_one:
                raise GenerationError(GenerationErrorCase.ONE_TO_ONE, f'too many one-to-one connections {num_one_to_one=} {nodes_from=}')

            # check for fraction that have single input/output
            num_single_input = ((rank_production > 0).astype(int).sum(axis=0) == 1).sum()
            if num_single_input / nodes_to > max_fraction_single_input:
                raise GenerationError(GenerationErrorCase.SINGLE_INPUT, f'too many nodes have a single input {num_single_input / nodes_to}')

            num_single_output = ((rank_production > 0).astype(int).sum(axis=1) == 1).sum()
            if  num_single_output / nodes_from > max_fraction_single_output:
                raise GenerationError(GenerationErrorCase.SINGLE_OUTPUT, f'too many nodes have a single output {num_single_output / nodes_from}')

            adj_matrix[so_far:so_far+nodes_from, so_far+nodes_from:so_far+nodes_from+nodes_to] = rank_production
            so_far += nodes_from

        # add in final products
        ranks = np.pad(ranks, (0, 1), constant_values=len(min_final_prices))

        s_last_goods = np.s_[-1*ranks[-1] - ranks[-2]:-1*ranks[-1]]
        s_final_products = np.s_[-1*ranks[-1]:]

        # generate production needs for final products
        target_edges = self.r.integers(
                np.max((
                    ranks[-2],
                    ranks[-1]*min_final_inputs)),
                np.min((
                    ranks[-2]*ranks[-1],
                    ranks[-2]*max_outputs,
                    ranks[-1]*max_final_inputs
                ))+1
        )

        target_weight = np.mean((min_input_per_output, max_input_per_output)) * target_edges
        final_production = np.ceil(self._random_bipartite_graph(
                ranks[-2], ranks[-1], target_edges,
                np.min((ranks[-1], max_outputs)),
                np.min((ranks[-2], max_final_inputs)),
                target_weight, min_input_per_output, max_input_per_output,
                min_in=min_final_inputs))

        # adjust weights to hit target prices
        final_prices = self.r.uniform(min_final_prices, max_final_prices)
        final_production = final_production / final_production.sum(axis=0) * final_prices
        adj_matrix[s_last_goods, s_final_products] = final_production
        total_nodes = np.sum(ranks)

        # make sure all non-lastrank products have an output
        if np.any(adj_matrix[:-ranks[-1],:].sum(axis=1) == 0):
            raise GenerationError(GenerationErrorCase.NO_OUTPUTS, f'some intermediate products have no output')
        # make sure all non-firstrank products have an input
        assert np.all(adj_matrix[:,ranks[0]:].sum(axis=0) > 0)

        # set up prices

        raw_price = self.r.uniform(0, 1, ranks[0])
        raw_price = raw_price / raw_price.sum() * np.mean((min_raw_price, max_raw_price)) * np.mean((min_per_rank, max_per_rank))
        markup = self.r.uniform(min_markup, max_markup, total_nodes)
        prices = np.pad(raw_price, (0, total_nodes-ranks[0]))
        prices = np.ceil(prices)
        for (nodes_from, nodes_to), so_far in zip(zip(ranks, ranks[1:]), np.cumsum(ranks)[:-1]):
            # price of the next rank is the price of the prior rank times the
            # production matrix times the markup
            relevant_prod_matrix = adj_matrix[so_far-nodes_from:so_far, so_far:so_far+nodes_to]

            prices[so_far:so_far+nodes_to] = (np.reshape(prices[so_far-nodes_from:so_far], (nodes_from, 1)) * relevant_prod_matrix).sum(axis=0) * markup[so_far:so_far+nodes_to]

        # adjust final production weights to account for the prices of inputs and markup
        #TODO: the below throws type error in mypy since vstack takes a tuple to vstack, what's going on here? (two cases below)
        adj_matrix[s_last_goods, s_final_products] /= np.vstack(prices[s_last_goods]) # type: ignore
        adj_matrix[s_last_goods, s_final_products] = np.ceil(adj_matrix[s_last_goods, s_final_products])

        # make sure all non-lastrank products have an output
        assert np.all(adj_matrix[:-ranks[-1],:].sum(axis=1) > 0)
        # make sure all non-firstrank products have an input
        assert np.all(adj_matrix[:,ranks[0]:].sum(axis=0) > 0)

        prices[s_final_products] = (np.vstack(prices[so_far-nodes_from:so_far]) * adj_matrix[s_last_goods, s_final_products]).sum(axis=0) * markup[s_final_products] # type: ignore

        assert not np.any(np.isnan(prices))
        assert np.all(prices > (prices[:, np.newaxis] * adj_matrix).sum(axis=0))

        # set up production times and batch sizes
        # in one minute produce a batch of enough goods to produce a batch of
        # some number of the next items in the chain
        production_times = np.full((total_nodes,), 60.)
        batch_sizes = np.clip(3. * np.ceil(np.min(adj_matrix, axis=1, where=adj_matrix>0, initial=np.inf)), 1., 50)
        batch_sizes[-ranks[-1]:] = 1

        if assign_names:
            product_names = self._generate_product_names(ranks, adj_matrix)
        else:
            product_names = [f'product_{x}' for x in range(len(adj_matrix))]

        chain = core.ProductionChain()
        chain.ranks = ranks
        chain.adj_matrix = adj_matrix
        chain.markup = markup
        chain.prices = prices
        chain.product_names = product_names
        chain.production_times = production_times
        chain.batch_sizes = batch_sizes

        chain.initialize()

        for i, (price, name) in enumerate(zip(prices[s_final_products], product_names[-ranks[-1]:]), len(prices)-len(min_final_prices)):
            self.logger.info(f'price {name}:\t${price}')
        self.logger.info(f'total price:\t${prices[s_final_products].sum()}')

        return chain

    def generate_sector_cultures(self,
            sector_ids:npt.NDArray,
            sector_coords:npt.NDArray[np.float64],
            sector_edges:npt.NDArray[np.float64],
            edge_distances:npt.NDArray[np.float64],
            num_cultures:int,
    ) -> Mapping[uuid.UUID, str]:
        # choose some seed sectors

        cultures = list(self.universe_config.cultures)
        culture_map:MutableMapping[int, str] = {}
        # distances from seed
        distances:MutableMapping[int, Mapping[int, float]] = {}
        mean_distances = np.zeros(len(sector_ids))
        for i in range(len(sector_ids)):
            _, d = dijkstra(edge_distances, i, -1)
            for dist in d.values():
                mean_distances[i] += dist
        mean_distances /= len(sector_ids)
        # first sector is randomly chosen
        #TODO: perhaps we should choose the most remote culture in some sense?
        #i = self.r.integers(len(sector_coords))
        i = int(mean_distances.argmax())
        min_dists = np.ones(len(sector_coords))*np.inf
        for _ in range(num_cultures):
            #TODO: what if we run out of cultures?
            assert(len(cultures) > 0)
            # assign a culture to next choice
            culture = self.r.choice(cultures)
            culture_map[i] = culture
            cultures.remove(culture)
            _, d = dijkstra(edge_distances, i, -1)
            distances[i] = d
            # we assume the graph is fully connected, so every entry will be
            # overwritten with a non-inf value
            for k, dist in d.items():
                if min_dists[k] > dist:
                    min_dists[k] = dist
            # choose most remote sector from any current seed sector
            i = int(min_dists.argmax())
            # seed sectors should appear with dist 0
            assert(i not in culture_map or self.universe_config.num_sectors == 1)

        uncultured_sectors = set(i for i in range(len(sector_coords)) if i not in culture_map)

        # "grow" culture from the seed sectors along edges
        # always choose the next closest uncultured sector via total distance
        # to seed sector
        while len(uncultured_sectors) > 0:
            min_dist = np.inf
            min_dist_seed = -1
            min_dist_sector = -1
            for s in uncultured_sectors:
                for i,d in distances.items():
                    if d[s] < min_dist:
                        min_dist = d[s]
                        min_dist_seed = i
                        min_dist_sector = s
            assert(min_dist_seed >= 0)
            assert(min_dist_sector >= 0)
            culture_map[min_dist_sector] = culture_map[min_dist_seed]
            uncultured_sectors.remove(min_dist_sector)

        return dict(((sector_ids[k], v) for k,v in culture_map.items()))

    def generate_sector_topology(self,
            universe_radius:float,
            num_sectors:int,
            sector_radius:float,
            sector_radius_std:float,
            sector_ids:npt.NDArray,
            max_sector_edge_length:float,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        assert(self.gamestate)

        # generate locations for all sectors
        sector_coords = self.r.uniform(-universe_radius, universe_radius, (num_sectors, 2))
        sector_coords_by_id = dict(zip(sector_ids, sector_coords))
        sector_k = sector_radius**2/sector_radius_std**2
        sector_theta = sector_radius_std**2/sector_radius
        sector_radii = self.r.gamma(sector_k, sector_theta, num_sectors)

        sector_loc_index = rtree.index.Index()

        # clean up any overlapping sectors
        for idx, (coords, radius) in enumerate(zip(sector_coords, sector_radii)):
            reject = True
            while reject:
                reject = False
                for hit in sector_loc_index.intersection((coords[0]-radius, coords[1]-radius, coords[0]+radius, coords[1]+radius), True):
                    other_coords, other_radius = hit.object # type: ignore
                    if util.distance(coords, other_coords) < radius + other_radius:
                        reject = True
                        break
                if reject:
                    coords = self.r.uniform(-universe_radius, universe_radius, 2)
                    radius = self.r.gamma(sector_k, sector_theta)

            sector_coords[idx] = coords
            sector_radii[idx] = radius
            sector_loc_index.insert(
                    idx,
                    (
                        coords[0]-radius, coords[1]-radius,
                        coords[0]+radius, coords[1]+radius
                    ),
                    (coords, radius)
            )

        # set up connectivity between sectors
        distances = util.pairwise_distances(sector_coords)#distance.squareform(distance.pdist(sector_coords))
        sector_edges, edge_distances = prims_mst(distances, self.r.integers(0, len(distances)))

        # index of bboxes for edges
        edge_id = 0
        edge_index = rtree.index.Index()
        for (i, source_id), (j, dest_id) in itertools.product(enumerate(sector_ids), enumerate(sector_ids)):
            if sector_edges[i,j] == 1:
                a = sector_coords[i]#self.gamestate.sectors[source_id].loc
                b = sector_coords[j]#self.gamestate.sectors[dest_id].loc
                bbox = (
                    min(a[0],b[0]), min(a[1],b[1]),
                    max(a[0],b[0]), max(a[1],b[1]),
                )
                edge_index.insert(edge_id, bbox, ((*a, *b), (source_id, dest_id)))
                edge_id += 1

        # add edges for nearby sectors, shortest distance first
        for (i,j) in np.dstack(np.unravel_index(np.argsort(distances.flatten()), (len(distances),len(distances))))[0]:
            source_id = sector_ids[i]
            dest_id = sector_ids[j]
            # skip self edges
            if source_id == dest_id:
                continue
            # skip existing edges
            if sector_edges[i,j] == 1:
                continue

            dist = util.distance(sector_coords_by_id[source_id], sector_coords_by_id[dest_id])
            if dist > max_sector_edge_length:
                continue

            # do not add crossing edges
            a = sector_coords_by_id[source_id]
            b = sector_coords_by_id[dest_id]
            bbox = (
                min(a[0],b[0]), min(a[1],b[1]),
                max(a[0],b[0]), max(a[1],b[1]),
            )
            segment = (a[0],a[1],b[0],b[1])
            collision = False
            for (hit, ids) in edge_index.intersection(bbox, objects="raw"): # type: ignore
                # ignore segments that share an endpoint
                if source_id in ids or dest_id in ids: # type: ignore
                    continue
                if util.segments_intersect(segment, hit): # type: ignore
                    collision = True
                    break
            if collision:
                continue

            # do not add edges if the current best distance is not much more
            p, path_dist = dijkstra(edge_distances, i, j)
            if path_dist[j] < dist*1.5:
                continue

            # sometmes don't take the edge, related to its length
            if self.r.uniform() < 0.9:#(max_sector_edge_length*2 - dist)/(max_sector_edge_length*2):
                continue

            sector_edges[i,j] = 1
            sector_edges[j,i] = 1
            edge_distances[i,j] = dist
            edge_distances[j,i] = dist
            edge_index.insert(edge_id, bbox, ((*a, *b), (source_id, dest_id)))
            edge_id += 1

        self.gamestate.update_edges(sector_edges, sector_ids, sector_coords)

        return sector_coords, sector_edges, sector_radii, edge_distances

    def generate_sectors(self,
            universe_radius:float,
            num_sectors:int,
            sector_radius:float,
            sector_radius_std:float,
            max_sector_edge_length:float,
            n_habitable_sectors:int,
            mean_habitable_resources:float,
            mean_uninhabitable_resources:float,
            num_cultures:int,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        assert(self.gamestate)
        # set up pre-expansion sectors, resources

        # compute the raw resource needs for each product sink
        pchain = self.gamestate.production_chain
        slices = [np.s_[pchain.ranks[0:i].sum():pchain.ranks[0:i+1].sum()] for i in range(len(pchain.ranks))]
        raw_needs = pchain.adj_matrix[slices[0], slices[0+1]]
        for i in range(1, len(slices)-1):
            raw_needs = raw_needs @ pchain.adj_matrix[slices[i], slices[i+1]]

        self.logger.info(f'raw needs {raw_needs}')

        # generate sector ids and choose which are inhabited
        sector_ids = np.array([uuid.uuid4() for _ in range(num_sectors)])
        habitable_mask = np.zeros(num_sectors, bool)
        habitable_mask[self.r.choice(num_sectors, n_habitable_sectors, replace=False)] = 1

        sector_coords, sector_edges, sector_radii, edge_distances = self.generate_sector_topology(
            universe_radius,
            num_sectors,
            sector_radius,
            sector_radius_std,
            sector_ids,
            max_sector_edge_length,
        )

        for observer in self._observers:
            observer.generation_step(GenerationStep.CULTURES)
        self._culture_map = self.generate_sector_cultures(
            sector_ids,
            sector_coords,
            sector_edges,
            edge_distances,
            num_cultures
        )

        self._load_name_models(culture_filter=list(set(self._culture_map.values())))

        # choose habitable sectors
        # each of these will have more resources
        # implies a more robust and complete production chain
        # implies a bigger population
        for observer in self._observers:
            observer.generation_step(GenerationStep.INHABITED_SECTORS)
        for idx, entity_id, (x,y), radius in zip(np.argwhere(habitable_mask), sector_ids[habitable_mask], sector_coords[habitable_mask], sector_radii[habitable_mask]):
            # mypy thinks idx is an int
            sector = self.spawn_habitable_sector(x, y, entity_id, radius, idx[0]) # type: ignore
            for observer in self._observers:
                observer.generation_tick()

        # set up non-habitable sectors
        for observer in self._observers:
            observer.generation_step(GenerationStep.UNINHABITED_SECTORS)
        for idx, entity_id, (x,y), radius in zip(np.argwhere(~habitable_mask), sector_ids[~habitable_mask], sector_coords[~habitable_mask], sector_radii[~habitable_mask]):
            # mypy thinks idx is an int
            sector = self.spawn_uninhabited_sector(x, y, entity_id, radius, idx[0]) # type: ignore
            for observer in self._observers:
                observer.generation_tick()

        # add gates for the travel lanes
        #TODO: there's probably a clever way to get these indicies
        for (i, source_id), (j, dest_id) in itertools.product(enumerate(sector_ids), enumerate(sector_ids)):
            if sector_edges[i,j] != 0:
                self.spawn_gate(self.gamestate.sectors[source_id], self.gamestate.sectors[dest_id])

        #TODO: post-expansion decline
        # deplete resources at population centers
        # degrade production elements post-expansion

        #TODO: current-era
        # establish factions
        # establish post-expansion production elements and equipment
        # establish current-era characters and distribute roles

        return sector_ids, habitable_mask

    def generate_player(self) -> None:
        assert(self.gamestate)

        # mining start: working for someone else, doing mining for a refinery
        # player starts in a ship near a refinery, ship owned by refinery owner
        refinery:Optional[sector_entity.Station] = None
        for character in self.gamestate.characters.values():
            for a in character.assets:
                if isinstance(a, sector_entity.Station) and a.resource in self.gamestate.production_chain.first_product_ids():
                    refinery = a
                    break

        if refinery is None:
            raise ValueError("no suitable refinery could be found")

        refinery.cargo[refinery.resource] += math.floor((refinery.cargo_capacity - np.sum(refinery.cargo))*0.2)

        assert refinery.sector

        asteroid_loc = refinery.loc
        while not 5e3 < util.distance(asteroid_loc, refinery.loc) < 1e4:
            asteroid_loc = self.gen_sector_location(refinery.sector, center=refinery.loc, radius=2e3, occupied_radius=5e2)
        assert refinery.resource is not None
        asteroid = self.spawn_asteroid(refinery.sector, asteroid_loc[0], asteroid_loc[1], resource=self.gamestate.production_chain.inputs_of(refinery.resource)[0], amount=1e5)

        ship_loc = refinery.loc
        while not 5e2 < util.distance(ship_loc, refinery.loc) < 1e3:
            ship_loc = self.gen_sector_location(refinery.sector, center=refinery.loc, radius=2e3, occupied_radius=5e2)
        ship = self.spawn_ship(refinery.sector, ship_loc[0], ship_loc[1], v=np.array((0.,0.)), w=0., default_order_fn=order_fn_wait(self.gamestate))

        self.gamestate.player = self.spawn_player(ship, balance=2e3)
        assert(self.gamestate.player.character)
        player_character = self.gamestate.player.character

        player_character.add_agendum(agenda.CaptainAgendum.create_eoa(ship, player_character, self.gamestate, enable_threat_response=False))

        # set up tutorial flags
        assert refinery.captain
        assert refinery.captain.location
        asteroid.context.set_flag(self.gamestate.event_manager.ck(events.ContextKeys.TUTORIAL_ASTEROID), asteroid.short_id_int())
        refinery.captain.context.set_flag(self.gamestate.event_manager.ck(events.ContextKeys.TUTORIAL_GUY), refinery.captain.short_id_int())
        refinery.captain.context.set_flag(self.gamestate.event_manager.ck(events.ContextKeys.TUTORIAL_TARGET_PLAYER), player_character.short_id_int())
        player_character.context.set_flag(self.gamestate.event_manager.ck(events.ContextKeys.TUTORIAL_GUY), refinery.captain.short_id_int())
        player_character.context.set_flag(self.gamestate.event_manager.ck(events.ContextKeys.TUTORIAL_RESOURCE), asteroid.resource)
        player_character.context.set_flag(self.gamestate.event_manager.ck(events.ContextKeys.TUTORIAL_TARGET_PLAYER), player_character.short_id_int())
        player_character.context.set_flag(self.gamestate.event_manager.ck(events.ContextKeys.TUTORIAL_AMOUNT_TO_MINE), 500)
        player_character.context.set_flag(self.gamestate.event_manager.ck(events.ContextKeys.TUTORIAL_AMOUNT_TO_TRADE), 500)

        assert(player_character.location)
        self.logger.info(f'player is {player_character.short_id()} in {player_character.location.address_str()} {player_character.name}')
        self.logger.info(f'tutorial guy is {refinery.captain.short_id()} in {refinery.captain.location.address_str()} {refinery.captain.name}')
        self.logger.info(f'refinery is {refinery.address_str()} {refinery.name}')
        self.logger.info(f'asteroid is {asteroid.address_str()} {asteroid.name}')

    def generate_player_for_combat_test(self, sector_ids:npt.NDArray, habitable_mask:npt.NDArray) -> None:
        assert(self.gamestate)

        # choose an uninhabited sector
        sector_id = self.r.choice(sector_ids[~habitable_mask], 1)[0] # type: ignore
        sector = self.gamestate.sectors[sector_id]

        # spawn the player, character, ship, etc.
        ship_loc = np.array((0., 0.))
        ship = self.spawn_ship(sector, ship_loc[0], ship_loc[1], v=np.array((0.,0.)), w=0., default_order_fn=order_fn_wait(self.gamestate))

        self.gamestate.player = self.spawn_player(ship, balance=2e3)
        player_character = self.gamestate.player.character
        assert(player_character)

        player_character.add_agendum(agenda.CaptainAgendum.create_eoa(ship, player_character, self.gamestate, enable_threat_response=False, start_transponder=False))

        # spawn an NPC ship/character with orders to attack the player
        min_dist = 1e5
        max_dist = 1.5e5
        threat_loc = self.gen_sector_location(sector, center=ship_loc, radius=max_dist, occupied_radius=5e2, min_dist=min_dist, strict=True)
        threat = self.spawn_ship(sector, threat_loc[0], threat_loc[1], v=np.array((0.,0.)), w=0., default_order_fn=order_fn_wait(self.gamestate))
        character = self.spawn_character(threat)
        character.take_ownership(threat)
        character.add_agendum(agenda.CaptainAgendum.create_eoa(threat, character, self.gamestate, enable_threat_response=False))

        order = combat.HuntOrder.create_hunt_order(ship.entity_id, threat, self.gamestate, start_loc=ship.loc)
        threat.clear_orders()
        threat.prepend_order(order)

        # add in some weather near the player (so they can escape into it)
        weather_loc = ship_loc + (ship_loc-threat_loc)
        weather_loc_radius = 4e5
        weather_radius = 5e4
        weather_loc = self.gen_sector_location(sector, center=weather_loc, radius=weather_loc_radius, occupied_radius=5e2, min_dist=3e5, strict=True)
        region_outer = core.SectorWeatherRegion(weather_loc, weather_radius*2, 0.5)
        region_inner = core.SectorWeatherRegion(weather_loc, weather_radius*2*0.8, 0.5)
        sector.add_region(region_outer)
        sector.add_region(region_inner)


    def generate_starfields(self) -> None:
        assert(self.gamestate)

        self.logger.info(f'generating universe_starfield...')
        self.gamestate.starfield = generate_starfield(
            self.r,
            radius=4*self.universe_config.universe_radius,
            desired_stars_per_char=(4/80.)**2,
            min_zoom=self.universe_config.universe_radius/80.,
            max_zoom=self.universe_config.sector_radius_mean/80*8,
            layer_zoom_step=0.25,
        )
        self.logger.info(f'generated {sum(x.num_stars for x in self.gamestate.starfield)} universe stars in {len(self.gamestate.starfield)} layers')

        self.logger.info(f'generating sector starfield...')
        self.gamestate.sector_starfield = generate_starfield(
            self.r,
            radius=8*self.universe_config.sector_radius_mean,
            desired_stars_per_char=(3/80.)**2,
            min_zoom=(6*self.universe_config.sector_radius_std+self.universe_config.sector_radius_mean)/80,
            max_zoom=config.Settings.generate.SectorEntities.ship.RADIUS*2,
            layer_zoom_step=0.25,
        )

        self.logger.info(f'generated {sum(x.num_stars for x in self.gamestate.sector_starfield)} sector stars in {len(self.gamestate.sector_starfield)} layers')

    def estimate_generation_ticks(self) -> int:
        # 10 ticks for production chains
        # 1 tick for starfields
        # each culture model + up to max cultures
        # each sector
        # 1 tick for player
        self._production_chain_ticks = set(np.linspace(1, self.production_chain_max_tries, num=10, dtype=int))
        return len(self._production_chain_ticks) + 1 + self.universe_config.num_cultures[1] + self.universe_config.num_sectors + 1

    def generate_universe(self) -> core.Gamestate:
        self.logger.info(f'generating a universe...')
        self.gamestate = core.Gamestate()
        self.gamestate.event_manager = self.event_manager
        self.gamestate.event_manager.initialize_gamestate(events.EventState(), self.gamestate)

        self.gamestate.generator = self
        self.gamestate.random = self.r
        generation_start = time.perf_counter()

        #TODO: should have a generation tick for this
        self.logger.info(f'generating sprite starfield...')
        min_zoom = self.universe_config.universe_radius/48.
        max_zoom = self.universe_config.universe_radius/48.*0.25
        sprite_starfields = generate_starfield(
            self.r,
            radius=self.universe_config.universe_radius,
            desired_stars_per_char=(4/80.)**2*3.,
            min_zoom=min_zoom,
            max_zoom=max_zoom,
            layer_zoom_step=0.25,
            mu=0.6, sigma=0.15,
        )
        self.gamestate.portrait_starfield = sprite_starfields

        num_cultures = int(self.r.integers(*self.universe_config.num_cultures, endpoint=True)) # type: ignore

        #TODO: janky!
        estimated_ticks = self.estimate_generation_ticks()
        for observer in self._observers:
            observer.estimated_generation_ticks(estimated_ticks)

        # generate a production chain
        for observer in self._observers:
            observer.generation_step(GenerationStep.PRODUCTION_CHAIN)
        self.gamestate.production_chain = self.generate_chain()

        # generate pretty starfields for the background
        for observer in self._observers:
            observer.generation_step(GenerationStep.STARFIELDS)
            observer.generation_tick()
        self.generate_starfields()

        # generate sectors
        sector_ids, habitable_mask = self.generate_sectors(
            universe_radius=self.universe_config.universe_radius,
            num_sectors=self.universe_config.num_sectors,
            sector_radius=self.universe_config.sector_radius_mean,
            sector_radius_std=self.universe_config.sector_radius_std,
            max_sector_edge_length=self.universe_config.max_sector_edge_length,
            n_habitable_sectors=self.universe_config.num_habitable_sectors,
            mean_habitable_resources=self.universe_config.mean_habitable_resources,
            mean_uninhabitable_resources=self.universe_config.mean_uninhabitable_resources,
            num_cultures=num_cultures,
        )

        # generate the player
        for observer in self._observers:
            observer.generation_step(GenerationStep.PLAYER)
            observer.generation_tick()
        self.generate_player()
        #self.generate_player_for_combat_test(sector_ids, habitable_mask)

        self.logger.info(f'sectors: {len(self.gamestate.sectors)}')
        self.logger.info(f'sectors_edges: {np.sum(self.gamestate.sector_edges)}')
        self.logger.info(f'characters: {len(self.gamestate.characters)}')
        self.logger.info(f'econ_agents: {len(self.gamestate.econ_agents)}')
        self.logger.info(f'entities: {len(self.gamestate.entities)}')

        assert all(x == y for x,y in zip(self.gamestate.entities.values(), self.gamestate.entities_short.values()))

        generation_stop = time.perf_counter()
        logging.info(f'took {generation_stop - generation_start:.3f}s to generate universe')

        for observer in self._observers:
            observer.universe_generated(self.gamestate)

        # trigger the start of game event
        self.gamestate.trigger_event(
            self.gamestate.characters.values(),
            self.gamestate.event_manager.e(events.Events.START_GAME),
            {},
        )

        # set a fingerprint to identify this game across gameply and saves
        #TODO: should the fingerprint be related to the universe we just generated?
        self.gamestate.fingerprint = hashlib.sha256(self.r.bytes(1024)).digest()

        return self.gamestate

    def load_universe(self, gamestate:core.Gamestate) -> None:
        self.gamestate = gamestate
        self.gamestate.generator = self
        # we take random generator from the gamestate which should be
        # authoritative at this point
        self.r = self.gamestate.random

        self._culture_map = dict((sector_id, sector.culture) for sector_id, sector in gamestate.sectors.items())
        if any(x != self._empty_name_model_culture for x in self._culture_map.values()):
            self._load_name_models(culture_filter=list(set(self._culture_map.values())))

        #TODO: event manager needs to be created and initialized by this point

        for observer in self._observers:
            observer.universe_loaded(self.gamestate)

def main() -> None:

    names = [
        config.Settings.generate.ProductionChain.ORE_NAMES,
        [f'Refined {x}' for x in config.Settings.generate.ProductionChain.ORE_NAMES],
        config.Settings.generate.ProductionChain.INTERMEDIATE_NAMES,
        config.Settings.generate.ProductionChain.HIGHTECH_NAMES,
        config.Settings.generate.ProductionChain.SINK_NAMES,
    ]
    constraints:List[List[List[int]]] = [
        [[]] * len(config.Settings.generate.ProductionChain.ORE_NAMES),
        [[i] for i in range(len(config.Settings.generate.ProductionChain.ORE_NAMES))],
        config.Settings.generate.ProductionChain.INTERMEDIATE_INPUTS,
        config.Settings.generate.ProductionChain.HIGHTECH_INPUTS,
        config.Settings.generate.ProductionChain.SINK_INPUTS,
    ]
    UniverseGenerator.viz_product_name_graph(names, constraints).render("/tmp/product_name_graph", format="pdf")

if __name__ == "__main__":
    main()
