import logging
import itertools
import uuid
import math
from typing import Optional, List, Dict, Mapping, Tuple, Sequence, Union, overload
import importlib.resources
import itertools

import numpy as np
import numpy.typing as npt
from scipy.spatial import distance # type: ignore
import cymunk # type: ignore

from stellarpunk import util, core, orders, agenda, econ

#TODO: names: sectors, planets, stations, ships, characters, raw materials,
#   intermediate products, final products, consumer products, station products,
#   ship products

RESOURCE_REL_SHIP = 0
RESOURCE_REL_STATION = 1
RESOURCE_REL_CONSUMER = 2


def prims_mst(distances:npt.NDArray[np.float64], root_idx:int) -> npt.NDArray[np.float64]:
    # prim's algorithm to construct a minimum spanning tree
    # https://en.wikipedia.org/wiki/Prim%27s_algorithm
    # choose starting vertex arbitrarily
    V = np.zeros(len(distances), bool)
    E = np.zeros((len(distances), len(distances)))
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
    return E

"""
@overload
def peaked_bounded_random(r:np.random.Generator, mu:float, sigma:float, lb:float=0., ub:float=1.0) -> float: ...

@overload
def peaked_bounded_random(r:np.random.Generator, mu:float, sigma:float, size:Union[int, Sequence[int]], lb:float=0., ub:float=1.0) -> npt.NDArray[np.float64]: ...
"""

def peaked_bounded_random(
        r:np.random.Generator, mu:float, sigma:float,
        size:Optional[Union[int, Sequence[int]]]=None,
        lb:float=0., ub:float=1.0) -> Union[float, npt.NDArray[np.float64]]:
    if mu <= lb or mu >= ub:
        raise ValueError(f'mu={mu} must be lb<mu<ub')
    if sigma <= 0.:
        raise ValueError(f'sigma={sigma} must be > 0.')

    scale = ub-lb
    mu = (mu-lb)/scale
    sigma = (sigma-lb)/scale
    phi = mu * (1-mu)/(sigma**2.)-1.
    if phi <= 1./mu or phi <= 1./(1.-mu):
        raise ValueError(f'sigma={sigma} must be s.t. after transforming mu and sigma to 0,1, mu * (1-mu)/(sigma**2.)-1. < 1/mu and < 1/(1-mu)')
    alpha = mu * phi
    beta = (1-mu) * phi
    # make sure alpha/beta > 1, which makes beta unimodal between 0,1
    assert alpha > 1.
    assert beta > 1.

    return lb+scale*r.beta(alpha, beta, size=size)

class GenerationListener:
    def production_chain_complete(self, production_chain:core.ProductionChain) -> None:
        pass

    def sectors_complete(self, sectors:Mapping[Tuple[int,int], core.Sector]) -> None:
        pass

def order_fn_null(ship:core.Ship, gamestate:core.Gamestate) -> core.Order:
    return core.Order(ship, gamestate)

def order_fn_wait(ship:core.Ship, gamestate:core.Gamestate) -> core.Order:
    return orders.WaitOrder(ship, gamestate)

def order_fn_goto_random_station(ship:core.Ship, gamestate:core.Gamestate) -> core.Order:
    if ship.sector is None:
        raise Exception("cannot go to location if ship isn't in a sector")
    station = gamestate.random.choice(np.array(ship.sector.stations))
    return orders.GoToLocation.goto_entity(station, ship, gamestate)

def order_fn_disembark_to_random_station(ship:core.Ship, gamestate:core.Gamestate) -> core.Order:
    if ship.sector is None:
        raise Exception("cannot disembark to if ship isn't in a sector")
    station = gamestate.random.choice(np.array(ship.sector.stations))
    return orders.DisembarkToEntity.disembark_to(station, ship, gamestate)

"""
def order_fn_harvest_random_resource(ship:core.Ship, gamestate:core.Gamestate) -> core.Order:
    if ship.sector is None:
        raise Exception("cannot harvest if ship isn't in a sector")
    # choose a station to mine for. this is the resource we'll mine
    station = gamestate.random.choice(np.array(list(filter(
        lambda x: x.resource in gamestate.production_chain.first_product_ids(),
        ship.sector.stations
    ))))

    # find the id of the raw resource input for this station
    resource = gamestate.production_chain.inputs_of(station.resource)[0]

    return orders.HarvestOrder(station, resource, ship, gamestate)
"""

class UniverseGenerator:
    def __init__(self, gamestate:core.Gamestate, seed:Optional[int]=None) -> None:
        self.logger = logging.getLogger(util.fullname(self))

        self.gamestate = gamestate

        # random generator
        self.r = np.random.default_rng(seed)

        self.parallel_max_edges_tries = 10000

        # load character portraits
        self.portraits:List[core.Sprite] = []
        sheet = importlib.resources.read_text("stellarpunk.data", "portraits.txt").split("\n")
        size = (32//2, 32//4)
        offset_limit = (len(sheet[0])//size[0], len(sheet)//size[1])
        # portraits are 32x32 pixels, binary chars are 2x4 pixels per char
        for offset_x, offset_y in itertools.product(range(offset_limit[0]), range(offset_limit[1])):
            self.portraits.append(core.Sprite(
                [
                    x[offset_x*size[0]:offset_x*size[0]+size[0]] for x in sheet[offset_y*size[1]:offset_y*size[1]+size[1]]
                ]
            ))

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

    def _gen_sector_location(self, sector:core.Sector, unoccupied:bool=True)->npt.NDArray[np.float64]:
        loc = self.r.normal(0, 1, 2) * sector.radius
        while unoccupied and sector.is_occupied(loc[0], loc[1], eps=2e3):
            loc = self.r.normal(0, 1, 2) * sector.radius

        return loc

    def _gen_sector_name(self) -> str:
        return "Some Sector"

    def _gen_planet_name(self) -> str:
        return "Magusan"

    def _gen_station_name(self) -> str:
        return "Some Station"

    def _gen_ship_name(self) -> str:
        return "Some Ship"

    def _gen_character_name(self) -> str:
        return "Bob Dole"

    def _gen_asteroid_name(self) -> str:
        return "Asteroid X"

    def _gen_gate_name(self, destination:core.Sector) -> str:
        return f"Gate to {destination.short_id()}"

    def _choose_portrait(self) -> core.Sprite:
        return self.portraits[self.r.integers(0, len(self.portraits))]

    def _phys_body(self, mass:Optional[float]=None, radius:Optional[float]=None) -> cymunk.Body:
        if mass is None:
            body = cymunk.Body()#body_type=pymunk.Body.STATIC)
        else:
            assert radius is not None
            moment = cymunk.moment_for_circle(mass, 0, radius)
            body = cymunk.Body(mass, moment)
        return body

    def _phys_shape(self, body:cymunk.Body, entity:core.SectorEntity, obj_flag:core.ObjectFlag, radius:float) -> cymunk.Shape:
        shape = cymunk.Circle(body, radius)
        shape.friction=0.1
        shape.collision_type = entity.object_type
        #shape.filter = pymunk.ShapeFilter(categories=obj_flag)
        body.position = (entity.loc[0], entity.loc[1])
        #body.entity = entity
        body.data = entity
        entity.radius = radius
        entity.phys_shape = shape
        return shape

    def spawn_station(self, sector:core.Sector, x:float, y:float, resource:Optional[int]=None, entity_id:Optional[uuid.UUID]=None) -> core.Station:
        if resource is None:
            resource = self.r.integers(0, len(self.gamestate.production_chain.prices)-self.gamestate.production_chain.ranks[-1])

        assert resource < self.gamestate.production_chain.num_products

        station_radius = 300.

        #TODO: stations are static?
        #station_moment = pymunk.moment_for_circle(station_mass, 0, station_radius)
        station_body = self._phys_body()
        station = core.Station(np.array((x, y), dtype=np.float64), station_body, self.gamestate.production_chain.shape[0], self._gen_station_name(), entity_id=entity_id)
        station.resource = resource

        self._phys_shape(station_body, station, core.ObjectFlag.STATION, station_radius)

        sector.add_entity(station)

        return station

    def spawn_planet(self, sector:core.Sector, x:float, y:float, entity_id:Optional[uuid.UUID]=None) -> core.Planet:
        planet_radius = 1000.

        #TODO: stations are static?
        planet_body = self._phys_body()
        planet = core.Planet(np.array((x, y), dtype=np.float64), planet_body, self.gamestate.production_chain.shape[0], self._gen_planet_name(), entity_id=entity_id)
        planet.population = self.r.uniform(1e10*5, 1e10*15)

        self._phys_shape(planet_body, planet, core.ObjectFlag.PLANET, planet_radius)

        sector.add_entity(planet)

        return planet

    def spawn_ship(self, sector:core.Sector, ship_x:float, ship_y:float, v:Optional[npt.NDArray[np.float64]]=None, w:Optional[float]=None, theta:Optional[float]=None, default_order_fn:core.Ship.DefaultOrderSig=order_fn_null, entity_id:Optional[uuid.UUID]=None) -> core.Ship:

        #TODO: clean this up
        # set up physics stuff

        # soyuz 5000 - 10000kg
        # dragon capsule 4000kg
        # shuttle orbiter 78000kg
        ship_mass = 2e3

        # soyuz: 7-10m long
        # shuttle orbiter: 37m long
        # spacex dragon: 6.1m
        # spacex starship: 120m long
        ship_radius = 30.

        # one raptor: 1.81 MN
        # one SSME: 2.279 MN
        # OMS main engine: 26.7 kN
        # KTDU-80 main engine: 2.95 kN
        #max_thrust = 5e5
        # 5e5 translates to 250m/s^2 which is over 25 gs
        max_thrust = 2e5

        # one draco: 400 N (x16 on Dragon)
        # OMS aft RCS: 3.87 kN
        # KTDU-80 11D428A-16: 129.16 N (x16 on the Soyuz)
        # some speculation that starship thrusters can do 100-200 kN
        #max_fine_thrust = 5e3
        max_fine_thrust = 1.5e4

        # note about g-forces:
        # assuming circle of radius 30m, mass 2e3 kg
        # mass moment 18,000,000 kg m^2
        # centriptal acceleration = r * w^2
        # 1g at 30m with angular velocity of 0.57 rad/sec
        # 5000 * 30 N m can get 2e3kg, 30m circle up to half a g in 60 seconds
        # 10000 * 30 N m can get 2e3kg, 30m circle up to half a g in 30 seconds
        # 30000 * 30 N m can get 2e3kg, 30m circle up to half a g in 10 seconds
        # starting from zero
        # space shuttle doesn't exeed 3g during ascent
        max_torque = max_fine_thrust * 6 * ship_radius


        ship_body = self._phys_body(ship_mass, ship_radius)
        ship = core.Ship(np.array((ship_x, ship_y), dtype=np.float64), ship_body, self.gamestate.production_chain.shape[0], self._gen_ship_name(), entity_id=entity_id)

        self._phys_shape(ship_body, ship, core.ObjectFlag.SHIP, ship_radius)

        ship.mass = ship_mass
        ship.moment = ship_body.moment
        ship.radius = ship_radius
        ship.max_thrust = max_thrust
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
        ship.prepend_order(ship.default_order(self.gamestate))

        return ship

    def spawn_gate(self, sector: core.Sector, destination: core.Sector, entity_id:Optional[uuid.UUID]=None) -> core.TravelGate:

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

        body = self._phys_body()
        gate = core.TravelGate(destination, direction, np.array((x,y), dtype=np.float64), body, self.gamestate.production_chain.shape[0], self._gen_gate_name(destination), entity_id=entity_id)

        self._phys_shape(body, gate, core.ObjectFlag.GATE, gate_radius)

        sector.add_entity(gate)

        return gate

    def spawn_asteroid(self, sector: core.Sector, x:float, y:float, resource:int, amount:float, entity_id:Optional[uuid.UUID]=None) -> core.Asteroid:
        asteroid_radius = 100

        #TODO: stations are static?
        #station_moment = pymunk.moment_for_circle(station_mass, 0, station_radius)
        body = self._phys_body()
        asteroid = core.Asteroid(resource, amount, np.array((x,y), dtype=np.float64), body, self.gamestate.production_chain.shape[0], self._gen_asteroid_name(), entity_id=entity_id)

        self._phys_shape(body, asteroid, core.ObjectFlag.ASTEROID, asteroid_radius)

        sector.add_entity(asteroid)

        return asteroid

    def spawn_resource_field(self, sector: core.Sector, x: float, y: float, resource: int, total_amount: float, width: float=0., mean_per_asteroid: float=1e5, variance_per_asteroid: float=1e4) -> List[core.Asteroid]:
        """ Spawns a resource field centered on x,y.

        resource : the type of resource
        total_amount : the total amount of that resource in the field
        width : the "width" of the field (stdev of the normal distribution),
            default sector radius / 5
        mean_per_asteroid : mean of resources per asteroid, default 1e5
        stdev_per_asteroid : stdev of resources per asteroid, default 1e4
        """

        if total_amount < 0:
            return []

        field_center = np.array((x,y), dtype=np.float64)
        if width == 0.:
            width = sector.radius / 5

        # generate asteroids until we run out of resources
        asteroids = []
        while total_amount > 0:
            loc = self.r.normal(0, width, 2) + field_center
            amount = self.r.normal(mean_per_asteroid, variance_per_asteroid)
            if amount > total_amount:
                amount = total_amount

            asteroid = self.spawn_asteroid(sector, loc[0], loc[1], resource, amount)

            asteroids.append(asteroid)

            total_amount -= amount

        return asteroids

    def spawn_character(self, location:core.SectorEntity, balance:float=10e3) -> core.Character:
        character = core.Character(self._choose_portrait(), location, name=self._gen_character_name())
        character.balance = balance
        self.gamestate.add_character(character)
        return character

    def spawn_habitable_sector(self, x:float, y:float, entity_id:uuid.UUID, radius:float, sector_idx:int) -> core.Sector:
        pchain = self.gamestate.production_chain

        # compute how many raw resources are needed, transitively via
        # production chain, to build each good
        slices = [np.s_[pchain.ranks[0:i].sum():pchain.ranks[0:i+1].sum()] for i in range(len(pchain.ranks))]
        raw_needs = pchain.adj_matrix[slices[0], slices[0+1]]
        for i in range(1, len(slices)-1):
            raw_needs = raw_needs @ pchain.adj_matrix[slices[i], slices[i+1]]

        sector = core.Sector(np.array([x, y]), radius, cymunk.Space(), self._gen_sector_name(), entity_id=entity_id)
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
        num_stations = int((self.gamestate.production_chain.ranks.sum() - self.gamestate.production_chain.ranks[0])*2.5)
        resources_to_generate = raw_needs[:,RESOURCE_REL_STATION] *  self.r.uniform(num_stations, 2*num_stations)
        #resources_to_generate += raw_needs[:,RESOURCE_REL_SHIP] * 100
        #resources_to_generate += raw_needs[:,RESOURCE_REL_CONSUMER] * 100*100
        asteroids:Dict[int, List[core.Asteroid]] = {}
        for resource, amount in enumerate(resources_to_generate):
            self.logger.info(f'spawning resource fields for {amount} units of {resource} in sector {sector.short_id()}')
            # choose a number of fields to start off with
            # divide resource amount across them
            field_amounts = [amount]
            # generate a field for each one
            asteroids[resource] = []
            for field_amount in field_amounts:
                loc = self._gen_sector_location(sector)
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
            entity_loc = self._gen_sector_location(sector)
            for build_resource, amount in enumerate(raw_needs[:,RESOURCE_REL_STATION]):
                self.harvest_resources(sector, entity_loc[0], entity_loc[1], build_resource, amount)
            station = self.spawn_station(
                    sector, entity_loc[0], entity_loc[1], resource=resource)
            assets.append(station)

        # spend resources to build additional stations
        # consume resources to establish and support population

        # set up population according to production capacity
        entity_loc = self._gen_sector_location(sector)
        planet = self.spawn_planet(sector, entity_loc[0], entity_loc[1])
        assets.append(planet)

        # some factor mining ships for every refinery
        mining_ship_factor = 2.
        num_mining_ships = int(agent_goods[:,self.gamestate.production_chain.ranks.cumsum()[0]:self.gamestate.production_chain.ranks.cumsum()[1]].sum() * mining_ship_factor)

        self.logger.debug(f'adding {num_mining_ships} mining ships to sector {sector.short_id()}')
        mining_ships = set()
        for i in range(num_mining_ships):
            ship_x, ship_y = self._gen_sector_location(sector)
            ship = self.spawn_ship(sector, ship_x, ship_y, default_order_fn=order_fn_wait)
            assets.append(ship)
            mining_ships.add(ship)

        # some factor trading ships as there are station -> station trade routes
        trade_ship_factor = 1./3.
        trade_routes_by_good = ((self.gamestate.production_chain.adj_matrix > 0).sum(axis=1))
        num_trading_ships = int((trade_routes_by_good[np.newaxis,:] * agent_goods).sum() * trade_ship_factor)
        self.logger.debug(f'adding {num_trading_ships} trading ships to sector {sector.short_id()}')

        trading_ships = set()
        for i in range(num_trading_ships):
            ship_x, ship_y = self._gen_sector_location(sector)
            ship = self.spawn_ship(sector, ship_x, ship_y, default_order_fn=order_fn_wait)
            assets.append(ship)
            trading_ships.add(ship)

        sum_eta = 0.
        num_eta = 0
        for ship in sector.ships:
            for station in sector.stations:
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
        mean_ownership = peaked_bounded_random(self.r, mu_ownership, sigma_ownership, lb=min_ownership, ub=max_ownership)
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
                if isinstance(asset, core.Ship):
                    if asset in mining_ships:
                        character.add_agendum(agenda.MiningAgendum(ship=asset, character=character, gamestate=self.gamestate))
                    elif asset in trading_ships:
                        character.add_agendum(agenda.TradingAgendum(ship=asset, character=character, gamestate=self.gamestate))
                    else:
                        raise ValueError("got a ship that wasn't in mining_ships or trading_ships")
                elif isinstance(asset, core.Station):
                    character.add_agendum(agenda.StationManager(station=asset, character=character, gamestate=self.gamestate))
                elif isinstance(asset, core.Planet):
                    #TODO: what to do with planet assets?
                    pass
                else:
                    raise ValueError(f'got an asset of unknown type {asset}')

        self.gamestate.add_sector(sector, sector_idx)

        return sector

    def harvest_resources(self, sector: core.Sector, x: float, y: float, resource: int, amount: float) -> None:
        if amount <= 0:
            return

        asteroids = sector.asteroids[resource].copy()
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
            n_ranks:int=3,
            min_per_rank:Sequence[int]=(3,6,5), max_per_rank:Sequence[int]=(6,10,7),
            max_outputs:int=4, max_inputs:int=4,
            min_input_per_output:int=2, max_input_per_output:int=10,
            min_raw_price:float=1., max_raw_price:float=20.,
            min_markup:float=1.05, max_markup:float=2.5,
            min_final_inputs:int=3, max_final_inputs:int=5,
            min_final_prices:Sequence[float]=(1e6, 1e7, 1e5),
            max_final_prices:Sequence[float]=(3*1e6, 4*1e7, 3*1e5),
            sink_names:Sequence[str]=["ships", "stations", "consumers"]) -> core.ProductionChain:
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
        if len(sink_names) != len(min_final_prices):
            raise ValueError("sink_names and min_final_prices must be same length")

        # set up a rank of raw resources that mirrors the first product rank
        # these will feed 1:1 from raw resources to the first products
        ranks = np.pad(ranks, (1,0), mode="edge")

        total_nodes = np.sum(ranks)

        # generate production chain subject to target total value for each
        adj_matrix = np.zeros((total_nodes+len(min_final_prices),total_nodes+len(min_final_prices)))

        # set up 1:1 production from raw resources to first products
        adj_matrix[0:ranks[0], ranks[0]:ranks[0]+ranks[1]] = np.eye(ranks[0])

        # set up production for rest of products
        so_far = ranks[0]
        for (nodes_from, nodes_to) in zip(ranks[1:], ranks[2:]):
            target_edges = self.r.integers(
                np.max((nodes_from, nodes_to)),
                np.min((
                    nodes_from*nodes_to,
                    nodes_from*max_outputs,
                    nodes_to*max_inputs
                ))+1
            )
            target_weight = np.mean((min_input_per_output, max_input_per_output)) * target_edges
            rank_production = self._random_bipartite_graph(
                    nodes_from, nodes_to, target_edges,
                    np.min((nodes_to, max_outputs)),
                    np.min((nodes_from, max_inputs)),
                    target_weight,
                    min_input_per_output, max_input_per_output).round()
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
        final_production = self._random_bipartite_graph(
                ranks[-2], ranks[-1], target_edges,
                np.min((ranks[-1], max_outputs)),
                np.min((ranks[-2], max_final_inputs)),
                target_weight, min_input_per_output, max_input_per_output,
                min_in=min_final_inputs).round()

        # adjust weights to hit target prices
        final_prices = self.r.uniform(min_final_prices, max_final_prices)
        final_production = final_production / final_production.sum(axis=0) * final_prices
        adj_matrix[s_last_goods, s_final_products] = final_production.round()
        total_nodes = np.sum(ranks)

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
        adj_matrix[s_last_goods, s_final_products] = adj_matrix[s_last_goods, s_final_products].round()
        prices[s_final_products] = (np.vstack(prices[so_far-nodes_from:so_far]) * adj_matrix[s_last_goods, s_final_products]).sum(axis=0) * markup[s_final_products] # type: ignore

        #prices = prices.round()
        assert not np.any(np.isnan(prices))
        #TODO: this can fail because of the rounding we do with the prices I think
        # make sure that the prices are more than the cost to produce
        assert np.all(prices > (prices[:, np.newaxis] * adj_matrix).sum(axis=0))

        # set up production times and batch sizes
        # in one minute produce a batch of enough goods to produce a batch of
        # some number of the next items in the chain
        production_times = np.full((total_nodes,), 60.)
        batch_sizes = np.clip(3. * np.ceil(np.min(adj_matrix, axis=1, where=adj_matrix>0, initial=np.inf)), 1., 50)
        batch_sizes[-ranks[-1]:] = 1

        chain = core.ProductionChain()
        chain.ranks = ranks
        chain.adj_matrix = adj_matrix
        chain.markup = markup
        chain.prices = prices
        chain.sink_names = sink_names
        chain.production_times = production_times
        chain.batch_sizes = batch_sizes

        chain.initialize()


        for i, (price, name) in enumerate(zip(prices[s_final_products], sink_names), len(prices)-len(min_final_prices)):
            self.logger.info(f'price {name}:\t${price}')
        self.logger.info(f'total price:\t${prices[s_final_products].sum()}')

        return chain

    def generate_sectors(self,
            width:int=7, height:int=7,
            sector_radius:float=3e5,
            sector_radius_std:float=1e5*0.25,
            sector_edge_length:float=1e5*15,
            n_habitable_sectors:int=15,
            mean_habitable_resources:float=1e9,
            mean_uninhabitable_resources:float=1e7) -> None:
        # set up pre-expansion sectors, resources

        # compute the raw resource needs for each product sink
        pchain = self.gamestate.production_chain
        slices = [np.s_[pchain.ranks[0:i].sum():pchain.ranks[0:i+1].sum()] for i in range(len(pchain.ranks))]
        raw_needs = pchain.adj_matrix[slices[0], slices[0+1]]
        for i in range(1, len(slices)-1):
            raw_needs = raw_needs @ pchain.adj_matrix[slices[i], slices[i+1]]

        self.logger.info(f'raw needs {raw_needs}')

        # generate locations for all sectors
        #TODO: sectors should not collide (related to radius of each)
        sector_coords = self.r.uniform(-sector_radius*width*1e1, sector_radius*height*1e1, (width*height, 2))
        sector_ids = np.array([uuid.uuid4() for _ in range(len(sector_coords))])
        habitable_mask = np.zeros(len(sector_coords), bool)
        habitable_mask[self.r.choice(len(sector_coords), n_habitable_sectors, replace=False)] = 1

        sector_k = sector_radius**2/sector_radius_std**2
        sector_theta = sector_radius_std**2/sector_radius

        # choose habitable sectors
        # each of these will have more resources
        # implies a more robust and complete production chain
        # implies a bigger population
        for idx, entity_id, (x,y) in zip(np.argwhere(habitable_mask), sector_ids[habitable_mask], sector_coords[habitable_mask]):
            self.spawn_habitable_sector(x, y, entity_id, self.r.gamma(sector_k, sector_theta), idx[0])

        # set up non-habitable sectors
        for idx, entity_id, (x,y) in zip(np.argwhere(~habitable_mask), sector_ids[~habitable_mask], sector_coords[~habitable_mask]):
            sector = core.Sector(np.array([x, y]), self.r.gamma(sector_k, sector_theta), cymunk.Space(), self._gen_sector_name(), entity_id=entity_id)

            self.gamestate.add_sector(sector, idx[0])

        # set up connectivity between sectors
        distances = distance.squareform(distance.pdist(sector_coords))
        sector_edges = prims_mst(distances, self.r.integers(0, len(distances)))

        # add edges for nearby sectors
        #TODO: edges should not cross sectors
        for (i, source_id), (j, dest_id) in itertools.product(enumerate(sector_ids), enumerate(sector_ids)):
            if source_id == dest_id:
                continue
            if util.distance(self.gamestate.sectors[source_id].loc, self.gamestate.sectors[dest_id].loc) < sector_edge_length:
                sector_edges[i,j] = 1

        self.gamestate.update_edges(sector_edges, sector_ids)

        # add gates for the travel lanes
        #TODO: there's probably a clever way to get these indicies
        for (i, source_id), (j, dest_id) in itertools.product(enumerate(sector_ids), enumerate(sector_ids)):
            if sector_edges[i,j] == 1:
                self.spawn_gate(self.gamestate.sectors[source_id], self.gamestate.sectors[dest_id])

        #TODO: post-expansion decline
        # deplete resources at population centers
        # degrade production elements post-expansion

        #TODO: current-era
        # establish factions
        # establish post-expansion production elements and equipment
        # establish current-era characters and distribute roles

    def generate_universe(self) -> core.Gamestate:
        self.gamestate.random = self.r

        # generate a production chain
        production_chain = self.generate_chain()
        self.gamestate.production_chain = production_chain

        # generate sectors
        self.generate_sectors()

        return self.gamestate
