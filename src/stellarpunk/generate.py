import logging
import itertools
from typing import Optional

import numpy as np
import numpy.typing as npt
import pymunk

from stellarpunk import util, core, orders

#TODO: names: sectors, planets, stations, ships, characters, raw materials,
#   intermediate products, final products, consumer products, station products,
#   ship products

class GenerationListener:
    def production_chain_complete(self, production_chain):
        pass

    def sectors_complete(self, sectors):
        pass

def order_fn_null(ship, gamestate):
    return core.Order(ship, gamestate)

def order_fn_wait(ship, gamestate):
    return orders.WaitOrder(ship, gamestate)

def order_fn_goto_random_station(ship, gamestate):
    station = gamestate.random.choice(ship.sector.stations)
    loc, arrival_distance = orders.GoToLocation.choose_destination(gamestate, ship.loc, station)
    return orders.GoToLocation(station.loc.copy(), ship, gamestate)

class UniverseGenerator:
    def __init__(self, gamestate, seed=None, listener=None):
        self.logger = logging.getLogger(util.fullname(self))

        self.gamestate = gamestate

        # random generator
        self.r = np.random.default_rng(seed)

        if not listener:
            # if no listener, set up a no-op listener
            self.listener = GenerationListener()
        else:
            self.listener = listener

        self.parallel_max_edges_tries = 10000

    def _old_adj_algo(self):
        adj_matrix = np.zeros((total_nodes,total_nodes))

        so_far = 0
        # set up inputs needed per output at each rank
        for (nodes_from, nodes_to) in zip(ranks, ranks[1:]):
            prior_rank = np.arange(so_far, so_far+nodes_from)
            next_rank_max_inputs = np.min((max_inputs, nodes_from+1))
            self.logger.info(f'{min_inputs}, {next_rank_max_inputs}')
            # for each node in the next rank
            for t in range(so_far+nodes_from, so_far+nodes_from+nodes_to):
                # choose a number of inputs
                n_inputs = self.r.integers(min_inputs, next_rank_max_inputs)
                inputs = self.r.choice(prior_rank, n_inputs, replace=False)
                for s in inputs:
                    adj_matrix[s, t] = self.r.uniform(min_input_per_output, max_input_per_output)

            # make sure every product from the prior rank is used at least once
            for s in range(so_far, so_far+nodes_from):
                if np.count_nonzero(adj_matrix[s,]) == 0:
                    #TODO:do this by either adding a new edge or reassigning an existing one
                    t = np.argmin((adj_matrix[:,so_far+nodes_from:so_far+nodes_from+nodes_to] > 0).sum(axis=0))+so_far+nodes_from
                    t_input_count = (adj_matrix[:,t] > 0).sum()
                    if t_input_count > next_rank_max_inputs:
                        raise Exception("ohnoes")
                    self.logging.info(f'{s} is not an input, assigning to {t} with {t_input_count}')
                    adj_matrix[s, t] = self.r.uniform(min_input_per_output, max_input_per_output)

            so_far += nodes_from

    def _random_bipartite_graph(
            self,
            n, m, k,
            max_out, max_in,
            total_w, min_w, max_w,
            min_out=1, min_in=1):
        """ Creates a bipartite, weighted graph according to model parameters.

        n: number of top nodes
        m: number of bottom nodes
        k: number of edges
        max_out: max out degree on top nodes
        max_in: max in degree on bottom nodes
        total_w: sum of weights on edges from top to bottom
        min_w: min weight on edges
        max_w: max weight on edges
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

        def choose_seq(n, k, min_deg, max_deg):
            """ Generates a sequence of integers from [0,n) of length k where
            each number occurs at least once and at most max_deg """

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

    def _gen_sector_name(self):
        return "Some Sector"

    def _gen_sector_location(self, sector, unoccupied=True):
        loc = self.r.normal(0, 1, 2) * sector.radius
        while unoccupied and sector.is_occupied(*loc):
            loc = self.r.normal(0, 1, 2) * sector.radius

        return loc

    def _gen_planet_name(self):
        return "Magusan"

    def _gen_station_name(self):
        return "Some Station"

    def _gen_ship_name(self):
        return "Some Ship"

    def _gen_character_name(self):
        return "Somebody"

    def _gen_asteroid_name(self):
        return "Asteroid X"

    def spawn_station(self, sector, x, y, resource=None):
        if resource is None:
            resource = self.r.uniform(0, len(pchain.prices)-pchain.ranks[-1])

        station_radius = 300.

        #TODO: stations are static?
        #station_moment = pymunk.moment_for_circle(station_mass, 0, station_radius)
        station_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        station = core.Station(np.array((x, y), dtype=np.float64), station_body, self._gen_station_name())
        station.loc.flags.writeable = False
        station.resource = resource

        station_shape = pymunk.Circle(station_body, station_radius)
        station_shape.friction=0.1
        station_shape.collision_type = station.object_type
        station_shape.filter = pymunk.ShapeFilter(categories=core.ObjectFlag.STATION)
        station_body.position = (station.loc[0], station.loc[1])
        station_body.entity = station
        station.radius = station_radius

        sector.add_entity(station)

        return station

    def spawn_planet(self, sector, x, y):
        planet_radius = 1000.

        #TODO: stations are static?
        planet_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        planet = core.Planet(np.array((x, y), dtype=np.float64), planet_body, self._gen_planet_name())
        planet.loc.flags.writeable = False
        planet.population = self.r.uniform(sector.resources*5, sector.resources*15)

        planet_shape = pymunk.Circle(planet_body, planet_radius)
        planet_shape.friction=0.1
        planet_shape.collision_type = planet.object_type
        planet_shape.filter = pymunk.ShapeFilter(categories=core.ObjectFlag.PLANET)
        planet_body.position = (planet.loc[0], planet.loc[1])
        planet_body.entity = planet
        planet.radius = planet_radius

        sector.add_entity(planet)

        return planet

    def spawn_ship(self, sector:core.Sector, ship_x:float, ship_y:float, v:Optional[npt.NDArray[np.float64]]=None, w:Optional[float]=None, theta:Optional[float]=None, default_order_fn=order_fn_null) -> core.Ship:

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
        max_thrust = 5e5

        # one draco: 400 N (x16 on Dragon)
        # OMS aft RCS: 3.87 kN
        # KTDU-80 11D428A-16: 129.16 N (x16 on the Soyuz)
        # some speculation that starship thrusters can do 100-200 kN
        max_fine_thrust = 5e3

        # note about g-forces:
        # assuming circle of radius 30m, mass 2e3 kg
        # mass moment 18,000,000 kg m^2
        # centriptal acceleration = r * w^2
        # 1g at 30m with angular acceleration of 0.57 rad/sec
        # 5000 * 30 N m can get 2e3kg, 30m circle up to half a g in 60 seconds
        # 10000 * 30 N m can get 2e3kg, 30m circle up to half a g in 30 seconds
        # 30000 * 30 N m can get 2e3kg, 30m circle up to half a g in 10 seconds
        # starting from zero
        # space shuttle doesn't exeed 3g during ascent
        max_torque = max_fine_thrust * 6 * ship_radius

        ship_moment = pymunk.moment_for_circle(ship_mass, 0, ship_radius)

        ship_body = pymunk.Body(ship_mass, ship_moment)
        ship = core.Ship(np.array((ship_x, ship_y), dtype=np.float64), ship_body, self._gen_ship_name())

        ship_shape = pymunk.Circle(ship_body, ship_radius)
        ship_shape.friction=0.1
        ship_shape.collision_type = ship.object_type
        ship_shape.filter = pymunk.ShapeFilter(categories=core.ObjectFlag.SHIP)
        ship_body.position = ship.loc[0], ship.loc[1]
        ship_body.entity = ship

        ship.mass = ship_mass
        ship.moment = ship_moment
        ship.radius = ship_radius
        ship.max_thrust = max_thrust
        ship.max_fine_thrust = max_fine_thrust
        ship.max_torque = max_torque

        if v is None:
            v = (self.r.normal(0, 50, 2))
        ship_body.velocity = pymunk.vec2d.Vec2d(*v)
        ship_body.angle = ship_body.velocity.angle

        if theta is not None:
            ship_body.angle = theta

        if w is None:
            ship_body.angular_velocity = self.r.normal(0, 0.08)
        else:
            ship_body.angular_velocity = w

        ship.velocity = np.array(ship_body.velocity, dtype=np.float64)
        ship.angular_velocity = ship_body.angular_velocity
        ship.angle = ship_body.angle
        sector.add_entity(ship)

        ship.default_order_fn = default_order_fn

        return ship

    def spawn_asteroid(self, sector: core.Sector, x:float, y:float, resource:int, amount:float) -> core.Asteroid:
        asteroid_radius = 100

        #TODO: stations are static?
        #station_moment = pymunk.moment_for_circle(station_mass, 0, station_radius)
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        asteroid = core.Asteroid(resource, amount, np.array((x,y), dtype=np.float64), body, self._gen_asteroid_name())
        asteroid.loc.flags.writeable = False
        shape = pymunk.Circle(body, asteroid_radius)
        shape.friction=0.1
        shape.collision_type = asteroid.object_type
        shape.filter = pymunk.ShapeFilter(categories=core.ObjectFlag.ASTEROID)
        body.position = (asteroid.loc[0], asteroid.loc[1])
        body.entity = asteroid
        asteroid.radius = asteroid_radius

        sector.add_entity(asteroid)

        return asteroid

    def spawn_resource_field(self, sector: core.Sector, x: float, y: float, resource: int, total_amount: float, width: float=None, mean_per_asteroid: float=1e5, variance_per_asteroid: float=1e4) -> list[core.Asteroid]:
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
        if not width:
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

            amount_to_mine = min(amount, asteroid.amount)
            asteroid.amount -= amount_to_mine
            amount -= amount_to_mine

            # if we've used up this one, remove it from the sector
            # set its prob to zero and renormalize
            if asteroid.amount == 0:
                sector.remove_entity(asteroid)
                asteroid_probs[i] = 0
                asteroid_probs = asteroid_probs/asteroid_probs.sum()

    def generate_chain(
            self,
            n_ranks=3,
            min_per_rank=(3,6,5), max_per_rank=(6,10,7),
            max_outputs=4, max_inputs=4,
            min_input_per_output=2, max_input_per_output=10,
            min_raw_price=1, max_raw_price=20,
            min_markup=1.05, max_markup=2.5,
            min_final_inputs=3, max_final_inputs=5,
            min_final_prices=(1e6, 1e7, 1e5),
            max_final_prices=(3*1e6, 4*1e7, 3*1e5),
            sink_names=["ships", "stations", "consumers"]):
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

        total_nodes = 0
        if isinstance(min_per_rank, int):
            if not isinstance(max_per_rank, int):
                raise ValueError("min_per_rank and max_per_rank must both be ints or sequences")
            ranks = self.r.integers(min_per_rank, max_per_rank+1, n_ranks)
        elif isinstance(min_per_rank, (list, tuple, np.ndarray)):
            if not isinstance(max_per_rank, (list, tuple, np.ndarray)):
                raise ValueError("min_per_rank and max_per_rank must both be ints or sequences")
            if len(min_per_rank) != len(max_per_rank):
                raise ValueError("min_per_rank and max_per_rank must be the same length")
            ranks = self.r.integers(min_per_rank, max_per_rank)
        else:
            raise ValueError("min_per_rank and max_per_rank must both be ints or sequences")

        if len(min_final_prices) != len(max_final_prices):
            raise ValueError("min and max final prices must be same length")
        if len(sink_names) != len(min_final_prices):
            raise ValueError("sink_names and min_final_prices must be same length")

        total_nodes = np.sum(ranks)


        # generate production chain subject to target total value for each
        adj_matrix = np.zeros((total_nodes+len(min_final_prices),total_nodes+len(min_final_prices)))
        so_far = 0
        for (nodes_from, nodes_to) in zip(ranks, ranks[1:]):
            target_edges = self.r.integers(
                np.max((nodes_from, nodes_to)),
                np.min((
                    nodes_from*nodes_to,
                    nodes_from*max_outputs,
                    nodes_to*max_inputs
                ))
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
                ))
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

        # adjust final production weights to account for the prices of inputs
        adj_matrix[s_last_goods, s_final_products] /= np.vstack(prices[s_last_goods])
        adj_matrix[s_last_goods, s_final_products] = adj_matrix[s_last_goods, s_final_products].round()
        prices[s_final_products] = (np.vstack(prices[so_far-nodes_from:so_far]) * adj_matrix[s_last_goods, s_final_products]).sum(axis=0)

        prices = prices.round()
        assert not np.any(np.isnan(prices))

        chain = core.ProductionChain()
        chain.ranks = ranks
        chain.adj_matrix = adj_matrix
        chain.markup = markup
        chain.prices = prices
        chain.sink_names = sink_names

        for i, (price, name) in enumerate(zip(prices[s_final_products], sink_names), len(prices)-len(min_final_prices)):
            self.logger.info(f'price {name}:\t${price}')
        self.logger.info(f'total price:\t${prices[s_final_products].sum()}')

        return chain

    def generate_sectors(self,
            width=6, height=6,
            sector_radius=1e5,
            n_habitable_sectors=5,
            mean_habitable_resources=1e9,
            mean_uninhabitable_resources=1e7):
        # set up pre-expansion sectors, resources

        RESOURCE_REL_SHIP = 0
        RESOURCE_REL_STATION = 1
        RESOURCE_REL_CONSUMER = 2

        # compute the raw resource needs for each product sink
        pchain = self.gamestate.production_chain
        slices = [np.s_[pchain.ranks[0:i].sum():pchain.ranks[0:i+1].sum()] for i in range(len(pchain.ranks))]
        raw_needs = pchain.adj_matrix[slices[0], slices[0+1]]
        for i in range(1, len(slices)-1):
            raw_needs = raw_needs @ pchain.adj_matrix[slices[i], slices[i+1]]

        self.logger.info(f'raw needs {raw_needs}')

        # choose habitable sectors
        # each of these will have more resources
        # implies a more robust and complete production chain
        # implies a bigger population
        habitable_coordinates = self.r.choice(
                list(itertools.product(range(width), range(height))),
                n_habitable_sectors,
                replace=False)
        for x,y in habitable_coordinates:
            sector = core.Sector(x, y, sector_radius, self._gen_sector_name())
            self.logger.info(f'generating habitable sector {sector.name} at ({x}, {y})')
            sector.space = pymunk.Space()

            # habitable planet
            # plenty of resources
            # plenty of production

            #TODO: resources we get should be enough to build out a full
            #production chain, plus some more in neighboring sectors, plus
            #fleet of ships, plus support population for a long time (from
            #expansion through the fall)
            sector.resources = self.r.uniform(
                    mean_habitable_resources/2,
                    mean_habitable_resources*1.5,
                    pchain.ranks[0])

            self.logger.info(f'starting resources: {sector.resources}')

            #TODO: set up resource fields
            # random number of fields per resource
            # random sizes
            # random allocation to each that sums to desired total
            num_stations = len(pchain.prices)-pchain.ranks[-1]
            resources_to_generate = raw_needs[:,RESOURCE_REL_STATION] *  self.r.uniform(num_stations, 2*num_stations)
            #resources_to_generate += raw_needs[:,RESOURCE_REL_SHIP] * 100
            #resources_to_generate += raw_needs[:,RESOURCE_REL_CONSUMER] * 100*100
            asteroids = {}
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


            #TODO: production and population
            # set up production stations according to resources
            # every inhabited sector should have a complete production chain
            for i in range(len(pchain.prices)-pchain.ranks[-1]):
                entity_loc = self._gen_sector_location(sector)

                # deplete enough resources from asteroids to pay for this station
                for resource, amount in enumerate(raw_needs[:,RESOURCE_REL_STATION]):
                    self.harvest_resources(sector, entity_loc[0], entity_loc[1], resource, amount)
                self.spawn_station(sector, entity_loc[0], entity_loc[1], resource=i)

                sector.resources -= raw_needs[:,RESOURCE_REL_STATION]
            # spend resources to build additional stations
            # consume resources to establish and support population

            # set up population according to production capacity
            entity_loc = self._gen_sector_location(sector)
            self.spawn_planet(sector, entity_loc[0], entity_loc[1])

            self.logger.info(f'ending resources: {sector.resources} ending entities: {len(sector.entities)}')

            self.gamestate.sectors[(x,y)] = sector

        # set up non-habitable sectors
        for x in range(width):
            for y in range(height):
                # skip inhabited sectors
                if (x,y) in self.gamestate.sectors:
                    continue

                sector = core.Sector(x, y, sector_radius, self._gen_sector_name())
                sector.space = pymunk.Space()

                sector.resources = self.r.uniform(
                        mean_uninhabitable_resources/2,
                        mean_uninhabitable_resources*1.5,
                        pchain.ranks[0])
                self.gamestate.sectors[(x,y)] = sector

        #TODO: post-expansion decline
        # deplete resources at population centers
        # degrade production elements post-expansion

        #TODO: current-era
        # establish factions
        # establish post-expansion production elements and equipment
        # establish current-era characters and distribute roles

        # quick hack to populate some ships
        for x,y in habitable_coordinates:
            sector = self.gamestate.sectors[(x,y)]
            num_ships = self.r.integers(15,35)
            self.logger.debug(f'adding {num_ships} ships to sector {sector.short_id()}')
            for i in range(num_ships):
                ship_x, ship_y = self._gen_sector_location(sector)
                self.spawn_ship(sector, ship_x, ship_y, default_order_fn=order_fn_goto_random_station)

    def generate_universe(self):
        self.gamestate.random = self.r

        # generate a production chain
        production_chain = self.generate_chain()
        self.gamestate.production_chain = production_chain
        self.listener.production_chain_complete(production_chain)

        # generate sectors
        self.generate_sectors()

        self.listener.sectors_complete(self.gamestate.sectors)

        return self.gamestate
