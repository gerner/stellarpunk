import sys
import logging

import ipdb
import numpy as np

import stellarpunk.util as util
import stellarpunk.core as core

#TODO: names: sectors, planets, stations, ships, characters, raw materials,
#   intermediate products, final products, consumer products, station products,
#   ship products

class UniverseGenerator:
    def __init__(self):
        self.logger = logging.getLogger(util.fullname(self))

        # random generator
        self.r = None

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

        if total_w < k*min_w or total_w > k*max_w:
            raise ValueError("total_w must be >= k*min_w and <= k*max_w")

        def choose_seq(n, k, min_deg, max_deg):
            """ Generates a sequence of integers from [0,n) of length k where
            each number occurs at least once and at most max_deg """

            nseq = np.zeros(n, dtype=int) + min_deg

            k_left = k-nseq.sum()
            while k_left > 0:
                v = self.r.choice(np.where(nseq < max_deg)[0], 1)[0]
                nseq[v] += 1
                k_left -= 1

            return nseq

        # prepare edge assignments between top and bottom
        nseq = choose_seq(n, k, min_out, max_out)
        mseq = choose_seq(m, k, min_in, max_in)

        stubs = [[v] * nseq[v] for v in range(0, n)]
        nstubs = [x for subseq in stubs for x in subseq]
        stubs = [[v] * mseq[v] for v in range(0, m)]
        mstubs = [x for subseq in stubs for x in subseq]

        assert len(nstubs) == len(mstubs)
        assert len(nstubs) == k

        self.r.shuffle(nstubs)
        self.r.shuffle(mstubs)

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

    def generate_chain(
            self,
            n_ranks=3,
            min_per_rank=(3,6,4), max_per_rank=(6,10,7),
            max_outputs=4, max_inputs=4,
            min_input_per_output=2, max_input_per_output=10,
            min_raw_price=1, max_raw_price=20,
            min_markup=1.05, max_markup=2.5,
            min_final_inputs=2, max_final_inputs=5,
            min_final_prices=(1e6, 1e7, 1e5),
            max_final_prices=(3*1e6, 4*1e7, 3*1e5)):
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

        total_nodes = np.sum(ranks)

        raw_price = self.r.uniform(min_raw_price, max_raw_price, ranks[0])

        # generate production chain for ship, station, consumer needs
        # subject to target total value for each
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
            target_weight = ((max_input_per_output - min_input_per_output)/2+min_input_per_output) * target_edges
            rank_production = self._random_bipartite_graph(
                    nodes_from, nodes_to, target_edges,
                    max_outputs, max_inputs,
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
                    ranks[-1]*max_outputs,
                    ranks[-1]*max_inputs
                ))
        )

        target_weight = ((max_input_per_output - min_input_per_output)/2+min_input_per_output) * target_edges
        final_production = self._random_bipartite_graph(
                ranks[-2], ranks[-1], target_edges,
                max_outputs, max_inputs,
                target_weight, min_input_per_output, max_input_per_output,
                min_in=min_final_inputs).round()

        # adjust weights to hit target prices
        final_prices = self.r.uniform(min_final_prices, max_final_prices)
        final_production = final_production / final_production.sum(axis=0) * final_prices
        adj_matrix[s_last_goods, s_final_products] = final_production.round()

        total_nodes = np.sum(ranks)
        markup = self.r.uniform(min_markup, max_markup, total_nodes)

        prices = np.pad(raw_price, (0, total_nodes-ranks[0])) * markup
        prices = prices.round()
        for (nodes_from, nodes_to), so_far in zip(zip(ranks, ranks[1:]), np.cumsum(ranks)[:-1]):
            # price of the next rank is the price of the prior rank times the
            # production matrix times the markup
            relevant_prod_matrix = adj_matrix[so_far-nodes_from:so_far, so_far:so_far+nodes_to]

            prices[so_far:so_far+nodes_to] = (np.reshape(prices[so_far-nodes_from:so_far], (nodes_from, 1)) * relevant_prod_matrix).sum(axis=0) * markup[so_far:so_far+nodes_to]

        # adjust final production weights to account for the prices of inputs
        adj_matrix[s_last_goods, s_final_products] /= np.vstack(prices[s_last_goods])
        adj_matrix[s_last_goods, s_final_products] = adj_matrix[s_last_goods, s_final_products].round()
        prices[s_final_products] = (np.vstack(prices[so_far-nodes_from:so_far]) * adj_matrix[s_last_goods, s_final_products]).sum(axis=0)

        chain = core.ProductionChain()
        chain.ranks = ranks
        chain.adj_matrix = adj_matrix
        chain.markup = markup
        chain.prices = prices.round()

        for i, price in enumerate(prices[s_final_products], len(prices)-len(min_final_prices)):
            self.logger.info(f'price {i}:\t${price}')
        self.logger.info(f'total price:\t${prices[s_final_products].sum()}')

        return chain

    def generate_sectors(self,
            game_state,
            width, height,
            n_habitable_sectors,
            mean_habitable_resources=1e9,
            mean_uninhabitable_resources=1e7):
        # set up pre-expansion sectors, resources

        # choose habitable sectors
        # each of these will have more resources
        # implies a more robust and complete production chain
        # implies a bigger population
        for i in range(n_habitable_sectors):
            x = self.r.integers(0, width)
            y = self.r.integers(0, height)

            sector = core.Sector()

            # habitable planet
            # plenty of resources
            # plenty of production

            sector.resources = self.r.chisquare(mean_habitable_resources)

            # set up production stations according to resources
            # every inhabited sector should have a 
            # more resources => more production stations, more complete production chain
            #TODO: set up production

            # set up population according to production capacity
            planet = core.Planet()
            planet.population = self.r.chisquare(sector.resources*10)
            sector.planets.append(planet)


            game_state.sectors[(x,y)] = sector

        # set up non-habitable sectors
        for x in range(width):
            for y in range(height):
                # skip inhabited sectors
                if (x,y) in game_state.sectors:
                    continue

                sector = core.Sector()
                sector.resources = self.r.chisquare(mean_uninhabitable_resources)
                game_state.sectors[(x,y)] = sector

        #TODO: post-expansion decline
        # deplete resources at population centers
        # degrade production elements post-expansion

        #TODO: current-era
        # establish factions
        # establish post-expansion production elements and equipment
        # establish current-era characters and distribute roles

    def generate_universe(self,
            width, height,
            n_habitable_sectors,
            seed=None,
            mean_habitable_resources=1e9,
            mean_uninhabitable_resources=1e7,
            production_chain=None):

        game_state = core.StellarPunk()

        self.r = np.random.default_rng(seed)
        game_state.random = self.r

        # generate a production chain
        if not production_chain:
            production_chain = self.generate_chain()
            production_chain.sink_names = ["ships", "stations", "consumers"]
        game_state.production_chain = production_chain

        # generate sectors
        self.generate_sectors(
                game_state,
                width, height,
                n_habitable_sectors,
                mean_habitable_resources, mean_uninhabitable_resources)

        return game_state

def main():
    with ipdb.launch_ipdb_on_exception():
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)
        logging.info("generating universe...")
        generator = UniverseGenerator()
        stellar_punk = generator.generate_universe(5, 5, 3)

        #logging.info("running simulation...")
        #stellar_punk.run()

        stellar_punk.production_chain.viz().render("production_chain", format="pdf")

        logging.info("done.")

if __name__ == "__main__":
    main()
