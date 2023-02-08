""" Stellarpunk production chain """

from typing import Sequence, Tuple

import graphviz # type: ignore

import numpy as np
import numpy.typing as npt


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
