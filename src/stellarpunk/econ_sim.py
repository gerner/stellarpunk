""" Tool to run simple simulation of a market. """

from __future__ import annotations

import sys
import os
import io
import logging
import contextlib
import warnings
import typing
from typing import TextIO, BinaryIO, Optional, Tuple, Sequence, Type, Any
from types import TracebackType

import numpy as np
import numpy.typing as npt
import pandas as pd # type: ignore
from scipy import sparse # type: ignore
import msgpack # type: ignore
import tqdm # type: ignore

from stellarpunk import util, core, generate, serialization

# sometimes we're willing to manufacture a very small amount of cash to avoid
# precision errors
PRICE_EPS = 1e-05

# setup:
# production chain (as in generate.py/sim.py/core.py) defines how goods are
# used to create other goods
# a bunch of agents each play the role of producing one good, have inventory
# simulate in rounds where agents will trade, trying to form a market
# at the start of the round agents at the base of chain will get resources at a
#   fixed price
# and agents at tips of the chain will sell resources at a fixed price

# measures/questions:
# are agents profitable? (and how profitable)
# how are market setups like number of agents in each good related to agent profitability?
# what's the surplus of the system? (total and distribution)
# how do good prices change over time?
# how much do invdividual trade prices vary from global average
# how does surplus change over time? (total and spread, say IQR or stdev)
# how many goods are produced? (total and which are outliers)

@typing.no_type_check
def _df_from_spmatrix(data:Any, index:Any=None, columns:Optional[Sequence[Any]]=None, fill_values:Optional[Any]=None) -> pd.DataFrame:
    """ Taken from https://github.com/pandas-dev/pandas/blob/5c66e65d7b9fef47ccb585ce2fd0b3ea18dc82ea/pandas/core/arrays/sparse/accessor.py 

    modified to allow setting fill_values """

    from pandas._libs.sparse import IntIndex # type: ignore

    from pandas import DataFrame # type: ignore

    data = data.tocsc()
    index, columns = DataFrame.sparse._prep_index(data, index, columns)
    n_rows, n_columns = data.shape
    # We need to make sure indices are sorted, as we create
    # IntIndex with no input validation (i.e. check_integrity=False ).
    # Indices may already be sorted in scipy in which case this adds
    # a small overhead.
    data.sort_indices()
    indices = data.indices
    indptr = data.indptr
    array_data = data.data
    arrays = []

    if fill_values is None:
        fill_values = [0] * n_columns
    elif not isinstance(fill_values, Sequence):
        fill_values = [fill_values] * n_columns

    for i, fill_value in zip(range(n_columns), fill_values):
        dtype = pd.SparseDtype(array_data.dtype, fill_value)
        sl = slice(indptr[i], indptr[i + 1])
        idx = IntIndex(n_rows, indices[sl], check_integrity=False)
        arr = pd.arrays.SparseArray._simple_new(array_data[sl], idx, dtype)
        arrays.append(arr)
    return DataFrame._from_arrays(
        arrays, columns=columns, index=index, verify_integrity=False
    )

def read_tick_log_to_df(f:BinaryIO, index_name:Optional[str]=None, column_names:Optional[Sequence[str]]=None, fill_values:Optional[Any]=None) -> pd.DataFrame:
    reader = serialization.TickMatrixReader(f)
    matrixes = []
    row_count = 0
    col_count = 0
    ticks = []
    while (ret := reader.read()) is not None:
        tick, m = ret
        if row_count > 0:
            rows = m.shape[0]
            if len(m.shape) == 1:
                cols = 1
            else:
                cols = m.shape[1]
            if (row_count, col_count) != (rows, cols):
                raise ValueError(f'expected each matrix to have same shape {(row_count, col_count)} vs {m.shape}')
        else:
            assert col_count == 0
            row_count = m.shape[0]
            if len(m.shape) == 1:
                col_count = 1
            else:
                col_count = m.shape[1]
        if col_count == 1:
            matrixes.append(sparse.csc_array(m[:,np.newaxis]))
        else:
            matrixes.append(sparse.csc_array(m))
        ticks.append(np.full((row_count,), tick))

    df = _df_from_spmatrix(sparse.vstack(matrixes), columns=column_names, fill_values=fill_values)

    df["tick"] = pd.Series(np.concatenate(ticks))
    df.index = pd.Series(np.tile(np.arange(row_count), len(matrixes)))
    if index_name is not None:
        df.index.set_names(index_name, inplace=True)
    df.set_index("tick", append=True, inplace=True)

    return df

class EconomyDataLogger(contextlib.AbstractContextManager):
    def __init__(self, enabled:bool=True, logdir:str="/tmp/", buffersize:int=4*1024) -> None:
        self.enabled = enabled
        self.logdir = logdir
        self.buffersize = buffersize

        self.transaction_log:TextIO = None #type:ignore[assignment]
        self.inventory_log:serialization.TickMatrixWriter = None #type:ignore[assignment]
        self.balance_log:serialization.TickMatrixWriter = None #type:ignore[assignment]
        self.buy_prices_log:serialization.TickMatrixWriter = None #type:ignore[assignment]
        self.buy_budget_log:serialization.TickMatrixWriter = None #type:ignore[assignment]
        self.max_buy_prices_log:serialization.TickMatrixWriter = None #type:ignore[assignment]
        self.sell_prices_log:serialization.TickMatrixWriter = None #type:ignore[assignment]
        self.min_sell_prices_log:serialization.TickMatrixWriter = None #type:ignore[assignment]
        self.production_efficiency_log:serialization.TickMatrixWriter = None #type:ignore[assignment]

        self.exit_stack:contextlib.ExitStack = contextlib.ExitStack()
        self.sim:EconomySimulation = None #type: ignore[assignment]

    def _open_txt_log(self, name:str) -> TextIO:
        return self.exit_stack.enter_context(open(os.path.join(self.logdir, f'{name}.log'), "wt", self.buffersize))

    def _open_bin_log(self, name:str) -> BinaryIO:
        return self.exit_stack.enter_context(open(os.path.join(self.logdir, f'{name}.log'), "wb", self.buffersize))

    def __enter__(self) -> EconomyDataLogger:
        """ Opens underlying files in a way that they will close on exit. """

        if self.enabled:
            self.transaction_log = self._open_txt_log("transactions")
            self.inventory_log = serialization.TickMatrixWriter(self._open_bin_log("inventory"))
            self.balance_log = serialization.TickMatrixWriter(self._open_bin_log("balance"))
            self.buy_prices_log = serialization.TickMatrixWriter(self._open_bin_log("buy_prices"))
            self.buy_budget_log = serialization.TickMatrixWriter(self._open_bin_log("buy_budget"))
            self.max_buy_prices_log = serialization.TickMatrixWriter(self._open_bin_log("max_buy_prices"))
            self.sell_prices_log = serialization.TickMatrixWriter(self._open_bin_log("sell_prices"))
            self.min_sell_prices_log = serialization.TickMatrixWriter(self._open_bin_log("min_sell_prices"))
            self.production_efficiency_log = serialization.TickMatrixWriter(self._open_bin_log("production_efficiency"))

        return self

    def __exit__(self, exc_type:Optional[Type[BaseException]], exc_value:Optional[BaseException], traceback:Optional[TracebackType]) -> Optional[bool]:
        """ Closes underlying log files. """
        self.exit_stack.close()
        return None

    def initialize(self, sim:EconomySimulation) -> None:
        self.sim = sim
        if self.enabled:
            with open(os.path.join(self.logdir, "agent_goods.log"), "wb") as agent_goods_log:
                agent_goods_log.write(msgpack.packb(self.sim.agent_goods, default=serialization.encode_matrix))

    def end_simulation(self) -> None:
        if self.enabled:
            with open(os.path.join(self.logdir, "production_chain.log"), "wb") as production_chain_log:
                production_chain_log.write(serialization.save_production_chain(self.sim.gamestate.production_chain))

    def produce_goods(self, goods_produced:npt.NDArray[np.float64]) -> None:
        if self.enabled:
            self.production_efficiency_log.write(
                self.sim.ticks,
                np.divide(
                    goods_produced,
                    self.sim.batch_sizes,
                    where=self.sim.batch_sizes > 0,
                    out=np.zeros((self.sim.num_agents, self.sim.num_products))
                )
            )

    def transact(self, diff:float, product_id:int, seller:int, buyer:int, price:float, sale_amount:float) -> None:
        if self.enabled:
            self.transaction_log.write(f'{self.sim.ticks}\t{seller}\t{buyer}\t{product_id}\t{sale_amount}\t{price}\n')

    def start_trading(self) -> None:
        if self.enabled:
            self.buy_prices_log.write(self.sim.ticks, self.sim.buy_prices)
            self.buy_budget_log.write(self.sim.ticks, self.sim.buy_budget)
            self.sell_prices_log.write(self.sim.ticks, self.sim.sell_prices)
            self.max_buy_prices_log.write(self.sim.ticks, self.sim.max_buy_prices)
            self.min_sell_prices_log.write(self.sim.ticks, self.sim.min_sell_prices)

    def end_trading(self) -> None:
        if self.enabled:
            self.balance_log.write(self.sim.ticks, self.sim.balance)
            self.inventory_log.write(self.sim.ticks, self.sim.inventory)

class EconomySimulation:
    def __init__(self, data_logger:EconomyDataLogger) -> None:
        self.logger = logging.getLogger(util.fullname(self))
        self.data_logger = data_logger

        self.ticks = 0

        self.gamestate:core.Gamestate = None #type: ignore[assignment]
        self.num_agents = 0
        self.num_products = 0

        # indicator matrix for goods each agent produces/sells
        self.agent_goods = np.zeros((self.num_agents, self.num_products))
        # indicator matrix for goods each agent wants to buy (inputs)
        self.buy_interest = np.zeros((self.num_agents, self.num_products))

        # the size of one production batch for each agent and each good
        self.batch_sizes = np.zeros((self.num_agents, self.num_products))

        # number of inputs each agent needs to produce one unit of outputs
        self.production_goods = np.zeros((self.num_agents, self.num_products))

        # the number of goods each agent has
        self.inventory = np.zeros((self.num_agents, self.num_products))
        self.last_inventory = self.inventory.copy()
        # how much money each agent has
        self.balance = np.zeros((self.num_agents, ))

        # price expectations
        self.buy_prices = np.zeros((self.num_agents, self.num_products))
        self.buy_budget = np.zeros((self.num_agents, self.num_products))
        self.sell_prices = np.zeros((self.num_agents, self.num_products))

        self.last_buys = np.zeros((self.num_agents, self.num_products))
        self.last_buy_prices = np.zeros((self.num_agents, self.num_products))
        self.last_buy_pudget = np.zeros((self.num_agents, self.num_products))
        self.last_sell_prices = np.zeros((self.num_agents, self.num_products))
        self.last_sales_value = np.zeros((self.num_agents, self.num_products))

        # price min/max
        self.min_buy_prices = np.zeros((self.num_agents, self.num_products))
        self.max_buy_prices = np.zeros((self.num_agents, self.num_products))
        self.min_sell_prices = np.zeros((self.num_agents, self.num_products))
        self.max_sell_prices = np.zeros((self.num_agents, self.num_products))

        # estimates each agent has of market prices (VEMA)
        self.price_vema_alpha = 2./(52.+1.)
        self.input_value_estimates = np.zeros((self.num_agents, self.num_products))
        self.input_volume_estimates = np.zeros((self.num_agents, self.num_products))
        self.output_value_estimates = np.zeros((self.num_agents, self.num_products))
        self.output_volume_estimates = np.zeros((self.num_agents, self.num_products))

        # number of ticks since a sale/buy
        self.no_sale_ticks = np.zeros((self.num_agents, self.num_products), dtype=int)
        self.no_buy_ticks = np.zeros((self.num_agents, self.num_products), dtype=int)
        # number of ticks we've been at max/min price and have failed to
        # conduct a necessary transaction
        self.cannot_buy_ticks = np.zeros((self.num_agents, self.num_products), dtype=int)
        self.cannot_sell_ticks = np.zeros((self.num_agents, self.num_products), dtype=int)

    def initialize(self,
            num_agents:int=-1,
            gamestate:Optional[core.Gamestate]=None,
            production_chain:Optional[core.ProductionChain]=None) -> None:
        self.logger.info("generating universe...")

        # set up the production chain
        if gamestate is None:
            self.gamestate = core.Gamestate()
        else:
            self.gamestate = gamestate

        if production_chain is None:
            generator = generate.UniverseGenerator(self.gamestate)
            self.gamestate.production_chain = generator.generate_chain(
                #n_ranks=1,
                #min_per_rank=(2,),
                #max_per_rank=(2,),
                #min_final_inputs=1,
            )
            self.gamestate.production_chain.viz().render("/tmp/production_chain", format="pdf")
        else:
            self.gamestate.production_chain = production_chain

        num_products = self.gamestate.production_chain.adj_matrix.shape[0]

        if num_agents < 0:
            num_agents = num_products
        elif num_agents < num_products:
            raise ValueError(f'{num_agents=} must be >= {num_products=}')

        self.num_agents = num_agents

        self.num_products = num_products

        # set up agents, at least one in every good, the rest can be random
        self.agent_goods = np.zeros((self.num_agents, self.num_products))
        #TODO: how to choose resource each agent is involved in?
        self.agent_goods[0:self.num_products,0:self.num_products] = np.eye(self.num_products)
        self.agent_goods[np.arange(self.num_agents)[self.num_products:,None],self.gamestate.random.uniform(size=(self.num_agents-self.num_products, self.num_products)).argsort(1)[:,:1]] = 1


        # make sure only one product per agent (for now, we might want to lift
        # this simplifying restriction in the future...)
        assert np.all(self.agent_goods.sum(axis=1) == 1)
        assert np.all(self.agent_goods.sum(axis=1) == 1)

        self.batch_sizes = self.gamestate.production_chain.batch_sizes[np.newaxis,:] * self.agent_goods

        self.production_goods = (self.agent_goods @ self.gamestate.production_chain.adj_matrix.T)

        #self.inventory = np.zeros((self.num_agents, self.num_products))
        self.inventory = self.batch_sizes.copy() * 52
        self.last_inventory = self.inventory.copy()
        self.max_inventory = 1e3

        #TODO: set up initial balance, inventory, prices
        self.balance = ((
                self.agent_goods * (self.gamestate.production_chain.prices * self.gamestate.production_chain.batch_sizes)[np.newaxis,:]
            ).sum(axis=1) * 50
        )
        self.inventory = np.zeros((self.num_agents, self.num_products))

        # vector indicating agent interest in buying
        self.buy_interest = (self.agent_goods @ (self.gamestate.production_chain.adj_matrix.T > 0).astype(int) > 0).astype(int)
        sell_interest = self.agent_goods

        self.buy_prices = self.buy_interest * self.gamestate.production_chain.prices[np.newaxis, :]
        self.buy_budget = np.zeros((self.num_agents, self.num_products))
        self.sell_prices = sell_interest * self.gamestate.production_chain.prices[np.newaxis, :]
        self.sell_prices[(1-sell_interest) > 0] = np.inf

        self.last_buys = np.zeros((self.num_agents, self.num_products))
        self.last_buy_budget = np.zeros((self.num_agents, self.num_products))
        self.last_buy_prices = self.buy_prices
        self.last_sell_prices = self.sell_prices
        self.last_sales_value = np.zeros((self.num_agents, self.num_products))

        #TODO: these should be set to guarantee break-even
        # max buy prices set to markup * base price
        self.max_buy_prices = self.buy_interest * (self.gamestate.production_chain.prices * self.gamestate.production_chain.markup)
        # min sell prices set to 1/markup * base price
        self.min_sell_prices = sell_interest * (self.gamestate.production_chain.prices * 1/self.gamestate.production_chain.markup)
        self.min_sell_prices[sell_interest == 0] = np.inf

        self.min_buy_prices = self.buy_interest * (self.gamestate.production_chain.prices * 1/self.gamestate.production_chain.markup * 0.9)
        self.max_sell_prices = sell_interest * (self.gamestate.production_chain.prices * self.gamestate.production_chain.markup * 1.1)
        self.max_sell_prices[sell_interest == 0] = np.inf

        self.no_sale_ticks = np.zeros((self.num_agents, self.num_products), dtype=int)
        self.no_buy_ticks = np.zeros((self.num_agents, self.num_products), dtype=int)
        self.cannot_buy_ticks = np.zeros((self.num_agents, self.num_products), dtype=int)
        self.cannot_sell_ticks = np.zeros((self.num_agents, self.num_products), dtype=int)

        # initialze our estimates of prices with the production chain prices
        self.input_value_estimates = np.repeat(self.gamestate.production_chain.prices[np.newaxis, :], self.num_agents, axis=0)
        self.input_volume_estimates = np.ones((self.num_agents, self.num_products))
        self.output_value_estimates = np.repeat(self.gamestate.production_chain.prices[np.newaxis, :], self.num_agents, axis=0)
        self.output_volume_estimates = np.ones((self.num_agents, self.num_products))

        # output price estimates should start out above estimated cost to produce
        input_price_estimates = self.input_value_estimates / self.input_volume_estimates
        output_price_estimates = self.output_value_estimates / self.output_volume_estimates
        assert np.all(output_price_estimates >= (input_price_estimates @ self.gamestate.production_chain.adj_matrix))
        assert np.all(~(self.max_sell_prices < self.min_sell_prices))
        assert np.all(~(self.max_buy_prices < self.min_buy_prices))

        self.ticks = 0

        self.data_logger.initialize(self)

    def produce_goods(self) -> npt.NDArray[np.float64]:
        """ Consumes input to produce output. """

        # compute how many units each agent can produce
        # NOTE:this assumes each agent only produces a single good
        # we'll only produce complete units, so take the floor
        units_to_produce = np.floor(np.amin(
            np.divide(self.inventory, self.production_goods,
                out=np.zeros_like(self.inventory),
                where=self.production_goods!=0),
            axis=1, initial=np.inf, where=self.production_goods!=0))

        # agents producing first rank goods don't have any inputs, so they'll
        # be infs here. instead, we want them to be marked as producing 0 goods
        # right now. their goods get sourced elsewhere
        units_to_produce[~(units_to_produce < np.inf)] = 0

        # clamp units_to_produce down to zero or one batch
        goods_produced = (units_to_produce[:, np.newaxis] * self.agent_goods >= self.batch_sizes).astype(float) * self.batch_sizes

        # compute how many of each input we need to produce that many units of
        # output
        inputs_needed = (goods_produced @ self.gamestate.production_chain.adj_matrix.T)

        self.inventory += goods_produced - inputs_needed
        self.gamestate.production_chain.goods_produced += goods_produced.sum(axis=0)

        self.data_logger.produce_goods(goods_produced)

        return goods_produced

    def source_resources(self, scale:float=1.) -> None:
        resource_injection = np.zeros((self.num_agents, self.num_products))
        # every agent that sells the source resources gets a load of scale
        # resources
        resource_injection[:,0:self.gamestate.production_chain.ranks[0]] += self.agent_goods[:,0:self.gamestate.production_chain.ranks[0]] * scale

        # price is fixed, discounted by the markup
        #   this is sort of a hard constraint on the economy
        injection_prices = ((self.gamestate.production_chain.prices / self.gamestate.production_chain.markup)[np.newaxis,:] * self.agent_goods)

        # cap the injection to what sourcers can afford
        # NOTE: adding PRICE_EPS here to allow for approximately equal
        # NOTE: we assume every agent only has a single good they are buying here
        # transactions
        resource_injection = np.floor(
            np.clip(
                resource_injection,
                0.,
                ((self.balance + PRICE_EPS) / injection_prices.sum(axis=1))[:,np.newaxis],
            )
        )

        # clip injection to some max
        #TODO: how to set max inventory desired? (see buy_budget in set_prices)
        resource_injection = np.floor(
            np.clip(
                resource_injection,
                0.,
                np.clip(self.output_volume_estimates*10-self.inventory, 0, np.inf)
            )
        )

        self.inventory += resource_injection
        self.balance -= (resource_injection * injection_prices).sum(axis=1)
        self.balance[np.isclose(self.balance, 0.)] = 0
        assert np.all(self.balance >= 0.)

        self.gamestate.production_chain.resources_mined += resource_injection.sum(axis=0)[:self.gamestate.production_chain.ranks[0]]
        self.gamestate.production_chain.value_mined += (resource_injection * injection_prices).sum(axis=0)[:self.gamestate.production_chain.ranks[0]]

    def sink_products(self, scale:float=1.) -> None:

        # see https://demonstrations.wolfram.com/ConstantPriceElasticityOfDemand/
        # we model demand for final goods as a large-scale market.
        # NOTE: kind of fudging this since the entire market should have this
        # quantity and we divide that over our sellers
        # we assume constant price elasticity as per:
        # Q = a*P^(1/c)
        # a is a market size parameter
        # c is the elasticity (-1<c<0 => inelastic demand, -1
        # As the price falls, the revenue area decreases for inelastic demand
        # (-1<c<0), remains constant for unit elastic demand (-1 = c), and
        # increases for elastic demand (c < -1).
        # @price = production_chain.prices, Q = scale * batchsize
        # -1 = c,
        # scale * batchsize = a / base_prices
        # a = scale * batchsize * base_prices
        # Q = (scale * batchsize * prices) * P ^ (1/c)
        # below Q = resource_sink, P = self.sell_prices, c = price_elasticity

        price_elasticity = -1
        base_prices = self.gamestate.production_chain.prices[np.newaxis,:] * self.agent_goods

        market_size_param = scale * self.batch_sizes * base_prices
        # we add PRICE_EPS here because of numerical precision issues
        resource_sink = np.floor(market_size_param * self.sell_prices ** (1/price_elasticity) + PRICE_EPS)
        assert resource_sink.shape == (self.num_agents, self.num_products)
        # zero out non-final-rank goods, we're not sinking them
        resource_sink[:,:-self.gamestate.production_chain.ranks[-1]] = 0
        # clip to inventory we have
        resource_sink = np.clip(resource_sink, 0, self.inventory, out=resource_sink)

        sink_prices = self.sell_prices.copy()
        sink_prices[resource_sink == 0.] = 0.

        """
        resource_sink = np.zeros((self.num_agents, self.num_products))

        # every agent that sells the sink goods gets to sell up to the scale
        resource_sink[:,-self.gamestate.production_chain.ranks[-1]:] += self.agent_goods[:,-self.gamestate.production_chain.ranks[-1]:] * scale

        # cap sinks at the inventory
        resource_sink = np.clip(resource_sink, 0, self.inventory, out=resource_sink)

        # goods are fix priced at the final output prices
        #   this is sort of a hard constraint on the economy
        sink_prices = self.gamestate.production_chain.prices
        """

        """
        # try setting price to min sell price times the markup for that product
        sink_prices = np.multiply(self.min_sell_prices, self.gamestate.production_chain.markup[np.newaxis,:], where=self.min_sell_prices < np.inf, out=np.zeros_like(self.min_sell_prices))
        """

        self.inventory -= resource_sink
        self.inventory[np.isclose(self.inventory, 0.)] = 0
        assert np.all(self.inventory >= 0.)

        self.balance += (resource_sink * sink_prices).sum(axis=1)
        self.gamestate.production_chain.goods_sunk += resource_sink.sum(axis=0)
        self.gamestate.production_chain.value_sunk += (resource_sink * sink_prices).sum(axis=0)

    def _compute_min_sell_prices(self, input_price_estimates:npt.NDArray[np.float64], output_price_estimates:npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # estimate min sell price based on the price we're paying for inputs
        # price_output_i >= sum_inputs_for_i(price_input_j * need_input_j)
        # NOTE: this assumes every agent produces/sells exactly one kind of item
        #TODO: if we never made a sale (or buy), this is unreliable
        min_sell_prices = (input_price_estimates @ self.gamestate.production_chain.adj_matrix * self.agent_goods)
        # force agents that sell rank 0 goods min sell price to be the price
        # they have to buy resources for
        min_sell_prices[:,:self.gamestate.production_chain.ranks[0]] += self.agent_goods[:,:self.gamestate.production_chain.ranks[0]] * (self.gamestate.production_chain.prices[:self.gamestate.production_chain.ranks[0]] / self.gamestate.production_chain.markup[:self.gamestate.production_chain.ranks[0]])
        min_sell_prices[self.agent_goods == 0] = np.inf

        #TODO: what do we do if we have output goods that aren't selling, even
        # at the minimum price?
        #self.min_sell_prices[(self.sell_prices == self.min_sell_prices) & (self.inventory[sell_interest] > 0)] = drop somehow, but cap it

        return min_sell_prices

    def _compute_max_buy_prices(self, input_price_estimates:npt.NDArray[np.float64], output_price_estimates:npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # estimate max buy prices based on the price we're selling output for
        # this is a total spend we can afford on inputs, we can distribute that
        # over the input goods according to how many of each we need, along with
        # some estimate of how we've observed input prices are related to each
        # other.

        # max buy price = scalar * observed_prices
        # because sum_inputs( max_price_j * need_j) = price_output
        # and we want to preserve the ratios between input prices, i.e.
        #   max_prices = scalar * observed_prices
        # we also ignore cases where we're not selling or there are no inputs
        # shape is self.num_agents by self.num_products
        output_has_inputs = self.agent_goods * self.gamestate.production_chain.adj_matrix.sum(axis=0)[np.newaxis, :] > 0
        assert output_has_inputs.shape == (self.num_agents, self.num_products)
        agent_markup_by_outputs = np.divide(
            output_price_estimates,
            (input_price_estimates @ self.gamestate.production_chain.adj_matrix),
            where=output_has_inputs,
            out=np.zeros((self.num_agents, self.num_products))
        )
        # we computed a sclar for each out, but then we can collapse back to per agent
        # NOTE: this assumes each agent only sells one good
        #max_buy_prices = agent_markup_by_outputs.max(axis=1)[:,np.newaxis] * input_price_estimates


        # and we apply that across the input goods, according to the input prices
        inventory_balance = np.divide(self.inventory, self.production_goods, where=self.production_goods > 0, out=np.zeros_like(self.inventory))
        deficit = np.clip(self.production_goods * self.batch_sizes.sum(axis=1)[:,np.newaxis] - self.inventory, 0, np.inf)

        # take a floor of deficit of 1 to avoid prices going to zero
        deficit[self.production_goods > 0] = np.clip(deficit[self.production_goods > 0], 1, np.inf)
        deficit_proportion = (deficit/np.divide(
                deficit.sum(axis=1),
                (self.production_goods>0).sum(axis=1),
                where=self.production_goods.sum(axis=1)>0,
                out=np.ones((self.num_agents,))
            )[:,np.newaxis]
        )
        price_scale = np.divide(
                output_price_estimates,
                (deficit_proportion * input_price_estimates @ self.gamestate.production_chain.adj_matrix),
                where=output_has_inputs,
                out=np.zeros((self.num_agents, self.num_products))
        )
        max_buy_prices = price_scale.sum(axis=1)[:,np.newaxis] * deficit_proportion * input_price_estimates


        max_buy_prices[self.buy_interest == 0] = 0.

        # the max price times production goods should our output price estimate
        assert np.all(
            np.isclose(
                (max_buy_prices * self.production_goods).sum(axis=1),
                (output_price_estimates * self.agent_goods).sum(axis=1)
            )
            | ~self.agent_goods[:,self.gamestate.production_chain.ranks[0]:].sum(axis=1).astype(bool)
        )

        # for goods that chronically cannot be purchased, increase the max price
        # by the same amount that the underlying price will increase
        gamma = 0.001

        # also calculate our expected production capacity at the current prices
        # (assume those are the max buy price)
        input_production_cost = (self.production_goods * self.buy_prices).sum(axis=1)
        expected_production_capacity = np.divide(self.balance + (self.inventory * self.buy_prices).sum(axis=1), input_production_cost, where=input_production_cost > PRICE_EPS, out=np.zeros((self.num_agents,)))

        # need to make sure we do not raise prices so much that we can't afford
        # a batch
        #TODO: how to set the number of ticks to wait at cannot buy before we violate the cost of inputs < price of output?
        cannot_buy_ticks_before_investment = 100
        needed_for_one_batch = (self.batch_sizes.max(axis=1)[:,np.newaxis] * self.production_goods)
        affordable_amount = (expected_production_capacity[:,np.newaxis] * self.production_goods)
        max_buy_prices[(self.cannot_buy_ticks > cannot_buy_ticks_before_investment) & (affordable_amount > needed_for_one_batch)] = self.max_buy_prices[(self.cannot_buy_ticks > cannot_buy_ticks_before_investment) & (affordable_amount > needed_for_one_batch)] * (1+gamma)

        return max_buy_prices

    def set_prices(self, buys_value:npt.NDArray[np.float64], buys:npt.NDArray[np.float64], sales_value:npt.NDArray[np.float64], sales:npt.NDArray[np.float64]) -> None:
        """ Agents set prices for the next round.

        Happens after trades happen but before resulting production.
        """

        # try: two step
        # 1. choose sale price inverse with inventory
        # 2. choose input prices consistent with that sale price

        # sale price must be at least the sum of the input amounts times our
        # best estimate of the input costs (which could be a moving average of
        # the price we've paid)
        # outside of that, we want maximize our revenue (price * volume)
        # we can assume that volume moves in the opposite direction of price
        # so we can experiment until we achieve some maximum

        input_price_estimates = self.input_value_estimates / self.input_volume_estimates
        output_price_estimates = self.output_value_estimates / self.output_volume_estimates

        self.no_sale_ticks[(sales == 0) & (self.agent_goods > 0)] += 1
        self.no_sale_ticks[(sales > 0) & (self.agent_goods > 0)] = 0
        self.no_buy_ticks[(buys == 0) & (self.buy_interest > 0)] += 1
        self.no_buy_ticks[(buys > 0) & (self.buy_interest > 0)] = 0

        cannot_sell = np.isclose(self.sell_prices, self.min_sell_prices) & (self.inventory > 0) & (self.agent_goods > 0) & (sales == 0)
        self.cannot_sell_ticks[cannot_sell] += 1
        self.cannot_sell_ticks[~cannot_sell] = 0

        #if np.any(self.cannot_sell_ticks > 100):
        #    breakpoint()

        cannot_buy = np.isclose(self.buy_prices, self.max_buy_prices) & (self.last_buy_budget > 0) & (buys == 0)
        self.cannot_buy_ticks[cannot_buy] += 1
        self.cannot_buy_ticks[~cannot_buy] = 0

        #if np.any(self.cannot_buy_ticks > 100):
        #    breakpoint()

        self.min_sell_prices = self._compute_min_sell_prices(input_price_estimates, output_price_estimates)

        self.max_buy_prices = self._compute_max_buy_prices(input_price_estimates, output_price_estimates)
        self.min_buy_prices = np.clip(self.min_buy_prices, 0, self.max_buy_prices)

        # choose a price to sell goods
        # we want to maximize revenue (price * volume)
        # so we do gradient ascent on the revenue curve, using the most recent
        # two ticks to estimate the gradient
        # price_n+1 = price_n + gamma * grad_rev(price_n)
        # grad_rev(price_n) ~ (rev_n - rev_n-1) / (price_n - price_n-1)
        gamma = 0.005
        sell_price_delta = np.subtract(
                self.sell_prices, self.last_sell_prices,
                where=self.agent_goods > 0,
                out=np.full((self.num_agents, self.num_products), np.inf)
        )
        zero_price_delta = np.isclose(sell_price_delta, 0.)
        revenue_delta = sales_value - self.last_sales_value
        step = gamma * np.divide(
                revenue_delta, sell_price_delta,
                where=~zero_price_delta,
                out=np.zeros_like(self.sell_prices)
        )
        step = np.clip(step, -self.sell_prices/5, self.sell_prices/5)
        new_sell_prices = self.sell_prices + step
        self.last_sell_prices = self.sell_prices
        self.last_sales_value = sales_value.copy()

        # if we didn't have a price delta, increase prices
        new_sell_prices[zero_price_delta] = self.sell_prices[zero_price_delta] * (1 + gamma)
        # if we sold nothing, drop prices
        new_sell_prices[sales == 0] = self.sell_prices[sales == 0] * (1 - gamma)
        # if we had nothing to sell, leave price
        new_sell_prices[self.last_inventory == 0] = self.sell_prices[self.last_inventory == 0]
        # and then clip prices to min prices (which handles goods we don't sell)
        self.sell_prices = np.clip(new_sell_prices, self.min_sell_prices, np.inf)

        # choose a price to buy goods
        # modify buy prices to maximize the amount per budget
        buy_price_delta = self.buy_prices - self.last_buy_prices
        amount_per_budget_delta = (
            np.divide(buys, self.last_buy_budget, where=self.last_buy_budget>0, out=np.zeros_like(buys))
        )
        amount_per_budget_delta = (
            amount_per_budget_delta
            - np.divide(self.last_buys, self.last_buy_budget, where=self.last_buy_budget > 0, out=np.zeros_like(self.last_buys))
        )

        step = gamma * np.divide(
                amount_per_budget_delta, buy_price_delta,
                where=~np.isclose(buy_price_delta, 0.),
                out=np.zeros_like(self.buy_prices)
        )

        # don't step up or down too much
        step = np.clip(step, -(self.buy_prices/2), self.buy_prices/2)

        new_buy_prices = self.buy_prices + step
        #if np.any(new_buy_prices[buys > 0] - self.buy_prices[buys > 0] > 0):
        #    breakpoint()

        # if we bought nothing, increase prices
        new_buy_prices[buys == 0] = (self.buy_prices[buys == 0] * (1+gamma))
        # if we had no budget, leave price
        new_buy_prices[self.last_buy_budget == 0] = self.buy_prices[self.last_buy_budget == 0]

        assert np.all(new_buy_prices[self.buy_interest > 0] > 0)

        # for debugging assert failures below, hang on to this
        old_last_buy_prices = self.last_buy_prices
        self.last_buy_prices = self.buy_prices
        self.buy_prices = np.clip(new_buy_prices, 0, self.max_buy_prices)
        assert np.all(self.buy_prices[self.buy_interest > 0] > 0)

        # set our buy budget to spend all our money, balancing our inventory
        # that is,
        # sum_inputs(budget_j) = balance
        # budget/price + inventory = some_constant * input_need
        # budget + inventory * price = some_constant * input_need * price
        # some_constant = (balance + sum_inputs(inventory_j * price_j)) / sum_inputs(need_j * price_j)
        # so budget = some_constant * need * price - inventory * price
        # some_constant here is our expected production capacity after we spend
        # our budget, assuming we buy at our set price

        input_production_cost = (self.production_goods * self.buy_prices).sum(axis=1)
        expected_production_capacity = np.divide(self.balance + (self.inventory * self.buy_prices).sum(axis=1), input_production_cost, where=input_production_cost > PRICE_EPS, out=np.zeros((self.num_agents,)))

        # clip the expected production capacity to limit how many extra goods
        # we keep on hand. This is sort of like a just-in-time supply chain.
        #TODO: how to set choose how many batches we want to keep on hand?
        #TODO: what do we do if the expected production capacity given the
        #balance and buy prices is less than one batch?
        desired_batches_of_input = 2
        expected_production_capacity = np.clip(expected_production_capacity, 0, self.batch_sizes.sum(axis=1)*desired_batches_of_input)

        self.buy_budget = expected_production_capacity[:,np.newaxis] * self.production_goods * self.buy_prices - self.inventory * self.buy_prices

        self.last_inventory = self.inventory.copy()

        #TODO: this assert can fail if expected_production_capacity < 1, which is a bad situation, but not a bug exactly
        #assert np.all(((self.buy_budget > 0) | (self.inventory >= (self.production_goods * self.batch_sizes.max(axis=1)[:,np.newaxis])))[self.production_goods > 0])

        # stop production if we're sitting on a lot of output already
        #TODO: how many batches of output do we want to keep on hand?
        desired_batches_of_output = 5
        self.buy_budget[((self.batch_sizes * desired_batches_of_output - (self.inventory * self.agent_goods)).max(axis=1)[:,np.newaxis] * self.buy_interest) <= 0] = 0

        #assert np.all(self.balance + PRICE_EPS - self.buy_budget.sum(axis=1) >= 0.)
        assert np.all(self.sell_prices > 0)
        assert np.all(self.buy_prices >= 0)
        assert np.all(~(self.sell_prices < self.min_sell_prices))
        assert np.all(~(self.buy_prices > self.max_buy_prices))

        self.last_buys = buys.copy()
        self.last_buy_budget = self.buy_budget.copy()

    def make_market(self, buy_prices:npt.NDArray[np.float64], sell_prices:npt.NDArray[np.float64], budget:npt.NDArray[np.float64]) -> Tuple[float, int, int, int, float, float]:
        """
        returns tuple:
            price difference (positive => sale is feasible)
            product id
            buyer id
            seller id
            price
            sale amount
        """

        min_sale_amount = 1.
        # get the per product best buy/sell price differences that can be
        # funded/supplied

        valid_buys = (buy_prices <= budget/min_sale_amount) & (buy_prices <= self.balance[:,np.newaxis])
        valid_sells = self.inventory >= min_sale_amount

        diffs = (
                np.amax(buy_prices, axis=0, where=valid_buys, initial=0)
                - np.amin(sell_prices, axis=0, where=valid_sells, initial=np.inf)
        )

        # and find the product with the best deal
        product = diffs.argmax()

        # if the biggest difference in sale price and buy price is negative,
        # there's no profitable sales
        if diffs[product] < 0:
            return (diffs[product], product, -1, -1, 0., 0.)

        # pick the buyer with max price for that product
        buyer = np.where(valid_buys[:,product])[0][buy_prices[valid_buys[:,product],product].argmax()]
        # pick the seller with the min price for that product
        seller = np.where(valid_sells[:,product])[0][sell_prices[valid_sells[:,product],product].argmin()]

        assert np.isclose(buy_prices[buyer, product] - sell_prices[seller, product], diffs[product])

        #TODO: how to set a price?
        price = (buy_prices[buyer, product] + sell_prices[seller,product])/2

        # transact the biggest volume possible (inventory vs budget/balance)
        sale_amount = np.floor(min(self.inventory[seller, product], budget[buyer, product]/price, self.balance[buyer]/price))

        assert buyer != seller
        assert buy_prices[buyer, product] > 0
        assert sell_prices[seller, product] < np.inf
        assert sell_prices[seller, product] <= buy_prices[buyer, product]
        assert sale_amount > 0
        assert budget[buyer, product] >= sale_amount * price
        assert self.balance[buyer] >= sale_amount * price
        assert self.inventory[seller, product] >= sale_amount
        assert self.balance[buyer] >= sale_amount * price

        return diffs[product], product, buyer, seller, price, sale_amount

    def transact(self, diff:float, product_id:int, buyer:int, seller:int, price:float, sale_amount:float) -> None:
        """ Conducts the transaction indicated by parameters. """

        self.inventory[buyer, product_id] += sale_amount
        self.balance[buyer] -= sale_amount * price

        self.inventory[seller, product_id] -= sale_amount
        self.balance[seller] += sale_amount * price

        self.buy_budget[buyer, product_id] -= sale_amount * price

        # update agent beliefs about prices
        self.input_value_estimates[buyer, product_id], self.input_volume_estimates[buyer, product_id] = util.update_vema(
                self.input_value_estimates[buyer, product_id], self.input_volume_estimates[buyer, product_id],
                self.price_vema_alpha, price, sale_amount)
        self.output_value_estimates[seller, product_id], self.output_volume_estimates[seller, product_id] = util.update_vema(
                self.output_value_estimates[seller, product_id], self.output_volume_estimates[seller, product_id],
                self.price_vema_alpha, price, sale_amount)
        self.gamestate.production_chain.observe_transaction(product_id, price, sale_amount)

        self.data_logger.transact(diff, product_id, buyer, seller, price, sale_amount)

    def run(self, max_ticks:int) -> None:
        self.logger.info("running simluation...")

        self.logger.info(f'beginning balance: {self.balance.sum()}')
        self.logger.info(f'beginning inventory: {self.inventory.sum()}')

        # keeps track of the number of units bought/sold each turn
        buys = np.empty((self.num_agents, self.num_products))
        buys_value = np.empty((self.num_agents, self.num_products))
        sales = np.empty((self.num_agents, self.num_products))
        sales_value = np.empty((self.num_agents, self.num_products))

        for _ in tqdm.tqdm(range(max_ticks)):
            # source resources into base of production chain and sink cost of
            # those resources
            self.source_resources(scale=10000)

            # sink resources from tips of production chain and source money of
            # those resources
            self.sink_products(scale=100)

            # try to make a market, do trade

            # TODO: what are min/max prices each agent has? (are these based on
            # production chain to make profit likely and equal across all
            # agents producing that good?

            # take trades that are matched until there are no possible trades
            # (no valid price matches with non-zero inventory)
            buys[:,:] = 0.
            buys_value[:,:] = 0
            sales[:,:] = 0.
            sales_value[:,:] = 0.
            #buy_prices = self.buy_prices.copy()
            #sell_prices = self.sell_prices.copy()

            self.data_logger.start_trading()

            transactions_this_tick = 0
            while (ret := self.make_market(self.buy_prices, self.sell_prices, self.buy_budget))[0] > 0:
                (diff, product, buyer, seller, price, sale_amount) = ret
                self.transact(diff, product, buyer, seller, price, sale_amount)
                buys[buyer, product] += sale_amount
                buys_value[buyer, product] += sale_amount * price
                sales[seller, product] += sale_amount
                sales_value[seller, product] += sale_amount * price
                transactions_this_tick += 1

            assert np.all(self.inventory * (1-self.agent_goods) <= self.production_goods * self.batch_sizes.sum(axis=1)[:, np.newaxis] * 20)
            assert np.all(buys.sum(axis=0) == sales.sum(axis=0))

            #if np.any(
            #        np.amax(self.buy_prices[:, np.where(buys.sum(axis=0) == 0)], axis=0)
            #        - np.amin(self.sell_prices[:, np.where(sales.sum(axis=0) == 0)], axis=0) > 0
            #    ):
            #    breakpoint()

            # compute demand and supply surplus
            #demand_surplus = (buy_prices > 0).astype(float).sum(axis=0)
            #supply_surplus = (sell_prices < np.inf).astype(float).sum(axis=0)

            # produce goods according to the production chain
            self.produce_goods()
            self.set_prices(buys_value, buys, sales_value, sales)

            self.data_logger.end_trading()


            self.ticks += 1

        self.logger.info(f'transactions: {self.gamestate.production_chain.transaction_count.sum()}')
        self.logger.info(f'transaction amount: {self.gamestate.production_chain.transaction_amount.sum()}')
        self.logger.info(f'transaction value: {self.gamestate.production_chain.transaction_value.sum()}')
        self.logger.info(f'ending balance: {self.balance.sum()}')
        self.logger.info(f'ending inventory: {self.inventory.sum()}')

        self.logger.info(f'resources mined: {self.gamestate.production_chain.resources_mined}')
        self.logger.info(f'value mined: {self.gamestate.production_chain.value_mined.sum()}')
        self.logger.info(f'goods produced: {self.gamestate.production_chain.goods_produced.sum()}')
        self.logger.info(f'final rank goods produced: {self.gamestate.production_chain.goods_produced[-self.gamestate.production_chain.ranks[-1]:]}')
        self.logger.info(f'goods sunk: {self.gamestate.production_chain.goods_sunk.sum()}')
        self.logger.info(f'value sunk: {self.gamestate.production_chain.value_sunk.sum()}')

        ranks = self.gamestate.production_chain.ranks
        for level, ((nodes_from, nodes_to), so_far) in enumerate(zip(zip(ranks, np.pad(ranks[1:], (0,1))), np.cumsum(ranks))):
            relevant_balances = self.balance[np.where(self.agent_goods[:,so_far-nodes_from:so_far].max(axis=1)>0)]
            min_balance = relevant_balances.min()
            mean_balance = relevant_balances.mean()
            max_balance = relevant_balances.max()
            relevant_inventories = (self.agent_goods[:,so_far-nodes_from:so_far] * self.inventory[:,so_far-nodes_from:so_far]).max(axis=1)[np.where(self.agent_goods[:,so_far-nodes_from:so_far].max(axis=1)>0)]
            min_inventory = relevant_inventories.min()
            mean_inventory = relevant_inventories.mean()
            max_inventory = relevant_inventories.max()
            self.logger.info(f'rank {level} balance: {mean_balance} ({min_balance},{max_balance})')
            self.logger.info(f'rank {level} pdct inventory: {mean_inventory} ({min_inventory},{max_inventory})')

def main() -> None:
    with contextlib.ExitStack() as context_stack:
        logging.basicConfig(
                format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
                stream=sys.stderr,
                level=logging.INFO
        )
        warnings.filterwarnings("error")
        mgr = context_stack.enter_context(util.PDBManager())

        data_logger = context_stack.enter_context(EconomyDataLogger())

        econ = EconomySimulation(data_logger)
        econ.initialize(num_agents=100)


        #econ.run(max_ticks=100000)
        econ.run(max_ticks=2000)

        data_logger.end_simulation()

if __name__ == "__main__":
    main()
