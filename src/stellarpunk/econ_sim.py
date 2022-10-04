""" Tool to run simple simulation of a market. """

from __future__ import annotations

import sys
import os
import io
import logging
import contextlib
import warnings
import typing
from typing import TextIO, BinaryIO, IO, Optional, Tuple, MutableSequence, Sequence, Type, Any, Iterator
from types import TracebackType

import numpy as np
import numpy.typing as npt
import msgpack # type: ignore
import tqdm # type: ignore
from numba import jit # type: ignore

from stellarpunk import util, core, generate, serialization

# sometimes we're willing to manufacture a very small amount of cash to avoid
# precision errors
PRICE_EPS = 1e-05

# the tick period we're targeting our various exponential moving averages for
# gets used as in alpha = 2/(EMA_TICKS + 1) see this article:
# https://en.wikipedia.org/wiki/Moving_average#Relationship_between_SMA_and_EMA
EMA_TICKS = 365 #365, as if every tick is a day and we're getting 1 year

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

class EconomyDataLogger(contextlib.AbstractContextManager):
    """ Logs data on the economy over time.

    Has a listener model for interesting events. Logs a lot of state when
    approriate event happens in the economy and the tick for that state.. """

    def __init__(self, enabled:bool=True, logdir:str="/tmp/", buffersize:int=4*1024, flush_interval:int=1000) -> None:
        """ Creates an object that facilitates logging the lifecycle of the
        economic simulation.

        Parameters
        ----------
        enabled : bool, default True
            controls if logging is enabled (otherwise be silent)
        logdir : str, default "/tmp/"
            directory to place log files (this dir must already exist)
        buffersize : int, default 4*1024
            size of the buffers for outputting to disk
        flush_interval : int, default 1000
            logs are flushed to disk every flush_interval ticks (set to 0 to
            disable)
        """

        self.enabled = enabled
        self.logdir = logdir
        self.buffersize = buffersize
        self.flush_interval = flush_interval

        self.transaction_log:TextIO = None #type:ignore[assignment]
        self.inventory_log:serialization.TickMatrixWriter = None #type:ignore[assignment]
        self.balance_log:serialization.TickMatrixWriter = None #type:ignore[assignment]
        self.buy_prices_log:serialization.TickMatrixWriter = None #type:ignore[assignment]
        self.buy_budget_log:serialization.TickMatrixWriter = None #type:ignore[assignment]
        self.max_buy_prices_log:serialization.TickMatrixWriter = None #type:ignore[assignment]
        self.sell_prices_log:serialization.TickMatrixWriter = None #type:ignore[assignment]
        self.min_sell_prices_log:serialization.TickMatrixWriter = None #type:ignore[assignment]
        self.production_efficiency_log:serialization.TickMatrixWriter = None #type:ignore[assignment]
        self.cannot_buy_log:serialization.TickMatrixWriter = None #type:ignore[assignment]
        self.cannot_sell_log:serialization.TickMatrixWriter = None #type:ignore[assignment]

        self.files:MutableSequence[IO] = []
        self.exit_stack:contextlib.ExitStack = contextlib.ExitStack()
        self.sim:EconomySimulation = None #type: ignore[assignment]

    def _open_txt_log(self, name:str) -> TextIO:
        f = self.exit_stack.enter_context(open(os.path.join(self.logdir, f'{name}.log'), "wt", self.buffersize))
        self.files.append(f)
        return f

    def _open_bin_log(self, name:str) -> BinaryIO:
        f = self.exit_stack.enter_context(open(os.path.join(self.logdir, f'{name}.log'), "wb", self.buffersize))
        self.files.append(f)
        return f

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
            self.cannot_buy_log = serialization.TickMatrixWriter(self._open_bin_log("cannot_buy"))
            self.cannot_sell_log = serialization.TickMatrixWriter(self._open_bin_log("cannot_sell"))

        return self

    def __exit__(self, exc_type:Optional[Type[BaseException]], exc_value:Optional[BaseException], traceback:Optional[TracebackType]) -> Optional[bool]:
        """ Closes underlying resources (e.g. log files). """
        self.exit_stack.close()
        return None

    def flush(self) -> None:
        """ Flushes output buffers. """
        for f in self.files:
            f.flush()

    def initialize(self, sim:EconomySimulation) -> None:
        self.sim = sim
        if self.enabled:
            with open(os.path.join(self.logdir, "agent_goods.log"), "wb") as agent_goods_log:
                agent_goods_log.write(msgpack.packb(self.sim.agent_goods, default=serialization.encode_matrix))
            self.sim.gamestate.production_chain.viz().render(os.path.join(self.logdir, "production_chain"), format="pdf")

    def end_simulation(self) -> None:
        if self.enabled:
            with open(os.path.join(self.logdir, "production_chain.log"), "wb") as production_chain_log:
                production_chain_log.write(serialization.save_production_chain(self.sim.gamestate.production_chain))

    def produce_goods(self, goods_produced:npt.NDArray[np.float64]) -> None:
        return
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

    def transact(self, diff:float, product_id:int, buyer:int, seller:int, price:float, sale_amount:float) -> None:
        if self.enabled:
            self.transaction_log.write(f'{self.sim.ticks}\t{product_id}\t{buyer}\t{seller}\t{price}\t{sale_amount}\n')

    def source_resources(self, price:npt.NDArray[np.float64], amount:npt.NDArray[np.float64]) -> None:
        seller = -1
        raw_product_id = -1
        nonzero = np.where(amount > 0)
        buyers = nonzero[0]
        products = nonzero[1]
        for buyer, product_id in zip(buyers, products):
            p = price[buyer, product_id]
            sale_amount = amount[buyer, product_id]
            # collapse raw resources to a single raw product id
            product_id = raw_product_id
            self.transaction_log.write(f'{self.sim.ticks}\t{product_id}\t{buyer}\t{seller}\t{p}\t{sale_amount}\n')

    def sink_products(self, price:npt.NDArray[np.float64], amount:npt.NDArray[np.float64]) -> None:
        buyer = -1
        nonzero = np.where(amount > 0) 
        sellers = nonzero[0]
        products = nonzero[1]
        for seller, product_id in zip(sellers, products):
            p = price[seller, product_id]
            sale_amount = amount[seller, product_id]
            self.transaction_log.write(f'{self.sim.ticks}\t{product_id}\t{buyer}\t{seller}\t{p}\t{sale_amount}\n')

    def start_trading(self) -> None:
        pass

    def set_prices(self) -> None:
        if self.enabled:
            self.buy_prices_log.write(self.sim.ticks, self.sim.buy_prices)
            self.buy_budget_log.write(self.sim.ticks, self.sim.buy_budget)
            self.sell_prices_log.write(self.sim.ticks, self.sim.sell_prices)
            self.max_buy_prices_log.write(self.sim.ticks, self.sim.max_buy_prices)
            self.min_sell_prices_log.write(self.sim.ticks, self.sim.min_sell_prices)

            self.balance_log.write(self.sim.ticks, self.sim.balance)
            self.inventory_log.write(self.sim.ticks, self.sim.inventory)
            self.cannot_buy_log.write(self.sim.ticks, self.sim.cannot_buy_ticks)
            self.cannot_sell_log.write(self.sim.ticks, self.sim.cannot_sell_ticks)

    def end_trading(self) -> None:
        if self.flush_interval and self.sim.ticks % self.flush_interval == 0:
            self.flush()
        pass

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
        # ema of inventory, smooths out some places we need to generally know what our inventory was
        self.smoothed_inventory_alpha = 2./(EMA_TICKS+1.)
        self.smoothed_inventory = np.zeros((self.num_agents, self.num_products))
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
        self.price_vema_alpha = 2./(EMA_TICKS+1.)
        self.input_value_estimates = np.zeros((self.num_agents, self.num_products))
        self.input_volume_estimates = np.zeros((self.num_agents, self.num_products))
        self.output_value_estimates = np.zeros((self.num_agents, self.num_products))
        self.output_volume_estimates = np.zeros((self.num_agents, self.num_products))

        # estimates of transaction rates for each agent, for each good, for each agent
        self.rate_alpha = 2./(EMA_TICKS+1.)
        self.buy_rate_estimates = np.zeros((self.num_agents, self.num_products))
        self.sell_rate_estimates = np.zeros((self.num_agents, self.num_products))

        # number of ticks since a sale/buy
        self.no_sale_ticks = np.zeros((self.num_agents, self.num_products), dtype=int)
        self.no_buy_ticks = np.zeros((self.num_agents, self.num_products), dtype=int)
        # number of ticks we've been at max/min price and have failed to
        # conduct a necessary transaction
        self.cannot_buy_ticks = np.zeros((self.num_agents, self.num_products), dtype=int)
        self.cannot_sell_ticks = np.zeros((self.num_agents, self.num_products), dtype=int)

        # Total market indicators

        # ema for total economy supply/demand volumes per tick
        self.supply_alpha = 2./(EMA_TICKS+1.)
        self.supply_estimate = np.zeros((self.num_products,))
        self.demand_estimate = np.zeros((self.num_products,))

        # ema for price
        self.global_price_alpha = 2./(EMA_TICKS+1.)
        self.global_value_estimate = np.zeros((self.num_products,))
        self.global_volume_estimate = np.zeros((self.num_products,))
        self.global_rate_estimate = np.zeros((self.num_products,))

        # parameters that tune behaviors

        self.price_setting_period = 25

        # how much money to start with in terms of number of batches of input
        #   assuming production chain initial prices
        self.starting_balance_batches = 50

        # max price step in either direction to take relative to current price
        self.rel_sell_price_step_max = 0.2
        self.rel_buy_price_step_max = 0.5

        # learning rate for setting prices (to maximize revenue and maximize
        # input amount per dollar
        self.price_setting_gamma = 0.005

        # how many batches of input/output to target keeping on hand at once
        # for output this translate to a cap on input buying
        self.desired_input_batches = 3 * self.price_setting_period
        self.desired_output_batches = 5 * self.price_setting_period

        # adverse condition parameters

        # how many ticks of no input buys before we violate profitability
        self.cannot_buy_ticks_investment_cutoff = EMA_TICKS
        self.cannot_buy_ticks_leave_market = EMA_TICKS*2
        self.cannot_buy_gamma = 0.003

        # does it make sense to let agents violate profitability in buy prices
        # and sell prices? if we don't then the one will naturally move to match
        # the actual prices implied by the other, so probably not
        self.cannot_sell_ticks_investment_cutoff = EMA_TICKS#sys.maxsize
        self.cannot_sell_ticks_leave_market = EMA_TICKS*2
        self.cannot_sell_gamma = 0.003

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

        # randomly assign goods to the rest of the agents where higher ranked
        # goods are exponentially less likely
        ranks = self.gamestate.production_chain.ranks
        p = np.hstack([[3**(len(ranks)-i)]*v for (i,v) in enumerate(ranks)])
        #p = 1.0 / self.gamestate.production_chain.prices
        p = p/p.sum()
        single_products = self.gamestate.random.choice(a=np.arange(self.num_products), size=self.num_agents-self.num_products, replace=True, p=p)
        self.agent_goods[np.arange(self.num_products, self.num_agents), single_products] = 1

        # make sure only one product per agent (for now, we might want to lift
        # this simplifying restriction in the future...)
        assert np.all(self.agent_goods.sum(axis=1) == 1)
        assert np.all(self.agent_goods.sum(axis=1) == 1)

        self.batch_sizes = self.gamestate.production_chain.batch_sizes[np.newaxis,:] * self.agent_goods

        self.production_goods = (self.agent_goods @ self.gamestate.production_chain.adj_matrix.T)

        self.inventory = np.zeros((self.num_agents, self.num_products))
        self.last_inventory = self.inventory.copy()
        self.smoothed_inventory = np.zeros((self.num_agents, self.num_products))
        self.max_inventory = 1e3

        # set up initial balance, inventory, prices
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

        self.buy_rate_estimates = np.zeros((self.num_agents, self.num_products))
        self.sell_rate_estimates = np.zeros((self.num_agents, self.num_products))

        # initialize market-level estimates
        self.supply_estimate = np.zeros((self.num_products,))
        self.demand_estimate = np.zeros((self.num_products,))
        self.global_value_estimate = np.zeros((self.num_products,))
        self.global_volume_estimate = np.zeros((self.num_products,))
        self.global_rate_estimate = np.zeros((self.num_products,))

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

        self.data_logger.source_resources(injection_prices, resource_injection)

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

        resource_sink_value = (resource_sink * sink_prices)
        self.balance += resource_sink_value.sum(axis=1)

        total_sinks = (resource_sink > 0).astype(int).sum(axis=0)

        # update global transaction estimates to include the sinking
        self.global_value_estimate[total_sinks > 0], self.global_volume_estimate[total_sinks > 0] = util.update_vema(
            self.global_value_estimate[total_sinks > 0], self.global_volume_estimate[total_sinks > 0],
            self.global_price_alpha,
            resource_sink_value.sum(axis=0)[total_sinks > 0] / total_sinks[total_sinks>0], resource_sink.sum(axis=0)[total_sinks > 0] / total_sinks[total_sinks>0]
        )
        self.global_rate_estimate[-self.gamestate.production_chain.ranks[-1]:] = util.update_ema(self.global_rate_estimate[-self.gamestate.production_chain.ranks[-1]:], self.global_price_alpha, total_sinks[-self.gamestate.production_chain.ranks[-1]:])

        self.data_logger.sink_products(sink_prices, resource_sink)

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

        # drop min sell price for stuff we chronically cannot sell and have never sold
        min_sell_prices[(self.cannot_sell_ticks > self.cannot_sell_ticks_investment_cutoff) & np.isclose(self.sell_rate_estimates, 0)] = self.min_sell_prices[(self.cannot_sell_ticks > self.cannot_sell_ticks_investment_cutoff) & np.isclose(self.sell_rate_estimates, 0)] * (1-self.cannot_sell_gamma)


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

        # one approach:
        # distributing markup across inputs according to input cost
        #agent_markup_by_outputs = np.divide(
        #    output_price_estimates,
        #    (input_price_estimates @ self.gamestate.production_chain.adj_matrix),
        #    where=output_has_inputs,
        #    out=np.zeros_like(output_price_estimates)
        #)
        # we computed a sclar for each out, but then we can collapse back to per agent
        # NOTE: this assumes each agent only sells one good
        #max_buy_prices = agent_markup_by_outputs.max(axis=1)[:,np.newaxis] * input_price_estimates
        # and we apply that across the input goods, according to the input prices
        #inventory_balance = np.divide(self.inventory, self.production_goods, where=self.production_goods > 0, out=np.zeros_like(self.inventory))

        # another approach:
        # compute a deficit vs what we need to produce and distribute what we
        # think we can sell for across those according to that proportion
        #TODO: this approach can lead to big swings in max buy price when we
        # happen to make a transaction to fill up our inventory
        # and then when we need that good again it can take a while for prices
        # to increase

        # desire just one batch of input/output if we've never sold output
        desired_input_batches = np.full((self.num_agents,), self.desired_input_batches)
        desired_input_batches[np.isclose((self.sell_rate_estimates * self.agent_goods).max(axis=1), 0.)] = 1

        deficit = self.production_goods * self.batch_sizes.sum(axis=1)[:,np.newaxis] * desired_input_batches[:,np.newaxis] - self.smoothed_inventory
        deficit = deficit.clip(0, np.inf, out=deficit)

        # take a floor of deficit of 1 to avoid prices going to zero
        deficit[self.production_goods > 0] = np.clip(deficit[self.production_goods > 0], 1, np.inf)
        deficit_proportion = (deficit/np.divide(
                deficit.sum(axis=1),
                (self.production_goods>0).sum(axis=1),
                where=self.production_goods.sum(axis=1)>0,
                out=np.ones((self.num_agents,))
            )[:,np.newaxis]
        )

        # take the maximum here in case the floor of our sell price has risen
        # above historical market levels
        output_price_estimates = np.maximum(output_price_estimates, self.min_sell_prices)
        price_scale = np.divide(
                output_price_estimates,
                (deficit_proportion * input_price_estimates @ self.gamestate.production_chain.adj_matrix),
                where=output_has_inputs,
                out=np.zeros_like(output_price_estimates)
        )

        max_buy_prices = price_scale.sum(axis=1)[:,np.newaxis] * deficit_proportion * input_price_estimates

        max_buy_prices[self.buy_interest == 0] = 0.

        # the max price times production goods should our output price estimate
        assert np.all(
            np.isclose(
                (max_buy_prices * self.production_goods).sum(axis=1),
                np.multiply(output_price_estimates, self.agent_goods, where=output_price_estimates<np.inf, out=np.zeros_like(output_price_estimates)).sum(axis=1)
            )
            | ~self.agent_goods[:,self.gamestate.production_chain.ranks[0]:].sum(axis=1).astype(bool)
        )

        # for goods that chronically cannot be purchased, increase the max price
        # by the same amount that the underlying price will increase

        # also calculate our expected production capacity at the current prices
        # (assume those are the max buy price)
        input_production_cost = (self.production_goods * self.buy_prices).sum(axis=1)
        expected_production_capacity = np.divide(self.balance + (self.inventory * self.buy_prices).sum(axis=1), input_production_cost, where=input_production_cost > PRICE_EPS, out=np.zeros_like(self.balance))

        # need to make sure we do not raise prices so much that we can't afford
        # a batch
        #TODO: how to set the number of ticks to wait at cannot buy before we violate the cost of inputs < price of output?
        needed_for_one_batch = (self.batch_sizes.max(axis=1)[:,np.newaxis] * self.production_goods)
        affordable_amount = (expected_production_capacity[:,np.newaxis] * self.production_goods)
        max_buy_prices[(self.cannot_buy_ticks > self.cannot_buy_ticks_investment_cutoff) & (affordable_amount > needed_for_one_batch)] = self.max_buy_prices[(self.cannot_buy_ticks > self.cannot_buy_ticks_investment_cutoff) & (affordable_amount > needed_for_one_batch)] * (1+self.cannot_buy_gamma)

        return max_buy_prices

    def set_prices(self, buys_value:npt.NDArray[np.float64], buys:npt.NDArray[np.float64], sales_value:npt.NDArray[np.float64], sales:npt.NDArray[np.float64], tick_delta:int) -> None:
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

        input_price_estimates = np.divide(self.input_value_estimates, self.input_volume_estimates, where=self.input_volume_estimates > 0, out=np.zeros_like(self.input_value_estimates))
        output_price_estimates = np.divide(self.output_value_estimates, self.output_volume_estimates, where=self.output_volume_estimates > 0, out=np.zeros_like(self.output_value_estimates))

        self.no_sale_ticks[(sales == 0) & (self.agent_goods > 0)] += tick_delta
        self.no_sale_ticks[(sales > 0) & (self.agent_goods > 0)] = 0
        self.no_buy_ticks[(buys == 0) & (self.buy_interest > 0)] += tick_delta
        self.no_buy_ticks[(buys > 0) & (self.buy_interest > 0)] = 0

        cannot_sell = (np.subtract(self.sell_prices, self.min_sell_prices, where=self.agent_goods > 0, out=np.zeros_like(self.sell_prices)) < PRICE_EPS) & (self.inventory > 0) & (self.agent_goods > 0) & (sales == 0)
        self.cannot_sell_ticks[cannot_sell] += tick_delta
        self.cannot_sell_ticks[~cannot_sell] = 0

        assert np.all(self.cannot_sell_ticks[sales > 0] == 0.)

        cannot_buy = (np.subtract(self.buy_prices, self.max_buy_prices, where=self.buy_interest>0, out=np.zeros_like(self.buy_prices)) > -PRICE_EPS) & (self.last_buy_budget > 0) & (buys == 0)

        # if we were already non-zero and we had no buys, then we still cannot buy
        #((self.cannot_buy_ticks > 0) & (buys == 0)) => cannot_buy
        #P => Q <=> ~P | Q
        # also allow cases where our last budget was non-positive, but we have enough goods on hand to produce a batch of output
        #if np.any(~((~((self.cannot_buy_ticks > 0) & (buys == 0)) | cannot_buy) | ((self.last_buy_budget <= 0) & (self.inventory * self.buy_interest - self.production_goods*self.batch_sizes.max(axis=1)[:,np.newaxis] >= 0)))):
        #    breakpoint()

        self.cannot_buy_ticks[cannot_buy] += tick_delta
        self.cannot_buy_ticks[~cannot_buy] = 0.

        assert np.all(self.cannot_buy_ticks[buys > 0] == 0.)

        #last_min_sell_prices = self.min_sell_prices
        self.min_sell_prices = self._compute_min_sell_prices(input_price_estimates, output_price_estimates)
        #assert np.all(np.divide(self.min_sell_prices, last_min_sell_prices, where=self.min_sell_prices < np.inf, out=np.zeros_like(self.min_sell_prices)) < 10)

        #last_max_buy_prices = self.max_buy_prices
        self.max_buy_prices = self._compute_max_buy_prices(input_price_estimates, output_price_estimates)
        self.min_buy_prices = np.clip(self.min_buy_prices, 0, self.max_buy_prices)
        #assert np.all(np.divide(self.max_buy_prices, last_max_buy_prices, where=self.max_buy_prices > 0, out=np.zeros_like(self.max_buy_prices)) < 10)

        # choose a price to sell goods
        # we want to maximize revenue (price * volume)
        # so we do gradient ascent on the revenue curve, using the most recent
        # two ticks to estimate the gradient
        # price_n+1 = price_n + gamma * grad_rev(price_n)
        # grad_rev(price_n) ~ (rev_n - rev_n-1) / (price_n - price_n-1)

        sell_price_delta = np.subtract(
                self.sell_prices, self.last_sell_prices,
                where=self.agent_goods > 0,
                out=np.full((self.num_agents, self.num_products), np.inf)
        )
        zero_price_delta = np.abs(sell_price_delta) < 1e-08# np.isclose(sell_price_delta, 0.)
        revenue_delta = sales_value - self.last_sales_value
        step = self.price_setting_gamma * np.divide(
                revenue_delta, sell_price_delta,
                where=~zero_price_delta,
                out=np.zeros_like(self.sell_prices)
        )
        step = step.clip(-self.sell_prices * self.rel_sell_price_step_max, self.sell_prices * self.rel_sell_price_step_max, out=step)
        new_sell_prices = self.sell_prices + step
        self.last_sell_prices = self.sell_prices
        self.last_sales_value = sales_value.copy()

        # if we didn't have a price delta, increase prices
        new_sell_prices[zero_price_delta] = self.sell_prices[zero_price_delta] * (1 + self.price_setting_gamma)
        # if we sold nothing, drop prices
        new_sell_prices[sales == 0] = self.sell_prices[sales == 0] * (1 - self.price_setting_gamma)
        # if we had nothing to sell, leave price
        new_sell_prices[self.last_inventory == 0] = self.sell_prices[self.last_inventory == 0]
        # and then clip prices to min prices (which handles goods we don't sell)
        self.sell_prices = new_sell_prices.clip(self.min_sell_prices, np.inf, out=new_sell_prices)

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

        step = self.price_setting_gamma * np.divide(
                amount_per_budget_delta, buy_price_delta,
                where=~np.isclose(buy_price_delta, 0.),
                out=np.zeros_like(self.buy_prices)
        )

        # where we had zero buys but had budget, increase prices
        # if we were less than the estimate, increase by half the distance
        step[(buys == 0) & (self.last_buy_budget > 0) & (self.buy_prices < input_price_estimates)] = (input_price_estimates[(buys == 0) & (self.last_buy_budget > 0) & (self.buy_prices < input_price_estimates)] - self.buy_prices[(buys == 0) & (self.last_buy_budget > 0) & (self.buy_prices < input_price_estimates)])/2.
        step[(buys == 0) & (self.last_buy_budget > 0) & (self.buy_prices >= input_price_estimates)] = self.buy_prices[(buys == 0) & (self.last_buy_budget > 0) & (self.buy_prices >= input_price_estimates)] * self.price_setting_gamma

        # don't step up or down too much
        step = step.clip(-(self.buy_prices * self.rel_buy_price_step_max), self.buy_prices * self.rel_buy_price_step_max, out=step)

        # make sure we don't raise our prices if we were able to spend all our budget on the prior step
        assert np.all((step <= 0) | (buys_value < self.last_buy_budget))

        new_buy_prices = self.buy_prices + step

        # if we bought nothing, increase prices
        #new_buy_prices[buys == 0] = (self.buy_prices[buys == 0] * (1+self.price_setting_no_buy_gamma))
        # if we had no budget, leave price
        new_buy_prices[self.last_buy_budget == 0] = self.buy_prices[self.last_buy_budget == 0]
        # keep cannot buy clamped to max buy price
        new_buy_prices[cannot_buy] = self.max_buy_prices[cannot_buy]

        assert np.all(new_buy_prices[self.buy_interest > 0] > 0)

        # for debugging assert failures below, hang on to this
        old_last_buy_prices = self.last_buy_prices
        self.last_buy_prices = self.buy_prices
        self.buy_prices = new_buy_prices.clip(0, self.max_buy_prices, out=new_buy_prices)
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
        #TODO: what do we do if the expected production capacity given the
        # balance and buy prices is less than one batch?
        # this basically means the agent can't afford to produce a single batch
        # they are kind of stuck unless the market price for inputs drops

        # desire just one batch of input/output if we've never sold output
        desired_input_batches = np.full((self.num_agents,), self.desired_input_batches)
        desired_input_batches[np.isclose((self.sell_rate_estimates * self.agent_goods).max(axis=1), 0.)] = 1
        desired_output_batches = np.full((self.num_agents,), self.desired_output_batches)
        desired_output_batches[np.isclose((self.sell_rate_estimates * self.agent_goods).max(axis=1), 0.)] = 1.

        if np.any((expected_production_capacity < 1.) & (self.agent_goods.argmax(axis=1) >= self.gamestate.production_chain.ranks[0]) & ((self.inventory * self.agent_goods).sum(axis=1) <= 0)):
            breakpoint()
        expected_production_capacity = np.floor(expected_production_capacity.clip(0, self.batch_sizes.sum(axis=1)*desired_input_batches, out=expected_production_capacity))

        self.buy_budget = expected_production_capacity[:,np.newaxis] * self.production_goods * self.buy_prices - self.inventory * self.buy_prices

        self.last_inventory = self.inventory.copy()

        # stop production if we're sitting on a lot of output already

        self.buy_budget[((self.batch_sizes * desired_output_batches[:, np.newaxis] - (self.inventory * self.agent_goods)).max(axis=1)[:,np.newaxis] * self.buy_interest) <= 0] = 0

        assert np.all(self.buy_budget.sum(axis=1) <= self.balance)
        assert np.all(self.sell_prices > 0)
        assert np.all(self.buy_prices >= 0)
        assert np.all(~(self.sell_prices < self.min_sell_prices))
        assert np.all(~(self.buy_prices > self.max_buy_prices))

        self.last_buys = buys.copy()
        self.last_buy_budget = self.buy_budget.copy()

    def valid_trades(self, buy_prices:npt.NDArray[np.float64], sell_prices:npt.NDArray[np.float64], budget:npt.NDArray[np.float64]) -> Iterator[Tuple[float, int, int, int, float, float]]:
        """
        Given market parameters, yields sequence of best transactions that
        continue to be valid after taking each.

        Best here means largest price difference between buyer and seller.

        yields tuples:
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

        valid_buys = (buy_prices <= budget/min_sale_amount) & (buy_prices*min_sale_amount <= self.balance[:,np.newaxis])
        valid_sells = self.inventory >= min_sale_amount

        max_buy_prices = np.amax(buy_prices, axis=0, where=valid_buys, initial=0)
        min_sell_prices = np.amin(sell_prices, axis=0, where=valid_sells, initial=np.inf)

        diffs = max_buy_prices - min_sell_prices

        # and find the product with the best deal
        product = util.choose_argmax(self.gamestate.random, diffs)
        #product = diffs.argmax()

        while diffs[product] > 0:
            # pick the buyer with max price for that product
            buyer_idx = util.choose_argmax(self.gamestate.random, buy_prices[valid_buys[:,product],product])
            buyer = np.where(valid_buys[:,product])[0][buyer_idx]
            # pick the seller with the min price for that product
            seller_idx = util.choose_argmin(self.gamestate.random, sell_prices[valid_sells[:,product],product])
            seller = np.where(valid_sells[:,product])[0][seller_idx]

            price = (buy_prices[buyer, product] + sell_prices[seller,product])/2

            # transact the biggest volume possible (inventory vs budget/balance)
            sale_amount = np.floor(min(self.inventory[seller, product], budget[buyer, product]/price, self.balance[buyer]/price))

            assert buyer != seller
            assert buy_prices[buyer, product] > 0
            assert sell_prices[seller, product] < np.inf
            assert sell_prices[seller, product] <= buy_prices[buyer, product]
            assert sale_amount > 0
            assert budget[buyer, product] >= sale_amount * price - PRICE_EPS
            assert self.balance[buyer] >= sale_amount * price - PRICE_EPS
            assert self.inventory[seller, product] >= sale_amount

            yield diffs[product], product, buyer, seller, price, sale_amount

            # notice that nothing about can change except entries for this
            # product, buyer, seller
            buyer_valid = (buy_prices[buyer, product] <= budget[buyer, product]/min_sale_amount) and (buy_prices[buyer, product]*min_sale_amount <= self.balance[buyer])
            seller_valid = self.inventory[seller, product] >= min_sale_amount
            if not buyer_valid:
                valid_buys[buyer, product] = buyer_valid
                max_buy_prices[product] = np.amax(buy_prices[:, product], axis=0, where=valid_buys[:,product], initial=0)
            if not seller_valid:
                valid_sells[seller, product] = seller_valid
                min_sell_prices[product] = np.amax(sell_prices[:, product], axis=0, where=valid_sells[:,product], initial=np.inf)
            if not buyer_valid or not seller_valid:
                diffs[product] = max_buy_prices[product] - min_sell_prices[product]
                product = util.choose_argmax(self.gamestate.random, diffs)

    def make_market(self, buy_prices:npt.NDArray[np.float64], sell_prices:npt.NDArray[np.float64], budget:npt.NDArray[np.float64]) -> Tuple[float, int, int, int, float, float]:
        """
        Given market parameters, find the "best" potential transaction.

        Best here means largest price difference between buyer and seller.

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

        assert util.isclose(buy_prices[buyer, product] - sell_prices[seller, product], diffs[product])

        #TODO: how to set a price?
        price = (buy_prices[buyer, product] + sell_prices[seller,product])/2

        # transact the biggest volume possible (inventory vs budget/balance)
        sale_amount = np.floor(min(self.inventory[seller, product], budget[buyer, product]/price, self.balance[buyer]/price))

        assert buyer != seller
        assert buy_prices[buyer, product] > 0
        assert sell_prices[seller, product] < np.inf
        assert sell_prices[seller, product] <= buy_prices[buyer, product]
        assert sale_amount > 0
        assert budget[buyer, product] >= sale_amount * price - PRICE_EPS
        assert self.balance[buyer] >= sale_amount * price - PRICE_EPS
        assert self.inventory[seller, product] >= sale_amount

        return diffs[product], product, buyer, seller, price, sale_amount

    def transact(self, diff:float, product_id:int, buyer:int, seller:int, price:float, sale_amount:float) -> None:
        """ Conducts the transaction indicated by parameters. """

        #NOTE: we assume every agent will have zero or one transaction per product per tick

        self.inventory[buyer, product_id] += sale_amount
        self.balance[buyer] -= sale_amount * price

        self.inventory[seller, product_id] -= sale_amount
        self.balance[seller] += sale_amount * price

        self.buy_budget[buyer, product_id] -= sale_amount * price

        self.gamestate.production_chain.observe_transaction(product_id, price, sale_amount)

        self.data_logger.transact(diff, product_id, buyer, seller, price, sale_amount)

    def estimate_profit(self) -> npt.NDArray[np.float64]:
        # price estimate (dollar per unit) = value (dollars) / volume (units)
        global_price_estimate = np.divide(self.global_value_estimate, self.global_volume_estimate, where=self.global_volume_estimate>0, out=np.zeros_like(self.global_value_estimate))

        # input volumes necessary to provide output volumes (outputs x inputs per output):
        units_per_tick  = self.global_volume_estimate * self.global_rate_estimate
        market_volume_breakdown = ((units_per_tick[:,np.newaxis] * self.gamestate.production_chain.adj_matrix.T))

        cost_per_tick = (market_volume_breakdown * global_price_estimate).sum(axis=1)

        # augment costs for first rank goods to include the fixed cost paid
        # during resource_injection
        # back into this from the output volume / tick times the fixed prices
        injection_prices = (self.gamestate.production_chain.prices / self.gamestate.production_chain.markup)
        cost_per_tick[:self.gamestate.production_chain.ranks[0]] = (
                injection_prices
                * self.global_volume_estimate
                * self.global_rate_estimate
        )[:self.gamestate.production_chain.ranks[0]]

        profit_per_tick = units_per_tick * global_price_estimate - cost_per_tick

        return profit_per_tick

    def run(self, max_ticks:int) -> None:
        self.logger.info("running simluation...")

        self.logger.info(f'beginning balance: {self.balance.sum()}')
        self.logger.info(f'beginning inventory: {self.inventory.sum()}')

        # keeps track of the number of units bought/sold each turn
        buys = np.empty((self.num_agents, self.num_products))
        buys_volume = np.empty((self.num_agents, self.num_products))
        buys_value = np.empty((self.num_agents, self.num_products))
        sales = np.empty((self.num_agents, self.num_products))
        sales_volume = np.empty((self.num_agents, self.num_products))
        sales_value = np.empty((self.num_agents, self.num_products))

        price_setting_buys_value = np.zeros_like(buys)
        price_setting_buys_volume = np.zeros_like(buys)
        price_setting_sales_value = np.zeros_like(buys)
        price_setting_sales_volume = np.zeros_like(buys)

        for _ in tqdm.tqdm(range(max_ticks)):
            # source resources into base of production chain and sink cost of
            # those resources
            self.source_resources(scale=10000)

            # sink resources from tips of production chain and source money of
            # those resources
            self.sink_products(scale=100)

            # try to make a market, do trade

            # take trades that are matched until there are no possible trades
            # (no valid price matches with non-zero inventory)
            buys[:,:] = 0.
            buys_volume[:,:] = 0.
            buys_value[:,:] = 0
            sales[:,:] = 0.
            sales_volume[:,:] = 0.
            sales_value[:,:] = 0.

            self.data_logger.start_trading()

            # update global supply/demand estimates
            self.supply_estimate = util.update_ema(
                self.supply_estimate,
                self.supply_alpha,
                (self.inventory * self.agent_goods).sum(axis=0)
            )
            self.demand_estimate = util.update_ema(
                self.demand_estimate,
                self.supply_alpha,
                np.divide(self.buy_budget, self.buy_prices, where=self.buy_prices>0, out=np.zeros_like(self.buy_budget)).sum(axis=0)
            )

            transactions_this_tick = 0
            #while (ret := self.make_market(self.buy_prices, self.sell_prices, self.buy_budget))[0] > 0:
            for ret in self.valid_trades(self.buy_prices, self.sell_prices, self.buy_budget):
                (diff, product, buyer, seller, price, sale_amount) = ret
                self.transact(diff, product, buyer, seller, price, sale_amount)
                buys[buyer, product] += 1
                buys_volume[buyer, product] += sale_amount
                buys_value[buyer, product] += sale_amount * price
                sales[seller, product] += 1
                sales_volume[seller, product] += sale_amount
                sales_value[seller, product] += sale_amount * price
                transactions_this_tick += 1

            price_setting_buys_value += buys_value
            price_setting_buys_volume += buys_volume
            price_setting_sales_value += buys_value
            price_setting_sales_volume += buys_volume

            # budget > 0, seller_inventory > 0, buy_prices.max() < sell_prices.min(), budget > 0
            #if np.any(
            #        (price_setting_buys_value.sum(axis=0) == 0)
            #        & ((self.inventory * self.agent_goods).sum(axis=0) > 0)
            #        & (self.buy_prices.max(axis=0) > np.amin(
            #            self.sell_prices,
            #            axis=0,
            #            where=(self.inventory * self.agent_goods) > 0,
            #            initial=np.inf)
            #        )
            #        & (self.last_buy_budget.sum(axis=0) > 0)
            #):
            #    breakpoint()

            #if transactions_this_tick == 0 and self.ticks > 1000:
            #    breakpoint()

            # make sure that agent's are not overproducing
            # this is a little rough since it's plausible they might acquire a
            # huge amount of cheap inputs that would exceed this, but we'll do
            # the assert really high and that becomes vanishingly unlikely.
            # OTOH, a failure to limit production seems likely to cause the
            # inventory to grow very very large, so there's a big difference
            # between a small violation and the kind of issue I'm trying to
            # catch

            assert np.all(self.inventory * (1-self.agent_goods) <= self.production_goods * self.batch_sizes.sum(axis=1)[:, np.newaxis] * self.desired_output_batches * 1e3)

            # the buys and sales data are reciprocal and should contain the
            # same aggregate information
            assert np.all(np.isclose(buys_value.sum(axis=0), sales_value.sum(axis=0)))
            assert np.all(np.isclose(buys_volume.sum(axis=0), sales_volume.sum(axis=0)))
            assert np.all(np.isclose(buys.sum(axis=0), sales.sum(axis=0)))

            # update agent beliefs about transactions
            self.input_value_estimates[buys>0], self.input_volume_estimates[buys>0] = util.update_vema(
                self.input_value_estimates[buys>0], self.input_volume_estimates[buys>0],
                self.price_vema_alpha,
                buys_value[buys>0]/buys[buys>0], buys_volume[buys>0]/buys[buys>0])
            self.output_value_estimates[sales>0], self.output_volume_estimates[sales>0] = util.update_vema(
                    self.output_value_estimates[sales>0], self.output_volume_estimates[sales>0],
                    self.price_vema_alpha,
                    sales_value[sales>0] / sales[sales>0], sales_volume[sales>0]/sales[sales>0])
            self.buy_rate_estimates = util.update_ema(
                    self.buy_rate_estimates, self.rate_alpha, buys)
            self.sell_rate_estimates = util.update_ema(
                    self.sell_rate_estimates, self.rate_alpha, sales)

            # update global transaction estimates
            total_sales = sales.sum(axis=0)
            total_buys = buys.sum(axis=0)
            # except not for final goods, which we handle in sink_products
            non_last_products = np.full((self.num_products,), False)
            non_last_products[0:-self.gamestate.production_chain.ranks[-1]] = True
            self.global_value_estimate[(total_buys>0) & non_last_products], self.global_volume_estimate[(total_buys>0) & non_last_products] = util.update_vema(
                self.global_value_estimate[(total_buys>0) & non_last_products], self.global_volume_estimate[(total_buys>0) & non_last_products],
                self.global_price_alpha,
                buys_value.sum(axis=0)[(total_buys>0) & non_last_products]/total_buys[(total_buys>0) & non_last_products], buys_volume.sum(axis=0)[(total_buys>0) & non_last_products]/total_buys[(total_buys>0) & non_last_products]
            )
            self.global_rate_estimate[non_last_products] = util.update_ema(self.global_rate_estimate[non_last_products], self.global_price_alpha, total_buys[non_last_products])

            # produce goods according to the production chain
            self.produce_goods()

            if self.ticks % self.price_setting_period == 0:
                self.set_prices(price_setting_buys_value, price_setting_buys_volume, price_setting_sales_value, price_setting_sales_volume, self.price_setting_period)
                price_setting_buys_value[:,:] = 0.
                price_setting_buys_volume[:,:] = 0.
                price_setting_sales_value[:,:] = 0.
                price_setting_sales_volume[:,:] = 0.

                self.data_logger.set_prices()

            self.smoothed_inventory = util.update_ema(self.smoothed_inventory, self.smoothed_inventory_alpha, self.inventory)
            self.data_logger.end_trading()

            self.ticks += 1

    def log_report(self) -> None:

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

        self.logger.info(f'profit per tick: {self.estimate_profit()}')

def main() -> None:
    with contextlib.ExitStack() as context_stack:
        logging.basicConfig(
                format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
                stream=sys.stderr,
                level=logging.INFO
        )
        warnings.filterwarnings("error")
        mgr = context_stack.enter_context(util.PDBManager())

        data_logger = context_stack.enter_context(EconomyDataLogger(enabled=True))

        econ = EconomySimulation(data_logger)
        econ.initialize(num_agents=300)


        # warm up anything (helps if we're profiling)
        econ .run(max_ticks=1)
        econ.run(max_ticks=50000)
        #econ.run(max_ticks=2000)

        econ.log_report()

        data_logger.end_simulation()

if __name__ == "__main__":
    main()
