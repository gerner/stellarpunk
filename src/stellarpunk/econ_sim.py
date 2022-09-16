""" Tool to run simple simulation of a market. """

import sys
import io
import logging
import contextlib
import warnings
from typing import TextIO, BinaryIO, Optional, Tuple, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd # type: ignore
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

def read_tick_log_to_df(f:BinaryIO, index_name:Optional[str]=None, column_names:Optional[Sequence[str]]=None) -> pd.DataFrame:
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
        matrixes.append(m)
        ticks.append(np.full((row_count,), tick))

    df = pd.DataFrame(np.concatenate(matrixes), columns=column_names)
    df["tick"] = pd.Series(np.concatenate(ticks))
    df.index = pd.Series(np.tile(np.arange(row_count), len(matrixes)))
    if index_name is not None:
        df.index.set_names(index_name, inplace=True)
    df.set_index("tick", append=True, inplace=True)

    return df

class EconomySimulation:
    def __init__(self,
            transaction_log:Optional[TextIO]=None,
            inventory_log:Optional[BinaryIO]=None,
            balance_log:Optional[BinaryIO]=None,
            buy_prices_log:Optional[BinaryIO]=None,
            sell_prices_log:Optional[BinaryIO]=None,
            production_efficiency_log:Optional[BinaryIO]=None,
            ) -> None:
        self.logger = logging.getLogger(util.fullname(self))

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
        self.last_sell_prices = np.zeros((self.num_agents, self.num_products))
        self.last_sales_value = np.zeros((self.num_agents, self.num_products))
        self.sell_prices = np.zeros((self.num_agents, self.num_products))
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

        self.transaction_log = transaction_log
        self.inventory_log = serialization.TickMatrixWriter(inventory_log) if inventory_log is not None else None
        self.balance_log = serialization.TickMatrixWriter(balance_log) if balance_log is not None else None
        self.buy_prices_log = serialization.TickMatrixWriter(buy_prices_log) if buy_prices_log is not None else None
        self.sell_prices_log = serialization.TickMatrixWriter(sell_prices_log) if sell_prices_log is not None else None
        self.production_efficiency_log = serialization.TickMatrixWriter(production_efficiency_log) if production_efficiency_log is not None else None

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
                n_ranks=1,
                min_per_rank=(2,),
                max_per_rank=(2,),
                min_final_inputs=1,
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

        self.inventory = np.zeros((self.num_agents, self.num_products))
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

        self.buy_prices = self.gamestate.random.uniform(self.buy_interest * self.gamestate.production_chain.prices, self.buy_interest * self.gamestate.production_chain.prices * 1.1)
        self.buy_budget = np.full((self.num_agents, self.num_products), np.inf)
        self.sell_prices = self.gamestate.random.uniform(sell_interest * self.gamestate.production_chain.prices * 1/1.1, sell_interest * self.gamestate.production_chain.prices)
        self.sell_prices[(1-sell_interest) > 0] = np.inf

        self.last_sell_prices = self.sell_prices
        self.last_sales_value = np.zeros((self.num_agents, self.num_products))

        #TODO: these should be set to guarantee break-even
        # max buy prices set to markup * base price
        self.max_buy_prices = self.buy_interest * (self.gamestate.production_chain.prices * self.gamestate.production_chain.markup)
        # min sell prices set to 1/markup * base price
        self.min_sell_prices = sell_interest * (self.gamestate.production_chain.prices * 1/self.gamestate.production_chain.markup)

        self.min_buy_prices = self.buy_interest * (self.gamestate.production_chain.prices * 1/self.gamestate.production_chain.markup * 0.9)
        self.max_sell_prices = sell_interest * (self.gamestate.production_chain.prices * self.gamestate.production_chain.markup * 1.1)
        self.max_sell_prices[sell_interest == 0] = np.inf

        # initialze our estimates of prices with the production chain prices
        self.input_value_estimates = np.repeat(self.gamestate.production_chain.prices[np.newaxis, :], self.num_agents, axis=0)
        self.input_volume_estimates = np.ones((self.num_agents, self.num_products))
        self.output_value_estimates = np.repeat(self.gamestate.production_chain.prices[np.newaxis, :], self.num_agents, axis=0)
        self.output_volume_estimates = np.ones((self.num_agents, self.num_products))

        assert np.all(~(self.max_sell_prices < self.min_sell_prices))
        assert np.all(~(self.max_buy_prices < self.min_buy_prices))

        self.ticks = 0

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

        if self.production_efficiency_log is not None:
            self.production_efficiency_log.write(self.ticks, np.divide(goods_produced, self.batch_sizes, where=self.batch_sizes > 0, out=np.zeros((self.num_agents, self.num_products))))

        return goods_produced

    def source_resources(self, scale:float=1.) -> None:
        resource_injection = np.zeros((self.num_agents, self.num_products))
        # every agent that sells the source resources gets a load of scale
        # resources
        resource_injection[:,0:self.gamestate.production_chain.ranks[0]] += self.agent_goods[:,0:self.gamestate.production_chain.ranks[0]] * scale

        # price is fixed, discounted by the markup
        #   this is sort of a hard constraint on the economy
        injection_prices = self.gamestate.production_chain.prices / self.gamestate.production_chain.markup

        # cap the injection to what sourcers can afford
        # NOTE: adding PRICE_EPS here to allow for approximately equal
        # transactions
        resource_injection = np.floor(
            np.clip(
                resource_injection,
                0.,
                (self.balance + PRICE_EPS) / injection_prices, out=resource_injection
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
        resource_sink = np.zeros((self.num_agents, self.num_products))

        # every agent that sells the sink goods gets to sell up to the scale
        resource_sink[:,-self.gamestate.production_chain.ranks[-1]:] += self.agent_goods[:,-self.gamestate.production_chain.ranks[-1]:] * scale

        # cap sinks at the inventory
        resource_sink = np.clip(resource_sink, 0, self.inventory, out=resource_sink)

        # goods are fix priced at the final output prices
        #   this is sort of a hard constraint on the economy
        source_prices = self.gamestate.production_chain.prices

        self.inventory -= resource_sink

        self.inventory[np.isclose(self.inventory, 0.)] = 0
        assert np.all(self.inventory >= 0.)

        self.balance += (resource_sink * source_prices).sum(axis=1)
        self.gamestate.production_chain.goods_sunk += resource_sink.sum(axis=0)
        self.gamestate.production_chain.value_sunk += (resource_sink * source_prices).sum(axis=0)

    def set_prices(self, buys_value:npt.NDArray[np.float64], buys:npt.NDArray[np.float64], sales_value:npt.NDArray[np.float64], sales:npt.NDArray[np.float64]) -> None:
        """ Agents set prices for the next round.

        Happens after trades happen but before resulting production.
        """

        # price based on sales:
        # increase price if you made a sale (seller),
        # decrease price if you made a buy (buyer)

        #self.buy_prices[buys > 0] *= (1./1.05)
        #self.buy_prices[buys == 0] *= (1.05)
        #self.buy_prices = np.clip(self.buy_prices, 0, self.max_buy_prices)
        #self.sell_prices[sales > 0] *= 1.05
        #self.sell_prices[sales == 0] *= (1./1.05)
        #self.sell_prices = np.clip(self.sell_prices, self.min_sell_prices, np.inf)

        # production pricing:
        # maximize (output_price - sum_inputs(input_price * input_per_output)) * volume_sold
        # output price based on how much stock you have
        # input price based on stock and balance
        # if input price goes up, other input pricess must go down (in a
        # sustainable way) or output price must go up

        # try: two step
        # 1. choose sale price inverse with inventory
        # 2. choose input prices consistent with that sale price

        # sale price must be at least the sum of the input amounts times our
        # best estimate of the input costs (which could be a moving average of
        # the price we've paid)
        # outside of that, we want maximize our revenue (price * volume)
        # we can assume that volume moves in the opposite direction of price
        # so we can experiment until we achieve some maximum
        # two interesting edge cases consistent with this:
        #   1. we sold all our inventory => raise price
        #   2. we sold zero => lower price

        # estimate min sell price based on the price we're paying for inputs
        # price_output_i >= sum_inputs_for_i(price_input_j * need_input_j)
        # NOTE: this assumes every agent produces/sells exactly one kind of item
        input_price_estimates = self.input_value_estimates / self.input_volume_estimates
        self.min_sell_prices = (input_price_estimates @ self.gamestate.production_chain.adj_matrix * self.agent_goods)
        # force agents that sell rank 0 goods min sell price to be the price
        # they have to buy resources for
        self.min_sell_prices[:,:self.gamestate.production_chain.ranks[0]] += self.agent_goods[:,:self.gamestate.production_chain.ranks[0]] * (self.gamestate.production_chain.prices[:self.gamestate.production_chain.ranks[0]] / self.gamestate.production_chain.markup[:self.gamestate.production_chain.ranks[0]])

        # estimate max buy prices based on the price we're selling output for
        # this is a total spend we can afford on inputs, we can distribute that
        # over the input goods according to how many of each we need, along with
        # some estimate of how we've observed input prices are related to each
        # other.

        output_price_estimates = self.output_value_estimates / self.output_volume_estimates
        # max buy price = scalar * observed_prices
        # because sum_inputs( max_price_j * need_j) = price_output
        # and we want to preserve the ratios between input prices, i.e.
        #   max_prices = scalar * observed_prices
        # we also ignore cases where we're not selling or there are no inputs
        # shape is self.num_agents by self.num_products
        output_has_inputs = self.agent_goods * self.gamestate.production_chain.adj_matrix.sum(axis=0)[np.newaxis, :] > 0
        assert output_has_inputs.shape == (self.num_agents, self.num_products)
        agent_scalar_by_outputs = np.divide(
            output_price_estimates,
            (input_price_estimates @ self.gamestate.production_chain.adj_matrix),
            where=output_has_inputs,
            out=np.zeros((self.num_agents, self.num_products))
        )

        # we computed a sclar for each out, but then we can collapse back to per agent
        # NOTE: this assumes each agent only sells one good
        self.max_buy_prices = agent_scalar_by_outputs.max(axis=1)[:,np.newaxis] * input_price_estimates
        self.max_buy_prices[self.buy_interest == 0] = 0.
        self.min_buy_prices = np.clip(self.min_buy_prices, 0, self.max_buy_prices)

        # choose a price to sell goods
        # we want to maximize revenue (price * volume)
        # so we do gradient ascent on the revenue curve, using the most recent
        # two ticks to estimate the gradient
        # price_n+1 = price_n + gamma * grad_rev(price_n)
        # grad_rev(price_n) ~ (rev_n - rev_n-1) / (price_n - price_n-1)
        gamma = 0.01
        price_delta = np.subtract(
                self.sell_prices, self.last_sell_prices,
                where=self.agent_goods > 0,
                out=np.full((self.num_agents, self.num_products), np.inf)
        )
        revenue_delta = sales_value - self.last_sales_value
        new_sell_prices = self.sell_prices + gamma * np.divide(
                revenue_delta, price_delta,
                where=~np.isclose(price_delta, 0.),
                out=revenue_delta
        )
        self.last_sell_prices = self.sell_prices
        self.last_sales_value = sales_value.copy()
        # if we sold nothing, drop prices
        new_sell_prices[sales == 0] = self.sell_prices[sales == 0] * (1 - gamma)
        # if we had nothing to sell, leave price
        new_sell_prices[self.last_inventory == 0] = self.sell_prices[self.last_inventory == 0]
        # and then clip prices to min prices (which handles goods we don't sell)
        self.sell_prices = np.clip(new_sell_prices, self.min_sell_prices, np.inf)

        # choose a price to buy goods
        # we want to balance our inventory so we don't have a surplus of any
        # good relative to what we need for production

        input_balance = np.divide(self.inventory, self.production_goods, out=np.zeros_like(self.inventory), where=self.production_goods!=0)

        want_to_buy = (self.buy_interest > 0)#(input_balance < 1) & (self.buy_interest > 0)
        #self.buy_prices[want_to_buy] = self.gamestate.random.uniform(self.min_buy_prices[want_to_buy], self.max_buy_prices[want_to_buy])
        #self.buy_prices[(input_balance >= 1) & (self.buy_interest > 0)] = 0

        self.buy_prices[want_to_buy] = self.max_buy_prices[want_to_buy]

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
        expected_production_capacity = np.divide(self.balance + (self.inventory * self.buy_prices).sum(axis=1), input_production_cost, where=input_production_cost > 0, out=np.zeros((self.num_agents,)))

        # clip the expected production capacity to limit how many extra goods
        # we keep on hand. This is sort of like a just-in-time supply chain.
        #TODO: how to set choose how many batches we want to keep on hand?
        expected_production_capacity = np.clip(expected_production_capacity, 0, self.batch_sizes.sum(axis=1) * 10)

        self.buy_budget = expected_production_capacity[:,np.newaxis] * self.production_goods * self.buy_prices - self.inventory * self.buy_prices

        self.last_inventory = self.inventory.copy()

        assert np.all(self.balance + PRICE_EPS - self.buy_budget.sum(axis=1) >= 0.)
        assert np.all(self.sell_prices > 0)
        assert np.all(self.buy_prices >= 0)
        assert np.all(~(self.sell_prices < self.min_sell_prices))
        assert np.all(~(self.buy_prices > self.max_buy_prices))


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

        if self.transaction_log:
            self.transaction_log.write(f'{self.ticks}\t{seller}\t{buyer}\t{product_id}\t{sale_amount}\t{price}\n')

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
            self.sink_products()

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

            if self.buy_prices_log:
                self.buy_prices_log.write(self.ticks, self.buy_prices)
            if self.sell_prices_log:
                self.sell_prices_log.write(self.ticks, self.sell_prices)

            transactions_this_tick = 0
            while (ret := self.make_market(self.buy_prices, self.sell_prices, self.buy_budget))[0] > 0:
                (diff, product, buyer, seller, price, sale_amount) = ret
                self.transact(diff, product, buyer, seller, price, sale_amount)
                buys[buyer, product] += sale_amount
                buys_value[buyer, product] += sale_amount * price
                sales[seller, product] += sale_amount
                sales_value[seller, product] += sale_amount * price
                transactions_this_tick += 1

            #if transactions_this_tick == 0:
            #    breakpoint()

            # compute demand and supply surplus
            #demand_surplus = (buy_prices > 0).astype(float).sum(axis=0)
            #supply_surplus = (sell_prices < np.inf).astype(float).sum(axis=0)

            # produce goods according to the production chain
            self.produce_goods()
            self.set_prices(buys_value, buys, sales_value, sales)

            if self.balance_log:
                self.balance_log.write(self.ticks, self.balance)
            if self.inventory_log:
                self.inventory_log.write(self.ticks, self.inventory)

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

def main() -> None:
    with contextlib.ExitStack() as context_stack:
        logging.basicConfig(
                format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
                stream=sys.stderr,
                level=logging.INFO
        )
        warnings.filterwarnings("error")
        mgr = context_stack.enter_context(util.PDBManager())

        transaction_log = context_stack.enter_context(open("/tmp/transactions.log", "wt", 1024*1024))
        inventory_log = context_stack.enter_context(open("/tmp/inventory.log", "wb", 1024*1024))
        balance_log = context_stack.enter_context(open("/tmp/balance.log", "wb", 1024*1024))
        production_chain_log = context_stack.enter_context(open("/tmp/production_chain.log", "wb", 1024*1024))
        agent_goods_log = context_stack.enter_context(open("/tmp/agent_goods.log", "wb", 1024*1024))
        buy_prices_log = context_stack.enter_context(open("/tmp/buy_prices.log", "wb", 1024*1024))
        sell_prices_log = context_stack.enter_context(open("/tmp/sell_prices.log", "wb", 1024*1024))
        production_efficiency_log = context_stack.enter_context(open("/tmp/production_efficiency.log", "wb", 1024*1024))

        econ = EconomySimulation(transaction_log=transaction_log, inventory_log=inventory_log, balance_log=balance_log, buy_prices_log=buy_prices_log, sell_prices_log=sell_prices_log, production_efficiency_log=production_efficiency_log)
        econ.initialize(num_agents=-1)

        production_chain_log.write(serialization.save_production_chain(econ.gamestate.production_chain))
        agent_goods_log.write(msgpack.packb(econ.agent_goods, default=serialization.encode_matrix))
        #pd.DataFrame(econ.gamestate.production_chain.adj_matrix).to_csv(production_chain_log, sep="\t", header=False, index=False)
        #pd.DataFrame(econ.agent_goods).to_csv(agent_goods_log, sep="\t", header=False)
        econ.run(max_ticks=100000)

if __name__ == "__main__":
    main()
