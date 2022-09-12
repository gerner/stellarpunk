""" Tests for econ sim. """

import pytest
import numpy as np
import numpy.testing as nptest

from stellarpunk import econ_sim, core, generate

@pytest.fixture
def sim() -> econ_sim.EconomySimulation:
    s = econ_sim.EconomySimulation()
    gamestate = core.Gamestate()
    generator = generate.UniverseGenerator(gamestate)
    gamestate.production_chain = generator.generate_chain(
            n_ranks=1,
            min_per_rank=(2,),
            max_per_rank=(2,),
            min_final_inputs=2,
            )
    s.initialize(gamestate=gamestate, production_chain=gamestate.production_chain)

    return s

def test_source_resources(sim:econ_sim.EconomySimulation) -> None:
    # this assumes we have exactly one agent for every good
    sim.balance = (sim.gamestate.production_chain.prices / sim.gamestate.production_chain.markup * 5)
    sim.inventory[:,:] = 0

    # quick sanity checks on my test setup (which is subtle)
    assert len(sim.balance.shape) == 1
    assert sim.balance.shape[0] == sim.num_agents
    nptest.assert_array_less(0, sim.balance)
    assert len(sim.inventory.shape) == 2
    assert sim.inventory.shape[0] == sim.num_agents
    assert sim.inventory.shape[0] == sim.num_products

    sourcing_agents = sim.gamestate.production_chain.ranks[0]

    # source fundable resources
    original_balance = sim.balance.copy()
    original_inventory = sim.inventory.copy()
    sim.source_resources(scale=2)

    # non sources should be unchanged
    nptest.assert_equal(
            sim.balance[sourcing_agents:],
            original_balance[sourcing_agents:]
    )
    nptest.assert_equal(
            sim.inventory[sourcing_agents:],
            original_inventory[sourcing_agents:]
    )

    # sources should be changed
    nptest.assert_allclose(
            sim.balance[0:sourcing_agents],
            (original_balance - (sim.gamestate.production_chain.prices / sim.gamestate.production_chain.markup) * 2)[0:sourcing_agents]
    )
    nptest.assert_allclose(
            sim.inventory[0:sourcing_agents],
            original_inventory[0:sourcing_agents] + np.pad(np.eye(sourcing_agents)*2, ((0,0), (0,sim.num_agents - sourcing_agents)))
    )

    # source unfundable resources
    original_balance = sim.balance.copy()
    original_inventory = sim.inventory.copy()
    sim.source_resources(scale=4)

    # non sources should be unchanged
    nptest.assert_equal(sim.balance[sourcing_agents:], original_balance[sourcing_agents:])
    nptest.assert_equal(sim.inventory[sourcing_agents:], original_inventory[sourcing_agents:])

    # sources should be changed
    nptest.assert_allclose(
            sim.balance[0:sourcing_agents],
            (original_balance - (sim.gamestate.production_chain.prices / sim.gamestate.production_chain.markup) * 3)[0:sourcing_agents],
            atol=1e-08 # the default is 0 which messes this up since we're at zero
    )
    nptest.assert_allclose(
            sim.inventory[0:sourcing_agents],
            original_inventory[0:sim.gamestate.production_chain.ranks[0]] + np.pad(np.eye(sourcing_agents)*3, ((0,0),(0,sim.num_agents-sourcing_agents)))
    )

    # and that should zero out the balance of those folks
    nptest.assert_allclose(sim.balance[0:sim.gamestate.production_chain.ranks[0]], 0.)

def test_sink_products(sim:econ_sim.EconomySimulation) -> None:
    # this assumes we have exactly one agent for every good
    sim.balance[:] = 0.
    sim.inventory[:,:] = sim.agent_goods * 5

    # quick sanity checks on my test setup (which is subtle)
    assert len(sim.balance.shape) == 1
    assert sim.balance.shape[0] == sim.num_agents
    assert len(sim.inventory.shape) == 2
    assert sim.inventory.shape[0] == sim.num_agents
    assert sim.inventory.shape[0] == sim.num_products
    assert np.all((sim.inventory == 0).sum(axis=1) == sim.num_products - 1)

    sinking_agents = sim.gamestate.production_chain.ranks[-1]

    # source fundable resources
    original_balance = sim.balance.copy()
    original_inventory = sim.inventory.copy()
    sim.sink_products(scale=2)

    # non sinks should be unchanged
    nptest.assert_equal(
            sim.balance[:-sinking_agents],
            original_balance[:-sinking_agents]
    )
    nptest.assert_equal(
            sim.inventory[:-sinking_agents],
            original_inventory[:-sinking_agents]
    )

    # sources should be changed
    nptest.assert_allclose(
            sim.balance[-sinking_agents:],
            (original_balance + (sim.gamestate.production_chain.prices) * 2)[-sinking_agents:]
    )
    nptest.assert_allclose(
            sim.inventory[-sinking_agents:],
            original_inventory[-sinking_agents:] - np.pad(np.eye(sinking_agents)*2, ((0,0), (sim.num_agents - sinking_agents,0)))
    )

    # source unfundable resources
    original_balance = sim.balance.copy()
    original_inventory = sim.inventory.copy()
    sim.sink_products(scale=4)

    # non sources should be unchanged
    nptest.assert_equal(sim.balance[:-sinking_agents], original_balance[:-sinking_agents])
    nptest.assert_equal(sim.inventory[:-sinking_agents], original_inventory[:-sinking_agents])

    # sources should be changed
    nptest.assert_allclose(
            sim.balance[-sinking_agents:],
            (original_balance + (sim.gamestate.production_chain.prices) * 3)[-sinking_agents:],
            atol=1e-08 # the default is 0 which messes this up since we're at zero
    )
    nptest.assert_allclose(
            sim.inventory[-sinking_agents:],
            original_inventory[-sinking_agents:] - np.pad(np.eye(sinking_agents)*3, ((0,0),(sim.num_agents-sinking_agents,0)))
    )

    # and that should zero out the inventory of those folks
    nptest.assert_allclose(sim.inventory[-sinking_agents:], 0.)

def test_make_market(sim:econ_sim.EconomySimulation) -> None:

    # no positive trades
    buy_prices = np.zeros((sim.num_agents, sim.num_products))
    sell_prices = np.full((sim.num_agents, sim.num_products), np.inf)
    buy_budget = np.full((sim.num_agents, sim.num_products), np.inf)

    (price_diff, product, buyer, seller, price, amount) = sim.make_market(buy_prices, sell_prices, buy_budget)

    assert price_diff < 0

    # no supplyable trades
    sim.inventory[:,:] = 0
    sim.balance[:] = 1e3
    buy_prices = sim.buy_interest * 10
    sell_prices = sim.agent_goods * 5

    (price_diff, product, buyer, seller, price, amount) = sim.make_market(buy_prices, sell_prices, buy_budget)

    assert price_diff < 0

    # no affordable trades
    sim.inventory[:,:] = sim.agent_goods * 1e2
    sim.balance[:] = 0
    buy_prices = sim.buy_interest * 10
    sell_prices = sim.agent_goods * 5

    (price_diff, product, buyer, seller, price, amount) = sim.make_market(buy_prices, sell_prices, buy_budget)

    assert price_diff < 0

    # several feasible trades, make sure we get the best
    sim.inventory[:,:] = sim.agent_goods * 1e2
    sim.balance[:] = 1e3

    # sale prices are all the same
    sell_prices = sim.agent_goods * 1
    # higest prices will be the final row (agent with highest id)
    # we zero out self prices (`* 1-sim.agent_goods`)
    # so last agent for second to last product
    # highest price should be the highest agent minus one for the highest 
    buy_prices = np.arange((sim.num_agents*sim.num_products)).reshape((sim.num_agents, sim.num_products)) * (1-sim.agent_goods)

    (price_diff, product, buyer, seller, price, amount) = sim.make_market(buy_prices, sell_prices, buy_budget)

    # assumes exactly as many agents as products and each agent sells the
    # product with id equal to their agent id
    assert price_diff > 0
    assert price_diff == buy_prices[buyer, product] - sell_prices[seller, product]
    assert product == sim.num_products - 2
    assert buyer == sim.num_agents - 1
    assert seller == sim.num_agents - 2
    assert sim.agent_goods[seller, product] == 1
    assert price >= sell_prices[seller, product]
    assert price <= buy_prices[buyer, product]
    assert amount > 0
    assert amount < sim.inventory[seller, product]
    assert amount * price <= sim.balance[buyer]

def test_make_market_complex(sim:econ_sim.EconomySimulation) -> None:
    # several feasible trades, make sure we get the best
    buy_budget = np.full((sim.num_agents, sim.num_products), np.inf)
    sim.inventory[:,:] = sim.agent_goods * 1e2
    sim.balance[:] = 1e3


    # sale prices are all the same
    sell_prices = sim.agent_goods * 1
    # higest prices will be the final row (agent with highest id)
    # we zero out self prices (`* 1-sim.agent_goods`)
    # so last agent for second to last product
    # highest price should be the highest agent minus one for the highest 
    buy_prices = np.arange((sim.num_agents*sim.num_products)).reshape((sim.num_agents, sim.num_products)) * (1-sim.agent_goods)

    # the obvious best buyer is broke
    sim.balance[-1] = 0.
    # the obvious best seller has no inventory
    sim.inventory[-1,-1] = 0.

    (price_diff, product, buyer, seller, price, amount) = sim.make_market(buy_prices, sell_prices, buy_budget)

    # assumes exactly as many agents as products and each agent sells the
    # product with id equal to their agent id
    assert price_diff > 0
    assert price_diff == buy_prices[buyer, product] - sell_prices[seller, product]
    assert product == sim.num_products - 3
    assert buyer == sim.num_agents - 2
    assert seller == sim.num_agents - 3
    assert sim.agent_goods[seller, product] == 1
    assert price >= sell_prices[seller, product]
    assert price <= buy_prices[buyer, product]
    assert amount > 0
    assert amount < sim.inventory[seller, product]
    assert amount * price <= sim.balance[buyer]

def test_produce_goods(sim:econ_sim.EconomySimulation) -> None:
    first_rank_agents = np.where(sim.agent_goods[:, :sim.gamestate.production_chain.ranks[0]].sum(axis=1)>0)
    other_agents = np.where(sim.agent_goods[:, sim.gamestate.production_chain.ranks[0]:].sum(axis=1)>0)

    # no inputs, no goods produced
    sim.inventory[:,:] = 0.
    sim.produce_goods()

    nptest.assert_allclose(sim.inventory, 0.)

    # everyone has exactly enough inputs to produce one unit of output
    sim.inventory[:,:] = sim.production_goods.copy()
    sim.produce_goods()

    # first rank agents produce nothing, everyone else produces just one unit
    nptest.assert_allclose(sim.inventory[first_rank_agents], 0.)
    nptest.assert_allclose(sim.inventory[other_agents], sim.agent_goods[other_agents])

    # everyone has exactly enough inputs to produce several units of output
    sim.inventory[:,:] = sim.production_goods * 3
    sim.produce_goods()

    # first rank agents produce nothing, everyone else produces 3 units
    nptest.assert_allclose(sim.inventory[first_rank_agents], 0.)
    nptest.assert_allclose(sim.inventory[other_agents], sim.agent_goods[other_agents] * 3)

    # everyone has exactly enough inputs produce one unit of output
    # then they have some more leftover (imbalanced)

    sim.inventory[:,:] = sim.production_goods * 3
    extra_inventory = sim.production_goods * 0.5
    # find one case where multiple inputs are needed, add a bunch of one input
    multi_input_agent = np.where((sim.production_goods > 0).sum(axis=1) > 1)[0][0]
    multi_input_input = np.where(sim.production_goods[multi_input_agent] > 0)[0][0]
    extra_inventory[multi_input_agent, multi_input_input] = sim.production_goods[multi_input_agent, multi_input_input] * 1000
    sim.inventory += extra_inventory
    sim.produce_goods()

    # first rank agents produce nothing, everyone else produces 3 units
    nptest.assert_allclose(sim.inventory[first_rank_agents], extra_inventory[first_rank_agents])
    nptest.assert_allclose(sim.inventory[other_agents], sim.agent_goods[other_agents] * 3 + extra_inventory[other_agents])


