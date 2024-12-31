import numpy as np

from stellarpunk import econ, agenda

from . import write_history

@write_history
def test_mining_agendum(gamestate, generator, sector, testui, simulator):
    # a ship and a station to sell at
    resource = 0
    ship = generator.spawn_ship(sector, -3000, 0, v=(0,0), w=0, theta=0)
    ship.cargo_capacity = 5e2

    #station resource needs to be one that consumes resource 0
    station_resource = np.where(gamestate.production_chain.adj_matrix[resource] > 0)[0][0]
    station = generator.spawn_station(sector, 5000, 0, resource=station_resource)
    sector.radius = 3000.
    asteroid = generator.spawn_asteroid(sector, 0, 5000, resource, 12.5e2)

    ship_owner = generator.spawn_character(ship)
    ship_owner.take_ownership(ship)
    ship.captain = ship_owner
    mining_agendum = agenda.MiningAgendum.create_mining_agendum(ship, ship_owner, gamestate)
    ship_owner.add_agendum(mining_agendum)

    station_owner = generator.spawn_character(station)
    station_owner.take_ownership(station)
    station_agent = econ.StationAgent.create_station_agent(station_owner, station, gamestate.production_chain, gamestate)
    station_agent._buy_price[0] = 10.
    station_agent._budget[0] = 10. * 5e2 * 3
    gamestate.representing_agent(station.entity_id, station_agent)

    expected_price = station_agent.buy_price(0)
    assert expected_price == 10.
    trader_starting_balance = mining_agendum.agent.balance()
    station_starting_balance = station_agent.balance()
    station_starting_budget = station_agent.budget(0)

    assert ship.cargo[0] == 0.
    assert station.cargo[0] == 0.
    assert asteroid.cargo[asteroid.resource] == 12.5e2

    # disable production by setting production time to more than our time horizon
    station.next_batch_time = gamestate.timestamp + 201
    mining_agendum.max_trips = 2
    testui.agenda.append(mining_agendum)
    testui.margin_neighbors = [ship]
    testui.eta = 200

    simulator.run()

    assert mining_agendum.is_complete()
    assert mining_agendum.round_trips == 2

    assert ship.cargo[0] == 0.
    assert np.isclose(station.cargo[0], 10e2)
    assert np.isclose(asteroid.cargo[asteroid.resource], 12.5e2 - 10e2)

    assert mining_agendum.agent.balance() == ship_owner.balance
    assert station_agent.balance() == station_owner.balance
    assert station_agent.inventory(0) == station.cargo[0]

    expected_total_value = expected_price * 10e2
    assert np.isclose(mining_agendum.agent.balance(), trader_starting_balance + expected_total_value)
    assert np.isclose(station_agent.balance(), station_starting_balance - expected_total_value)
    assert np.isclose(station_agent.budget(0), station_starting_budget - expected_total_value)

def test_mining_partial_transfer(gamestate, generator, sector, testui, simulator):
    # make sure that if buyer runs out of money partway through transfer,
    # things wrap up ok

    # a ship and a station to sell at
    resource = 0
    ship = generator.spawn_ship(sector, -3000, 0, v=(0,0), w=0, theta=0)
    ship.cargo_capacity = 5e2

    #station resource needs to be one that consumes resource 0
    station_resource = np.where(gamestate.production_chain.adj_matrix[resource] > 0)[0][0]
    station = generator.spawn_station(sector, 5000, 0, resource=station_resource)
    sector.radius = 3000.
    asteroid = generator.spawn_asteroid(sector, 0, 5000, resource, 12.5e2)

    # make sure owner has enough for 1.5 batches
    price = 10.
    ship_owner = generator.spawn_character(ship)
    ship_owner.take_ownership(ship)
    ship.captain = ship_owner
    mining_agendum = agenda.MiningAgendum.create_mining_agendum(ship, ship_owner, gamestate)
    ship_owner.add_agendum(mining_agendum)

    station_owner = generator.spawn_character(station, balance=1.5 * 5e2 * price + price/2)
    station_owner.take_ownership(station)
    station_agent = econ.StationAgent.create_station_agent(station_owner, station, gamestate.production_chain, gamestate)
    station_agent._buy_price[0] = price
    station_agent._budget[0] = price * 5e2 * 3
    gamestate.representing_agent(station.entity_id, station_agent)

    expected_price = station_agent.buy_price(0)
    trader_starting_balance = mining_agendum.agent.balance()
    station_starting_balance = station_agent.balance()
    station_starting_budget = station_agent.budget(0)

    assert ship.cargo[0] == 0.
    assert station.cargo[0] == 0.

    # disable production by setting production time to more than our time horizon
    station.next_batch_time = gamestate.timestamp + 201
    mining_agendum.max_trips = 2
    testui.agenda.append(mining_agendum)
    testui.margin_neighbors = [ship]
    testui.eta = 200

    simulator.run()

    assert mining_agendum.is_complete()
    assert mining_agendum.round_trips == 2

    assert np.isclose(ship.cargo[0], 5e2/2)
    assert np.isclose(station.cargo[0], 5e2*1.5)

    assert station_agent.inventory(0) == station.cargo[0]

    expected_total_value = expected_price * 5e2 * 1.5
    assert np.isclose(mining_agendum.agent.balance(), trader_starting_balance + expected_total_value)
    assert np.isclose(station_agent.balance(), station_starting_balance - expected_total_value)
    assert np.isclose(station_agent.budget(0), station_starting_budget - expected_total_value)

def test_basic_trading(gamestate, generator, sector, testui, simulator, econ_logger):
    # simple setup: trader and two stations
    # one station produces a good, another one consumes it
    # we'll do two trade runs between the two

    trader_capacity = 5e2
    resource = gamestate.production_chain.ranks[0]
    buy_price = gamestate.production_chain.prices[resource]

    ship = generator.spawn_ship(sector, -3000, 0, v=(0,0), w=0, theta=0)
    ship.cargo_capacity = trader_capacity

    station_producer = generator.spawn_station(sector, 5000, 0, resource=resource, batches_on_hand=200)
    producer_initial_inventory = station_producer.cargo[resource]
    assert station_producer.cargo[resource] >= trader_capacity * 2

    consumer_resource = np.where(gamestate.production_chain.adj_matrix[resource] > 0)[0][0]
    assert resource in gamestate.production_chain.inputs_of(consumer_resource)
    station_consumer = generator.spawn_station(sector, 0, 5000, resource=consumer_resource)

    # set up the trader character with enough money to do one trade
    # they'll get enough from selling that first trade to do a second one
    initial_balance = trader_capacity*buy_price
    ship_owner = generator.spawn_character(ship, balance=initial_balance)
    ship_owner.take_ownership(ship)
    ship.captain = ship_owner
    trading_agendum = agenda.TradingAgendum.create_trading_agendum(ship, ship_owner, gamestate)
    trading_agendum.max_trips=2
    trader_agent = trading_agendum.agent
    ship_owner.add_agendum(trading_agendum)

    # set up prodcer character, no need to have money on hand
    producer_initial_balance = 0
    producer_owner = generator.spawn_character(station_producer, balance=producer_initial_balance)
    producer_owner.take_ownership(station_producer)
    station_producer.captain = producer_owner
    producer_agendum = agenda.StationManager.create_station_manager(station_producer, producer_owner, gamestate)
    producer_owner.add_agendum(producer_agendum)
    producer_agent = gamestate.econ_agents[station_producer.entity_id]

    assert resource in producer_agent.sell_resources()
    assert resource not in producer_agent.buy_resources()
    assert producer_agent.sell_price(resource) == gamestate.production_chain.prices[resource]

    # set up consumer to have enough money to buy two trade's worth of goods
    consumer_initial_balance = trader_capacity*2*buy_price*gamestate.production_chain.markup[consumer_resource]
    consumer_owner = generator.spawn_character(station_consumer, balance=consumer_initial_balance)
    consumer_owner.take_ownership(station_consumer)
    station_consumer.captain = consumer_owner
    consumer_agendum = agenda.StationManager.create_station_manager(station_consumer, consumer_owner, gamestate)
    consumer_owner.add_agendum(consumer_agendum)
    consumer_agent = gamestate.econ_agents[station_consumer.entity_id]

    assert resource in consumer_agent.buy_resources()
    assert resource not in consumer_agent.sell_resources()
    assert consumer_agent.buy_price(resource) > gamestate.production_chain.prices[resource]
    assert consumer_initial_balance >= consumer_agent.buy_price(resource) * trader_capacity * 2

    # check that buys and sales all make sense
    buys = agenda.possible_buys(gamestate, ship, trading_agendum.agent, trading_agendum.allowed_goods, trading_agendum.buy_from_stations)
    assert len(buys) == 1
    assert len(buys[resource]) == 1
    assert buys[resource][0][2] == station_producer
    sales = agenda.possible_sales(gamestate, ship, econ.YesAgent(gamestate.production_chain), trading_agendum.allowed_goods, trading_agendum.sell_to_stations)
    assert len(sales[resource]) == 1
    assert sales[resource][0][2] == station_consumer
    assert sales[resource][0][0] > buys[resource][0][0]

    assert len(set(consumer_agent.sell_resources()).intersection(set(producer_agent.buy_resources()))) == 0
    assert set(producer_agent.sell_resources()).intersection(set(consumer_agent.buy_resources())) == set((resource,))

    buy_ret = agenda.choose_station_to_buy_from(gamestate, ship, trading_agendum.agent, trading_agendum.allowed_goods, trading_agendum.buy_from_stations, trading_agendum.sell_to_stations)
    assert buy_ret is not None
    assert buy_ret[0] == resource
    assert buy_ret[1] == station_producer

    # keep track of some info about the expected buy/sale
    buy_price = buys[resource][0][0]
    buy_value =  buy_price * trader_capacity
    sale_price = sales[resource][0][0]
    sale_value = sale_price * trader_capacity

    # now actually run trading in the simulator
    testui.eta = 200
    testui.agenda.append(trading_agendum)
    testui.margin_neighbors = [ship]
    # delay production until after the test will finish
    station_consumer.next_batch_time = gamestate.timestamp + 201
    simulator.run()

    assert trading_agendum.is_complete()

    # make sure the right transactions were logged
    assert trading_agendum.trade_trips == 2
    assert len(econ_logger.transactions) == 4

    xact1 = econ_logger.transactions[0]
    xact2 = econ_logger.transactions[1]
    xact3 = econ_logger.transactions[2]
    xact4 = econ_logger.transactions[3]

    assert xact1.product_id == resource
    assert xact1.sale_amount == trader_capacity
    assert xact1.price == buy_price
    assert xact1.buyer == trader_agent.agent_id
    assert xact1.seller == producer_agent.agent_id

    assert xact2.product_id == resource
    assert xact2.sale_amount == trader_capacity
    assert xact2.price == sale_price
    assert xact2.buyer == consumer_agent.agent_id
    assert xact2.seller == trader_agent.agent_id

    assert xact3.product_id == resource
    assert xact3.sale_amount == trader_capacity
    assert xact3.price == buy_price
    assert xact3.buyer == trader_agent.agent_id
    assert xact3.seller == producer_agent.agent_id

    assert xact4.product_id == resource
    assert xact4.sale_amount == trader_capacity
    assert xact4.price == sale_price
    assert xact4.buyer == consumer_agent.agent_id
    assert xact4.seller == trader_agent.agent_id

    # make sure everyone has the right state given those transactions
    assert ship.cargo[resource] == 0.
    assert station_producer.cargo[resource] == producer_initial_inventory - trader_capacity*2
    assert station_consumer.cargo[resource] == trader_capacity * 2
    assert np.isclose(ship_owner.balance, initial_balance + (sale_value - buy_value) * 2)
    assert np.isclose(producer_owner.balance, producer_initial_balance + buy_value * 2)
    assert np.isclose(consumer_owner.balance, consumer_initial_balance - sale_value * 2)
