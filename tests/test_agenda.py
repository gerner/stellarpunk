import numpy as np

from stellarpunk import econ, agenda

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
    mining_agendum = agenda.MiningAgendum(ship, ship_owner, gamestate)
    ship_owner.add_agendum(mining_agendum)

    station_owner = generator.spawn_character(station)
    station_owner.take_ownership(station)
    station_agent = econ.StationAgent(station, gamestate.production_chain)
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
    mining_agendum = agenda.MiningAgendum(ship, ship_owner, gamestate)
    ship_owner.add_agendum(mining_agendum)

    station_owner = generator.spawn_character(station, balance=1.5 * 5e2 * price + price/2)
    station_owner.take_ownership(station)
    station_agent = econ.StationAgent(station, gamestate.production_chain)
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

