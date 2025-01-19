import tempfile
import logging
import collections
import functools
from collections.abc import MutableMapping

import numpy as np

from stellarpunk import sim, core, agenda, econ, orders, util
from stellarpunk.orders import steering, movement
from stellarpunk.core import sector_entity, combat
from stellarpunk.serialization import util as s_util

from . import write_history, add_sector_intel

def test_to_int():
    with tempfile.TemporaryFile() as fp:
        v1 = 42
        bytes_written = s_util.int_to_f(v1, fp)
        fp.flush()
        assert bytes_written == fp.tell()
        fp.seek(0)
        v2 = s_util.int_from_f(fp)
        assert v2 == v1
        assert bytes_written == fp.tell()

def test_save_load_registry(event_manager, intel_director, generator):
    game_saver = sim.initialize_save_game(generator, event_manager, intel_director, debug=True)
    with tempfile.TemporaryFile() as fp:
        bytes_written = game_saver._save_registry(fp)
        fp.flush()
        assert bytes_written == fp.tell()

        fp.seek(0)
        game_saver._load_registry(fp)
        assert bytes_written == fp.tell()

def test_trivial_gamestate(event_manager, intel_director, gamestate, generator, player):
    assert player == gamestate.player
    game_saver = sim.initialize_save_game(generator, event_manager, intel_director, debug=True)
    save_filename = "/tmp/stellarpunk_testfile.stpnk"
    filename = game_saver.save(gamestate, save_filename)
    g2 = game_saver.load(filename)
    #this won't work!
    #assert g2 == gamestate

    assert gamestate.random.integers(42) == g2.random.integers(42)
    assert gamestate.player.entity_id == g2.player.entity_id
    assert gamestate.production_chain == g2.production_chain

#def test_event_state(event_manager, gamestate, generator):
#    # trigger some events
#    pass

@write_history
def test_saving_in_goto(ship, player, gamestate, intel_director, generator, sector, testui, simulator):
    target_loc = np.array((8000., -2000.))
    goto_order = orders.GoToLocation.create_go_to_location(target_loc, ship, gamestate)
    ship.prepend_order(goto_order)
    game_saver = sim.initialize_save_game(generator, gamestate.event_manager, intel_director, debug=True)

    # set up a tick callback to save and load
    original_ship_id = ship.entity_id
    saved_once = False
    def periodic_save_load():
        nonlocal saved_once, gamestate, sector, player, game_saver, goto_order, ship, original_ship_id

        # some sanity checking
        assert gamestate == core.Gamestate.gamestate
        assert gamestate == goto_order.gamestate
        assert sector.entity_id in gamestate.entities
        assert sector == gamestate.entities[sector.entity_id]
        assert sector.entity_id in gamestate.sectors
        assert sector == gamestate.sectors[sector.entity_id]
        assert len(gamestate.sectors) == 1
        assert ship.entity_id in gamestate.entities
        assert ship == gamestate.entities[ship.entity_id]
        assert ship.entity_id in sector.entities
        assert ship == sector.entities[ship.entity_id]
        assert len(sector.entities) == 1
        assert player.entity_id in gamestate.entities
        assert player == gamestate.entities[player.entity_id]
        assert player.character.entity_id in gamestate.entities
        assert player.character == gamestate.entities[player.character.entity_id]
        assert player.agent.entity_id in gamestate.entities
        assert player.agent == gamestate.entities[player.agent.entity_id]
        assert len(gamestate.entities) == 5

        if not saved_once and gamestate.timestamp > goto_order.init_eta / 3.0:
            save_filename = "/tmp/stellarpunk_testfile.stpnk"
            save_filename = game_saver.save(gamestate, save_filename)
            gamestate = game_saver.load(save_filename)
            player = gamestate.get_entity(player.entity_id, core.Player)
            sector = gamestate.get_entity(sector.entity_id, core.Sector)
            ship = gamestate.get_entity(ship.entity_id, core.Ship)
            goto_order = gamestate.get_order(goto_order.order_id, orders.GoToLocation)
            saved_once = True
        #elif saved_once and gamestate.timestamp > goto_order.init_eta / 3.0:
        #    raise Exception()


    testui.tick_callback = periodic_save_load

    testui.orders.append(goto_order)
    simulator.run()


    assert goto_order.is_complete()
    assert util.distance(ship.loc, target_loc) < 2e3

    # these were experimentally determined to be the values achieved without a
    # save/load cycle. this might be fragile
    assert util.isclose(gamestate.timestamp, 30.699999809264416)
    assert util.isclose(util.distance(ship.loc, target_loc), 1466.2242390081483)

def test_saving_in_basic_trading(player, gamestate, generator, intel_director, sector, testui, simulator, econ_logger):
    game_saver = sim.initialize_save_game(generator, gamestate.event_manager, intel_director, debug=True)

    # simple setup: trader and two stations
    # one station produces a good, another one consumes it
    # we'll do two trade runs between the two
    # we'll also do a few save/loads during the run

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

    # setup station and econ agent intel
    add_sector_intel(ship, sector, ship_owner, gamestate)

    # check that buys and sales all make sense
    buys = agenda.possible_buys(ship_owner, gamestate, ship, trading_agendum.agent, trading_agendum.allowed_goods, trading_agendum.buy_from_stations)
    assert len(buys) == 1
    assert len(buys[resource]) == 1
    assert buys[resource][0][2] == station_producer
    sales = agenda.possible_sales(ship_owner, gamestate, ship, econ.YesAgent(gamestate.production_chain), trading_agendum.allowed_goods, trading_agendum.sell_to_stations)
    assert len(sales[resource]) == 1
    assert sales[resource][0][2] == station_consumer
    assert sales[resource][0][0] > buys[resource][0][0]

    assert len(set(consumer_agent.sell_resources()).intersection(set(producer_agent.buy_resources()))) == 0
    assert set(producer_agent.sell_resources()).intersection(set(consumer_agent.buy_resources())) == set((resource,))

    buy_ret = agenda.choose_station_to_buy_from(ship_owner, gamestate, ship, trading_agendum.agent, trading_agendum.allowed_goods, trading_agendum.buy_from_stations, trading_agendum.sell_to_stations)
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

    # set up a tick callback to save and load
    saved_once = False
    def periodic_save_load():
        nonlocal saved_once, game_saver, player, gamestate, generator, sector, trading_agendum, trader_agent, producer_agent, consumer_agent, ship, station_producer, station_consumer, ship_owner, producer_owner, consumer_owner

        # some sanity checking
        assert gamestate == core.Gamestate.gamestate
        assert gamestate == trading_agendum.gamestate
        assert sector.entity_id in gamestate.entities
        assert sector == gamestate.entities[sector.entity_id]
        assert sector.entity_id in gamestate.sectors
        assert sector == gamestate.sectors[sector.entity_id]

        if not saved_once and gamestate.timestamp > 30.0:
            save_filename = "/tmp/stellarpunk_testfile.stpnk"
            save_filename = game_saver.save(gamestate, save_filename)
            gamestate = game_saver.load(save_filename)
            player = gamestate.get_entity(player.entity_id, core.Player)
            sector = gamestate.get_entity(sector.entity_id, core.Sector)

            # fetch state so we can do the right asserts
            trading_agendum = gamestate.get_agendum(trading_agendum.agenda_id, agenda.TradingAgendum)
            trader_agent = gamestate.get_entity(trader_agent.entity_id, econ.ShipTraderAgent)
            producer_agent = gamestate.get_entity(producer_agent.entity_id, econ.StationAgent)
            consumer_agent = gamestate.get_entity(consumer_agent.entity_id, econ.StationAgent)
            ship = gamestate.get_entity(ship.entity_id, core.Ship)
            station_producer = gamestate.get_entity(station_producer.entity_id, sector_entity.Station)
            station_consumer = gamestate.get_entity(station_consumer.entity_id, sector_entity.Station)
            ship_owner = gamestate.get_entity(ship_owner.entity_id, core.Character)
            producer_owner = gamestate.get_entity(producer_owner.entity_id, core.Character)
            consumer_owner = gamestate.get_entity(consumer_owner.entity_id, core.Character)

            saved_once = True

    testui.tick_callback = periodic_save_load

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

    # this is fragile, but experimentally determined from running this test
    # without save/load. note, there is some variation even without save/load
    assert abs(gamestate.timestamp - 107.06666666666187) < 10.0

    # this is too fragile, but I've checked with/without saving and sometimes
    # this passes and sometimes it doesn't
    #TODO: why is this non-deterministic?
    #assert util.distance(ship.loc, np.array([-702.65356445, 3528.47949219])) < 1e1

def test_saving_in_mining(player, gamestate, generator, intel_director, sector, testui, simulator):
    game_saver = sim.initialize_save_game(generator, gamestate.event_manager, intel_director, debug=True)
    # ship and asteroid
    resource = 0
    ship = generator.spawn_ship(sector, -3000, 0, v=(0,0), w=0, theta=0)
    asteroid = generator.spawn_asteroid(sector, 0, 0, resource, 5e2)
    # ship mines the asteroid
    mining_order = orders.MineOrder.create_mine_order(asteroid, 3.5e2, ship, gamestate)
    ship.prepend_order(mining_order)

    testui.orders = [mining_order]
    testui.margin_neighbors = [ship]

    assert ship.cargo[0] == 0.
    assert np.isclose(asteroid.cargo[asteroid.resource], 5e2)

    # set up a tick callback to save and load
    saved_once = False
    def periodic_save_load():
        nonlocal saved_once, game_saver, gamestate, generator, player, sector, ship, asteroid, mining_order

        # some sanity checking
        assert gamestate == core.Gamestate.gamestate
        assert mining_order.is_complete() or mining_order.order_id in gamestate.orders
        assert mining_order.is_complete() or mining_order == gamestate.orders[mining_order.order_id]
        assert sector.entity_id in gamestate.sectors
        assert sector == gamestate.sectors[sector.entity_id]

        if not saved_once and gamestate.timestamp > 30.0:
            save_filename = "/tmp/stellarpunk_testfile.stpnk"
            save_filename = game_saver.save(gamestate, save_filename)
            gamestate = game_saver.load(save_filename)

            player, sector, ship, asteroid, mining_order = gamestate.recover_objects((player, sector, ship, asteroid, mining_order))

            saved_once = True

    testui.tick_callback = periodic_save_load

    simulator.run()
    assert mining_order.is_complete()

    # make sure ship ends up near enough to the asteroid
    assert np.linalg.norm(ship.loc - asteroid.loc) < 2e3 + asteroid.radius + steering.VELOCITY_EPS
    # make sure we got the resources
    assert np.isclose(ship.cargo[0], 3.5e2)
    # make sure asteroid lost the resources
    assert np.isclose(asteroid.cargo[asteroid.resource], 5e2 - 3.5e2)

def test_saving_during_attack(player, gamestate, generator, intel_director, sector, testui, simulator):
    game_saver = sim.initialize_save_game(generator, gamestate.event_manager, intel_director, debug=True)
    #TODO: this test can be flaky and sometimes keeps running.
    #  what is non-determinitistic about it???

    # simulates an attack run by a single ship on another single ship
    attacker = generator.spawn_ship(sector, -300000, 0, v=(0,0), w=0, theta=0)
    attacker.sensor_settings._sensor_power = attacker.sensor_settings._max_sensor_power
    attacker.sensor_settings._last_sensor_power = attacker.sensor_settings._max_sensor_power
    defender = generator.spawn_ship(sector, 0, 0, v=(0,0), w=0, theta=0)

    defender_owner = generator.spawn_character(defender)
    defender_owner.take_ownership(defender)
    defender_owner.add_agendum(agenda.CaptainAgendum.create_eoa(defender, defender_owner, gamestate))

    attack_order = combat.AttackOrder.create_attack_order(sector.sensor_manager.target(defender, attacker), attacker, gamestate, max_missiles=15)
    attacker.prepend_order(attack_order)

    testui.orders = [attack_order]
    testui.eta = 3600
    testui.collisions_allowed = True

    state_ticks:MutableMapping[combat.AttackOrder.State, int] = collections.defaultdict(int)
    attack_ticks = 0
    age_sum = 0.
    a_zero_forces = 0
    a_non_zero_forces = 0
    d_zero_forces = 0
    d_non_zero_forces = 0
    evade_max_thrust_sum = 0.
    ticks_evading = 0
    ticks_fleeing = 0
    distinct_flee_orders = 0
    dist_sum = 0
    max_thrust_sum = 0.

    last_loc = defender.loc
    last_force = np.array(defender.phys.force)
    last_velocity = defender.velocity

    saved_once = False
    def tick_callback():
        nonlocal attacker, state_ticks, a_zero_forces, a_non_zero_forces, d_zero_forces, d_non_zero_forces, age_sum, attack_ticks
        nonlocal defender, ticks_fleeing, distinct_flee_orders, ticks_evading, evade_max_thrust_sum, dist_sum, max_thrust_sum
        nonlocal last_loc, last_force, last_velocity
        nonlocal saved_once, game_saver, gamestate, generator, player, sector, attack_order
        if util.distance(last_loc, defender.loc) > max(np.linalg.norm(defender.velocity)*simulator.dt*3.0, 1.):
            raise Exception()
        if util.magnitude(*defender.velocity) > 15000*1.5:
            raise Exception()
        last_loc = defender.loc
        last_velocity = defender.velocity
        last_force = np.array(defender.phys.force)
        assert attacker.sector
        if not attack_order.is_complete():
            state_ticks[attack_order.state] += 1
            age_sum += attack_order.target.age
            attack_ticks += 1
        if defender.phys.force.length == 0.:
            d_zero_forces += 1
        else:
            d_non_zero_forces += 1
        if attacker.phys.force.length == 0.:
            a_zero_forces += 1
        else:
            a_non_zero_forces += 1
        defender_top_order = defender.top_order()
        if isinstance(defender_top_order, combat.FleeOrder):
            ticks_fleeing += 1
            dist_sum += util.distance(defender.loc, attacker.loc)
            max_thrust_sum += defender_top_order.max_thrust
            if defender_top_order not in testui.orders:
                testui.add_order(defender_top_order)
                distinct_flee_orders += 1
        defender_current_order = defender.current_order()
        if isinstance(defender_current_order, movement.EvadeOrder):
            ticks_evading += 1
            evade_max_thrust_sum += defender_current_order.max_thrust

        if attack_order.is_complete():
            simulator.raise_breakpoint()


        # some sanity checking
        assert gamestate == core.Gamestate.gamestate
        assert sector.entity_id in gamestate.sectors
        assert sector == gamestate.sectors[sector.entity_id]

        if not saved_once and gamestate.timestamp > 30.0:
            save_filename = "/tmp/stellarpunk_testfile.stpnk"
            save_filename = game_saver.save(gamestate, save_filename)
            gamestate = game_saver.load(save_filename)

            player, sector, attacker, defender, attack_order = gamestate.recover_objects((player, sector, attacker, defender, attack_order))

            saved_once = True

    testui.tick_callback = tick_callback

    simulator.notify_on_collision = True
    simulator.run()

    # either the defender is destroyed or we had to give up
    assert not attack_order.target.is_active() or attack_order.state == combat.AttackOrder.State.GIVEUP

    assert attacker not in set(functools.reduce(lambda x, y: x + [y[0], y[1]], testui.collisions, list()))

    # should only have the attack order and a flee order
    assert len(testui.orders) == 2
    flee_order = testui.orders[1]
    assert isinstance(flee_order, combat.FleeOrder)

    logging.info(f'{attacker.sector=} {defender.sector=} in {gamestate.timestamp}s')
    logging.info(f'target avg age: {age_sum/attack_ticks}s avg dist: {util.human_distance(dist_sum/ticks_fleeing)}')
    logging.info(f'missiles fired: {attack_order.missiles_fired}')
    logging.info(f'target active: {attack_order.target.is_active()}')
    logging.info(f'threats destroyed: {flee_order.point_defense.targets_destroyed}')

