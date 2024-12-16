""" Station UI tests """

import curses

from stellarpunk import agenda
from stellarpunk.interface import station as v_station

def test_force_pause(gamestate, generator, testui, sector, resource_station, resource_station_agendum):
    ship = generator.spawn_ship(sector, 0, 2400, v=(0,0), w=0, theta=0)
    station_view = v_station.StationView(resource_station, ship, gamestate, testui)

    testui.open_view(station_view, deactivate_views=True)

    # can pause
    assert not gamestate.paused
    gamestate.pause()
    assert gamestate.paused
    gamestate.pause()
    assert not gamestate.paused

    station_view.enter_mode(v_station.Mode.TRADE)

    # now we're paused and we can't unpause it
    assert gamestate.paused
    gamestate.pause()
    assert gamestate.paused


def test_trade_menu_validator(gamestate, generator, testui, econ_logger, sector, resource_station, resource_station_agendum):
    buy_resource = resource_station_agendum.station.resource
    sell_resource = gamestate.production_chain.inputs_of(buy_resource)[0]

    # set up prices and cargo available for testing
    resource_station.owner.balance = 1.2e3
    resource_station_agendum.agent._buy_price[sell_resource] = 5e2
    resource_station_agendum.agent._sell_price[buy_resource] = 7.5e2
    resource_station.cargo[buy_resource] = 5.

    # set up a ship with some cargo to sell
    ship = generator.spawn_ship(sector, 0, 2400, v=(0,0), w=0, theta=0)
    ship.cargo[sell_resource] = 10.
    player = generator.spawn_player(ship, balance=2.5e3)
    gamestate.player = player

    station_view = v_station.StationView(resource_station, ship, gamestate, testui)

    testui.open_view(station_view, deactivate_views=True)

    station_view.enter_mode(v_station.Mode.TRADE)

    # make sure the menus have the right options
    assert len(station_view.sell_menu.options) == 1
    assert station_view.sell_menu.options[0].data == sell_resource
    assert station_view.sell_menu.options[0].value == ship.cargo[sell_resource]
    assert station_view.sell_menu.options[0].setting == ship.cargo[sell_resource]
    assert len(station_view.buy_menu.options) == 1
    assert station_view.buy_menu.options[0].data == buy_resource
    assert station_view.buy_menu.options[0].value == 0
    assert station_view.buy_menu.options[0].setting == 0
    assert station_view.buy_menu.options[0].pool == int(resource_station.cargo[buy_resource])

    # TODO: validate more cases?
    # verify validation criteria:
    # validate ship and station have total cargo capacity for buys/sells
    # validate station budgets for each resource for sells
    # validate player and station agent can afford all buys/sells

    # can't buy the sell stuff
    station_view.sell_menu.select_more()
    assert station_view.sell_menu.options[0].setting == 10
    assert testui.last_status_message == "The station does not sell that good"
    testui.status_message("")

    station_view.sell_menu.select_option(0)
    station_view.sell_menu.select_less()
    assert station_view.sell_menu.options[0].setting == 9
    station_view.sell_menu.select_less()
    assert station_view.sell_menu.options[0].setting == 8
    # station doesn't have the budget to buy more
    station_view.sell_menu.select_less()
    assert station_view.sell_menu.options[0].setting == 8
    assert testui.last_status_message == "The station doesn't have enough money to buy those goods"
    testui.status_message("")

    station_view.buy_menu.select_option(0)
    station_view.buy_menu.select_more()
    assert station_view.buy_menu.options[0].setting == 1
    station_view.buy_menu.select_more()
    assert station_view.buy_menu.options[0].setting == 2
    station_view.buy_menu.select_more()
    assert station_view.buy_menu.options[0].setting == 3
    # player doesn't have enough money to buy more
    station_view.buy_menu.select_more()
    assert station_view.buy_menu.options[0].setting == 3
    assert testui.last_status_message == "You don't have enough money to buy those goods"
    testui.status_message("")

    # verify some pre-transaction setup
    assert resource_station.owner.balance == 1.2e3
    assert player.character.balance == 2.5e3
    assert len(econ_logger.transactions) == 0

    # execute the trade
    # TODO: a little janky to execute the trade with a keypress
    testui.handle_input(curses.ascii.CR, 1.0)
    assert testui.last_status_message.startswith("Trade completed")
    testui.status_message("")

    assert len(econ_logger.transactions) == 2
    assert econ_logger.transactions[0].product_id == sell_resource
    assert econ_logger.transactions[0].sale_amount == 2.
    assert econ_logger.transactions[0].price == 5e2
    assert econ_logger.transactions[0].buyer == resource_station_agendum.agent.agent_id
    assert econ_logger.transactions[0].seller == player.agent.agent_id
    assert econ_logger.transactions[1].product_id == buy_resource
    assert econ_logger.transactions[1].sale_amount == 3.
    assert econ_logger.transactions[1].price == 7.5e2
    assert econ_logger.transactions[1].buyer == player.agent.agent_id
    assert econ_logger.transactions[1].seller == resource_station_agendum.agent.agent_id

    assert resource_station.cargo[sell_resource] == 2.
    assert resource_station.cargo[buy_resource] == 5. - 3.
    assert ship.cargo[sell_resource] == 10. - 2.
    assert ship.cargo[buy_resource] == 3.
    assert resource_station.owner.balance == 1.2e3 - 5e2 * 2. + 7.5e2 * 3.
    assert player.character.balance == 2.5e3 + 5e2 * 2. - 7.5e2 * 3.
