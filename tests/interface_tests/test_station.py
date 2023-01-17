""" Station UI tests """

from stellarpunk import agenda
from stellarpunk.interface import station as v_station

def test_trade_menu_validator(gamestate, generator, sector, testui):
    station = generator.spawn_station(sector, 0, 0, resource=0)
    station_character = generator.spawn_character(station, balance=2e3)
    station_character.take_ownership(station)
    station_character.add_agendum(agenda.StationManager(station=station, character=station_character, gamestate=gamestate))

    ship = generator.spawn_ship(sector, 0, 2400, v=(0,0), w=0, theta=0)
    ship.cargo[0] = 1e2
    player_character = generator.spawn_character(ship, balance=2e3)
    gamestate.player.character = player_character

    station_view = v_station.StationView(station, ship, testui)

    testui.open_view(station_view, deactivate_views=True)

    station_view.enter_mode(v_station.Mode.TRADE)

    # verify validation criteria:
    # validate ship and station have total cargo capacity for buys/sells
    # validate station budgets for each resource for sells
    # validate player and station agent can afford all buys/sells


