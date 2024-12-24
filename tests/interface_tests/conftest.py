import pytest

from stellarpunk import core, agenda, generate
from stellarpunk.core import sector_entity

@pytest.fixture
def resource_station(
    gamestate: core.Gamestate,
    generator: generate.UniverseGenerator,
    sector: core.Sector
) -> sector_entity.Station:
    buy_resource = gamestate.production_chain.ranks[0]
    station = generator.spawn_station(sector, 1e3, 1e3, resource=buy_resource)
    return station

@pytest.fixture
def resource_station_agendum(
    gamestate: core.Gamestate,
    generator: generate.UniverseGenerator,
    resource_station: sector_entity.Station,
) -> agenda.StationManager:
    station_character = generator.spawn_character(resource_station, balance=2e3)
    station_character.take_ownership(resource_station)
    station_agendum = agenda.StationManager(
        station=resource_station,
        character=station_character,
        gamestate=gamestate,
    )
    station_character.add_agendum(station_agendum)

    return station_agendum

