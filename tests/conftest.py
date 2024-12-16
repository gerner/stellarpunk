import pytest

import cymunk # type: ignore
import numpy as np

from stellarpunk import core, sim, generate, interface, sensors
from . import MonitoringUI, MonitoringEconDataLogger, MonitoringSimulator

@pytest.fixture
def gamestate(econ_logger:MonitoringEconDataLogger) -> core.Gamestate:
    gamestate = core.Gamestate()
    gamestate.econ_logger = econ_logger
    gamestate.player = core.Player(gamestate)
    return gamestate

@pytest.fixture
def generator(gamestate:core.Gamestate) -> generate.UniverseGenerator:
    ug = generate.UniverseGenerator(seed=0)
    ug.gamestate = gamestate
    ug.initialize(empty_name_model_culture="test")
    gamestate.random = ug.r
    gamestate.generator = ug
    gamestate.production_chain = ug.generate_chain(
            max_fraction_one_to_one=1.,
            max_fraction_single_input=1.,
            max_fraction_single_output=1.,
            assign_names=False,
            #n_ranks=1, min_per_rank=(1,), max_per_rank=(1,), min_final_inputs=1)
    )
    return ug

@pytest.fixture
def sector(gamestate:core.Gamestate) -> core.Sector:
    sector_radius=1e5
    sector_name = "Sector"

    sector = core.Sector(np.array([0, 0]), sector_radius, cymunk.Space(), gamestate, sector_name, culture="test")
    sector.sensor_manager = sensors.SensorManager(sector)
    gamestate.sectors[sector.entity_id] = sector

    return sector

@pytest.fixture
def ship(gamestate: core.Gamestate, generator: generate.UniverseGenerator, sector: core.Sector) -> core.Ship:
    return generator.spawn_ship(sector, 0, 2400, v=np.array((0,0)), w=0, theta=0)

@pytest.fixture
def player(gamestate: core.Gamestate, generator: generate.UniverseGenerator, ship:core.Ship) -> core.Player:
    player = generator.spawn_player(ship, balance=2.5e3)
    gamestate.player = player
    return player

@pytest.fixture
def testui(gamestate:core.Gamestate, generator:generate.UniverseGenerator, sector:core.Sector) -> MonitoringUI:
    testui = MonitoringUI(sector, generator, interface.AbstractMixer())
    testui.gamestate = gamestate
    return testui

@pytest.fixture
def econ_logger() -> MonitoringEconDataLogger:
    return MonitoringEconDataLogger()

@pytest.fixture
def simulator(gamestate:core.Gamestate, generator:generate.UniverseGenerator, testui:MonitoringUI) -> sim.Simulator:
    simulation = MonitoringSimulator(generator, testui)
    simulation.min_tick_sleep = np.inf
    #testui.min_ui_timeout = -np.inf
    testui.runtime = simulation

    simulation.pre_initialize()
    simulation.gamestate = gamestate
    simulation.initialize()
    simulation.start_game()

    return simulation
