from typing import Generator

import pytest
import cymunk # type: ignore
import numpy as np

from stellarpunk import core, sim, generate, interface, sensors, events, intel, config
from stellarpunk.agenda import intel as aintel
from . import MonitoringUI, MonitoringEconDataLogger, MonitoringSimulator

@pytest.fixture
def event_manager() -> events.EventManager:
    em = events.EventManager()
    events.register_events(em)
    sensors.pre_initialize(em)
    intel.pre_initialize(em)
    #TODO: all events (e.g. ui manager events) should be registered by this point
    #TODO: this is where events are setup from config, do we need to worry about that?
    #em.pre_initialize({})
    return em

@pytest.fixture
def intel_director() -> aintel.IntelCollectionDirector:
    return sim.initialize_intel_director()

@pytest.fixture
def gamestate(econ_logger:MonitoringEconDataLogger, event_manager:events.EventManager) -> Generator[core.Gamestate, None, None]:
    gamestate = core.Gamestate()
    gamestate.econ_logger = econ_logger
    event_manager.initialize_gamestate(events.EventState(), gamestate)
    yield gamestate
    gamestate.sanity_check_orders()
    gamestate.sanity_check_effects()

@pytest.fixture
def generator(event_manager:events.EventManager, intel_director:aintel.IntelCollectionDirector, gamestate:core.Gamestate) -> generate.UniverseGenerator:
    ug = generate.UniverseGenerator(seed=0)
    ug.gamestate = gamestate
    ug.pre_initialize(event_manager, intel_director, empty_name_model_culture="test")
    gamestate.random = ug.r
    gamestate.generator = ug
    gamestate.production_chain = ug.generate_chain(
            max_fraction_one_to_one=1.,
            max_fraction_single_input=1.,
            max_fraction_single_output=1.,
            assign_names=False,
            #n_ranks=1, min_per_rank=(1,), max_per_rank=(1,), min_final_inputs=1)
    )

    # just pull out intel events
    e = {k:v for k,v in config.Events.items() if "group" in v and v["group"] == "intel"}
    event_manager.pre_initialize(e)
    event_manager.initialize_gamestate(events.EventState(), gamestate)
    intel_director.initialize_gamestate(gamestate)

    return ug

@pytest.fixture
def sector(gamestate:core.Gamestate) -> core.Sector:
    sector_radius=1e5
    hex_size = 1e4
    sector_name = "Sector"

    sector = core.Sector(np.array([0, 0]), sector_radius, hex_size, cymunk.Space(), gamestate, sector_name, culture="test")
    sector.sensor_manager = sensors.SensorManager(sector)
    gamestate.add_sector(sector, 0)

    return sector

@pytest.fixture
def ship(gamestate: core.Gamestate, generator: generate.UniverseGenerator, sector: core.Sector) -> core.Ship:
    return generator.spawn_ship(sector, 0, 2400, v=np.array((0,0)), w=0, theta=0, initial_transponder=True, initial_sensor_power_ratio=1.0)

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
def simulator(event_manager:events.EventManager, gamestate:core.Gamestate, generator:generate.UniverseGenerator, testui:MonitoringUI) -> sim.Simulator:
    simulation = MonitoringSimulator(generator, testui)
    simulation.min_tick_sleep = np.inf
    #testui.min_ui_timeout = -np.inf
    testui.runtime = simulation

    simulation.pre_initialize()

    simulation.initialize_gamestate(gamestate)
    simulation.start_game()

    return simulation
