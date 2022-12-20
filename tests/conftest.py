import pytest

import cymunk # type: ignore
import numpy as np

from stellarpunk import core, sim, generate
from . import MonitoringUI, MonitoringEconDataLogger

@pytest.fixture
def gamestate(econ_logger:MonitoringEconDataLogger) -> core.Gamestate:
    gamestate = core.Gamestate()
    gamestate.econ_logger = econ_logger
    return gamestate

@pytest.fixture
def generator(gamestate:core.Gamestate) -> generate.UniverseGenerator:
    ug = generate.UniverseGenerator(gamestate, seed=0)
    gamestate.random = ug.r
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

    sector = core.Sector(np.array([0, 0]), sector_radius, cymunk.Space(), sector_name)
    gamestate.sectors[sector.entity_id] = sector

    return sector

@pytest.fixture
def testui(gamestate:core.Gamestate, sector:core.Sector) -> MonitoringUI:
    return MonitoringUI(gamestate, sector)

@pytest.fixture
def econ_logger() -> MonitoringEconDataLogger:
    return MonitoringEconDataLogger()

@pytest.fixture
def simulator(gamestate:core.Gamestate, testui:MonitoringUI) -> sim.Simulator:
    simulation = sim.Simulator(gamestate, testui, ticks_per_hist_sample=1)
    gamestate.min_tick_sleep = np.inf
    #testui.min_ui_timeout = -np.inf

    simulation.initialize()

    return simulation
