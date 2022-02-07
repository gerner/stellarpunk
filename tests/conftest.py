import pytest

import pymunk
import numpy as np

from stellarpunk import core, sim, generate
from . import MonitoringUI

@pytest.fixture
def gamestate() -> core.Gamestate:
    return core.Gamestate()

@pytest.fixture
def generator(gamestate:core.Gamestate) -> generate.UniverseGenerator:
    ug = generate.UniverseGenerator(gamestate, seed=0)
    gamestate.random = ug.r
    return ug

@pytest.fixture
def sector(gamestate:core.Gamestate) -> core.Sector:
    sector_radius=1e5
    sector_name = "Sector"

    sector = core.Sector(0, 0, sector_radius, sector_name)
    sector.space = pymunk.Space()
    gamestate.sectors[(0,0)] = sector

    return sector

@pytest.fixture
def testui(gamestate:core.Gamestate, sector:core.Sector) -> MonitoringUI:
    return MonitoringUI(gamestate, sector)

@pytest.fixture
def simulator(gamestate:core.Gamestate, testui:MonitoringUI) -> sim.Simulator:
    simulation = sim.Simulator(gamestate, testui)
    simulation.min_tick_sleep = np.inf
    simulation.min_ui_timeout = -np.inf

    simulation.initialize()

    return simulation
