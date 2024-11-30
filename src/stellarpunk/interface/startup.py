import time
from typing import Any

from stellarpunk import core, interface
from stellarpunk.interface import pilot

class StartupView(interface.View):
    """ Startup screen for giving player loading feedback.

    Watches universe generation and gives player info about progress.

    TODO: also startup menu? """

    def __init__(self, *args:Any, **kwargs:Any):
        super().__init__(*args, **kwargs)
        self._start_time = time.time()
        self._target_startup_time = 5.0

    def initialize(self) -> None:
        self.logger.info(f'starting startup view')
        self.interface.reinitialize_screen(name="Stellarpunk")
        self._start_time = time.time()
        self.gamestate.force_pause(self)

    def update_display(self) -> None:
        if time.time() > self._start_time + self._target_startup_time:
            assert isinstance(self.gamestate.player.character.location, core.Ship)
            self.gamestate.force_unpause(self)
            pilot_view = pilot.PilotView(core.Gamestate.gamestate.player.character.location, self.interface)
            self.interface.swap_view(pilot_view, self)
