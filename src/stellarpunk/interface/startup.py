import time
import threading
import enum
import curses.ascii
from collections.abc import Collection
from typing import Any, Optional

from stellarpunk import core, interface, generate
from stellarpunk.interface import pilot, ui_util
from stellarpunk.serialization import save_game

class Mode(enum.Enum):
    """ Startup Menu Modes """
    NONE = enum.auto()
    MAIN_MENU = enum.auto()
    RESUME = enum.auto()
    NEW_GAME = enum.auto()
    LOAD_GAME = enum.auto()
    EXIT_GAME = enum.auto()
    EXIT = enum.auto()

class StartupView(interface.View, generate.UniverseGeneratorObserver):
    """ Startup screen for giving player loading feedback.

    Watches universe generation and gives player info about progress.

    TODO: also startup menu? """

    def __init__(self, generator:generate.UniverseGenerator, game_saver:save_game.GameSaver, *args:Any, **kwargs:Any):
        super().__init__(*args, **kwargs)

        self._generator = generator
        self._generator_thread:Optional[threading.Thread] = None

        self._game_saver = game_saver

        self._start_time = time.time()
        self._target_startup_time = 5.0
        self._universe_loaded = False
        self._mode = Mode.NONE
        self._current_generation_step = generate.GenerationStep.NONE
        self._estimated_generation_ticks = 0
        self._generation_ticks = 0

    def estimated_generation_ticks(self, ticks:int) -> None:
        self._estimated_generation_ticks = ticks

    def generation_step(self, step:generate.GenerationStep) -> None:
        self.logger.debug(f'step: {step} {self._generation_ticks}/{self._estimated_generation_ticks}')
        self._current_generation_step = step

    def generation_tick(self) -> None:
        self._generation_ticks += 1
        self.logger.debug(f'tick: {self._current_generation_step} {self._generation_ticks}/{self._estimated_generation_ticks}')

    def universe_generated(self, gamestate:core.Gamestate) -> None:
        self._universe_loaded = True

    def _generate_universe(self) -> None:
        gamestate = self._generator.generate_universe()
        self.interface.log_message("new universe created")
        gamestate.force_pause(self)
        # the gamestate will get sent to people via an event on universe
        # generator

    def _enter_main_menu(self) -> None:
        self.viewscreen.erase()
        menu_items:ui_util.TextMenuItem = []
        if self.interface.gamestate:
            menu_items.append(ui_util.TextMenuItem(
                "Resume", lambda: self._enter_mode(Mode.RESUME)
            ))
        menu_items.extend([
            ui_util.TextMenuItem(
                "New Game", lambda: self._enter_mode(Mode.NEW_GAME)
            ),
            ui_util.TextMenuItem(
                "Load Game", lambda: self._enter_mode(Mode.LOAD_GAME)
            ),
            ui_util.TextMenuItem(
                "Exit Game", lambda: self._enter_mode(Mode.EXIT_GAME)
            ),
        ])
        self._main_menu = ui_util.Menu(
            "Main Menu",
            ui_util.number_text_menu_items(menu_items)
        )

    def _enter_resume(self) -> None:
        self.interface.close_view(self)

    def _enter_new_game(self) -> None:
        self._generator.observe(self)
        self._generator_thread = threading.Thread(target=self._generate_universe)
        self._generator_thread.start()

    def _exit_new_game(self) -> None:
        assert(self._universe_loaded)
        self._generator.unobserve(self)
        #TODO: should we have a timeout here?
        assert(self._generator_thread is not None)
        self._generator_thread.join()
        assert(self._generator.gamestate)
        self._generator.gamestate.production_chain.viz().render("/tmp/production_chain", format="pdf")
        assert(self.interface.gamestate)

        # finish the startup sequence which will trigger starting the full
        # game loop
        self.interface.runtime.exit_startup()
        self.interface.runtime.start_game()

        assert self.interface.player.character and isinstance(self.interface.player.character.location, core.Ship)
        pilot_view = pilot.PilotView(self.interface.player.character.location, self._generator.gamestate, self.interface)
        self.interface.close_all_views()
        if len(self.interface.views) > 0:
            raise Exception()
        self.interface.open_view(pilot_view)

    def _enter_load_game(self) -> None:
        self.viewscreen.erase()

        def load_game(save:save_game.SaveGame) -> None:
            self.interface.log_message(f'loading savegame "{save.filename}"')
            gamestate = self._game_saver.load(save.filename)
            gamestate.force_pause(self)
            # the gamestate will get sent to people via an event from universe
            # generator
            self.interface.log_message(f'game loaded.')
            self._universe_loaded = True
            #TODO: should we let the user press a key first?
            self._enter_mode(Mode.EXIT)

        # get savegame options
        load_options = []
        for s in self._game_saver.list_save_games():
            load_options.append(ui_util.TextMenuItem(
                s.filename,
                lambda s=s: load_game(s) # type: ignore
            ))

        self._load_menu = ui_util.Menu(
            "Load Game",
            ui_util.number_text_menu_items(load_options)
        )

    def _exit_load_game(self) -> None:
        assert(self._universe_loaded)

        self.interface.runtime.exit_startup()
        self.interface.runtime.start_game()

        assert self.interface.player.character and isinstance(self.interface.player.character.location, core.Ship)
        pilot_view = pilot.PilotView(self.interface.player.character.location, self._generator.gamestate, self.interface)
        self.interface.close_all_views()
        self.interface.open_view(pilot_view)

    def _enter_exit_game(self) -> None:
        self.interface.runtime.exit_startup()
        self.interface.runtime.quit()
        self.interface.close_view(self)

    def _enter_mode(self, mode:Mode) -> None:
        """ Leaves current mode and enters a new mode. """

        self.logger.debug(f'exiting {self._mode} entering {mode}')
        # leave the old mode
        if self._mode == Mode.NEW_GAME:
            self._exit_new_game()
        elif self._mode == Mode.LOAD_GAME:
            self._exit_load_game()

        # enter the new mode
        self._mode = mode
        if mode == Mode.MAIN_MENU:
            self._enter_main_menu()
        elif mode == Mode.RESUME:
            self._enter_resume()
        elif mode == Mode.NEW_GAME:
            self._enter_new_game()
        elif mode == Mode.LOAD_GAME:
            self._enter_load_game()
        elif mode == Mode.EXIT_GAME:
            self._enter_exit_game()
        elif mode == Mode.EXIT:
            self.logger.debug(f'exiting startup menu')
        else:
            raise ValueError(f'cannot enter mode {mode}')

    def initialize(self) -> None:
        self.logger.info(f'starting startup view')

        if self.interface.gamestate:
            self.interface.gamestate.force_pause(self)

        self.viewscreen.erase()
        self.interface.reinitialize_screen(name="Stellarpunk")
        self._start_time = time.time()

        self._enter_mode(Mode.MAIN_MENU)

    def terminate(self) -> None:
        if self.interface.gamestate:
            self.interface.gamestate.force_unpause(self)

    def _draw_main_menu(self) -> None:
        y = 15
        x = 15
        self._main_menu.draw(self.viewscreen, y, x)

    def _draw_new_game(self) -> None:
        self.viewscreen.erase()
        self.viewscreen.addstr(15, 15, f'generating a universe...')
        self.viewscreen.addstr(16, 15, f'{self._current_generation_step} {self._generation_ticks}/{self._estimated_generation_ticks}')
        #TODO: janky hack to draw a progress bar
        m = ui_util.MeterMenu("foo", [])
        m._draw_meter(self.viewscreen, ui_util.MeterItem("test", self._generation_ticks, maximum=max(self._generation_ticks, self._estimated_generation_ticks)), 17, 15)
        #self.viewscreen.addstr(17, 15, "."*self._generation_ticks)
        if self._universe_loaded:
            self.viewscreen.addstr(18, 15, f'universe generated.')
            self.viewscreen.addstr(19, 15, f'<press esc or return to start>')

    def _draw_load_game(self) -> None:
        # menu to choose which save game to load
        # selecting one loads that game and then transitions 
        y = 15
        x = 15
        self._load_menu.draw(self.viewscreen, y, x)

    def update_display(self) -> None:
        # TODO: have some clever graphic for the main menu

        if self._mode == Mode.MAIN_MENU:
            self._draw_main_menu()
        elif self._mode == Mode.NEW_GAME:
            self._draw_new_game()
        elif self._mode == Mode.LOAD_GAME:
            self._draw_load_game()
        else:
            raise ValueError(f'cannot draw mode {self._mode}')
        self.interface.refresh_viewscreen()

    def _key_list_main_menu(self) -> Collection[interface.KeyBinding]:
        return self._main_menu.key_list()

    def _key_list_new_game(self) -> Collection[interface.KeyBinding]:
        if self._universe_loaded:
            def begin() -> None:
                self._enter_mode(Mode.EXIT)
            key_list = self.bind_aliases(
                [curses.ascii.ESC, curses.ascii.CR], begin, help_key="startup_start_game"
            )
            return key_list
        else:
            return []

    def _key_list_load_game(self) -> Collection[interface.KeyBinding]:
        def cancel() -> None:
            self._enter_mode(Mode.MAIN_MENU)

        key_list = list(self._load_menu.key_list())
        key_list.extend(self.bind_aliases(
            [curses.ascii.ESC], cancel, help_key="startup_load_cancel"
        ))
        return key_list

    def key_list(self) -> Collection[interface.KeyBinding]:
        if self._mode == Mode.MAIN_MENU:
            return self._key_list_main_menu()
        elif self._mode == Mode.NEW_GAME:
            return self._key_list_new_game()
        elif self._mode == Mode.LOAD_GAME:
            return self._key_list_load_game()
        else:
            raise ValueError(f'unknown mode {self._mode}')

