import time
import threading
import enum
import curses.ascii
import datetime
from collections.abc import Collection
from typing import Any, Optional

import hashime # type: ignore

from stellarpunk import core, interface, generate, util
from stellarpunk.interface import pilot, ui_util
from stellarpunk.serialization import save_game

class Mode(enum.Enum):
    """ Startup Menu Modes """
    NONE = enum.auto()
    MAIN_MENU = enum.auto()
    RESUME = enum.auto()
    NEW_GAME = enum.auto()
    CREATE_NEW_GAME = enum.auto()
    LOAD_GAME = enum.auto()
    LOADING = enum.auto()
    EXIT_GAME = enum.auto()
    EXIT = enum.auto()

class ConfigOption(enum.Enum):
    UNIVERSE_SCALE = enum.auto()
    SECTOR_SCALE = enum.auto()
    SECTOR_STD = enum.auto()
    MAX_SECTOR_EDGE_LENGTH = enum.auto()
    NUM_SECTORS = enum.auto()
    NUM_INHABITED_SECTORS = enum.auto()
    MEAN_INHABITED_RESOURCES = enum.auto()
    MEAN_UNINHABITED_RESOURCES = enum.auto()
    STATION_FACTOR = enum.auto()
    MINER_FACTOR = enum.auto()
    TRADER_FACTOR = enum.auto()

class StartupView(generate.UniverseGeneratorObserver, save_game.GameSaverObserver, interface.View):
    """ Startup screen for giving player loading feedback.

    Watches universe generation and gives player info about progress.

    TODO: also startup menu? """

    def __init__(self, generator:generate.UniverseGenerator, game_saver:save_game.GameSaver, *args:Any, **kwargs:Any):
        super().__init__(*args, **kwargs)

        self._generator = generator
        self._generator_thread:Optional[threading.Thread] = None
        self._generator_exception:Optional[Exception] = None
        self._threaded_generation = False

        self._game_saver = game_saver

        self._start_time = time.time()
        self._target_startup_time = 5.0
        self._mode = Mode.NONE

        # these fields are shared by multiple threads
        # their access should be protected by the lock
        self._generation_lock = util.TimeoutLock()
        self._generator_abort:bool = False
        self._estimated_generation_ticks = 0
        self._current_generation_step = generate.GenerationStep.NONE
        self._generation_ticks = 0
        self._universe_loaded = False

        self._main_menu = ui_util.Menu("null", [])
        self._new_game_config_menu = ui_util.MeterMenu("null", [])
        self._load_menu = ui_util.Menu("null", [])
        self._saves:list[save_game.SaveGame] = []

        self._save_game:Optional[save_game.SaveGame] = None
        self._loaded_gamestate:Optional[core.Gamestate] = None

    # generate.UniverseGeneratorObserver
    # these methods might be called from the generation thread
    def estimated_generation_ticks(self, ticks:int) -> None:
        with self._generation_lock.acquire():
            self._estimated_generation_ticks = ticks

    def generation_step(self, step:generate.GenerationStep) -> None:
        with self._generation_lock.acquire():
            self.logger.debug(f'step: {step} {self._generation_ticks}/{self._estimated_generation_ticks}')
            self._current_generation_step = step

    def generation_tick(self) -> None:
        with self._generation_lock.acquire():
            if self._generator_abort:
                raise generate.GenerationError(generate.GenerationErrorCase.ABORT)
            self._generation_ticks += 1
            self.logger.debug(f'generation tick: {self._current_generation_step} {self._generation_ticks}/{self._estimated_generation_ticks}')

    # save_game.GameSaverObserver
    # these methods might be called from the generation thread
    def load_start(self, ticks:int, game_saver:save_game.GameSaver) -> None:
        with self._generation_lock.acquire():
            self._estimated_generation_ticks = ticks

    def load_tick(self, game_saver:save_game.GameSaver) -> None:
        with self._generation_lock.acquire():
            if self._generator_abort:
                raise save_game.LoadError(save_game.LoadErrorCase.ABORT)
            self._generation_ticks += 1
            self.logger.debug(f'load tick: {self._generation_ticks}/{self._estimated_generation_ticks}')


    def _generate_universe(self) -> None:
        self.interface.log_message("generating a universe...")
        gamestate = self._generator.generate_universe()
        self.interface.log_message("new universe created")
        gamestate.force_pause(self)
        # the gamestate will get sent to people via an event on universe
        # generator
        self._universe_loaded = True

    def _generate_universe_threaded(self) -> None:
        try:
            self._generate_universe()
        except Exception as e:
            self._generator_exception = e

    def _load_game(self) -> None:
        assert(self._save_game)
        self.interface.log_message(f'loading savegame "{self._save_game.filename}" ...')
        start_time = time.perf_counter()
        gamestate = self._game_saver.load(self._save_game.filename)
        gamestate.force_pause(self)
        # the gamestate will get sent to people via an event from universe
        # generator
        self._loaded_gamestate = gamestate
        end_time = time.perf_counter()
        self.interface.log_message(f'game loaded in {end_time-start_time:.2f}s.')
        self._universe_loaded = True

    def _load_game_threaded(self) -> None:
        try:
            self._load_game()
        except Exception as e:
            self._generator_exception = e

    def _enter_main_menu(self) -> None:
        self.viewscreen.erase()
        menu_items:list[ui_util.TextMenuItem] = []
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
        if self.interface.gamestate:
            menu_items.append(ui_util.TextMenuItem(
                "Resume", lambda: self._enter_mode(Mode.RESUME)
            ))
        self._main_menu = ui_util.Menu(
            "Main Menu",
            ui_util.number_text_menu_items(menu_items)
        )

    def _enter_resume(self) -> None:
        self.interface.close_view(self)

    def _enter_new_game(self) -> None:
        self.viewscreen.erase()

        # start with default options
        self._generator.universe_config = generate.UniverseConfig()

        #TODO: get savegame options
        self.config_options:dict[ConfigOption, ui_util.MeterItem] = {}
        self.config_options[ConfigOption.UNIVERSE_SCALE] = ui_util.MeterItem(
            "Universe scale",
            self._generator.universe_config.universe_radius,
            minimum=self._generator.universe_config.universe_radius*0.1,
            maximum=self._generator.universe_config.universe_radius*5.0,
            increment=self._generator.universe_config.universe_radius/50,
            number_format=".2",
        )
        self.config_options[ConfigOption.SECTOR_SCALE] = ui_util.MeterItem(
            "Sector scale",
            self._generator.universe_config.sector_radius_mean,
            minimum=self._generator.universe_config.sector_radius_mean*0.1,
            maximum=self._generator.universe_config.sector_radius_mean*5.0,
            increment=self._generator.universe_config.sector_radius_mean/50,
            number_format=".2",
        )
        self.config_options[ConfigOption.SECTOR_STD] = ui_util.MeterItem(
            "Sector radius rel stdev ",
            self._generator.universe_config.sector_radius_std/self._generator.universe_config.sector_radius_mean,
            minimum=0.05,
            maximum=1.0,
            increment=0.05,
            number_format=".2f",
        )
        self.config_options[ConfigOption.MAX_SECTOR_EDGE_LENGTH] = ui_util.MeterItem(
            "Sector edge max length",
            self._generator.universe_config.max_sector_edge_length,
            minimum=self._generator.universe_config.max_sector_edge_length*0.1,
            maximum=self._generator.universe_config.max_sector_edge_length*5.0,
            increment=self._generator.universe_config.max_sector_edge_length/50,
            number_format=".2",
        )

        self.config_options[ConfigOption.NUM_SECTORS] = ui_util.MeterItem(
            "Total sectors",
            self._generator.universe_config.num_sectors,
            minimum=self._generator.universe_config.num_sectors*0.1,
            maximum=self._generator.universe_config.num_sectors*5.0,
            increment=1,
        )
        self.config_options[ConfigOption.NUM_INHABITED_SECTORS] = ui_util.MeterItem(
            "Total inhabited sectors",
            self._generator.universe_config.num_habitable_sectors,
            minimum=self._generator.universe_config.num_habitable_sectors*0.1,
            maximum=self._generator.universe_config.num_habitable_sectors*5.0,
            increment=1,
        )
        min_resources = 0.0
        max_resources = max(self._generator.universe_config.mean_habitable_resources*5.0, self._generator.universe_config.mean_uninhabitable_resources*5.0)
        self.config_options[ConfigOption.MEAN_INHABITED_RESOURCES] = ui_util.MeterItem(
            "Mean inhabited resources",
            self._generator.universe_config.mean_habitable_resources,
            minimum=min_resources,
            maximum=max_resources,
            increment=max_resources/50.0,
            number_format=".2",
        )
        self.config_options[ConfigOption.MEAN_UNINHABITED_RESOURCES] = ui_util.MeterItem(
            "Mean uninhabited resources",
            self._generator.universe_config.mean_uninhabitable_resources,
            minimum=min_resources,
            maximum=max_resources,
            increment=max_resources/50.0,
            number_format=".2",
        )

        self.config_options[ConfigOption.STATION_FACTOR] = ui_util.MeterItem(
            "Stations per sector per good",
            self._generator.universe_config.station_factor,
            minimum=0.1,
            maximum=10.0,
            increment=0.1,
            number_format=".2f",
        )
        self.config_options[ConfigOption.MINER_FACTOR] = ui_util.MeterItem(
            "Miners per sector per resource",
            self._generator.universe_config.mining_ship_factor,
            minimum=0.1,
            maximum=10.0,
            increment=0.1,
            number_format=".2f",
        )
        self.config_options[ConfigOption.TRADER_FACTOR] = ui_util.MeterItem(
            "Traders per sector per station",
            self._generator.universe_config.trading_ship_factor,
            minimum=0.1,
            maximum=10.0,
            increment=0.1,
            number_format=".2f",
        )

        self._new_game_config_menu = ui_util.MeterMenu(
            "Configure a New Game",
            list(self.config_options.values()),
            number_width=9,
        )

    def _validate_config_options(self) -> None:
        if self.config_options[ConfigOption.NUM_SECTORS].setting < self.config_options[ConfigOption.NUM_INHABITED_SECTORS].setting:
            raise ui_util.ValidationError("not enough total sectors for that many inhabited sectors")

    def _enter_create_new_game(self) -> None:
        self._validate_config_options()

        # transfer values from the config menu to the universe config
        self._generator.universe_config.universe_radius = self.config_options[ConfigOption.UNIVERSE_SCALE].setting
        self._generator.universe_config.sector_radius_mean = self.config_options[ConfigOption.SECTOR_SCALE].setting
        self._generator.universe_config.sector_radius_std = self.config_options[ConfigOption.SECTOR_STD].setting
        self._generator.universe_config.max_sector_edge_length = self.config_options[ConfigOption.MAX_SECTOR_EDGE_LENGTH].setting

        self._generator.universe_config.num_sectors = int(self.config_options[ConfigOption.NUM_SECTORS].setting)
        self._generator.universe_config.num_habitable_sectors = int(self.config_options[ConfigOption.NUM_INHABITED_SECTORS].setting)
        self._generator.universe_config.mean_habitable_resources = self.config_options[ConfigOption.MEAN_INHABITED_RESOURCES].setting
        self._generator.universe_config.mean_uninhabitable_resources = self.config_options[ConfigOption.MEAN_UNINHABITED_RESOURCES].setting

        self._generator.universe_config.station_factor = self.config_options[ConfigOption.STATION_FACTOR].setting
        self._generator.universe_config.mining_ship_factor = self.config_options[ConfigOption.MINER_FACTOR].setting
        self._generator.universe_config.trading_ship_factor = self.config_options[ConfigOption.TRADER_FACTOR].setting

        self._generation_ticks = 0
        self._estimated_generation_ticks = 100
        self._generator.observe(self)
        if self._threaded_generation:
            self._generator_thread = threading.Thread(target=self._generate_universe_threaded)
            self._generator_thread.start()
        else:
            self._generate_universe()

    def _exit_create_new_game(self) -> None:
        assert(self._universe_loaded)
        self._generator.unobserve(self)
        #TODO: should we have a timeout here?
        if self._threaded_generation:
            assert(self._generator_thread is not None)
            self._generator_thread.join()
            self._generator_thread = None
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
            self._save_game = save
            self._enter_mode(Mode.LOADING)

        # get savegame options
        load_options = []
        self._saves.clear()
        for s in self._game_saver.list_save_games():
            self._saves.append(s)
            load_options.append(ui_util.TextMenuItem(
                s.filename,
                lambda s=s: load_game(s) # type: ignore
            ))

        self._load_menu = ui_util.Menu(
            "Load Game",
            ui_util.number_text_menu_items(load_options)
        )

    def _enter_loading(self) -> None:
        self.viewscreen.erase()

        self._generation_ticks = 0
        self._estimated_generation_ticks = 100
        self._game_saver.observe(self)

        if self._threaded_generation:
            self._generator_thread = threading.Thread(target=self._load_game_threaded)
            self._generator_thread.start()
        else:
            self._load_game()

    def _exit_loading(self) -> None:
        assert(self._universe_loaded)

        if self._threaded_generation:
            assert(self._generator_thread is not None)
            self._generator_thread.join()
            self._generator_thread = None

        self._game_saver.unobserve(self)

        self.interface.runtime.exit_startup()
        self.interface.runtime.start_game()

        assert self._loaded_gamestate
        assert self.interface.player.character and isinstance(self.interface.player.character.location, core.Ship)
        assert self.interface.player == self._loaded_gamestate.player
        pilot_view = pilot.PilotView(self.interface.player.character.location, self._loaded_gamestate, self.interface)
        self.interface.close_all_views()
        if len(self.interface.views) > 0:
            raise Exception()
        self.interface.open_view(pilot_view)


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
        if self._mode == Mode.CREATE_NEW_GAME:
            self._exit_create_new_game()
        elif self._mode == Mode.LOADING:
            self._exit_loading()

        # enter the new mode
        self._mode = mode
        if mode == Mode.MAIN_MENU:
            self._enter_main_menu()
        elif mode == Mode.RESUME:
            self._enter_resume()
        elif mode == Mode.NEW_GAME:
            self._enter_new_game()
        elif mode == Mode.CREATE_NEW_GAME:
            self._enter_create_new_game()
        elif mode == Mode.LOAD_GAME:
            self._enter_load_game()
        elif mode == Mode.LOADING:
            self._enter_loading()
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

    def focus(self) -> None:
        super().focus()
        self.interface.reinitialize_screen(name="Stellarpunk")
        self.active=True

    def _force_cleanup_generator_thread(self) -> None:
        if self._generator_thread is None:
            return
        if not self._generator_thread.is_alive():
            self.logger.warning(f'generator thread is not None, but is not running')
            self._generator_thread = None
        else:
            self.logger.warning(f'generation thread is still running, attempting to terminate...')
            # abort the generator thread if it's still running
            # this might happen if the user quits while generating/loading
            with self._generation_lock.acquire(timeout=5.0) as result:
                if result:
                    self._generator_abort = True
                else:
                    self.logger.error(f'could not acquire generation lock to trigger generation/load abort. aborting thread cleanup.')
            if result:
                self._generator_thread.join(5.0)
                if self._generator_thread.is_alive():
                    self.logger.error(f'could not join generator thread. aborting thread cleanup')
                else:
                    self._generator_thread = None
                    self.logger.warning('generation thread cleaned up')

    def terminate(self) -> None:
        self._force_cleanup_generator_thread()

        if self.interface.gamestate:
            self.interface.gamestate.force_unpause(self)

    def _draw_main_menu(self) -> None:
        y = 15
        x = 15
        self._main_menu.draw(self.viewscreen, y, x)

    def _draw_new_game(self) -> None:
        """ new game configuration screen. """
        self.viewscreen.erase()
        y = 15
        x = 15
        self._new_game_config_menu.draw(self.viewscreen, y, x)

        y = y + self._new_game_config_menu.height + 4
        self.viewscreen.addstr(y, x, "Press <ENTER> to create new game or <ESC> to cancel")

    def _draw_create_new_game(self) -> None:
        """ new game creation loading screen. """
        with self._generation_lock.acquire():
            self.viewscreen.erase()
            self.viewscreen.addstr(15, 15, f'generating a universe...')
            self.viewscreen.addstr(16, 15, f'{self._current_generation_step} {self._generation_ticks}/{self._estimated_generation_ticks}')
            #TODO: janky hack to draw a progress bar
            m = ui_util.MeterItem("test", self._generation_ticks, maximum=max(self._generation_ticks, self._estimated_generation_ticks))
            m.draw(self.viewscreen, 17, 15)
            #self.viewscreen.addstr(17, 15, "."*self._generation_ticks)
            if self._universe_loaded:
                self.viewscreen.addstr(18, 15, f'universe generated.')
                self.viewscreen.addstr(19, 15, f'<press esc or return to start>')
            elif self._generator_exception:
                raise self._generator_exception

    def _draw_load_game(self) -> None:
        """ choose save to load menu """
        self.viewscreen.erase()
        # menu to choose which save game to load
        # selecting one loads that game and then transitions 
        y = 15
        x = 15
        if len(self._load_menu.options) > 0:
            self._load_menu.draw(self.viewscreen, y, x)

            # draw some info about the saved game
            selected_save = self._saves[self._load_menu.selected_option]

            randomart = hashime.DrunkenBishop(selected_save.game_fingerprint).to_art()
            i = 15
            for line in randomart.split("\n"):
                self.viewscreen.addstr(i, x+64, line)
                i += 1

            self.viewscreen.addstr(16, x+64+20, selected_save.pc_name)
            self.viewscreen.addstr(17, x+64+20, selected_save.pc_sector_name)
            self.viewscreen.addstr(18, x+64+20, f'{selected_save.game_date}')

        else:
            self.viewscreen.addstr(18, 15, f'no save games to load.')
            self.viewscreen.addstr(19, 15, f'<press esc or return to start>')

    def _draw_loading(self) -> None:
        with self._generation_lock.acquire():
            self.viewscreen.erase()
            self.viewscreen.addstr(15, 15, f'loading game...')
            self.viewscreen.addstr(16, 15, f'{self._generation_ticks}/{self._estimated_generation_ticks}')
            #TODO: janky hack to draw a progress bar
            m = ui_util.MeterItem("test", self._generation_ticks, maximum=max(self._generation_ticks, self._estimated_generation_ticks))
            m.draw(self.viewscreen, 17, 15)
            #self.viewscreen.addstr(17, 15, "."*self._generation_ticks)
            if self._universe_loaded:
                self.viewscreen.addstr(18, 15, f'game loaded.')
                self.viewscreen.addstr(19, 15, f'<press esc or return to start>')
            elif self._generator_exception:
                raise self._generator_exception

    def update_display(self) -> None:
        # TODO: have some clever graphic for the main menu

        if self._mode == Mode.MAIN_MENU:
            self._draw_main_menu()
        elif self._mode == Mode.NEW_GAME:
            self._draw_new_game()
        elif self._mode == Mode.CREATE_NEW_GAME:
            self._draw_create_new_game()
        elif self._mode == Mode.LOAD_GAME:
            self._draw_load_game()
        elif self._mode == Mode.LOADING:
            self._draw_loading()
        else:
            raise ValueError(f'cannot draw mode {self._mode}')
        self.interface.refresh_viewscreen()

    def _key_list_main_menu(self) -> Collection[interface.KeyBinding]:
        return self._main_menu.key_list()

    def _key_list_new_game(self) -> Collection[interface.KeyBinding]:
        def cancel() -> None:
            self._enter_mode(Mode.MAIN_MENU)
        def create() -> None:
            try:
                self._validate_config_options()
            except ui_util.ValidationError as e:
                self.interface.status_message(
                        e.message,
                        self.interface.get_color(interface.Color.ERROR)
                )
                return
            self._enter_mode(Mode.CREATE_NEW_GAME)
        def quick_gen() -> None:
            """ sets up universe generation to be fast, creates a very simple
            universe. """

            self._generator.universe_config.cultures=[self._generator._empty_name_model_culture]
            self._generator.universe_config.num_cultures = [1,1]
            self._generator._load_empty_name_models(self._generator._empty_name_model_culture)
            self.config_options[ConfigOption.NUM_SECTORS].setting = 1
            self.config_options[ConfigOption.NUM_INHABITED_SECTORS].setting = 1

            self._generator.universe_config.production_chain_config.n_ranks=3
            self._generator.universe_config.production_chain_config.min_per_rank=(2,2,2)
            self._generator.universe_config.production_chain_config.max_per_rank=(2,2,2)
            self._generator.universe_config.production_chain_config.min_final_inputs=1
            self._generator.universe_config.production_chain_config.max_fraction_one_to_one=1.0
            self._generator.universe_config.production_chain_config.max_fraction_single_input=1.0
            self._generator.universe_config.production_chain_config.max_fraction_single_output=1.0

            self._enter_mode(Mode.CREATE_NEW_GAME)

        key_list = list(self._new_game_config_menu.key_list())
        key_list.extend(self.bind_aliases(
            [ord('q')], quick_gen, help_key="startup_new_game_quick_gen"
        ))
        key_list.extend(self.bind_aliases(
            [curses.ascii.ESC], cancel, help_key="startup_new_game_cancel"
        ))
        key_list.extend(self.bind_aliases(
            [curses.ascii.CR], create, help_key="startup_new_game_create"
        ))
        return key_list

    def _key_list_create_new_game(self) -> Collection[interface.KeyBinding]:
        with self._generation_lock.acquire():
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

        if len(self._load_menu.options) > 0:
            key_list = list(self._load_menu.key_list())
            key_list.extend(self.bind_aliases(
                [curses.ascii.ESC], cancel, help_key="startup_load_cancel"
            ))
            return key_list
        else:
            return self.bind_aliases(
                [curses.ascii.ESC, curses.ascii.CR], cancel, help_key="startup_load_cancel"
            )


    def _key_list_loading(self) -> Collection[interface.KeyBinding]:
        with self._generation_lock.acquire():
            if self._universe_loaded:
                def begin() -> None:
                    self._enter_mode(Mode.EXIT)
                key_list = self.bind_aliases(
                    [curses.ascii.ESC, curses.ascii.CR], begin, help_key="startup_start_game"
                )
                return key_list
            else:
                return []

    def key_list(self) -> Collection[interface.KeyBinding]:
        if self._mode == Mode.MAIN_MENU:
            return self._key_list_main_menu()
        elif self._mode == Mode.NEW_GAME:
            return self._key_list_new_game()
        elif self._mode == Mode.CREATE_NEW_GAME:
            return self._key_list_create_new_game()
        elif self._mode == Mode.LOAD_GAME:
            return self._key_list_load_game()
        elif self._mode == Mode.LOADING:
            return self._key_list_loading()
        else:
            raise ValueError(f'unknown mode {self._mode}')

