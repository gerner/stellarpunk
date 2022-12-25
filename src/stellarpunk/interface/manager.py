""" Interface Manager gluing together the interface elements """

from typing import Optional, Sequence, Any, Mapping, Callable, Collection
import cProfile
import pstats
import curses
import uuid

from stellarpunk import core, interface, generate, util, config, dialog
from stellarpunk.interface import universe, sector, pilot, command_input, character, comms

class InterfaceManager:
    def __init__(self, gamestate:core.Gamestate, generator:generate.UniverseGenerator) -> None:
        self.interface = interface.Interface(gamestate, generator)
        self.gamestate = gamestate

        self.profiler:Optional[cProfile.Profile] = None

        self._command_list = {x.command:x for x in self.command_list()}

    def __enter__(self) -> "InterfaceManager":
        self.interface.key_list = {x.key:x for x in self.key_list()}
        self.interface.__enter__()

        # TODO: fix the way we set up the rest of the interface
        self.interface.initialize()
        uv = universe.UniverseView(self.gamestate, self.interface)
        self.interface.open_view(uv)
        assert self.gamestate.player.character.location.sector is not None
        sv = uv.open_sector_view(self.gamestate.player.character.location.sector)
        assert isinstance(self.gamestate.player.character.location, core.Ship)
        sv.open_pilot_view(self.gamestate.player.character.location)

        return self

    def __exit__(self, *args:Any) -> None:
        self.interface.__exit__(*args)

    def focused_view(self) -> Optional[interface.View]:
        """ Get the topmost view that's not the topmost CommandInput """
        assert len(self.interface.views) > 0
        if isinstance(self.interface.views[-1], command_input.CommandInput):
            if len(self.interface.views) == 1:
                return None
            else:
                return self.interface.views[-2]
        else:
            return self.interface.views[-1]

    def time_accel(self) -> None:
        old_accel_rate, _ = self.gamestate.get_time_acceleration()
        new_accel_rate = old_accel_rate * 1.25
        if util.isclose_flex(new_accel_rate, 1.0, atol=0.1):
            new_accel_rate = 1.0
        if new_accel_rate >= interface.Settings.MAX_TIME_ACCEL:
            new_accel_rate = interface.Settings.MAX_TIME_ACCEL
        self.gamestate.time_acceleration(new_accel_rate, False)

    def time_decel(self) -> None:
        old_accel_rate, _ = self.gamestate.get_time_acceleration()
        new_accel_rate = old_accel_rate / 1.25
        if util.isclose_flex(new_accel_rate, 1.0, atol=0.1):
            new_accel_rate = 1.0
        if new_accel_rate <= interface.Settings.MIN_TIME_ACCEL:
            new_accel_rate = interface.Settings.MIN_TIME_ACCEL
        self.gamestate.time_acceleration(new_accel_rate, False)

    def help(self) -> None:
        command_list = {x.command: x for x in self.command_list()}

        h = object()
        view = self.focused_view()
        if view is not None:
            command_list.update({x.command: x for x in view.command_list()})

        self.interface.log_message("press \":\" to enter command mode")
        self.interface.log_message("type a command and press <ENTER> to execute it")
        self.interface.log_message("available commands:")
        for k,v in command_list.items():
            self.interface.log_message(f'\t{k}\t{v.help}')

        self.interface.log_message("")

    def keys(self) -> None:
        key_list = self.interface.key_list.copy()

        h = object()
        view = self.focused_view()
        if view is not None:
            key_list.update({x.key: x for x in view.key_list()})

        self.interface.log_message("keys:")
        for k,v in key_list.items():
            if k != curses.KEY_MOUSE:
                if chr(k).isprintable():
                    if chr(k) == " ":
                        self.interface.log_message(f'\t<SPACE>\t{v.help}')
                    else:
                        self.interface.log_message(f'\t{chr(k)}\t{v.help}')
                elif k == ord('\r'):
                    self.interface.log_message(f'\t<ENTER>\t{v.help}')

        self.interface.log_message("")

    def bind_key(self, k:int, f:Callable[[], None]) -> interface.KeyBinding:
        try:
            h = getattr(getattr(config.Settings.help.interface, self.__class__.__name__).keys, chr(k))
        except AttributeError:
            h = "NO HELP"
        return interface.KeyBinding(k, f, h)

    def bind_command(self, command:str, f: Callable[[Sequence[str]], None], tab_completer:Optional[Callable[[str, str], str]]=None) -> interface.CommandBinding:
        try:
            h = getattr(getattr(config.Settings.help.interface, self.__class__.__name__).commands, command)
        except AttributeError:
            h = "NO HELP"
        return interface.CommandBinding(command, f, h, tab_completer)


    def key_list(self) -> Collection[interface.KeyBinding]:
        def open_command_prompt() -> None:
            command_list = self._command_list.copy()
            v = self.focused_view()
            if v is not None:
                command_list.update({x.command: x for x in v.command_list()})
            self.interface.open_view(command_input.CommandInput(self.interface, commands=command_list))

        return [
            self.bind_key(ord(" "), self.gamestate.pause),
            self.bind_key(ord(">"), self.time_accel),
            self.bind_key(ord("<"), self.time_decel),
            self.bind_key(ord(":"), open_command_prompt),
            self.bind_key(ord("?"), self.help),
        ]

    def command_list(self) -> Collection[interface.CommandBinding]:
        """ Global commands that should be valid in any context. """
        def fps(args:Sequence[str]) -> None: self.interface.show_fps = not self.interface.show_fps
        def quit(args:Sequence[str]) -> None: self.interface.gamestate.quit()
        def raise_exception(args:Sequence[str]) -> None: self.gamestate.should_raise = True
        def colordemo(args:Sequence[str]) -> None: self.interface.open_view(interface.ColorDemo(self.interface))
        def attrdemo(args:Sequence[str]) -> None: self.interface.open_view(interface.AttrDemo(self.interface))
        def profile(args:Sequence[str]) -> None:
            if self.profiler:
                self.profiler.disable()
                pstats.Stats(self.profiler).dump_stats("/tmp/profile.prof")
            else:
                self.profiler = cProfile.Profile()
                self.profiler.enable()

        def fast(args:Sequence[str]) -> None:
            _, fast_mode = self.gamestate.get_time_acceleration()
            self.gamestate.time_acceleration(1.0, fast_mode=not fast_mode)

        def decrease_fps(args:Sequence[str]) -> None:
            if self.interface.max_fps == self.interface.desired_fps:
                self.interface.desired_fps+=1
                self.interface.max_fps = self.interface.desired_fps
            else:
                self.interface.desired_fps+=1

        def increase_fps(args:Sequence[str]) -> None:
            self.interface.desired_fps-=1
            if self.interface.max_fps > self.interface.desired_fps:
               self.interface.max_fps = self.interface.desired_fps

        def open_pilot(args:Sequence[str]) -> None:
            """ Opens a PilotView on the ship the player is piloting """
            if not isinstance(self.gamestate.player.character.location, core.Ship):
                #TODO: what if the character is just a passenger? surely they cannot just take the helm
                raise command_input.UserError(f'player is not in a ship to pilot')
            self.interface.swap_view(
                pilot.PilotView(self.gamestate.player.character.location, self.interface),
                self.focused_view()
            )

        def open_sector(args:Sequence[str]) -> None:
            """ Opens a sector view on the sector the player is in """
            assert self.gamestate.player.character.location.sector is not None
            sector_view = sector.SectorView(self.gamestate.player.character.location.sector, self.interface)
            self.interface.swap_view(
                sector_view,
                self.focused_view()
            )
            sector_view.select_target(
                self.gamestate.player.character.location.entity_id,
                self.gamestate.player.character.location,
                focus=True
            )

        def open_universe(args:Sequence[str]) -> None:
            assert self.gamestate.player.character.location.sector is not None
            universe_view = universe.UniverseView(self.gamestate, self.interface)
            self.interface.swap_view(
                universe_view,
                self.focused_view()
            )
            universe_view.select_sector(
                self.gamestate.player.character.location.sector,
                focus=True
            )

        def open_character(args:Sequence[str]) -> None:
            if len(args) == 0:
                target_character = self.gamestate.player.character
            else:
                try:
                    chr_id = uuid.UUID(args[0])
                except:
                    raise command_input.UserError(f'{args[0]} is not a valid uuid')
                if chr_id in self.gamestate.characters:
                    target_character = self.gamestate.characters[chr_id]
                else:
                    raise command_input.UserError(f'no character found for id {args[0]}')
            character_view = character.CharacterView(target_character, self.interface)
            self.interface.swap_view(
                character_view,
                self.focused_view()
            )

        def open_comms(args:Sequence[str]) -> None:
            comms_view = comms.CommsView(dialog.load_dialog("dialog_demo"), self.interface)
            self.interface.open_view(comms_view, deactivate_views=True)

        def open_dialog(args:Sequence[str]) -> None:
            d = dialog.load_dialog(args[0])

        return [
            self.bind_command("pause", lambda x: self.gamestate.pause()),
            self.bind_command("t_accel", lambda x: self.time_accel()),
            self.bind_command("t_decel", lambda x: self.time_decel()),
            self.bind_command("fps", fps),
            self.bind_command("quit", quit),
            self.bind_command("raise", raise_exception),
            self.bind_command("colordemo", colordemo),
            self.bind_command("attrdemo", attrdemo),
            self.bind_command("profile", profile),
            self.bind_command("fast", fast),
            self.bind_command("decrease_fps", decrease_fps),
            self.bind_command("increase_fps", increase_fps),
            self.bind_command("help", lambda x: self.help()),
            self.bind_command("keys", lambda x: self.keys()),
            self.bind_command("pilot", open_pilot),
            self.bind_command("sector", open_sector),
            self.bind_command("universe", open_universe),
            self.bind_command("character", open_character, util.tab_completer(map(str, self.interface.gamestate.characters.keys()))),
            self.bind_command("comms", open_comms),
            self.bind_command("dialog", open_dialog),
        ]

