""" Interface Manager gluing together the interface elements """

import cProfile
import pstats
import curses
import uuid
import collections
import logging
from typing import Optional, Sequence, Any, Callable, Collection, Dict, Tuple, List, Mapping

import numpy as np

from stellarpunk import core, interface, generate, util, config, events, narrative
from stellarpunk.interface import audio, universe, sector, pilot, command_input, character, comms, station, ui_events


KEY_DISPLAY = {
    ord('\r') :         "<ENTER>",
    ord('\n') :         "<ENTER>",
    curses.ascii.ESC :  "<ESC>",
    ord(' ') :          "<SPACE>",
    curses.KEY_LEFT :   "<LEFT>",
    curses.KEY_UP :     "<UP>",
    curses.KEY_RIGHT :  "<RIGHT>",
    curses.KEY_DOWN :   "<DOWN>",
    ord(',') :          "\",\"",
}


class ColorDemo(interface.View):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def update_display(self) -> None:
        self.interface.viewscreen.erase()
        self.interface.viewscreen.addstr(0, 35, "COLOR DEMO")

        for c in range(256):
            self.interface.viewscreen.addstr(
                int(c/8)+1, c % 8 * 9,
                f'...{c:03}...', curses.color_pair(c)
            )
        self.interface.viewscreen.addstr(34, 1, "Press any key to continue")
        self.interface.refresh_viewscreen()

    def handle_input(self, key: int, dt: float) -> bool:
        if key == ord(" "):
            curses.flash()
            return True
        elif key != -1:
            self.interface.close_view(self)
            return True
        else:
            return False


class AttrDemo(interface.View):
    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

    def update_display(self) -> None:
        self.interface.viewscreen.erase()
        self.interface.viewscreen.addstr(0, 35, "ATTR DEMO");

        attrs = [
            (curses.A_ALTCHARSET, "Alternate character set mode"),
            (curses.A_BLINK, "Blink mode"),
            (curses.A_BOLD, "Bold mode"),
            (curses.A_DIM, "Dim mode"),
            (curses.A_INVIS, "Invisible or blank mode"),
            (curses.A_ITALIC, "Italic mode"),
            (curses.A_NORMAL, "Normal attribute"),
            (curses.A_PROTECT, "Protected mode"),
            (curses.A_REVERSE, "Reverse background and foreground colors"),
            (curses.A_STANDOUT, "Standout mode"),
            (curses.A_UNDERLINE, "Underline mode"),
        ]

        i = 1
        for a in attrs:
            self.interface.viewscreen.addstr(i, 1, a[1], a[0])
            i+=1
        self.interface.viewscreen.addstr(i+1, 1, "Press any key to continue")
        self.interface.refresh_viewscreen()

    def handle_input(self, key:int, dt:float) -> bool:
        if key != -1:
            self.interface.close_view(self)
            return True
        else:
            return False


class KeyDemo(interface.View):
    @staticmethod
    def get_curses_keys() -> Dict[int, str]:
        curses_keys = {}
        for k in curses.__dict__.keys():
            if k.startswith("KEY_") and isinstance(curses.__dict__[k], int):
                curses_keys[curses.__dict__[k]] = k

        return curses_keys

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.curses_keys = KeyDemo.get_curses_keys()

    def initialize(self) -> None:
        self.interface.viewscreen.erase()
        self.interface.viewscreen.addstr(1, 1, "type any key and see its code in the log window")
        self.interface.viewscreen.addstr(2, 1, "this tool captures all keys, so help won't work here")
        self.interface.viewscreen.addstr(5, 1, "press <ESC> to exit")
        self.interface.refresh_viewscreen()

    def handle_input(self, key: int, dt: float) -> bool:
        if key == curses.ascii.ESC:
            self.interface.close_view(self)
            return True

        print_view = "(unprintable)"
        if chr(key).isprintable():
            print_view = f'"{chr(key)}"'

        curses_key = ""
        if key in self.curses_keys:
            curses_key = self.curses_keys[key]

        self.interface.log_message(f'pressed {key} {print_view} {curses_key} at {core.Gamestate.gamestate.ticks}')

        return True


class CircleDemo(interface.View):
    """ Testing tool showing a circle drawn on the screen inside a bounding box

    Useful for debugging the non-trivial logic choosing which parts of the
    circle to draw. """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.scale = 15.
        self.radius = self.scale
        self.bbox = (-1.5*self.scale, -1.5*self.scale, 1.5*self.scale, 1.5*self.scale)

    def update_display(self) -> None:
        self.interface.viewscreen.erase()
        self.interface.viewscreen.addstr(0, 35, "CIRCLE DEMO")

        # make a rectangle
        c = util.make_rectangle_canvas(self.bbox, 1, 2)
        assert isinstance(self.viewscreen, interface.Canvas)
        util.draw_canvas_at(c, self.viewscreen.window, int(self.scale+5), int(self.scale+15), bounds=self.viewscreen_bounds)

        # make a circle
        c = util.make_circle_canvas(self.radius, 1, 2, bbox=self.bbox)
        assert isinstance(self.viewscreen, interface.Canvas)
        util.draw_canvas_at(c, self.viewscreen.window, int(self.scale+5), int(self.scale+15), bounds=self.viewscreen_bounds)

        self.interface.viewscreen.addstr(int(self.scale+15), 1, "Press any key to continue")
        self.interface.refresh_viewscreen()

    def handle_input(self, key: int, dt: float) -> bool:
        if key == curses.ascii.ESC:
            self.interface.close_view(self)
            return True
        elif key == ord("+"):
            self.bbox = (self.bbox[0]-1, self.bbox[1]-1, self.bbox[2]+1, self.bbox[3]+1)
        elif key == ord("-"):
            self.bbox = (self.bbox[0]+1, self.bbox[1]+1, self.bbox[2]-1, self.bbox[3]-1)
        elif key == ord(">"):
            self.radius += 1
        elif key == ord("<"):
            self.radius -= 1
        elif key == ord("w"):
            self.bbox = (self.bbox[0], self.bbox[1]-1, self.bbox[2], self.bbox[3]-1)
        elif key == ord("a"):
            self.bbox = (self.bbox[0]-1, self.bbox[1], self.bbox[2]-1, self.bbox[3])
        elif key == ord("s"):
            self.bbox = (self.bbox[0], self.bbox[1]+1, self.bbox[2], self.bbox[3]+1)
        elif key == ord("d"):
            self.bbox = (self.bbox[0]+1, self.bbox[1], self.bbox[2]+1, self.bbox[3])
        else:
            return False
        return True


class PolygonDemo(interface.View):
    """ Testing tool showing a polygon drawn on the screen inside a bounding box

    Useful for debugging the drawing logic. """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.scale = 15.
        self.sidelength = self.scale
        self.bbox = (-1.5*self.scale, -1.5*self.scale, 1.5*self.scale, 1.5*self.scale)
        self.offset = (0,0)

    def update_display(self) -> None:
        self.interface.viewscreen.erase()
        self.interface.viewscreen.addstr(0, 35, "CIRCLE DEMO")

        # make a rectangle
        c = util.make_rectangle_canvas(self.bbox, 1, 2)
        assert isinstance(self.viewscreen, interface.Canvas)
        util.draw_canvas_at(c, self.viewscreen.window, int(self.scale+5), int(self.scale+15), bounds=self.viewscreen_bounds)

        # make a polygon
        vertices = [(0,0), (self.sidelength, 0), (self.sidelength/2, np.sin(np.pi/3) * self.sidelength)]
        c = util.make_polygon_canvas(vertices, 1, 2, bbox=self.bbox, offset_x=self.offset[0], offset_y=self.offset[1])
        assert isinstance(self.viewscreen, interface.Canvas)
        util.draw_canvas_at(c, self.viewscreen.window, int(self.scale+5), int(self.scale+15), bounds=self.viewscreen_bounds)

        self.interface.viewscreen.addstr(int(self.scale+15), 1, "Press any key to continue")
        self.interface.refresh_viewscreen()

    def handle_input(self, key: int, dt: float) -> bool:
        if key == curses.ascii.ESC:
            self.interface.close_view(self)
            return True
        elif key == ord("+"):
            self.bbox = (self.bbox[0]-1, self.bbox[1]-1, self.bbox[2]+1, self.bbox[3]+1)
        elif key == ord("-"):
            self.bbox = (self.bbox[0]+1, self.bbox[1]+1, self.bbox[2]-1, self.bbox[3]-1)
        elif key == ord(">"):
            self.sidelength += 1
        elif key == ord("<"):
            self.sidelength -= 1
        elif key == ord("w"):
            self.bbox = (self.bbox[0], self.bbox[1]-1, self.bbox[2], self.bbox[3]-1)
        elif key == ord("a"):
            self.bbox = (self.bbox[0]-1, self.bbox[1], self.bbox[2]-1, self.bbox[3])
        elif key == ord("s"):
            self.bbox = (self.bbox[0], self.bbox[1]+1, self.bbox[2], self.bbox[3]+1)
        elif key == ord("d"):
            self.bbox = (self.bbox[0]+1, self.bbox[1], self.bbox[2]+1, self.bbox[3])
        elif key == ord("i"):
            self.offset = (self.offset[0], self.offset[1]-1)
        elif key == ord("k"):
            self.offset = (self.offset[0], self.offset[1]+1)
        elif key == ord("j"):
            self.offset = (self.offset[0]-1, self.offset[1])
        elif key == ord("l"):
            self.offset = (self.offset[0]+1, self.offset[1])
        else:
            return False
        return True


class InterfaceManager(core.CharacterObserver):
    def __init__(self, gamestate:core.Gamestate, generator:generate.UniverseGenerator, event_manager:events.EventManager) -> None:
        self.mixer = audio.Mixer()
        self.interface = interface.Interface(gamestate, generator, self.mixer)
        self.gamestate = gamestate
        self.generator = generator
        self.event_manager = event_manager

        self.profiler:Optional[cProfile.Profile] = None
        self.mouse_on = True

    def __enter__(self) -> "InterfaceManager":
        self.interface.key_list = {x.key:x for x in self.key_list()}
        self.mixer.__enter__()
        self.interface.__enter__()
        self.register_events()
        return self

    def __exit__(self, *args:Any) -> None:
        self.interface.__exit__(*args)
        self.mixer.__exit__(*args)

    def initialize(self) -> None:
        self.interface.initialize()
        self.gamestate.player.character.observe(self)
        assert isinstance(self.gamestate.player.character.location, core.Ship)
        pilot_view = pilot.PilotView(self.gamestate.player.character.location, self.interface)
        self.interface.open_view(pilot_view)

    def register_events(self) -> None:
        events.register_action(ui_events.DialogAction(self.interface, self.event_manager))
        events.register_action(ui_events.PlayerNotification(self.interface))
        events.register_action(ui_events.PlayerReceiveBroadcast(self.interface))
        events.register_action(ui_events.PlayerReceiveMessage(self.interface))

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

    def open_universe(self) -> None:
        universe_view = universe.UniverseView(self.gamestate, self.interface)
        self.interface.swap_view(
            universe_view,
            self.focused_view()
        )
        if self.gamestate.player.character.location is not None and self.gamestate.player.character.location.sector is not None:
            universe_view.select_sector(
                self.gamestate.player.character.location.sector,
                focus=True
            )

    def character_destroyed(self, character:core.Character) -> None:
        if character == self.gamestate.player.character:
            self.gamestate.force_pause(self)
            self.interface.log_message("you've been killed")
            # TODO: what should we do when the player's character dies?
            self.open_universe()

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

    def tick_step(self) -> None:
        self.gamestate.pause(False)
        self.gamestate.one_tick = True

    def help(self) -> None:
        command_list = {x.command: x for x in self.command_list()}

        h = object()
        view = self.focused_view()
        if view is not None:
            command_list.update({x.command: x for x in view.command_list()})

        help_lines = []
        help_lines.append("context sensitive help:")
        help_lines.append("press \":\" to enter command mode")
        help_lines.append("type a command and press <ENTER> to execute it")
        help_lines.append("available commands:")
        for k,v in command_list.items():
            help_lines.append(f'\t{k}\t{v.help}')

        self.interface.log_message("\n".join(help_lines))

    def keys(self) -> None:
        key_list = self.interface.key_list.copy()

        h = object()
        view = self.focused_view()
        if view is not None:
            key_list.update({x.key: x for x in view.key_list()})

        help_entries:Dict[str, Tuple[List[str], str]] = collections.defaultdict(lambda: ([], ""))
        for k,v in key_list.items():
            if k in KEY_DISPLAY:
                k_label = KEY_DISPLAY[k]
            elif k == curses.KEY_MOUSE or not chr(k).isprintable():
                continue
            else:
                k_label = chr(k)

            key_items, help_text = help_entries[v.help_key]
            key_items.append(k_label)
            help_entries[v.help_key] = (key_items, help_text or v.help or "NO HELP")

        help_lines = []
        help_lines.append("keys:")
        for _, (key_items, help_text) in help_entries.items():
            help_lines.append(f'\t{",".join(key_items)}\t{help_text}')

        self.interface.log_message("\n".join(help_lines))

    def bind_key(self, k:int, f:Callable[[], None], help_key:Optional[str]=None) -> interface.KeyBinding:
        try:
            h = getattr(getattr(config.Settings.help.interface, self.__class__.__name__).keys, chr(k))
        except AttributeError:
            h = "NO HELP"
        return interface.KeyBinding(k, f, h, help_key=help_key)

    def bind_command(self, command:str, f: Callable[[Sequence[str]], None], tab_completer:Optional[Callable[[str, str], str]]=None) -> interface.CommandBinding:
        try:
            h = getattr(getattr(config.Settings.help.interface, self.__class__.__name__).commands, command)
        except AttributeError:
            h = "NO HELP"
        return interface.CommandBinding(command, f, h, tab_completer)


    def key_list(self) -> Collection[interface.KeyBinding]:
        def open_command_prompt() -> None:
            command_list = {x.command:x for x in self.command_list()}
            v = self.focused_view()
            if v is not None:
                command_list.update({x.command: x for x in v.command_list()})
            self.interface.open_view(command_input.CommandInput(self.interface, commands=command_list))

        return [
            self.bind_key(ord(" "), self.gamestate.pause),
            self.bind_key(ord(">"), self.time_accel),
            self.bind_key(ord("<"), self.time_decel),
            self.bind_key(ord("."), self.tick_step),
            self.bind_key(ord(":"), open_command_prompt),
            self.bind_key(ord("?"), self.help),
        ]

    def command_list(self) -> Collection[interface.CommandBinding]:
        """ Global commands that should be valid in any context. """
        def fps(args:Sequence[str]) -> None: self.interface.show_fps = not self.interface.show_fps
        def quit(args:Sequence[str]) -> None: self.interface.gamestate.quit()
        def raise_exception(args:Sequence[str]) -> None: self.gamestate.should_raise = True
        def raise_breakpoint(args:Sequence[str]) -> None: self.gamestate.should_raise_breakpoint = True
        def colordemo(args:Sequence[str]) -> None: self.interface.open_view(ColorDemo(self.interface), deactivate_views=True)
        def attrdemo(args:Sequence[str]) -> None: self.interface.open_view(AttrDemo(self.interface), deactivate_views=True)
        def keydemo(args:Sequence[str]) -> None: self.interface.open_view(KeyDemo(self.interface), deactivate_views=True)
        def circledemo(args:Sequence[str]) -> None: self.interface.open_view(CircleDemo(self.interface), deactivate_views=True)
        def polygondemo(args:Sequence[str]) -> None: self.interface.open_view(PolygonDemo(self.interface), deactivate_views=True)
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
            if self.gamestate.player.character.location is None or self.gamestate.player.character.location.sector is None:
                raise command_input.UserError("player character not in a sector")
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
            self.open_universe()

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
            self.interface.open_view(character_view, deactivate_views=True)

        def open_station(args:Sequence[str]) -> None:
            if len(args) < 1:
                raise command_input.UserError(f'need to specify station to view')

            if self.gamestate.player.character.location is None or self.gamestate.player.character.location.sector is None:
                raise command_input.UserError("character is not in a sector")

            try:
                station_id = uuid.UUID(args[0])
                target_station = next(x for x in self.gamestate.player.character.location.sector.stations if x.entity_id == station_id)
            except:
                raise command_input.UserError(f'{args[0]} not a recognized station id')

            ship = self.gamestate.player.character.location if isinstance(self.gamestate.player.character.location, core.Ship) else next(x for x in self.gamestate.player.character.assets if isinstance(x, core.Ship))
            station_view = station.StationView(target_station, ship, self.interface)
            self.interface.open_view(station_view, deactivate_views=True)

        def open_comms(args:Sequence[str]) -> None:
            if len(args) < 1:
                raise command_input.UserError(f'need to specify message to reply to')

            try:
                msg_id = uuid.UUID(args[0])
                message = self.gamestate.player.messages[msg_id]
            except:
                raise command_input.UserError(f'{args[0]} not a recognized message id')

            if message.reply_to is None:
                raise command_input.UserError(f'{message.short_id()} has no reply to')
            if message.replied_at is not None:
                raise command_input.UserError(f'already replied to {message.short_id()}')
            assert message.reply_dialog
            speaker = message.reply_to
            dialog = message.reply_dialog
            message.replied_at = self.interface.gamestate.timestamp

            comms_view = comms.CommsView(
                events.DialogManager(dialog, self.gamestate, self.event_manager, self.interface.player.character, speaker),
                speaker,
                self.interface,
            )
            self.interface.open_view(comms_view, deactivate_views=True)

        def toggle_mouse(args:Sequence[str]) -> None:
            if self.mouse_on:
                self.interface.disable_mouse()
            else:
                self.interface.enable_mouse()
            self.mouse_on = not self.mouse_on

        def debug_collision(args:Sequence[str])->None:
            if len(self.interface.collisions) == 0:
                raise command_input.UserError("no collisions to debug")

            collision = self.interface.collisions[-1]
            if isinstance(collision[0], core.Ship):
                ship = collision[0]
            elif isinstance(collision[1], core.Ship):
                ship = collision[1]
            else:
                raise Exception("expected one of colliding objects to be a ship")

            assert ship.sector

            sector_view = sector.SectorView(ship.sector, self.interface)
            self.interface.swap_view(
                sector_view,
                self.focused_view()
            )
            sector_view.select_target(ship.entity_id, ship, focus=True)

        in_location = self.gamestate.player.character.location is not None and self.gamestate.player.character.location.sector is not None

        return [
            self.bind_command("pause", lambda x: self.gamestate.pause()),
            self.bind_command("t_accel", lambda x: self.time_accel()),
            self.bind_command("t_decel", lambda x: self.time_decel()),
            self.bind_command("fps", fps),
            self.bind_command("quit", quit),
            self.bind_command("raise", raise_exception),
            self.bind_command("breakpoint", raise_breakpoint),
            self.bind_command("colordemo", colordemo),
            self.bind_command("attrdemo", attrdemo),
            self.bind_command("keydemo", keydemo),
            self.bind_command("circledemo", circledemo),
            self.bind_command("polygondemo", polygondemo),
            self.bind_command("profile", profile),
            self.bind_command("fast", fast),
            self.bind_command("decrease_fps", decrease_fps),
            self.bind_command("increase_fps", increase_fps),
            self.bind_command("help", lambda x: self.help()),
            self.bind_command("keys", lambda x: self.keys()),
            self.bind_command("pilot", open_pilot),
            self.bind_command("sector", open_sector),
            self.bind_command("universe", open_universe),
            self.bind_command("character", open_character, util.tab_completer(map(str, self.gamestate.characters.keys()))),
            self.bind_command("comms", open_comms, util.tab_completer(map(str, self.interface.gamestate.player.messages.keys()))),
            self.bind_command("station", open_station, util.tab_completer(str(x.entity_id) for x in self.gamestate.player.character.location.sector.stations) if in_location else None),
            self.bind_command("toggle_mouse", toggle_mouse),
            self.bind_command("debug_collision", debug_collision),
        ]

