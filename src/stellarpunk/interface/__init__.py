""" Curses based interface for Stellar Punk. """

from __future__ import annotations

import logging
import curses
import curses.ascii
from curses import textpad
import enum
import array
import fcntl
import termios
import time
import collections
import math
import collections.abc
import abc
import textwrap
import uuid
import weakref
from typing import Deque, Any, Dict, Sequence, List, Callable, Optional, Mapping, Tuple, Union, MutableMapping, Set, Collection

import numpy as np
import numpy.typing as npt

from stellarpunk import util, core, config, generate
from stellarpunk.serialization import save_game
from stellarpunk.core import combat, sector_entity

class Layout(enum.Enum):
    LEFT_RIGHT = enum.auto()
    UP_DOWN = enum.auto()

class Settings:
    MIN_LOGSCREEN_WIDTH = 80
    MIN_LOGSCREEN_HEIGHT = 8
    MIN_VIEWSCREEN_WIDTH = 80
    MIN_VIEWSCREEN_HEIGHT = 8
    MIN_LR_LAYOUT_WIDTH = MIN_LOGSCREEN_WIDTH + MIN_VIEWSCREEN_WIDTH + 4
    MIN_SCREEN_WIDTH = MIN_LOGSCREEN_WIDTH + 2
    MIN_SCREEN_HEIGHT_LR = min((MIN_LOGSCREEN_HEIGHT, MIN_VIEWSCREEN_HEIGHT)) + 2 + 1
    MIN_SCREEN_HEIGHT_UD = MIN_LOGSCREEN_HEIGHT + MIN_VIEWSCREEN_HEIGHT + 4 + 1

    UMAP_SECTOR_WIDTH=34
    UMAP_SECTOR_HEIGHT=17
    UMAP_SECTOR_XSEP=0
    UMAP_SECTOR_YSEP=0

    VIEWSCREEN_BUFFER_WIDTH = 300
    VIEWSCREEN_BUFFER_HEIGHT = 100

    MAX_TIME_ACCEL = 20.0
    MIN_TIME_ACCEL = 0.025

    MAX_FRAME_HISTORY_SEC = 0.5
    MIN_FPS = 2
    MAX_FPS = 30

class Color(enum.Enum):
    ERROR = enum.auto()
    RADAR_RING = enum.auto()
    SENSOR_RING = enum.auto()
    PROFILE_RING = enum.auto()

class Icons:

    SHIP_N = "\u25B2" # black up pointing triangle
    SHIP_E = "\u25B6"
    SHIP_S = "\u25BC"
    SHIP_W = "\u25C0"
    SHIP_SE = "\u25E2"
    SHIP_SW = "\u25E3"
    SHIP_NW = "\u25E4"
    SHIP_NE = "\u25E5"

    PLANET = "\u25CB" #"○" \u25CB white circle
    STATION = "\u25A1" #"□" \u25A1 white square
    DERELICT = "\u2302" # "⌂" house symbol (kinda looks like a gravestone?)
    ASTEROID = "\u25C7" # "◇" \u25C7 white diamond
    TRAVEL_GATE = "\u25CC" # "◌" \u25CC dotted circle
    PROJECTILE = "·"

    UNKNOWN = "?"

    MULTIPLE = "*"

    EFFECT_MINING = "\u2726" # "✦" \u2726 black four pointed star
    EFFECT_TRANSFER = "\u2327" # "⌧" \u2327 X in a rectangle box
    EFFECT_UNKNOWN = "?"

    HEADING_INDICATOR = "h"
    VELOCITY_INDICATOR = "v"
    TARGET_DIRECTION_INDICATOR = "t"

    LOCATION_INDICATOR = "X"
    TARGET_INDICATOR = "X"

    STAR_LARGE = "*" #"🞣"
    STAR_SMALL = "." #"🞟"
    STAR_SMALL_ALTS = ["˙", "·", "."]

    """
    "△" \u25B3 white up pointing triangle
    "" \u25B7 white right pointing triangle
    "" \u25BD white down pointing triangle
    "" \u25C1 white left pointing triangle
    "◸" \u25F8 upper left triangle
    "" \u25F9 upper right triangle
    "" \u25FA lower left triangle
    "◿" \u25FF lower right triangle


    "◯" \u25EF large circle
    "○" \u25CB white circle
    "●" \u25Cf black circle
    "◌" \u25CC dotted circle
    "◐" \u25D0 circle with left half black
    "" \u25D1 circle with right half black
    "" \u25D2 circle with lower half black
    "" \u25D3 circle with upper half black
    "" \u25C7 white diamond
    "◆" \u25C6 black diamond
    "□" \u25A1 white square
    "■" \u25A0 black square
    "◰" \u25F0 white square with upper left quadrant
    "" \u25F1 white square with lower left quadrant
    "" \u25F2 white square with lower right quadrant
    "" \u25F3 white square with upper right quadrant

    "∵" therefore
    "∴" because
    "✨" sparkles
    "✦" \u2726 black four pointed star
    "✧" white four pointed star
    "✻" teardrop spoked asterisk
    "⌧" \u2327 X in a rectangle box
    "⌬" benzene ring
    "⬡" white hexagon
    "⬢" black hexagon
    "🞣" medium greek cross
    "🞟" medium small lozenge
    "·" middle dot
    "˙" dot above
    "." arabic dot below
    """

    COLOR_UNKNOWN = 227
    RESOURCE_COLORS = [95, 6, 143, 111, 22, 169]
    COLOR_TRAVEL_GATE = 220
    COLOR_CARGO = 243
    COLOR_HEADING_INDICATOR = 47
    COLOR_VELOCITY_INDICATOR = 47
    COLOR_TARGET_DIRECTION_INDICATOR = 47
    COLOR_LOCATION_INDICATOR = 47
    COLOR_TARGET_INDICATOR = 10
    COLOR_TARGET_IMAGE_INDICATOR = 100

    COLOR_UNIVERSE_SECTOR = 29
    COLOR_UNIVERSE_EDGE = 40

    COLOR_STAR_ALTS = [203, 209, 228, 231, 123, 52, 28]
    COLOR_STAR_SMALL = 203#245
    COLOR_STAR_LARGE = 249

    COLOR_CULTURES = [94, 102, 110, 118, 126, 134, 142, 150, 158, 166, 174, 182, 190, 198, 206, 214, 222]

    @staticmethod
    def angle_to_ship(angle:float) -> str:
        """ Returns ship icon pointing in angle (radians) direction. """

        # pos x is E
        # pos y is S (different than standard axes!
        # 0 radians -> E
        # pi/2 radians -> S
        icons = [
                Icons.SHIP_E,
                Icons.SHIP_SE,
                Icons.SHIP_S,
                Icons.SHIP_SW,
                Icons.SHIP_W,
                Icons.SHIP_NW,
                Icons.SHIP_N,
                Icons.SHIP_NE
        ]
        return icons[round(util.normalize_angle(angle)/(2*math.pi)*len(icons))%len(icons)]

    @staticmethod
    def sensor_image_icon(image:core.AbstractSensorImage) -> str:
        if not image.identified:
            return Icons.UNKNOWN

        identity = image.identity
        if issubclass(identity.object_type, core.Ship):
            icon = Icons.angle_to_ship(identity.angle)
        elif issubclass(identity.object_type, sector_entity.Station):
            icon = Icons.STATION
        elif issubclass(identity.object_type, sector_entity.Planet):
            icon = Icons.PLANET
        elif issubclass(identity.object_type, sector_entity.Asteroid):
            icon = Icons.ASTEROID
        elif issubclass(identity.object_type, sector_entity.TravelGate):
            icon = Icons.TRAVEL_GATE
        elif issubclass(identity.object_type, sector_entity.Projectile):
            icon = Icons.PROJECTILE
        else:
            icon = Icons.UNKNOWN
        return icon


    @staticmethod
    def sensor_image_attr(image:core.AbstractSensorImage) -> int:
        if not image.identified:
            return 0

        identity = image.identity
        if issubclass(identity.object_type, sector_entity.Asteroid):
            #TODO: how do we want to handle learning about these kinds of
            # specific properties of sensor images?
            entity = core.Gamestate.gamestate.entities[identity.entity_id]
            assert(isinstance(entity, sector_entity.Asteroid))
            return curses.color_pair(Icons.RESOURCE_COLORS[entity.resource]) if entity.resource < len(Icons.RESOURCE_COLORS) else 0
        elif issubclass(identity.object_type, sector_entity.TravelGate):
            return curses.color_pair(Icons.COLOR_TRAVEL_GATE)
        else:
            return 0

    @staticmethod
    def sector_entity_icon(entity:core.SectorEntity, angle:Optional[float]=None) -> str:
        if isinstance(entity, core.Ship) or isinstance(entity, combat.Missile):
            icon = Icons.angle_to_ship(angle if angle is not None else entity.angle)
        elif isinstance(entity, sector_entity.Station):
            icon = Icons.STATION
        elif isinstance(entity, sector_entity.Planet):
            icon = Icons.PLANET
        elif isinstance(entity, sector_entity.Asteroid):
            icon = Icons.ASTEROID
        elif isinstance(entity, sector_entity.TravelGate):
            icon = Icons.TRAVEL_GATE
        elif isinstance(entity, sector_entity.Projectile):
            icon = Icons.PROJECTILE
        else:
            icon = Icons.UNKNOWN
        return icon

    @staticmethod
    def sector_entity_attr(entity:core.SectorEntity) -> int:
        if isinstance(entity, sector_entity.Asteroid):
            return curses.color_pair(Icons.RESOURCE_COLORS[entity.resource]) if entity.resource < len(Icons.RESOURCE_COLORS) else 0
        elif isinstance(entity, sector_entity.TravelGate):
            return curses.color_pair(Icons.COLOR_TRAVEL_GATE)
        else:
            return 0

    @staticmethod
    def culture_attr(culture:str) -> int:
        if culture in config.Settings.generate.Universe.CULTURES:
            return curses.color_pair(Icons.COLOR_CULTURES[config.Settings.generate.Universe.CULTURES.index(culture) % len(Icons.COLOR_CULTURES)])
        else:
            return 0

class BasicCanvas:
    def __init__(self, height:int, width:int, y:int, x:int, aspect_ratio:float) -> None:
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.aspect_ratio = aspect_ratio

    def erase(self) -> None:
        pass
    def addstr(self, y:int, x:int, string:str, attr:int=0) -> None:
        pass
    def rectangle(self, uly:int, ulx:int, lry:int, lrx:int) -> None:
        pass
    def noutrefresh(self, pminrow:int, pmincol:int) -> None:
        pass

class Canvas(BasicCanvas):
    def __init__(self, window:curses.window, *args:Any) -> None:
        super().__init__(*args)
        self.window = window

    def erase(self) -> None:
        self.window.erase()

    def noutrefresh(self, pminrow:int, pmincol:int) -> None:
        self.window.noutrefresh(
                pminrow, pmincol,
                self.y, self.x,
                self.y+self.height-1,
                self.x+self.width-1
        )

    def addstr(self, y:int, x:int, string:str, attr:int=0) -> None:
        """ Draws a string to the window, clipping as necessary for offscreen. """

        if y < 0 or y >= self.height or x >= self.width:
            #TODO: do we care about embedded newlines? (some lines might be visible)
            return

        if x < 0:
            string = string[-x:]
            x = 0

        if x + len(string) > self.width:
            string = string[:self.width-x]

        self.window.addstr(y, x, string, attr)

    def rectangle(self, uly:int, ulx:int, lry:int, lrx:int) -> None:
        textpad.rectangle(self.window, uly, ulx, lry, lrx)

class PerspectiveObserver(abc.ABC):
    def perspective_updated(self, perspective:Perspective) -> None: ...

class Perspective:
    """ Represents a view on space, at some position, at some zoom """
    def __init__(self, canvas:BasicCanvas, zoom:float, min_zoom:float, max_zoom:float) -> None:
        self.canvas = canvas

        # expressed in meters per character width
        self.zoom = zoom

        # most zoomed out (largest value)
        self.min_zoom = min_zoom
        # most zoomed in (smallest value)
        self.max_zoom = max_zoom

        # min x, min y, max x, max y
        self.bbox = (0., 0., 0., 0.)
        self.meters_per_char = (0., 0.)

        self._cursor = (0., 0.)

        self.observers:weakref.WeakSet[PerspectiveObserver] = weakref.WeakSet()

    def observe(self, observer:PerspectiveObserver) -> None:
        self.observers.add(observer)

    def get_cursor(self) -> Tuple[float, float]:
        return self._cursor

    def set_cursor(self, c:Tuple[float, float]) -> None:
        self._cursor = c
        self.update_bbox()

    cursor = property(get_cursor, set_cursor)

    def move_cursor(self, direction:int) -> None:
        # ~4 characters horzontally or 2 characters vertically
        stepsize = self.meters_per_char[0]*4.

        x,y = self.cursor

        if direction == ord('w'):
            y -= stepsize
        elif direction == ord('a'):
            x -= stepsize
        elif direction == ord('s'):
            y += stepsize
        elif direction == ord('d'):
            x += stepsize
        else:
            raise ValueError(f'unknown direction {direction}')

        self.cursor = (x,y)

    def zoom_cursor(self, direction:int) -> None:
        if direction == ord('+'):
            self.zoom *= 0.9
            if self.zoom < self.max_zoom:
                self.zoom = self.max_zoom
        elif direction == ord('-'):
            self.zoom *= 1.1
            if self.zoom > self.min_zoom:
                self.zoom = self.min_zoom
        else:
            raise ValueError(f'unknown direction {direction}')

        self.update_bbox()

    def update_bbox(self) -> None:
        if self.zoom <= 0.:
            raise ValueError(f'zoom must be positive {self.zoom=}')
        self.meters_per_char = (
                self.zoom,
                self.zoom * self.canvas.aspect_ratio
        )

        vsw = self.canvas.width
        vsh = self.canvas.height

        self.bbox = (
            self.cursor[0] - (vsw/2 * self.meters_per_char[0]),
            self.cursor[1] - (vsh/2 * self.meters_per_char[1]),
            self.cursor[0] + (vsw/2 * self.meters_per_char[0]),
            self.cursor[1] + (vsh/2 * self.meters_per_char[1]),
        )

        for o in self.observers:
            o.perspective_updated(self)

    def screen_to_sector(self, screen_loc_x:int, screen_loc_y:int) -> Tuple[float, float]:
        return  util.screen_to_sector(
            screen_loc_x, screen_loc_y,
            self.bbox[0], self.bbox[1],
            self.meters_per_char[0], self.meters_per_char[1],
            self.canvas.x, self.canvas.y
        )

    def sector_to_screen(self, sector_loc_x:float, sector_loc_y:float) -> Tuple[int, int]:
        return util.sector_to_screen(
            sector_loc_x, sector_loc_y,
            self.bbox[0], self.bbox[1],
            self.meters_per_char[0], self.meters_per_char[1]
        )

CommandSig = Union[
        Callable[[Sequence[str]], None],
        Tuple[
            Callable[[Sequence[str]], None],
            Callable[[str, str], str]]
]

class CommandBinding:
    def __init__(self, command:str, f:Callable[[Sequence[str]], None], h:str, tab_completer:Optional[Callable[[str, str, int], str]]=None, help_key:Optional[str]=None) -> None:
        self.command = command
        self.f = f
        self.help = h
        self.tab_completer = tab_completer
        if help_key is None:
            help_key = str(uuid.uuid4())
        self.help_key = help_key

    def __call__(self, args:Sequence[str]) -> None:
        self.f(args)

    def complete(self, partial:str, command:str, direction:int=1) -> str:
        if self.tab_completer:
            return self.tab_completer(partial, command, direction) or " "
        else:
            return self.command

class KeyBinding:
    def __init__(self, key:int, f:Callable[[], Any], h:Optional[str], help_key:Optional[str]=None) -> None:
        self.key = key
        self.f = f
        self.help = h
        if help_key is None:
            help_key = str(uuid.uuid4())
        self.help_key = help_key

    def __call__(self) -> None:
        self.f()

class View(abc.ABC):
    def __init__(self, interface: AbstractInterface) -> None:
        self.logger = logging.getLogger(util.fullname(self))
        self.has_focus = False
        self.active = True
        self.fast_render = False
        self.interface = interface
        self.closed = False

    @property
    def viewscreen(self) -> BasicCanvas:
        return self.interface.viewscreen

    @property
    def viewscreen_dimensions(self) -> Tuple[int, int]:
        return (self.interface.viewscreen.width, self.interface.viewscreen.height)

    @property
    def viewscreen_bounds(self) -> Tuple[int, int, int, int]:
        return (0, 0, self.interface.viewscreen.width, self.interface.viewscreen.height)

    def bind_key(self, k:int, f:Callable[[], Any], help_key:Optional[str]=None) -> KeyBinding:
        h = config.get_key_help(self, help_key or chr(k))
        return KeyBinding(k, f, h, help_key=help_key)

    def bind_aliases(self, keys: Collection[int], f: Callable[[], Any], help_key:Optional[str]=None) -> Collection[KeyBinding]:
        bindings = []
        for k in keys:
            bindings.append(self.bind_key(k, f, help_key))

        return bindings

    def bind_command(self, command:str, f: Callable[[Sequence[str]], None], tab_completer:Optional[Callable[[str, str, int], str]]=None) -> CommandBinding:
        try:
            h = getattr(getattr(config.Settings.help.interface, self.__class__.__name__).commands, command)
        except AttributeError:
            h = "NO HELP"
        return CommandBinding(command, f, h, tab_completer)

    def initialize(self) -> None:
        pass

    def terminate(self) -> None:
        pass

    def focus(self) -> None:
        self.logger.debug(f'{self} got focus')
        self.has_focus = True

    def unfocus(self) -> None:
        self.has_focus = False

    def update_display(self) -> None:
        pass

    def handle_mouse(self, m_id: int, m_x: int, m_y: int, m_z: int, bstate: int) -> bool:
        return False

    def handle_input(self, key:int, dt:float) -> bool:
        key_list = {x.key: x for x in self.key_list()}
        if key in key_list:
            key_list[key]()
            return True
        else:
            return False

    def command_list(self) -> Collection[CommandBinding]:
        return []

    def key_list(self) -> Collection[KeyBinding]:
        return []

class GameView(View):
    def __init__(self, gamestate:core.Gamestate, *args:Any, **kwargs:Any):
        super().__init__(*args, **kwargs)
        self.gamestate = gamestate

class AbstractMixer:
    @property
    def sample_rate(self) -> int:
        return 44100

    def play_sample(self, sample: npt.NDArray[np.float64], callback: Optional[Callable[[], Any]] = None, loops:int=0) -> int:
        """ Plays an audio sample encoded in an np array of floats from -1 to 1

        We assume the sample is at our sampel rate. """
        return -1

    def halt_channel(self, channel:int) -> None:
        pass

class AbstractInterface(abc.ABC):
    def __init__(self, generator:generate.UniverseGenerator, mixer: AbstractMixer) -> None:
        self.logger = logging.getLogger(util.fullname(self))
        self.runtime = core.AbstractGameRuntime()
        self.gamestate:core.Gamestate = None # type: ignore
        self.generator = generator
        self.mixer = mixer
        self.views:List[View] = []

    def collision_detected(self, entity_a:core.SectorEntity, entity_b:core.SectorEntity, impulse:Tuple[float, float], ke:float) -> None:
        pass

    @abc.abstractmethod
    def tick(self, timeout:float, dt:float) -> None:
        pass

    def reinitialize_screen(self, name:str="Main Viewscreen") -> None:
        pass

    def refresh_viewscreen(self) -> None:
        pass

    def handle_input(self, key:int, dt:float) -> bool:
        self.status_message()
        if len(self.views) == 0:
            return False
        v = self.views[-1]
        assert v.has_focus
        if key == curses.KEY_MOUSE:
            try:
                m_tuple = curses.getmouse()
                m_id, m_x, m_y, m_z, bstate = m_tuple
                self.logger.debug(f'getmouse: {m_tuple}')
            except curses.error as e:
                self.logger.warning(f'error getting mouse {e}')
                return False
            return v.handle_mouse(m_id, m_x, m_y, m_z, bstate)
        else:
            return v.handle_input(key, dt)

    def open_view(self, view:View, deactivate_views:bool=False) -> None:
        self.logger.debug(f'opening view {view}')
        if len(self.views):
            self.views[-1].unfocus()
            if deactivate_views:
                for v in self.views:
                    v.active = False
        view.initialize()
        view.focus()
        self.views.append(view)

    def close_view(self, view:View, skip_focus:bool=False) -> None:
        if view.closed:
            # don't double terminate
            # this allows one code deep path to close a view while the caller
            # tries to clean itself up (e.g. CommandInput executing a command
            # that closes the active view)
            assert(view not in self.views)
            return
        self.logger.debug(f'closing view {view}')
        assert view in self.views
        self.views.remove(view)
        view.terminate()
        view.closed = True
        if not skip_focus and len(self.views) > 0:
            self.views[-1].focus()

    def close_all_views(self) -> None:
        for view in self.views.copy():
            self.close_view(view, skip_focus=True)
        assert(len(self.views) == 0)

    def swap_view(self, new_view:View, old_view:Optional[View]) -> None:
        if old_view is not None:
            self.close_view(old_view)
        self.open_view(new_view)

    def log_message(self, message:str) -> None:
        pass

    def status_message(self, message:str="", attr:int=0, cursor:bool=False) -> None:
        pass

    def get_color(self, color:Color) -> int:
        return 0

    @property
    @abc.abstractmethod
    def player(self) -> core.Player: ...

    @abc.abstractmethod
    def newpad(
        self,
        pad_lines:int, pad_cols:int,
        height:int, width:int,
        y:int, x:int,
        aspect_ratio:float) -> BasicCanvas: ...

    @property
    @abc.abstractmethod
    def viewscreen(self) -> BasicCanvas: ...

    @property
    @abc.abstractmethod
    def aspect_ratio(self) -> float: ...

class FPSCounter:
    """ Measures FPS by keeping track of frame render times. """

    def __init__(self, max_history_sec:float) -> None:
        self.frame_history:Deque[float] = collections.deque()
        self.current_fps = 0.
        self.max_history_sec = max_history_sec

    def add_frame(self, now:float) -> float:
        self.frame_history.append(now)
        return self.update_fps(now)

    def update_fps(self, now:float) -> float:
        while len(self.frame_history) > 0 and now - self.frame_history[0] > self.max_history_sec:
            self.frame_history.popleft()

        self.current_fps = len(self.frame_history) / self.max_history_sec
        return self.current_fps

    @property
    def fps(self) -> float:
        return self.current_fps

class Interface(AbstractInterface):
    def __init__(self, game_saver:save_game.GameSaver, *args:Any, **kwargs:Any):
        super().__init__(*args, **kwargs)
        #self.game_saver = game_saver
        #self.next_autosave_timestamp = 0.
        #self.next_autosave_real_timestamp = 0.
        self.stdscr:curses.window = None # type: ignore[assignment]

        self.desired_fps = Settings.MAX_FPS
        self.max_fps = self.desired_fps
        self.min_fps = Settings.MIN_FPS
        self.min_ui_timeout = 0.#gamestate.desired_dt/4

        self.fps_counter = FPSCounter(Settings.MAX_FRAME_HISTORY_SEC)
        self.fast_fps_counter = FPSCounter(Settings.MAX_FRAME_HISTORY_SEC)

        # the size of the global screen, containing other viewports
        self.screen_width = 0
        self.screen_height = 0

        # width/height of the font in pixels
        self.font_width = 0.
        self.font_height = 0.

        # viewport sizes and positions in the global screen
        # this is what's visible
        self._viewscreen:Canvas = None # type: ignore[assignment]
        self.viewscreen_width = 0
        self.viewscreen_height = 0
        self.viewscreen_x = 0
        self.viewscreen_y = 0
        self.viewscreen_bounds = (0,0,0,0)

        self.logscreen:curses.window = None # type: ignore[assignment]
        self.logscreen_width = 0
        self.logscreen_height = 0
        self.logscreen_x = 0
        self.logscreen_y = 0
        self.logscreen_padding = 1
        self.logscreen_buffer:Deque[str] = collections.deque(maxlen=100)

        # keep track of collisions we've seen
        self.collisions:List[tuple[core.SectorEntity, core.SectorEntity, Tuple[float, float], float]] = []

        # last view has focus for input handling
        self.views:List[View] = []

        self.show_fps = True

        self.one_time_step = False


        self.status_message_lifetime:float = 7.
        self.status_message_clear_time:float = np.inf

        self.key_list:Dict[int, KeyBinding] = {}

    @property
    def player(self) -> core.Player:
        return self.gamestate.player

    @property
    def viewscreen(self) -> BasicCanvas:
        return self._viewscreen

    @property
    def aspect_ratio(self) -> float:
        return self.font_height/self.font_width

    def __enter__(self) -> Interface:
        """ Does most simple interface initialization.

        In particular, this does enought initialization of the curses interface
        so that it can be cleaned up properly in __exit__. """

        self.logger.info("starting the inferface")

        self.stdscr = curses.initscr()

        curses.noecho()
        curses.cbreak()

        # make getch non-blocking, only check if there is input available
        #   this is important for running the game loop
        #self.stdscr.nodelay(True)

        # have curses interpret escape sequences for us
        self.stdscr.keypad(True)

        # we can check if there's color later
        try:
            curses.start_color()
            curses.use_default_colors()
            for i in range(0, curses.COLORS):
                curses.init_pair(i + 1, i, -1)
            curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_RED)
        except:
            pass
        self.logger.debug(f'extended color support? {curses.can_change_color()} {curses.COLORS}')

        return self

    def __exit__(self, type:Any, value:Any, traceback:Any) -> None:
        """ Cleans up the interface as much as possible so the terminal, etc.
        is returned to its original state

        This allows, e.g., us to break into the ipdb debugger on an error. """

        self.logger.info("exiting the inferface")
        # Set everything back to normal
        if self.stdscr:
            self.logger.info("resetting screen")
            self.stdscr.keypad(False)
            curses.echo()
            curses.nocbreak()
            curses.endwin()
            self.logger.info("done")

    def choose_viewport_sizes(self) -> None:
        """ Chooses viewport sizes and locations for viewscreen and the log."""

        # first get some basic terminal screen dimensions

        # calculate the font width/height in pixels
        # useful for displaying distances properly
        # from: https://stackoverflow.com/a/43947507/553580
        #struct winsize
        #{
        #   unsigned short ws_row;	/* rows, in characters */
        #   unsigned short ws_col;	/* columns, in characters */
        #   unsigned short ws_xpixel;	/* horizontal size, pixels */
        #   unsigned short ws_ypixel;	/* vertical size, pixels */
        #};
        #TODO: font-size in pixels seems pretty fragile
        # maybe we should just have a setting for it?
        buf = array.array('H', [0, 0, 0, 0])
        fcntl.ioctl(1, termios.TIOCGWINSZ, buf)
        ioctl_rows = buf[0]
        ioctl_cols = buf[1]
        ioctl_pwidth = buf[2]
        ioctl_pheight = buf[3]
        self.font_height = ioctl_pheight / ioctl_rows
        self.font_width = ioctl_pwidth / ioctl_cols

        self.screen_height, self.screen_width = self.stdscr.getmaxyx()
        self.logger.info(f'screen dimensions are {(self.screen_height, self.screen_width)} characters and {(ioctl_pheight, ioctl_pwidth)} pixels (y,x)')

        # ioctl and curses should agree, this is probably fragile
        assert ioctl_rows == self.screen_height
        assert ioctl_cols == self.screen_width

        if self.screen_width < Settings.MIN_SCREEN_WIDTH:
            raise Exception(f'minimum screen width is {Settings.MIN_SCREEN_WIDTH} columns (currently {self.screen_width})')

        # decide if we're doing left/right or up/down layout of view screen and
        # log screen
        if self.screen_width >= Settings.MIN_LR_LAYOUT_WIDTH:
            if self.screen_height < Settings.MIN_SCREEN_HEIGHT_LR:
                raise Exception(f'minimum screen height at this width is {Settings.MIN_SCREEN_HEIGHT_LR} columns (currently {self.screen_height})')

            # reserve y, x space:
            # viewscreen border = 1 char top/bottom, 1 char left/right
            # logscreen border = 1 char top/bottom, 1 char left/right
            # status line = 1 veritcal line
            self.layout = Layout.LEFT_RIGHT

            self.logscreen_width = Settings.MIN_LOGSCREEN_WIDTH
            self.logscreen_height = self.screen_height - 2 - 1

            self.viewscreen_width = self.screen_width - self.logscreen_width - 4
            self.viewscreen_height = self.screen_height - 2 - 1
            self.viewscreen_x = 1
            self.viewscreen_y = 1

            self.logscreen_x = self.viewscreen_width + 3
            self.logscreen_y = 1
        else:
            if self.screen_height < Settings.MIN_SCREEN_HEIGHT_UD:
                raise Exception(f'minimum screen height at this width is {Settings.MIN_SCREEN_HEIGHT_UD} columns (currently {self.screen_height})')

            self.layout = Layout.UP_DOWN

            self.logscreen_width = self.screen_width - 2
            self.logscreen_height = Settings.MIN_LOGSCREEN_HEIGHT

            self.viewscreen_width = self.screen_width - 2
            self.viewscreen_height = self.screen_height - self.logscreen_height - 4 - 1
            self.viewscreen_x = 1
            self.viewscreen_y = 1

            self.logscreen_x = 1
            self.logscreen_y = self.viewscreen_height + 3

        self.viewscreen_bounds = (0, 0, self.viewscreen_width, self.viewscreen_height)

        self.logger.info(f'chose layout {self.layout}')
        self.logger.info(f'viewscreen (x,y) {(self.viewscreen_y, self.viewscreen_x)} (h,w): {(self.viewscreen_height, self.viewscreen_width)}')
        self.logger.info(f'logscreen (x,y) {(self.logscreen_y, self.logscreen_x)} (h,w): {(self.logscreen_height, self.logscreen_width)}')

    def reinitialize_screen(self, name:str="Main Viewscreen") -> None:
        self.logger.debug(f'reinitializing the screen')
        self.choose_viewport_sizes()
        textpad.rectangle(
                self.stdscr,
                self.viewscreen_y-1,
                self.viewscreen_x-1,
                self.viewscreen_y+self.viewscreen_height,
                self.viewscreen_x+self.viewscreen_width
        )
        self.stdscr.addstr(self.viewscreen_y-1, self.viewscreen_x+1, " "+name+" ")
        textpad.rectangle(
                self.stdscr,
                self.logscreen_y-1,
                self.logscreen_x-1,
                self.logscreen_y+self.logscreen_height,
                self.logscreen_x+self.logscreen_width
        )
        self.stdscr.addstr(self.logscreen_y-1, self.logscreen_x+1, " Message Log ")

        # make the viewscreen 1 extra row to avoid curses error when writing to
        # the bottom right character
        self._viewscreen = Canvas(curses.newpad(self.viewscreen_height+1, self.viewscreen_width), self.viewscreen_height, self.viewscreen_width, self.viewscreen_y, self.viewscreen_x, self.aspect_ratio)
        self.logscreen = curses.newpad(self.logscreen_height+1, self.logscreen_width)
        self.logscreen.scrollok(True)
        for message in self.logscreen_buffer:
            self.draw_log_message(message)

        self.stdscr.noutrefresh()
        self.refresh_viewscreen()
        self.refresh_logscreen()
        #self.stdscr.addstr(self.screen_height-1, 0, " "*(self.screen_width-1))

    def initialize(self) -> None:
        self.enable_mouse()
        # enable terminal reporting of mouse position
        # as per https://stackoverflow.com/a/64809709/553580
        print('\033[?1003h')
        # setting mouseinterval to 0 means no lag on mouse events, but means we
        # will not get click vs mousedown vs mouseup events handled by curses
        curses.mouseinterval(0)
        curses.set_escdelay(1)
        self.stdscr.timeout(0)

        curses.nonl()
        curses.curs_set(0)

        self.reinitialize_screen()

    def enable_mouse(self) -> None:
        curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)

    def disable_mouse(self) -> None:
        curses.mousemask(0)

    def get_color(self, color:Color) -> int:
        if color == Color.ERROR:
            return curses.color_pair(1)
        elif color == Color.RADAR_RING:
            return curses.color_pair(29)
        elif color == Color.SENSOR_RING:
            return curses.color_pair(34)
        elif color == Color.PROFILE_RING:
            return curses.color_pair(198)
        else:
            raise ValueError(f'unknown color {color}')

    def newpad(
            self,
            pad_lines:int, pad_cols:int,
            height:int, width:int,
            y:int, x:int,
            aspect_ratio:float
    ) -> BasicCanvas:
        return Canvas(
                curses.newpad(pad_lines, pad_cols),
                height, width,
                y, x,
                aspect_ratio,
            )

    def refresh_viewscreen(self) -> None:
        self._viewscreen.noutrefresh(0, 0)

    def refresh_logscreen(self) -> None:
        self.logscreen.noutrefresh(
                0, 0,
                self.logscreen_y, self.logscreen_x,
                self.logscreen_y+self.logscreen_height-1,
                self.logscreen_x+self.logscreen_width-1
        )

    def collision_detected(self, entity_a:core.SectorEntity, entity_b:core.SectorEntity, impulse:Tuple[float, float], ke:float) -> None:
        #TODO: how does this ever get cleared!?
        self.collisions.append((
            entity_a,
            entity_b,
            impulse,
            ke,
        ))
        self.status_message(
                f'collision detected {entity_a.address_str()}, {entity_b.address_str()}',
                attr=self.get_color(Color.ERROR)
        )

    def draw_log_message(self, message:str) -> None:
        for message_line in message.split("\n"):
            if message_line == "":
                self.logscreen.addstr(self.logscreen_height, self.logscreen_padding, "\n")
            else:
                lines = textwrap.wrap(message_line, width=self.logscreen_width-self.logscreen_padding*2, subsequent_indent="  ")
                for line in lines:
                    self.logscreen.addstr(self.logscreen_height,self.logscreen_padding, line+"\n")
        self.logscreen.addstr(self.logscreen_height,self.logscreen_padding, "\n")

    def log_message(self, message:str) -> None:
        """ Adds a message to the log, scrolling everything else up. """
        self.logscreen_buffer.append(message)
        self.draw_log_message(message)
        self.refresh_logscreen()

    def status_message(self, message:str="", attr:int=0, cursor:bool=False) -> None:
        """ Adds a status message. """
        self.stdscr.addstr(self.screen_height-1, 0, " "*(self.screen_width-1))
        self.stdscr.addstr(self.screen_height-1, 0, message, attr)

        if cursor:
            self.stdscr.addstr(self.screen_height-1, len(message), " ", curses.A_REVERSE)

        if message:
            self.status_message_clear_time = time.time() + self.status_message_lifetime
        else:
            self.status_message_clear_time = np.inf

    def diagnostics_message(self, message:str, attr:int=0) -> None:

        self.stdscr.addstr(self.screen_height-1, self.screen_width-len(message)-1, message, attr)

    def show_diagnostics(self) -> None:
        if self.runtime.game_running():
            ticks = self.gamestate.ticks
            timestamp = self.gamestate.timestamp
            paused = self.gamestate.paused
        else:
            ticks = 0
            timestamp = 0
            paused = False

        attr = 0
        diagnostics = []
        if self.show_fps:
            diagnostics.append(f'{ticks} ({self.runtime.get_missed_ticks()}) {timestamp:.2f} ({self.runtime.get_ticktime()*1000:>5.2f}ms +{(self.runtime.get_desired_dt() - self.runtime.get_ticktime())*1000:>5.2f}ms) {self.fps_counter.fps:>2.0f}fps')
        if paused:
            attr |= curses.color_pair(1)
            diagnostics.append("PAUSED")

        self.diagnostics_message(" ".join(diagnostics), attr)

    def show_date(self) -> None:
        date_string = ' '
        date_string += self.gamestate.current_time().strftime("%c")
        time_accel_rate, fast_mode = self.runtime.get_time_acceleration()
        if not util.isclose(time_accel_rate, 1.0) or fast_mode:
            if fast_mode:
                date_string += f' ( fast)'
            else:
                date_string += f' ({time_accel_rate:>5.2f})'
        else:
            date_string += f' ( 1.00)'
        date_string += ' '
        # if the date_string changes length it might mess up the frame which is
        # only drawn when the window is reinitialized
        assert len(date_string) == 1+4+4+3+9+5+7+1
        self.stdscr.addstr(
            self.viewscreen_y-1,
            self.viewscreen_x+self.viewscreen_width-len(date_string)-2,
            date_string
        )

    def show_cash(self) -> None:
        #TODO: what if the player is currently unattached to a character?
        assert(self.player.character)
        balance_string = f' ${self.player.character.balance:.2f} '
        self.stdscr.addstr(
            self.viewscreen.y+self.viewscreen.height,
            self.viewscreen_x+self.viewscreen_width-len(balance_string)-2,
            balance_string
        )

    def handle_input(self, key:int, dt:float) -> bool:
        if super().handle_input(key, dt):
            return True
        elif key in self.key_list:
            self.key_list[key]()
            return True
        else:
            return False

    """
    def _tick_autosave(self) -> None:
        if self.game_saver is None:
            return
        if self.gamestate is None:
            return
        if self.gamestate.is_force_paused():
            # no autosave whlie force paused, we'll pick it back up when it
            # gets force paused (even if it stays paused)
            return

        #TODO: do I want this to be game seconds or wall seconds?
        #TODO: what about time acceleration?
        #TODO: what about doing a ton of stuff while paused?
        if self.gamestate.timestamp > self.next_autosave_timestamp and time.time() > self.next_autosave_real_timestamp:
            self.log_message('saving game...')
            start_time = time.perf_counter()
            self.game_saver.autosave(self.gamestate)
            end_time = time.perf_counter()
            self.log_message(f'game saved in {end_time-start_time:.2f}s.')
            self.set_next_autosave_ts()

    def set_next_autosave_ts(self) -> None:
        self.next_autosave_timestamp = self.gamestate.timestamp + config.Settings.AUTOSAVE_PERIOD_SEC
        self.next_autosave_real_timestamp = time.time() + config.Settings.AUTOSAVE_PERIOD_SEC
    """

    def tick(self, timeout:float, dt:float) -> None:
        start_time = time.perf_counter()
        self.fps_counter.update_fps(start_time)
        # only render a frame if there's enough time and it won't exceed the fps cap
        if (timeout > self.min_ui_timeout and self.fps_counter.fps < self.max_fps) or (self.fps_counter.fps < self.min_fps):
            # update the display (i.e. draw_universe_map, draw_sector_map, draw_pilot_map)
            self.fps_counter.add_frame(start_time)
            self.fast_fps_counter.add_frame(start_time)

            if self.one_time_step:
                self.gamestate.paused = True
                self.one_time_step = False

            if time.time() > self.status_message_clear_time:
                self.status_message()

            for view in self.views:
                if view.active:
                    view.update_display()
            if self.runtime.game_running():
                self.show_date()
                self.show_cash()
            self.show_diagnostics()
            self.stdscr.noutrefresh()

            curses.doupdate()
        elif self.fast_fps_counter.fps < self.desired_fps:
            # only update a couple of fast things
            for view in self.views:
                if view.active and view.fast_render:
                    view.update_display()
            self.show_diagnostics()
            self.stdscr.noutrefresh()
            curses.doupdate()

        #TODO: this can block in the case of mouse clicks
        #TODO: see note above about setting mouseinterval to 0 which fixes this?
        # maybe we should offload getch to another thread that can always block
        # and read stuff from it from a queue? it's not clear about
        # threadsafety of getch vs getmouse tho and other curses stuff

        # process input according to what has focus (i.e. umap, smap, pilot, command)
        #self.stdscr.timeout(int(timeout*100))

        key = self.stdscr.getch()
        while key != -1:
            if key < 0:
                return
            elif key == curses.KEY_RESIZE:
                for view in self.views:
                    view.initialize()
                return

            #self.logger.debug(f'keypress {key}')
            self.handle_input(key, dt)
            key = self.stdscr.getch()
