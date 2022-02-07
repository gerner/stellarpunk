""" Curses based interface for Stellar Punk. """

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
import cProfile
import pstats
import abc

from stellarpunk import util, generate, core

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

    VIEWSCREEN_BUFFER_WIDTH = 500
    VIEWSCREEN_BUFFER_HEIGHT = 500

class Color(enum.Enum):
    ERROR = enum.auto()


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

    MULTIPLE = "*"

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
    """

    RESOURCE_COLORS = [95, 6, 143, 111, 22, 169]

    @staticmethod
    def angle_to_ship(angle):
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
    def sector_entity_icon(entity):
        if isinstance(entity, core.Ship):
            icon = Icons.angle_to_ship(entity.angle)
        elif isinstance(entity, core.Station):
            icon = Icons.STATION
        elif isinstance(entity, core.Planet):
            icon = Icons.PLANET
        elif isinstance(entity, core.Asteroid):
            icon = Icons.ASTEROID
        else:
            icon = "?"
        return icon

    @staticmethod
    def sector_entity_attr(entity):
        if isinstance(entity, core.Asteroid):
            return curses.color_pair(Icons.RESOURCE_COLORS[entity.resource]) if entity.resource < len(Icons.RESOURCE_COLORS) else 0
        else:
            return 0

class View:
    def __init__(self, interface):
        self.logger = logging.getLogger(util.fullname(self))
        self.has_focus = False
        self.interface = interface

    def initialize(self):
        pass

    def focus(self):
        self.logger.debug(f'{self} got focus')
        self.has_focus = True

    def unfocus(self):
        self.has_focus = False

    def update_display(self):
        pass

    def handle_input(self, key):
        pass

class ColorDemo(View):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def focus(self):
        super().focus()
        self.interface.camera_x = 0
        self.interface.camera_y = 0

    def update_display(self):
        self.interface.camera_x = 0
        self.interface.camera_y = 0
        self.interface.viewscreen.erase()
        self.interface.viewscreen.addstr(0, 35, "COLOR DEMO");

        for c in range(256):
            self.interface.viewscreen.addstr(int(c/8)+1, c%8*9,f'...{c:03}...', curses.color_pair(c));
        self.interface.viewscreen.addstr(34, 1, "Press any key to continue")
        self.interface.refresh_viewscreen()

    def handle_input(self, key):
        return key == -1

class CommandInput(View):
    """ Command mode: typing in a command to execute. """

    class UserError(Exception):
        pass

    def __init__(self, *args, commands=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.commands = commands or {}
        self.partial = ""
        self.command = ""

    def _command_name(self):
        i = self.command.strip(" ").find(" ")
        if i < 0:
            return self.command.strip()
        else:
            return self.command.strip()[0:i]

    def _command_args(self):
        return self.command.strip().split()[1:]

    def initialize(self):
        for c, f in self.interface.command_list().items():
            if c not in self.commands:
                self.commands[c] = f
        self.logger.info("entering command mode")

    def update_display(self):
        self.interface.status_message(f':{self.command}')

    def handle_input(self, key):
        if key in (ord('\n'), ord('\r')):
            self.logger.debug(f'read command {self.command}')
            self.interface.status_message()
            # process the command
            command_name = self._command_name()
            if command_name in self.commands:
                self.logger.info(f'executing {self.command}')
                try:
                    if isinstance(self.commands[command_name], collections.abc.Sequence):
                        self.commands[command_name][0](self._command_args())
                    else:
                        self.commands[command_name](self._command_args())
                except CommandInput.UserError as e:
                    self.logger.info(f'user error executing {self.command}: {e}')
                    self.interface.status_message(f'error in "{self.command}" {str(e)}', curses.color_pair(1))
            else:
                self.interface.status_message(f'unknown command "{self.command}" enter command mode with ":" and then "quit" to quit.', curses.color_pair(1))
            return False
        elif chr(key).isprintable():
            self.command += chr(key)
            self.partial = self.command
        elif key == curses.ascii.BS:
            self.command = self.command[:-1]
            self.partial = self.command
        elif key == curses.ascii.TAB:
            if " " not in self.command:
                self.command = util.tab_complete(self.partial, self.command, sorted(self.commands.keys())) or self.partial
            elif self._command_name() in self.commands and isinstance(self.commands[self._command_name()], collections.abc.Sequence):
                self.command = self.commands[self._command_name()][1](self.partial, self.command) or ""
        elif key == curses.ascii.ESC:
            self.interface.status_message()
            return False
        elif key == curses.KEY_RESIZE:
            self.interface.reinitialize_screen(name="???")

        return True

class GenerationUI(generate.GenerationListener):
    """ Handles the UI during universe generation. """

    def __init__(self, ui):
        self.ui = ui

    def production_chain_complete(self, production_chain):
        pass

    def sectors_complete(self, sectors):
        #self.ui.universe_mode()
        pass

class AbstractInterface(abc.ABC):
    @abc.abstractmethod
    def collision_detected(self, entity_a:core.SectorEntity, entity_b:core.SectorEntity, impulse:float, ke:float) -> None:
        pass

    @abc.abstractmethod
    def tick(self, timeout:float) -> None:
        pass

class Interface(AbstractInterface):
    def __init__(self, gamestate, generator):
        self.stdscr = None
        self.logger = logging.getLogger(util.fullname(self))

        # the size of the global screen, containing other viewports
        self.screen_width = 0
        self.screen_height = 0

        # width/height of the font in pixels
        self.font_width = 0
        self.font_height = 0

        # viewport sizes and positions in the global screen
        # this is what's visible
        self.viewscreen = None
        self.viewscreen_width = 0
        self.viewscreen_height = 0
        self.viewscreen_x = 0
        self.viewscreen_y = 0
        self.viewscreen_bounds = (0,0,0,0)

        self.logscreen = None
        self.logscreen_width = 0
        self.logscreen_height = 0
        self.logscreen_x = 0
        self.logscreen_y = 0

        # upper left corner of the camera view inside the viewscreen
        self.camera_x = 0
        self.camera_y = 0

        self.gamestate = gamestate

        self.generator = generator

        # last view has focus for input handling
        self.views = []

        # list of frame times
        self.frame_history = collections.deque()
        # max frame history to keep in seconds
        self.max_frame_history = 1

        self.show_fps = True

        self.step = False

        self.profiler = None

    def __enter__(self):
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

    def __exit__(self, type, value, traceback):
        """ Cleans up the interface as much as possible so the terminal, etc.
        is returned to its original state

        This allows, e.g., us to break into the ipdb debugger on an error. """

        self.logger.info("exiting the inferface")
        # Set everything back to normal
        if self.stdscr:
            self.logger.info("resetting screen")
            self.stdscr.keypad(0)
            curses.echo()
            curses.nocbreak()
            curses.endwin()
            self.logger.info("done")

    def fps(self):
        return len(self.frame_history) / self.max_frame_history

    def choose_viewport_sizes(self):
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

    def reinitialize_screen(self, name="Main Viewscreen"):
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

        self.viewscreen = curses.newpad(Settings.VIEWSCREEN_BUFFER_HEIGHT, Settings.VIEWSCREEN_BUFFER_WIDTH)
        self.logscreen = curses.newpad(self.logscreen_height+1, self.logscreen_width)
        self.logscreen.scrollok(True)

        self.stdscr.noutrefresh()
        self.refresh_viewscreen()
        self.refresh_logscreen()
        #self.stdscr.addstr(self.screen_height-1, 0, " "*(self.screen_width-1))

    def initialize(self):
        curses.mousemask(curses.ALL_MOUSE_EVENTS)
        #curses.mouseinterval(200)
        curses.set_escdelay(1)
        self.stdscr.timeout(0)

        curses.nonl()

        self.reinitialize_screen()

    def get_color(self, color:Color) -> int:
        if color == Color.ERROR:
            return curses.color_pair(1)
        else:
            raise ValueError(f'unknown color {color}')

    def refresh_viewscreen(self):
        self.viewscreen.noutrefresh(
                self.camera_y, self.camera_x,
                self.viewscreen_y, self.viewscreen_x,
                self.viewscreen_y+self.viewscreen_height-1,
                self.viewscreen_x+self.viewscreen_width-1
        )

    def refresh_logscreen(self):
        self.logscreen.noutrefresh(
                0, 0,
                self.logscreen_y, self.logscreen_x,
                self.logscreen_y+self.logscreen_height-1,
                self.logscreen_x+self.logscreen_width-1
        )

    def collision_detected(self, entity_a:core.SectorEntity, entity_b:core.SectorEntity, impulse:float, ke:float) -> None:
        self.status_message(
                f'collision detected {entity_a.address_str()}, {entity_b.address_str()}',
                attr=self.get_color(Color.ERROR)
        )

    def log_message(self, message:str) -> None:
        """ Adds a message to the log, scrolling everything else up. """
        self.logscreen.addstr(self.logscreen_height,0, message+"\n")
        self.refresh_logscreen()

    def status_message(self, message:str="", attr:int=0) -> None:
        """ Adds a status message. """
        self.stdscr.addstr(self.screen_height-1, 0, " "*(self.screen_width-1))
        self.stdscr.addstr(self.screen_height-1, 0, message, attr)

    def diagnostics_message(self, message:str, attr:int=0) -> None:

        self.stdscr.addstr(self.screen_height-1, self.screen_width-len(message)-1, message, attr)

    def show_diagnostics(self) -> None:
        attr = 0
        diagnostics = []
        if self.show_fps:
            diagnostics.append(f'{self.gamestate.ticks} ({self.gamestate.missed_ticks}) {self.gamestate.timestamp:.2f} ({self.gamestate.ticktime*1000:>5.2f}ms) {self.fps():>2.0f}fps')
        if self.gamestate.paused:
            attr |= curses.color_pair(1)
            diagnostics.append("PAUSED")

        self.diagnostics_message(" ".join(diagnostics), attr)

    def show_date(self) -> None:
        date_string = self.gamestate.current_time().strftime("%c")
        date_string = " "+date_string+" "
        self.stdscr.addstr(
                self.viewscreen_y-1,
                self.viewscreen_x+self.viewscreen_width-len(date_string)-2,
                date_string
        )

    def generation_listener(self) -> generate.GenerationListener:
        return GenerationUI(self)

    def open_view(self, view):
        self.logger.debug(f'opening view {view}')
        if len(self.views):
            self.views[-1].unfocus()
        view.initialize()
        view.focus()
        self.views.append(view)

    def command_list(self):
        def pause(args): self.gamestate.paused = not self.gamestate.paused
        def fps(args): self.show_fps = not self.show_fps
        def quit(args): self.gamestate.quit()
        def raise_exception(args): self.gamestate.should_raise = True
        def colordemo(args): self.open_view(ColorDemo(self))
        def profile(args):
            if self.profiler:
                self.profiler.disable()
                pstats.Stats(self.profiler).dump_stats("/tmp/profile.prof")
            else:
                self.profiler = cProfile.Profile()
                self.profiler.enable()
        return {
                "pause": pause,
                "fps": fps,
                "quit": quit,
                "raise": raise_exception,
                "colordemo": colordemo,
                "profile": profile,
        }

    def tick(self, timeout:float) -> None:
        # update the display (i.e. draw_universe_map, draw_sector_map, draw_pilot_map)
        start_time = time.perf_counter()
        self.frame_history.append(start_time)
        while self.frame_history[0] < start_time - self.max_frame_history:
            self.frame_history.popleft()

        if self.step:
            self.gamestate.paused = True
            self.step = False

        for view in self.views:
            view.update_display()
        self.show_date()
        self.show_diagnostics()
        self.stdscr.noutrefresh()

        curses.doupdate()


        #TODO: this can block in the case of mouse clicks
        # maybe we should offload getch to another thread that can always block
        # and read stuff from it from a queue? it's not clear about
        # threadsafety of getch vs getmouse tho and other curses stuff

        # process input according to what has focus (i.e. umap, smap, pilot, command)
        #self.stdscr.timeout(int(timeout*100))
        key = self.stdscr.getch()

        if key == -1:
            return
        elif key == curses.KEY_RESIZE:
            for view in self.views:
                view.initialize()
        elif key == ord("."):
            self.gamestate.paused = False
            self.step = True
        elif key >= 0:
            self.status_message()
            v = self.views[-1]
            if not v.handle_input(key):
                self.logger.debug(f'closing view {v}')
                self.views.remove(v)
                self.views[-1].focus()
