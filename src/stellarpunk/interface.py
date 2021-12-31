""" Curses based interface for Stellar Punk. """

import logging
import curses
from curses import textpad
import enum

from stellarpunk import util, generate


class Layout(enum.Enum):
    LEFT_RIGHT = enum.auto()
    UP_DOWN = enum.auto()

class Mode(enum.Enum):
    UNIVERSE = enum.auto()
    COMMAND = enum.auto()
    SECTOR = enum.auto()
    PILOT = enum.auto()

class Settings:
    MIN_LOGSCREEN_WIDTH = 80
    MIN_LOGSCREEN_HEIGHT = 8
    MIN_VIEWSCREEN_WIDTH = 80
    MIN_VIEWSCREEN_HEIGHT = 8
    MIN_LR_LAYOUT_WIDTH = MIN_LOGSCREEN_WIDTH + MIN_VIEWSCREEN_WIDTH + 4
    MIN_SCREEN_WIDTH = MIN_LOGSCREEN_WIDTH + 2
    MIN_SCREEN_HEIGHT_LR = min((MIN_LOGSCREEN_HEIGHT, MIN_VIEWSCREEN_HEIGHT)) + 2 + 1
    MIN_SCREEN_HEIGHT_UD = MIN_LOGSCREEN_HEIGHT + MIN_VIEWSCREEN_HEIGHT + 4 + 1

    VIEWSCREEN_BUFFER_WIDTH = 1000
    VIEWSCREEN_BUFFER_HEIGHT = 1000

    UMAP_SECTOR_WIDTH=34
    UMAP_SECTOR_HEIGHT=17
    UMAP_SECTOR_XSEP=0
    UMAP_SECTOR_YSEP=0

class Icons:

    SHIP_N = "\u25B2" # black up pointing triangle
    SHIP_E = "\u25B6"
    SHIP_S = "\u25BC"
    SHIP_W = "\u25C0"
    SHIP_SE = "\u25E2"
    SHIP_SW = "\u25E3"
    SHIP_NW = "\u25E4"
    SHIP_NE = "\u25E5"

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
    "" \u25C6 white diamond
    "◆" \u25C6 black diamond
    "□" \u25A1 white square
    "■" \u25A0 black square
    "◰" \u25F0 white square with upper left quadrant
    "" \u25F1 white square with lower left quadrant
    "" \u25F2 white square with lower right quadrant
    "" \u25F3 white square with upper right quadrant
    """

class GenerationUI(generate.GenerationListener):
    """ Handles the UI during universe generation. """

    def __init__(self, ui):
        self.ui = ui

    def production_chain_complete(self, production_chain):
        pass

    def sectors_complete(self, sectors):
        self.ui.universe_mode()


class Interface:
    def __init__(self, gamestate):
        self.stdscr = None
        self.logger = logging.getLogger(util.fullname(self))

        # the size of the global screen, containing other viewports
        self.screen_width = 0
        self.screen_height = 0

        # viewport sizes and positions in the global screen
        # this is what's visible
        self.viewscreen = None
        self.viewscreen_width = 0
        self.viewscreen_height = 0
        self.viewscreen_x = 0
        self.viewscreen_y = 0

        self.logscreen = None
        self.logscreen_width = 0
        self.logscreen_height = 0
        self.logscreen_x = 0
        self.logscreen_y = 0

        # upper left corner of the camera view inside the viewscreen
        self.camera_x = 0
        self.camera_y = 0

        # position of the viewscreen cursor
        self.ucursor_x = 0
        self.ucursor_y = 0
        self.sector_maxx = 0
        self.sector_maxy = 0

        self.gamestate = gamestate

        self.current_mode = None

    def __enter__(self):
        """ Does most simple interface initialization.

        In particular, this does enought initialization of the curses interface
        so that it can be cleaned up properly in __exit__. """

        self.logger.info("starting the inferface")
        self.stdscr = curses.initscr()

        curses.noecho()
        curses.cbreak()
        self.stdscr.keypad(0)

        # we can check if there's color later
        try:
            curses.start_color()
            curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_RED)
        except:
            pass

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

    def choose_viewport_sizes(self):
        """ Chooses viewport sizes and locations for viewscreen and the log."""

        self.screen_height, self.screen_width = self.stdscr.getmaxyx()
        self.logger.info(f'screen dimensions are {(self.screen_height, self.screen_width)}')

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

        self.logger.info(f'chose layout {self.layout}')
        self.logger.info(f'viewscreen (x,y) {(self.viewscreen_y, self.viewscreen_x)} (h,w): {(self.viewscreen_height, self.viewscreen_width)}')
        self.logger.info(f'logscreen (x,y) {(self.logscreen_y, self.logscreen_x)} (h,w): {(self.logscreen_height, self.logscreen_width)}')

    def initialize(self):

        self.choose_viewport_sizes()
        textpad.rectangle(
                self.stdscr,
                self.viewscreen_y-1,
                self.viewscreen_x-1,
                self.viewscreen_y+self.viewscreen_height,
                self.viewscreen_x+self.viewscreen_width
        )
        self.stdscr.addstr(self.viewscreen_y-1, self.viewscreen_x+1, " Main Viewscreen ")
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

        self.viewscreen.addstr(0,0, "V"+("v"*(self.viewscreen_width-2))+"V")
        self.viewscreen.addstr(1,0, "this is the viewscreen wtf")
        self.viewscreen.addstr(self.viewscreen_height-1,0, "V"+("v"*(self.viewscreen_width-2))+"V")

        self.logscreen.addstr(0,0, "L"+("l"*(self.logscreen_width-2))+"L")
        self.logscreen.addstr(1,0, "this is the logscreen")
        self.logscreen.addstr(self.logscreen_height-1,0, "L"+("l"*(self.logscreen_width-2))+"L")

        self.log_message("first line")
        for _ in range(10):
            self.log_message("more text")
        self.log_message("last line")


        self.stdscr.noutrefresh()
        self.refresh_viewscreen()
        self.logscreen.noutrefresh(
                0, 0,
                self.logscreen_y, self.logscreen_x,
                self.logscreen_y+self.logscreen_height-1,
                self.logscreen_x+self.logscreen_width-1
        )

        curses.doupdate()

        self.logger.info("waiting for final input")
        self.get_any_key()

    def refresh_viewscreen(self):
        self.viewscreen.noutrefresh(
                self.camera_y, self.camera_x,
                self.viewscreen_y, self.viewscreen_x,
                self.viewscreen_y+self.viewscreen_height-1,
                self.viewscreen_x+self.viewscreen_width-1
        )

    def log_message(self, message):
        """ Adds a message to the log, scrolling everything else up. """
        self.logscreen.addstr(self.logscreen_height,0, message+"\n")

    def status_message(self, message="", attr=0):
        """ Adds a status message. """
        self.stdscr.addstr(self.screen_height-1, 0, " "*(self.screen_width-1))
        self.stdscr.addstr(self.screen_height-1, 0, message, attr)

    def get_any_key(self):
        self.status_message("Press any key to continue")
        k = self.stdscr.getkey()
        self.logger.info(f'get any key "{k}"')

    def draw_umap_sector(self, y, x, sector):
        """ Draws a single sector to viewscreen starting at position (y,x) """
        textpad.rectangle(self.viewscreen, y, x, y+Settings.UMAP_SECTOR_HEIGHT-1, x+Settings.UMAP_SECTOR_WIDTH-1)

        if (self.ucursor_x, self.ucursor_y) == (sector.x, sector.y):
            self.viewscreen.addstr(y+1,x+1, sector.short_id(), curses.A_STANDOUT)
        else:
            self.viewscreen.addstr(y+1,x+1, sector.short_id())


    def draw_universe_map(self, sectors):
        """ Draws a map of all sectors. """

        self.viewscreen.erase()
        self.sector_maxx = -1
        self.sector_maxy = -1
        for (x,y), sector in sectors.items():
            self.sector_maxx = max(self.sector_maxx, x)
            self.sector_maxy = max(self.sector_maxy, y)
            # claculate screen_y and screen_x from x,y
            screen_x = x*(Settings.UMAP_SECTOR_WIDTH+Settings.UMAP_SECTOR_XSEP)
            screen_y = y*(Settings.UMAP_SECTOR_HEIGHT+Settings.UMAP_SECTOR_YSEP)

            self.draw_umap_sector(screen_y, screen_x, sector)

        view_y = self.ucursor_y*(Settings.UMAP_SECTOR_HEIGHT+Settings.UMAP_SECTOR_YSEP)
        view_x = self.ucursor_x*(Settings.UMAP_SECTOR_WIDTH+Settings.UMAP_SECTOR_XSEP)

        self.viewscreen.move(view_y+1, view_x+1)

        if view_x < self.camera_x:
            self.camera_x = view_x
        elif view_x > self.camera_x + self.viewscreen_width - Settings.UMAP_SECTOR_WIDTH:
            self.camera_x = view_x - self.viewscreen_width + Settings.UMAP_SECTOR_WIDTH
        if view_y < self.camera_y:
            self.camera_y = view_y
        elif view_y > self.camera_y + self.viewscreen_height - Settings.UMAP_SECTOR_HEIGHT:
            self.camera_y = view_y - self.viewscreen_height + Settings.UMAP_SECTOR_HEIGHT
        self.refresh_viewscreen()

    def draw_sector_map(self, sector):
        """ Draws a map of a sector. """

        # sectors laid out in x,y grid
        # we'll lay them out as small boxes with name and overview stats
        pass

    def generation_listener(self):
        return GenerationUI(self)

    def move_ucursor(self, direction):
        old_x = self.ucursor_x
        old_y = self.ucursor_y

        if direction == ord('w'):
            self.ucursor_y -= 1
        elif direction == ord('a'):
            self.ucursor_x -= 1
        elif direction == ord('s'):
            self.ucursor_y += 1
        elif direction == ord('d'):
            self.ucursor_x += 1
        else:
            raise ValueError(f'unknown direction {direction}')

        if self.ucursor_x < 0:
            self.ucursor_x = 0
            self.status_message("no more sectors to the left", curses.color_pair(1))
        elif self.ucursor_x > self.sector_maxx:
            self.ucursor_x = self.sector_maxx
            self.status_message("no more sectors to the right", curses.color_pair(1))

        if self.ucursor_y < 0:
            self.ucursor_y = 0
            self.status_message("no more sectors upward", curses.color_pair(1))
        elif self.ucursor_y > self.sector_maxy:
            self.ucursor_y = self.sector_maxy
            self.status_message("no more sectors downward", curses.color_pair(1))

    def universe_mode(self):
        """ universe mode: interacting with the universe map.

        Move the cursor around, pan camera at edge of viewscreen. Can select a
        sector and enter sector mode. """

        self.current_mode = Mode.UNIVERSE

        self.ucursor_x = 0
        self.ucursor_y = 0
        self.camera_x = 0
        self.camera_y = 0
        while(True):
            self.logger.debug("drawloop")
            self.draw_universe_map(self.gamestate.sectors)
            curses.doupdate()

            self.logger.debug("awaiting input...")
            key = self.stdscr.getch()
            self.status_message()
            self.logger.debug(f'got {key}')
            if key in (ord('w'), ord('a'), ord('s'), ord('d')):
                self.move_ucursor(key)
            elif key == 27:
                self.logger.info(f'breaking')
                return None
            elif key == ":":
                return Mode.COMMAND

    def command_mode(self):
        """ Command mode: typing in a command to execute. """

        pass
