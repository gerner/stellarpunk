""" Curses based interface for Stellar Punk. """

import logging
import curses
from curses import textpad
import enum

from stellarpunk import util, generate


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

class Interface:
    def __init__(self):
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

        self.camera_x = 0
        self.camera_y = 0

        self.logscreen = None
        self.logscreen_width = 0
        self.logscreen_height = 0
        self.logscreen_x = 0
        self.logscreen_y = 0

    def __enter__(self):
        """ Does most simple interface initialization.

        In particular, this does enought initialization of the curses interface
        so that it can be cleaned up properly in __exit__. """

        self.logger.info("starting the inferface")
        self.stdscr = curses.initscr()

        curses.noecho()
        curses.cbreak()
        self.stdscr.keypad(1)

        # we can check if there's color later
        try:
            curses.start_color()
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
        textpad.rectangle(
                self.stdscr,
                self.logscreen_y-1,
                self.logscreen_x-1,
                self.logscreen_y+self.logscreen_height,
                self.logscreen_x+self.logscreen_width
        )

        self.viewscreen = curses.newpad(self.viewscreen_height+1, self.viewscreen_width)
        self.logscreen = curses.newpad(self.logscreen_height, self.logscreen_width)
        self.logscreen.scrollok(True)

        self.viewscreen.addstr(0,0, "V"+("v"*(self.viewscreen_width-2))+"V")
        self.viewscreen.addstr(1,0, "this is the viewscreen")
        self.viewscreen.addstr(self.viewscreen_height-1,0, "V"+("v"*(self.viewscreen_width-2))+"V")

        self.logscreen.addstr(0,0, "L"+("l"*(self.logscreen_width-2))+"L")
        self.logscreen.addstr(1,0, "this is the logscreen")
        self.logscreen.addstr(self.logscreen_height-1,0, "L"+("l"*(self.logscreen_width-2))+"L")

        self.stdscr.addstr(self.screen_height-1, 0, "this is the overall screen")

        self.logscreen.addstr(self.logscreen_height-1,0, "first line\n")
        self.logscreen.addstr(self.logscreen_height-1,0, "and more text\n")
        self.logscreen.addstr(self.logscreen_height-1,0, "and more text\n")
        self.logscreen.addstr(self.logscreen_height-1,0, "and more text\n")
        self.logscreen.addstr(self.logscreen_height-1,0, "and more text\n")
        self.logscreen.addstr(self.logscreen_height-1,0, "and more text\n")
        self.logscreen.addstr(self.logscreen_height-1,0, "and more text\n")
        self.logscreen.addstr(self.logscreen_height-1,0, "and more text\n")
        self.logscreen.addstr(self.logscreen_height-1,0, "and more text\n")
        self.logscreen.addstr(self.logscreen_height-1,0, "and more text\n")
        self.logscreen.addstr(self.logscreen_height-1,0, "and more text\n")
        self.logscreen.addstr(self.logscreen_height-1,0, "last line\n")

        #y,x = (self.logscreen_y,self.logscreen_x)
        #pos_str = f'string positioned at {(y,x)}'
        #self.stdscr.addstr(y,x, pos_str)

        self.stdscr.noutrefresh()
        self.viewscreen.noutrefresh(
                0, 0,
                self.viewscreen_y, self.viewscreen_x,
                self.viewscreen_y+self.viewscreen_height-1,
                self.viewscreen_x+self.viewscreen_width-1
        )
        self.logscreen.noutrefresh(
                0, 0,
                self.logscreen_y, self.logscreen_x,
                self.logscreen_y+self.logscreen_height-1,
                self.logscreen_x+self.logscreen_width-1
        )

        curses.doupdate()

        self.logger.info("waiting for final input")
        self.stdscr.getch()

    def draw_universe_map(self, sectors):
        """ Draws a map of all sectors. """
        pass

    def draw_sector_map(self, sector):
        """ Draws a map of a sector. """

        # sectors laid out in x,y grid
        # we'll lay them out as small boxes with name and overview stats
        pass

class GenerationUI(generate.GenerationListener):
    """ Handles the UI during universe generation. """

    def __init__(self, ui):
        self.ui = ui

    def production_chain_complete(self, production_chain):
        pass
