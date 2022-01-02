""" Curses based interface for Stellar Punk. """

import logging
import curses
from curses import textpad
import enum
import array
import fcntl
import termios
import math

import drawille

from stellarpunk import util, generate, core


class Layout(enum.Enum):
    LEFT_RIGHT = enum.auto()
    UP_DOWN = enum.auto()

class Mode(enum.Enum):
    QUIT = -1
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

    PLANET = "\u25CB"
    STATION = "\u25A1"
    DERELICT = "\u2302" # house symbol (kina looks like a gravestone?)

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

        self.logscreen = None
        self.logscreen_width = 0
        self.logscreen_height = 0
        self.logscreen_x = 0
        self.logscreen_y = 0

        # upper left corner of the camera view inside the viewscreen
        self.camera_x = 0
        self.camera_y = 0

        # position of the universe sector cursor (in universe-sector coords)
        self.ucursor_x = 0
        self.ucursor_y = 0
        self.sector_maxx = 0
        self.sector_maxy = 0

        # position of the sector cursor (in sector space coords
        self.scursor_x = 0
        self.scursor_y = 0
        # sector zoom level, expressed in meters to fit on screen
        self.szoom = 0

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

        self.logger.info(f'chose layout {self.layout}')
        self.logger.info(f'viewscreen (x,y) {(self.viewscreen_y, self.viewscreen_x)} (h,w): {(self.viewscreen_height, self.viewscreen_width)}')
        self.logger.info(f'logscreen (x,y) {(self.logscreen_y, self.logscreen_x)} (h,w): {(self.logscreen_height, self.logscreen_width)}')

    def reinitialize_screen(self, name="Main Viewscreen"):
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
        self.logscreen.noutrefresh(
                0, 0,
                self.logscreen_y, self.logscreen_x,
                self.logscreen_y+self.logscreen_height-1,
                self.logscreen_x+self.logscreen_width-1
        )
        self.stdscr.addstr(self.screen_height-1, 0, " "*(self.screen_width-1))

    def initialize(self):
        self.color_demo()
        self.reinitialize_screen()

    def color_demo(self):
        self.stdscr.addstr(0, 35, "COLOR DEMO");

        for c in range(256):
            self.stdscr.addstr(int(c/8)+1, c%8*9,f'...{c:03}...', curses.color_pair(c));
        self.stdscr.addstr(34, 1, "Press any key to continue")

        self.stdscr.getch()

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
        self.stdscr.getch()

    def draw_umap_sector(self, y, x, sector):
        """ Draws a single sector to viewscreen starting at position (y,x) """
        textpad.rectangle(self.viewscreen, y, x, y+Settings.UMAP_SECTOR_HEIGHT-1, x+Settings.UMAP_SECTOR_WIDTH-1)

        if (self.ucursor_x, self.ucursor_y) == (sector.x, sector.y):
            self.viewscreen.addstr(y+1,x+1, sector.short_id(), curses.A_STANDOUT)
        else:
            self.viewscreen.addstr(y+1,x+1, sector.short_id())

        self.viewscreen.addstr(y+2,x+1, sector.name)
        self.viewscreen.addstr(y+Settings.UMAP_SECTOR_HEIGHT-2, x+1, f'{len(sector.entities)} objects')

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

    def draw_grid(self, y, x, meters_per_char_x, meters_per_char_y, ul_x, ul_y, lr_x, lr_y, target_screen, max_ticks=10):
        """ Draws a grid at tick lines. """

        # choose ticks
        #TODO: should choose maxTicks based on resolution

        major_ticks_x = util.NiceScale(ul_x, lr_x, maxTicks=max_ticks, constrain_to_range=True)
        minor_ticks_y = util.NiceScale(ul_y, lr_y, maxTicks=max_ticks*4, constrain_to_range=True)
        major_ticks_y = util.NiceScale(ul_y, lr_y, maxTicks=max_ticks, constrain_to_range=True)
        minor_ticks_x = util.NiceScale(ul_x, lr_x, maxTicks=max_ticks*4, constrain_to_range=True)

        self.logger.debug(f'bounds {(ul_x, lr_x)} vertical lines {(major_ticks_x.niceMin, major_ticks_x.niceMax)} every {major_ticks_x.tickSpacing}')
        self.logger.debug(f'bounds {(ul_y, lr_y)} horizonal lines {(major_ticks_y.niceMin, major_ticks_y.niceMax)} every {major_ticks_y.tickSpacing}')

        c = drawille.Canvas()

        # draw the vertical lines
        i = major_ticks_x.niceMin
        while i < lr_x:
            j = minor_ticks_y.niceMin
            while j < lr_y:
                d_x, d_y = util.sector_to_drawille(i, j, meters_per_char_x, meters_per_char_y)
                c.set(d_x, d_y)
                j += minor_ticks_y.tickSpacing
            i += major_ticks_x.tickSpacing

        # draw the horizonal lines
        j = major_ticks_y.niceMin
        while j < lr_y:
            i = minor_ticks_x.niceMin
            while i < lr_x:
                d_x, d_y = util.sector_to_drawille(i, j, meters_per_char_x, meters_per_char_y)
                c.set(d_x, d_y)
                i += minor_ticks_x.tickSpacing
            j += major_ticks_y.tickSpacing

        # draw the grid to the screen
        text = c.frame().split("\n")
        max_text_len = max(map(lambda t: len(t), text))

        s_ul_y = int(y - len(text)/2)
        s_ul_x = int(x - max_text_len/2)
        for i, line in enumerate(text):
            if s_ul_y+i < 0 or s_ul_x < 0:
                # off by one between sector to drawille and the fact that
                # adding a point inside a char means we need that entire char
                continue
            target_screen.addstr(s_ul_y+i, s_ul_x, line, curses.color_pair(29))

        # draw location indicators
        i = major_ticks_x.niceMin
        while i <= major_ticks_x.niceMax:
            s_i, _ = util.sector_to_screen(i, 0, ul_x, ul_y, meters_per_char_x, meters_per_char_y)
            target_screen.addstr(0, s_i, util.human_distance(i), curses.color_pair(29))
            i += major_ticks_x.tickSpacing
        j = major_ticks_y.niceMin
        while j <= major_ticks_y.niceMax:
            _, s_j = util.sector_to_screen(0, j, ul_x, ul_y, meters_per_char_x, meters_per_char_y)
            target_screen.addstr(s_j, 0, util.human_distance(j), curses.color_pair(29))
            j += major_ticks_y.tickSpacing

    def draw_radar(self, y, x, meters_per_char_x, meters_per_char_y, radius, target_screen):
        """ Draws a radar graphic to get sense of scale centered at y, x. """

        # choose ticks
        ticks = util.NiceScale(-1*radius, radius, constrain_to_range=True)
        stepsize = ticks.tickSpacing

        def polar_to_rectangular(r, theta):
            return (r*math.cos(theta), r*math.sin(theta))

        c = drawille.Canvas()

        # draw a cross
        i = 0
        while i < radius:
            drawille_x,_ = util.sector_to_drawille(i, 0, meters_per_char_x, meters_per_char_y)
            _,drawille_y = util.sector_to_drawille(0, i, meters_per_char_x, meters_per_char_y)
            c.set(drawille_x, 0)
            c.set(-1*drawille_x, 0)
            c.set(0, drawille_y)
            c.set(0, -1*drawille_y)
            i += stepsize/2

        # draw rings to fill up the square with sides 2*radius
        r = stepsize
        theta_step = math.pi/16
        while r < math.sqrt(2*radius*radius):
            theta = 0
            while theta < 2*math.pi:
                s_x, s_y = polar_to_rectangular(r, theta)
                if abs(s_x) < radius and abs(s_y) < radius:
                    d_x, d_y = util.sector_to_drawille(s_x, s_y, meters_per_char_x, meters_per_char_y)
                    c.set(d_x, d_y)
                theta += theta_step
            r += stepsize

        # draw radar to the screen
        text = c.frame().split("\n")
        max_text_len = max(map(lambda t: len(t), text))

        ul_y = int(y - len(text)/2)
        ul_x = int(x - max_text_len/2)
        for i, line in enumerate(text):
            if ul_y+i < 0 or ul_x < 0:
                # off by one between sector to drawille and the fact that
                # adding a point inside a char means we need that entire char
                continue
            target_screen.addstr(ul_y+i, ul_x, line, curses.color_pair(29))

        # draw distance indicators
        for r in range(int(stepsize), int(ticks.niceMax), int(stepsize)):
            target_screen.addstr(y+int(r/meters_per_char_y), x, util.human_distance(r), curses.color_pair(29))

    def draw_entity(self, y, x, entity, target_screen):
        """ Draws a single sector entity at screen position (y,x) """

        if isinstance(entity, core.Ship):
            icon = Icons.SHIP_N
        elif isinstance(entity, core.Station):
            icon = Icons.STATION
        elif isinstance(entity, core.Planet):
            icon = Icons.PLANET
        else:
            icon = "?"

        target_screen.addstr(y, x, icon)
        target_screen.addstr(y, x+1, f' {entity.short_id()}', curses.A_DIM)

    def draw_multiple_entities(self, y, x, entities, target_screen):
        target_screen.addstr(y, x, Icons.MULTIPLE)
        target_screen.addstr(y, x+1, f' {len(entities)} entities', curses.A_DIM)

    def draw_sector_map(self, sector):
        """ Draws a map of a sector. """

        self.viewscreen.erase()

        # get the bounding box for the viewscreen, determined by the zoom level
        # we try to fit at least the zoom level on screen, constrained by the
        # minimum of viewscreen width and height
        meters_per_char_x = self.szoom / min(self.viewscreen_width, math.floor(self.viewscreen_height/self.font_width*self.font_height))
        meters_per_char_y = meters_per_char_x / self.font_width * self.font_height

        assert self.szoom / meters_per_char_y <= self.viewscreen_height
        assert self.szoom / meters_per_char_x <= self.viewscreen_width

        self.logger.info(f'resolution is {meters_per_char_x:.0f}m x {meters_per_char_y:.0f}m')

        ul_x = self.scursor_x - (self.viewscreen_width/2 * meters_per_char_x)
        ul_y = self.scursor_y - (self.viewscreen_height/2 * meters_per_char_y)
        lr_x = self.scursor_x + (self.viewscreen_width/2 * meters_per_char_x)
        lr_y = self.scursor_y + (self.viewscreen_height/2 * meters_per_char_y)

        self.logger.info(f'drawing sector {sector.entity_id} with bounding box ({(ul_x, ul_y)}, {(lr_x, lr_y)}) with {sector.spatial.count((ul_x, ul_y, lr_x, lr_y))} objects visible of {len(sector.entities)} total')

        #self.draw_radar(
        #        int(self.viewscreen_height/2), int(self.viewscreen_width/2),
        #        meters_per_char_x, meters_per_char_y,
        #        self.szoom/2,
        #        self.viewscreen
        #)
        self.draw_grid(
            int(self.viewscreen_height/2), int(self.viewscreen_width/2),
            meters_per_char_x, meters_per_char_y,
            ul_x, ul_y, lr_x, lr_y,
            self.viewscreen
        )

        # list x,y coords of center of screen
        #self.viewscreen.addstr(1,3, f'{self.scursor_x:.0f},{self.scursor_y:.0f}', curses.color_pair(29))

        occupied = {}
        # sort the entities so we draw left to right, top to bottom
        # this ensures any annotations down and to the right of an entity on
        # the sector map will not cover up the icon for an entity
        # this assumes any extra annotations are down and to the right
        for hit in sorted(sector.spatial.intersection((ul_x, ul_y, lr_x, lr_y), objects=True), key=lambda h: h.bbox):
            entity = sector.entities[hit.object]
            screen_x, screen_y = util.sector_to_screen(entity.x, entity.y, ul_x, ul_y, meters_per_char_x, meters_per_char_y)
            #self.logger.debug(f'hit {entity.entity_id} at {(entity.x, entity.y)} translates to ({screen_x, screen_y})')
            if (screen_x, screen_y) in occupied:
                entities = occupied[(screen_x, screen_y)]
                entities.append(entity)
                self.draw_multiple_entities(screen_y, screen_x, entities, self.viewscreen)
            else:
                occupied[(screen_x, screen_y)] = [entity]
                self.draw_entity(screen_y, screen_x, entity, self.viewscreen)

        self.refresh_viewscreen()

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

    def move_scursor(self, direction, sector):
        old_x = self.scursor_x
        old_y = self.scursor_y

        stepsize = self.szoom/32

        if direction == ord('w'):
            self.scursor_y -= stepsize
        elif direction == ord('a'):
            self.scursor_x -= stepsize
        elif direction == ord('s'):
            self.scursor_y += stepsize
        elif direction == ord('d'):
            self.scursor_x += stepsize
        else:
            raise ValueError(f'unknown direction {direction}')

    def universe_mode(self):
        """ Universe mode: interacting with the universe map.

        Displays all the sectors in a grid with summary information about each.
        Player can move between sectors, pan the camera at edge of viewscreen.
        Can select a sector and enter sector mode. Can enter command mode.
        """

        self.logger.info(f'entering universe mode') 
        self.reinitialize_screen(name="Universe Map")

        self.ucursor_x = 0
        self.ucursor_y = 0
        self.camera_x = 0
        self.camera_y = 0
        while(True):
            self.current_mode = Mode.UNIVERSE
            self.logger.debug("universe drawloop")
            self.draw_universe_map(self.gamestate.sectors)
            curses.doupdate()

            self.logger.debug("awaiting input...")
            key = self.stdscr.getch()
            self.logger.debug(f'got {key}')

            if key in (ord('w'), ord('a'), ord('s'), ord('d')):
                self.status_message()
                self.move_ucursor(key)
            elif key == ord('\n'): #TODO: enter contstant
                sector_ret = self.sector_mode(self.gamestate.sectors[(self.ucursor_x, self.ucursor_y)])
                if Mode.QUIT == sector_ret:
                    return Mode.QUIT
                self.reinitialize_screen(name="Universe Map")
            elif key == ord(":"):
                command_ret = self.command_mode()
                if Mode.QUIT == command_ret:
                    return Mode.QUIT
            elif key == curses.KEY_RESIZE:
                self.reinitialize_screen(name="Universe Map")

    def sector_mode(self, target_sector):
        """ Sector mode: interacting with the sector map.

        Draw the contents of the sector: ships, stations, asteroids, etc.
        Player can move around the sector, panning the camera at the edges of
        the viewscreen, search for named entities within the sector (jumping to
        each a la vim search.) Can select a ship to interact with. Can enter
        pilot mode. Can return to universe mode or enter command mode.
        """

        self.logger.info(f'entering sector mode for {target_sector.entity_id}')
        self.reinitialize_screen(name="Sector Map")

        self.scursor_x = 0
        self.scursor_y = 0
        self.szoom = target_sector.radius*2
        self.camera_x = 0
        self.camera_y = 0

        while(True):
            self.logger.debug(f'sector drawloop at zoom {self.szoom}')
            self.draw_sector_map(target_sector)
            curses.doupdate()

            self.logger.debug("awaiting input...")
            key = self.stdscr.getch()
            self.logger.debug(f'got {key}')

            if key in (ord('w'), ord('a'), ord('s'), ord('d')):
                self.move_scursor(key, target_sector)
            elif key == ord("+"):
                self.szoom *= 0.9
            elif key == ord("-"):
                self.szoom *= 1.1
            elif key == 27: #TODO: should handle escape here
                return None
            elif key == ord(":"):
                command_ret = self.command_mode()
                if Mode.QUIT == command_ret:
                    return Mode.QUIT

    def command_mode(self):
        """ Command mode: typing in a command to execute. """

        self.logger.info("entering command mode")
        self.status_message(":")
        command = ""
        while(True):
            self.current_mode = Mode.COMMAND
            key = self.stdscr.getch()
            if key == ord('\n'): #TODO: enter constant
                # process the command
                if command == "quit":
                    self.logger.info("quitting")
                    return Mode.QUIT
                else:
                    self.status_message(f'unknown command "{command}" enter command mode with ":" and then "quit" to quit.', curses.color_pair(1))
                    return None
            elif chr(key).isprintable():
                command += chr(key)
                self.stdscr.addch(chr(key))
            elif key == 8: #TODO: backspace constant
                command = command[:-1]
                (y,x) = self.stdscr.getyx()
                if x > 1:
                    self.stdscr.move(y, x-1)
                    self.stdscr.delch()
