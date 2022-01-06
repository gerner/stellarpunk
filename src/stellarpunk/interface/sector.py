""" Interface methods dealing with displaying a single sector map. """

import logging
import math
import bisect
import curses
import curses.textpad
import curses.ascii
import time

import drawille

from stellarpunk import util, core, interface

class SectorView(interface.View):
    """ Sector mode: interacting with the sector map.

    Draw the contents of the sector: ships, stations, asteroids, etc.
    Player can move around the sector, panning the camera at the edges of
    the viewscreen, search for named entities within the sector (jumping to
    each a la vim search.) Can select a ship to interact with. Can enter
    pilot mode. Can return to universe mode or enter command mode.
    """

    def __init__(self, sector, interface, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #TODO: this circular dependency feels odd
        #   need to know dimensions of the viewscreen (should pass in?)
        #   need to write messages outside the viewscreen (do we?)
        self.interface = interface
        self.sector = sector

        # where the sector map is centered in sector coordinates
        self.scursor_x = 0
        self.scursor_y = 0

        # entity id of the currently selected target
        self.selected_target = None
        self.selected_entity = None

        # sector zoom level, expressed in meters to fit on screen
        self.szoom = self.sector.radius*2
        self.meters_per_char_x = 0
        self.meters_per_char_y = 0

        # sector coord bounding box (ul_x, ul_y, lr_x, lr_y)
        self.bbox = (0,0,0,0)

    @property
    def viewscreen(self):
        return self.interface.viewscreen

    def initialize(self):
        self.logger.info(f'entering sector mode for {self.sector.entity_id}')
        self.interface.camera_x = 0
        self.interface.camera_y = 0
        self.update_bbox()
        self.interface.reinitialize_screen(name="Sector Map")

    def focus(self):
        super().focus()
        self.interface.camera_x = 0
        self.interface.camera_y = 0
        self.interface.reinitialize_screen(name="Sector Map")

    def update_bbox(self):
        self.meters_per_char_x, self.meters_per_char_y = self.meters_per_char()

        vsw = self.interface.viewscreen_width
        vsh = self.interface.viewscreen_height

        ul_x = self.scursor_x - (vsw/2 * self.meters_per_char_x)
        ul_y = self.scursor_y - (vsh/2 * self.meters_per_char_y)
        lr_x = self.scursor_x + (vsw/2 * self.meters_per_char_x)
        lr_y = self.scursor_y + (vsh/2 * self.meters_per_char_y)

        self.bbox = (ul_x, ul_y, lr_x, lr_y)

        self.logger.debug(f'viewing sector {self.sector.entity_id} with bounding box ({(ul_x, ul_y)}, {(lr_x, lr_y)}) with per {self.meters_per_char_x:.0f}m x {self.meters_per_char_y:.0f}m char')

    def set_scursor(self, x, y):
        self.scursor_x = x
        self.scursor_y = y

        self.update_bbox()

    def move_scursor(self, direction):
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

        self.update_bbox()

    def zoom_scursor(self, direction):
        if direction == ord('+'):
            self.szoom *= 0.9
        elif direction == ord('-'):
            self.szoom *= 1.1
        else:
            raise ValueError(f'unknown direction {direction}')

        self.update_bbox()

    def meters_per_char(self):
        meters_per_char_x = self.szoom / min(self.interface.viewscreen_width, math.floor(self.interface.viewscreen_height/self.interface.font_width*self.interface.font_height))
        meters_per_char_y = meters_per_char_x / self.interface.font_width * self.interface.font_height

        assert self.szoom / meters_per_char_y <= self.interface.viewscreen_height
        assert self.szoom / meters_per_char_x <= self.interface.viewscreen_width

        return (meters_per_char_x, meters_per_char_y)

    def select_target(self, target_id, entity):
        if target_id == self.selected_target:
            # no-op selecting the same target
            return
        self.selected_target = target_id
        self.selected_entity = entity
        self.interface.log_message(f'{entity.short_id()}: {entity.name}')


    def draw_grid(self, max_ticks=10):
        """ Draws a grid at tick lines. """

        # choose ticks
        #TODO: should choose maxTicks based on resolution

        major_ticks_x = util.NiceScale(
                self.bbox[0], self.bbox[2],
                maxTicks=max_ticks, constrain_to_range=True)
        minor_ticks_y = util.NiceScale(
                self.bbox[1], self.bbox[3],
                maxTicks=max_ticks*4, constrain_to_range=True)
        major_ticks_y = util.NiceScale(
                self.bbox[1], self.bbox[3],
                maxTicks=max_ticks, constrain_to_range=True,
                tickSpacing=major_ticks_x.tickSpacing)
        minor_ticks_x = util.NiceScale(
                self.bbox[0], self.bbox[2],
                maxTicks=max_ticks*4, constrain_to_range=True,
                tickSpacing=minor_ticks_y.tickSpacing)

        c = drawille.Canvas()

        # draw the vertical lines
        i = major_ticks_x.niceMin
        while i < self.bbox[2]:
            j = minor_ticks_y.niceMin
            while j < self.bbox[3]:
                d_x, d_y = util.sector_to_drawille(
                        i, j,
                        self.meters_per_char_x, self.meters_per_char_y)
                c.set(d_x, d_y)
                j += minor_ticks_y.tickSpacing
            i += major_ticks_x.tickSpacing

        # draw the horizonal lines
        j = major_ticks_y.niceMin
        while j < self.bbox[3]:
            i = minor_ticks_x.niceMin
            while i < self.bbox[2]:
                d_x, d_y = util.sector_to_drawille(
                        i, j,
                        self.meters_per_char_x, self.meters_per_char_y)
                c.set(d_x, d_y)
                i += minor_ticks_x.tickSpacing
            j += major_ticks_y.tickSpacing

        # get upper left corner position so drawille canvas fills the screen
        (d_x, d_y) = util.sector_to_drawille(
                self.bbox[0], self.bbox[1],
                self.meters_per_char_x, self.meters_per_char_y)
        # draw the grid to the screen
        text = c.rows(d_x, d_y)

        for i, line in enumerate(text):
            self.viewscreen.addstr(i, 0, line, curses.color_pair(29))

        # draw location indicators
        i = major_ticks_x.niceMin
        while i <= major_ticks_x.niceMax:
            s_i, _ = util.sector_to_screen(
                    i, 0,
                    self.bbox[0], self.bbox[1],
                    self.meters_per_char_x, self.meters_per_char_y)
            self.viewscreen.addstr(0, s_i, util.human_distance(i), curses.color_pair(29))
            i += major_ticks_x.tickSpacing
        j = major_ticks_y.niceMin
        while j <= major_ticks_y.niceMax:
            _, s_j = util.sector_to_screen(
                    0, j,
                    self.bbox[0], self.bbox[1],
                    self.meters_per_char_x, self.meters_per_char_y)
            self.viewscreen.addstr(s_j, 0, util.human_distance(j), curses.color_pair(29))
            j += major_ticks_y.tickSpacing

    def draw_radar(self, radius):
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
            drawille_x,_ = util.sector_to_drawille(
                    i, 0, self.meters_per_char_x, self.meters_per_char_y)
            _,drawille_y = util.sector_to_drawille(
                    0, i, self.meters_per_char_x, self.meters_per_char_y)
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
                    d_x, d_y = util.sector_to_drawille(
                            s_x, s_y, self.meters_per_char_x, self.meters_per_char_y)
                    c.set(d_x, d_y)
                theta += theta_step
            r += stepsize

        # get upper left corner position so drawille canvas fills the screen
        (d_x, d_y) = util.sector_to_drawille(
                self.bbox[0], self.bbox[1],
                self.meters_per_char_x, self.meters_per_char_y)
        # draw the grid to the screen
        text = c.rows(d_x, d_y)

        for i, line in enumerate(text):
            self.viewscreen.addstr(i, 0, line, curses.color_pair(29))

        # draw distance indicators
        for r in range(int(stepsize), int(ticks.niceMax), int(stepsize)):
            self.viewscreen.addstr(y+int(r/meters_per_char_y), x, util.human_distance(r), curses.color_pair(29))

    def draw_entity(self, y, x, entity):
        """ Draws a single sector entity at screen position (y,x) """

        if isinstance(entity, core.Ship):
            icon = interface.Icons.angle_to_ship(entity.angle)
        elif isinstance(entity, core.Station):
            icon = interface.Icons.STATION
        elif isinstance(entity, core.Planet):
            icon = interface.Icons.PLANET
        else:
            icon = "?"

        icon_attr = 0
        description_attr = curses.color_pair(9)
        if entity.entity_id == self.selected_target:
            icon_attr |= curses.A_STANDOUT
            description_attr |= curses.A_STANDOUT

        self.viewscreen.addstr(y, x, icon, icon_attr)
        self.viewscreen.addstr(y, x+1, f' {entity.short_id()}', description_attr)
        self.viewscreen.addstr(y+1, x+1, f' s: {entity.x:.0f},{entity.y:.0f}', description_attr)
        self.viewscreen.addstr(y+2, x+1, f' v: {entity.velocity[0]:.0f},{entity.velocity[1]:.0f}', description_attr)

    def draw_multiple_entities(self, y, x, entities):
        self.viewscreen.addstr(y, x, interface.Icons.MULTIPLE)
        self.viewscreen.addstr(y, x+1, f' {len(entities)} entities', curses.color_pair(9))

    def draw_sector_map(self):
        """ Draws a map of a sector. """

        self.viewscreen.erase()

        #self.draw_radar(
        #        int(self.viewscreen_height/2), int(self.viewscreen_width/2),
        #        meters_per_char_x, meters_per_char_y,
        #        self.szoom/2,
        #        self.viewscreen
        #)
        self.draw_grid()

        # list x,y coords of center of screen
        #self.viewscreen.addstr(1,3, f'{self.scursor_x:.0f},{self.scursor_y:.0f}', curses.color_pair(29))

        occupied = {}
        # sort the entities so we draw left to right, top to bottom
        # this ensures any annotations down and to the right of an entity on
        # the sector map will not cover up the icon for an entity
        # this assumes any extra annotations are down and to the right
        for hit in sorted(self.sector.spatial.intersection(self.bbox, objects=True), key=lambda h: h.bbox):
            entity = self.sector.entities[hit.object]
            screen_x, screen_y = util.sector_to_screen(
                    entity.x, entity.y, self.bbox[0], self.bbox[1],
                    self.meters_per_char_x, self.meters_per_char_y)
            if (screen_x, screen_y) in occupied:
                entities = occupied[(screen_x, screen_y)]
                entities.append(entity)
                self.draw_multiple_entities(
                        screen_y, screen_x, entities)
            else:
                occupied[(screen_x, screen_y)] = [entity]
                self.draw_entity(screen_y, screen_x, entity)

        #TODO: draw an indicator for off-screen targeted entities
        #se_x = self.selected_entity.x
        #se_y = self.selected_entity.y
        #if (se_x < ul_x or se_x > lr_x or
        #        se_y < ul_y or se_y > lr_y):


        self.interface.refresh_viewscreen()

    def update_display(self):
        self.interface.camera_x = 0
        self.interface.camera_y = 0
        self.draw_sector_map()
        self.interface.refresh_viewscreen()

    def handle_input(self, key):
        if key in (ord('w'), ord('a'), ord('s'), ord('d')):
            self.move_scursor(key)
        elif key in (ord("+"), ord("-")):
            self.zoom_scursor(key)
        elif key == ord("t"):
            entity_id_list = sorted(self.sector.entities.keys())
            if len(entity_id_list) == 0:
                self.select_target(None, None)
            elif self.selected_target is None:
                self.select_target(entity_id_list[0], self.sector.entities[entity_id_list[0]])
            else:
                next_index = bisect.bisect_right(entity_id_list, self.selected_target)
                if next_index >= len(entity_id_list):
                    next_index = 0
                self.select_target(entity_id_list[next_index], self.sector.entities[entity_id_list[next_index]])
        elif key in (ord('\n'), ord('\r')):
            if self.selected_target:
                self.set_scursor(
                        self.sector.entities[self.selected_target].x,
                        self.sector.entities[self.selected_target].y
                )
        elif key == ord(":"):
            self.interface.open_view(interface.CommandInput(self.interface))
        elif key == curses.KEY_RESIZE:
            self.initialize()
        elif key == curses.KEY_MOUSE:
            m_tuple = curses.getmouse()
            m_id, m_x, m_y, m_z, bstate = m_tuple
            ul_x = self.scursor_x - (self.interface.viewscreen_width/2 * self.meters_per_char_x)
            ul_y = self.scursor_y - (self.interface.viewscreen_height/2 * self.meters_per_char_y)
            sector_x, sector_y = util.screen_to_sector(
                    m_x, m_y, ul_x, ul_y,
                    self.meters_per_char_x, self.meters_per_char_y,
                    self.interface.viewscreen_x, self.interface.viewscreen_y)

            self.logger.debug(f'got mouse: {m_tuple}, corresponding to {(sector_x, sector_y)} ul: {(ul_x, ul_y)}')

            # select a target within a cell of the mouse click
            hit = next(self.sector.spatial.nearest(
                (sector_x-self.meters_per_char_x, sector_y-self.meters_per_char_y,
                    sector_x+self.meters_per_char_x, sector_y+self.meters_per_char_y),
                1, objects=True), None)
            if hit:
                #TODO: check if the hit is close enough
                self.select_target(hit.object, self.sector.entities[hit.object])
        elif key == curses.ascii.ESC: #TODO: should handle escape here
            if self.selected_target is not None:
                self.selected_target = None
            else:
                return False

        return True
