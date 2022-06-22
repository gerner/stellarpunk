""" Ship piloting interface.

Sits within a sector view.
"""

import math
import curses

from typing import Tuple, Optional, Any, Callable

from stellarpunk import core, interface, util

class PilotView(interface.View):
    """ Piloting mode: direct command of a ship. """

    def __init__(self, ship:core.Ship, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        self.ship = ship

        # where the sector map is centered in sector coordinates
        self.scursor_x = 0.
        self.scursor_y = 0.

        # sector zoom level, expressed in meters to fit on screen
        self.szoom = self.ship.sector.radius*2
        self.meters_per_char_x = 0.
        self.meters_per_char_y = 0.

        # sector coord bounding box (ul_x, ul_y, lr_x, lr_y)
        self.bbox = (0.,0.,0.,0.)

        self_cached_radar = None

    def _compute_radar(self, max_ticks:int=10) -> None:
        self._cached_radar = util.compute_uiradar(
                (self.scursor_x, self.scursor_y),
                self.bbox, self.meters_per_char_x, self.meters_per_char_y)


    def meters_per_char(self) -> Tuple[float, float]:
        meters_per_char_x = self.szoom / min(self.interface.viewscreen_width, math.floor(self.interface.viewscreen_height/self.interface.font_width*self.interface.font_height))
        meters_per_char_y = meters_per_char_x / self.interface.font_width * self.interface.font_height

        assert self.szoom / meters_per_char_y <= self.interface.viewscreen_height
        assert self.szoom / meters_per_char_x <= self.interface.viewscreen_width

        return (meters_per_char_x, meters_per_char_y)


    def update_bbox(self) -> None:
        self.meters_per_char_x, self.meters_per_char_y = self.meters_per_char()

        vsw = self.interface.viewscreen_width
        vsh = self.interface.viewscreen_height

        ul_x = self.scursor_x - (vsw/2 * self.meters_per_char_x)
        ul_y = self.scursor_y - (vsh/2 * self.meters_per_char_y)
        lr_x = self.scursor_x + (vsw/2 * self.meters_per_char_x)
        lr_y = self.scursor_y + (vsh/2 * self.meters_per_char_y)

        self.bbox = (ul_x, ul_y, lr_x, lr_y)

        self._compute_radar()

    def draw_radar(self) -> None:
        """ Draws a grid at tick lines. """

        major_ticks_x, minor_ticks_y, major_ticks_y, minor_ticks_x, text = self._cached_radar

        for lineno, line in enumerate(text):
            self.viewscreen.addstr(lineno, 0, line, curses.color_pair(29))

        # draw location indicators
        i = major_ticks_x.niceMin
        while i <= major_ticks_x.niceMax:
            s_x, s_y = util.sector_to_screen(
                    i+self.scursor_x, self.scursor_y,
                    self.bbox[0], self.bbox[1],
                    self.meters_per_char_x, self.meters_per_char_y)
            self.viewscreen.addstr(s_y, s_x, util.human_distance(i), curses.color_pair(29))
            i += major_ticks_x.tickSpacing
        j = major_ticks_y.niceMin
        while j <= major_ticks_y.niceMax:
            s_x, s_y = util.sector_to_screen(
                    self.scursor_x, j+self.scursor_y,
                    self.bbox[0], self.bbox[1],
                    self.meters_per_char_x, self.meters_per_char_y)
            self.viewscreen.addstr(s_y, s_x, util.human_distance(j), curses.color_pair(29))
            j += major_ticks_y.tickSpacing

        # add a scale near corner
        scale_label = f'scale {util.human_distance(major_ticks_x.tickSpacing)}'
        scale_x = self.interface.viewscreen_width - len(scale_label) - 2
        scale_y = self.interface.viewscreen_height - 2
        self.viewscreen.addstr(scale_y, scale_x, scale_label, curses.color_pair(29))

        # add center position near corner
        pos_label = f'({self.scursor_x:.0f},{self.scursor_y:.0f})'
        pos_x = self.interface.viewscreen_width - len(pos_label) - 2
        pos_y = self.interface.viewscreen_height - 1
        self.viewscreen.addstr(pos_y, pos_x, pos_label, curses.color_pair(29))

    def initialize(self) -> None:
        self.logger.info(f'entering pilot mode for {self.ship.entity_id}')
        self.interface.camera_x = 0
        self.interface.camera_y = 0
        self.meters_per_char_x = 0.
        self.meters_per_char_y = 0.

        self.update_bbox()
        self.interface.reinitialize_screen(name="Pilot's Seat")

    def focus(self) -> None:
        self.interface.camera_x = 0
        self.interface.camera_y = 0
        self.interface.reinitialize_screen(name="Pilot's Seat")

    def handle_input(self, key:int) -> bool:
        return False

    def update_display(self) -> None:
        #TODO: we should be careful with setting cursor position and updating
        # the bbox which is expensive.
        self.scursor_x = self.ship.loc[0]
        self.scursor_y = self.ship.loc[1]
        self.update_bbox()

        self.interface.camera_x = 0
        self.interface.camera_y = 0
        #self.draw_sector_map()
        self.viewscreen.erase()
        self.draw_radar()
        self.interface.refresh_viewscreen()

