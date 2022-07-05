""" Ship piloting interface.

Sits within a sector view.
"""

import math
import curses

from typing import Tuple, Optional, Any, Callable, Mapping, MutableMapping

from stellarpunk import core, interface, util
from stellarpunk.interface import presenter

class PilotView(interface.View):
    """ Piloting mode: direct command of a ship. """

    def __init__(self, ship:core.Ship, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        self.ship = ship
        if self.ship.sector is None:
            raise ValueError("ship must be in a sector to pilot")

        # where the sector map is centered in sector coordinates
        self.scursor_x = 0.
        self.scursor_y = 0.

        # sector zoom level, expressed in meters to fit on screen
        self.szoom = self.ship.sector.radius*2
        self.meters_per_char_x = 0.
        self.meters_per_char_y = 0.

        # sector coord bounding box (ul_x, ul_y, lr_x, lr_y)
        self.bbox = (0.,0.,0.,0.)

        self._cached_radar = (util.NiceScale(0,0), util.NiceScale(0,0), util.NiceScale(0,0), util.NiceScale(0,0), "")

        self.presenter = presenter.Presenter(self.interface.gamestate, self, self.ship.sector, self.bbox, self.meters_per_char_x, self.meters_per_char_y)

        # indicates if the ship should follow its orders, or direct player
        # control
        self.auto_pilot_on = False

    def _command_list(self) -> Mapping[str, interface.CommandInput.CommandSig]:
        return {}

    def _open_command_prompt(self) -> bool:
        self.interface.open_view(interface.CommandInput(
            self.interface, commands=self._command_list()))
        return True

    def _toggle_autopilot(self) -> bool:
        """ Toggles autopilot state.

        If autopilot is on, we follow normal orders for the ship. Otherwise we
        suspend the current order queue and follow the user's input directly.
        """

        self.auto_pilot_on = not self.auto_pilot_on
        return True

    def _drive(self, key:int) -> bool:
        return True

    def _zoom_scursor(self, key:int) -> bool:
        if key == ord("+"):
            self.szoom *= 0.9
        elif key == ord("-"):
            self.szoom *= 1.1
        else:
            raise ValueError("can only zoom + or -")
        self._update_bbox()
        return True

    def handle_input(self, key:int) -> bool:
        if key == curses.ascii.ESC: return False
        elif key == ord(":"): return self._open_command_prompt()
        elif key in (ord("+"), ord("-")): return self._zoom_scursor(key)
        elif key == ord("p"): return self._toggle_autopilot()
        elif key in (ord("w"), ord("a"), ord("s"), ord("d")): return self._drive(key)
        else: return True

    def _compute_radar(self, max_ticks:int=10) -> None:
        self._cached_radar = util.compute_uiradar(
                (self.scursor_x, self.scursor_y),
                self.bbox, self.meters_per_char_x, self.meters_per_char_y)


    def _meters_per_char(self) -> Tuple[float, float]:
        meters_per_char_x = self.szoom / min(self.interface.viewscreen_width, math.floor(self.interface.viewscreen_height/self.interface.font_width*self.interface.font_height))
        meters_per_char_y = meters_per_char_x / self.interface.font_width * self.interface.font_height

        assert self.szoom / meters_per_char_y <= self.interface.viewscreen_height
        assert self.szoom / meters_per_char_x <= self.interface.viewscreen_width

        return (meters_per_char_x, meters_per_char_y)


    def _update_bbox(self) -> None:
        self.meters_per_char_x, self.meters_per_char_y = self._meters_per_char()

        vsw = self.interface.viewscreen_width
        vsh = self.interface.viewscreen_height

        ul_x = self.scursor_x - (vsw/2 * self.meters_per_char_x)
        ul_y = self.scursor_y - (vsh/2 * self.meters_per_char_y)
        lr_x = self.scursor_x + (vsw/2 * self.meters_per_char_x)
        lr_y = self.scursor_y + (vsh/2 * self.meters_per_char_y)

        self.bbox = (ul_x, ul_y, lr_x, lr_y)

        self._compute_radar()

        self.presenter.bbox = self.bbox
        self.presenter.meters_per_char_x = self.meters_per_char_x
        self.presenter.meters_per_char_y = self.meters_per_char_y

    def _auto_pan(self) -> None:
        """ Pans the viewscreen to center on the ship, but only if the ship has
        moved enough. Avoid's updating the bbox on every tick. """

        # exit early if recentering won't move by half a display cell
        if (abs(self.ship.loc[0] - self.scursor_x) < self.meters_per_char_x/2.
            and abs(self.ship.loc[1] - self.scursor_y) < self.meters_per_char_y/2.):
            return

        self.scursor_x = self.ship.loc[0]
        self.scursor_y = self.ship.loc[1]
        self._update_bbox()

    def _draw_radar(self) -> None:
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

    def _draw_map(self) -> None:
        pass

    def initialize(self) -> None:
        self.logger.info(f'entering pilot mode for {self.ship.entity_id}')
        self.interface.camera_x = 0
        self.interface.camera_y = 0
        self.meters_per_char_x = 0.
        self.meters_per_char_y = 0.

        self._update_bbox()
        self.interface.reinitialize_screen(name="Pilot's Seat")

    def focus(self) -> None:
        self.interface.camera_x = 0
        self.interface.camera_y = 0
        self.interface.reinitialize_screen(name="Pilot's Seat")

    def update_display(self) -> None:
        self._auto_pan()

        self.interface.camera_x = 0
        self.interface.camera_y = 0

        #TODO: would be great not to erase the screen on every tick
        self.viewscreen.erase()
        self._draw_radar()
        self.presenter.draw_sector_map()

        self.interface.refresh_viewscreen()

