""" Tools for generating and drawing a starfield. """

import curses
from typing import Sequence, Tuple

import numpy as np

from stellarpunk import util, interface

class Starfield:
    def __init__(self, starfield:Sequence[Tuple[Tuple[float, float], float]], perspective:interface.Perspective) -> None:
        # a set of layers corresponding to different zoom levels
        self.starfield = starfield
        self.perspective = perspective

    def draw_starfield(self, canvas:interface.Canvas) -> None:
        # draw the starfield with parallax

        # parallax basically means the background layer is more zoomed out
        zoom_factor = 1.5
        mpc_x = self.perspective.meters_per_char[0]*zoom_factor
        mpc_y = self.perspective.meters_per_char[1]*zoom_factor

        bbox_width = self.perspective.bbox[2]-self.perspective.bbox[0]
        bbox_height = self.perspective.bbox[3]-self.perspective.bbox[1]

        bbox = (
            self.perspective.cursor[0] - (bbox_width)/2*zoom_factor,
            self.perspective.cursor[1] - (bbox_height)/2*zoom_factor,
            self.perspective.cursor[0] + (bbox_width)/2*zoom_factor,
            self.perspective.cursor[1] + (bbox_height)/2*zoom_factor,
        )

        for (x,y), size in self.starfield:
            if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]:
                s_x, s_y = util.sector_to_screen(
                    x, y,
                    bbox[0], bbox[1],
                    mpc_x, mpc_y)

                if size > 0.5:
                    icon = interface.Icons.STAR_LARGE
                    attr = curses.color_pair(interface.Icons.COLOR_STAR_LARGE)
                else:
                    attr = curses.color_pair(interface.Icons.COLOR_STAR_SMALL)
                    if size < 0.167:
                        icon = interface.Icons.STAR_SMALL_ALTS[0]
                    elif size < 0.333:
                        icon = interface.Icons.STAR_SMALL_ALTS[1]
                    else:
                        icon = interface.Icons.STAR_SMALL_ALTS[2]

                canvas.viewscreen.addch(s_y, s_x, icon, attr)

