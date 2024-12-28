""" Tools for generating and drawing a starfield. """

import curses
from typing import Sequence, Tuple, Mapping, Dict, List, Any
import bisect
import logging
import math
import itertools

import numpy as np

from stellarpunk import util, interface, core

class Starfield(interface.PerspectiveObserver):
    def __init__(self, starfields:Sequence[core.StarfieldLayer], perspective:interface.Perspective, *args:Any, zoom_step:float=1.4, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(util.fullname(self))
        # a set of layers corresponding to different zoom levels
        self.starfields = starfields
        self.perspective = perspective
        self.zoom_step = zoom_step
        self.perspective.observe(self)

        # stars to draw to screen as screen coords -> icon, attr
        self._cached_star_layout:Mapping[Tuple[int,int],Tuple[str, int, int]]

    def perspective_updated(self, perspective:interface.Perspective) -> None:
        self._cached_star_layout = self.compute_star_layout()

    def _compute_star(self, loc:Tuple[float, float], size:float, color:int, bbox:Tuple[float, float, float, float], mpc:Tuple[float, float]) -> Tuple[Tuple[int,int], Tuple[str, int, int]]:
        s_x, s_y = util.sector_to_screen(
            loc[0], loc[1],
            bbox[0], bbox[1],
            mpc[0], mpc[1])

        attr = curses.A_DIM
        if size > 0.5:
            icon = interface.Icons.STAR_LARGE
        else:
            if size < 0.167:
                icon = interface.Icons.STAR_SMALL_ALTS[0]
            elif size < 0.333:
                icon = interface.Icons.STAR_SMALL_ALTS[1]
            else:
                icon = interface.Icons.STAR_SMALL_ALTS[2]

        return (s_y, s_x), (icon, attr, interface.Icons.COLOR_STAR_ALTS[color])

    def _compute_one_starfield(self, starfield:core.StarfieldLayer, conversion_factor:float, zoom_factor:float) -> Dict[Tuple[int,int],Tuple[str, int, int]]:
        # translate the starfield layer into coordinates as if it were at the
        # perspective's level

        computed_layout:Dict[Tuple[int,int],Tuple[str, int, int]] = {}

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

        x_tile_width = (starfield.bbox[2]-starfield.bbox[0])*conversion_factor
        y_tile_width = (starfield.bbox[3]-starfield.bbox[1])*conversion_factor

        x_tiles_min = math.floor((bbox[0] - starfield.bbox[0]*conversion_factor)/x_tile_width)
        x_tiles_max = math.floor((bbox[2] - starfield.bbox[0]*conversion_factor)/x_tile_width)
        y_tiles_min = math.floor((bbox[1] - starfield.bbox[1]*conversion_factor)/y_tile_width)
        y_tiles_max = math.floor((bbox[3] - starfield.bbox[1]*conversion_factor)/y_tile_width)

        tiles = list(itertools.product(range(x_tiles_min, x_tiles_max+1), range(y_tiles_min, y_tiles_max+1)))

        for (x,y), size, color in starfield._star_list:

            # convert the x,y coordinates of the star into the virtual layer
            # coordinates (scaling and tiling)
            for i,j in tiles:
                x_tiled = x * conversion_factor + i*x_tile_width
                y_tiled = y * conversion_factor + j*y_tile_width
                if bbox[0] < x_tiled < bbox[2] and bbox[1] < y_tiled < bbox[3]:
                    k,v = self._compute_star((x_tiled,y_tiled), size/zoom_factor, color, bbox, (mpc_x, mpc_y))
                    computed_layout[k] = v

        return computed_layout

    def compute_star_layout(self) -> Mapping[Tuple[int,int],Tuple[str, int, int]]:

        # draw the starfield with parallax

        # parallax basically means the background layer is more zoomed out

        # model is that we have an unlimited number of layers
        # as we zoom we're always going to be looking at two and zooming into
        # them
        # in reality we only have two fields which we rescale and tile
        # as we get too close for the upper layer, we'll rescale and swap it to
        # the back

        # map the perspective's zoom level to some discrete stops
        # this effectively indexes into the layer space

        zoom_index = int(math.log(self.perspective.zoom/self.perspective.min_zoom)/math.log(0.6))
        zoom_stop = self.perspective.min_zoom*0.6**zoom_index

        # swap starfields on consecutive zoom levels
        starfield_A = self.starfields[zoom_index % 2]
        starfield_B = self.starfields[(zoom_index + 1) % 2]

        #TODO: maintain consistency when zooming through layers
        # when we zoom "through" the starfields we want to maintain consistency
        # of the stars, so if we push the "front" layer to the "back" all the
        # stars in the front should still be visible in the back at their same
        # positions (even if they have a different size). This won't be the
        # case because of the zoom_factor parameter below I think.
        # if we don't do this then all stars will be shuffled whenever we swap zooms

        computed_layout = {}
        computed_layout.update(
            self._compute_one_starfield(
                starfield_A,
                conversion_factor=zoom_stop/starfield_A.zoom,
                zoom_factor=self.zoom_step,
            )
        )
        computed_layout.update(
            self._compute_one_starfield(
                starfield_B,
                conversion_factor=zoom_stop/starfield_B.zoom,
                zoom_factor=self.zoom_step*self.zoom_step,
            )
        )

        return computed_layout

    def draw_starfield(self, canvas:interface.BasicCanvas) -> None:
        for (y,x), (s, a, c) in self._cached_star_layout.items():
            canvas.addstr(y, x, s, a | curses.color_pair(c))

    def draw_starfield_to_sprite(self, width:int, height:int, x_start:int=0, y_start:int=0) -> core.Sprite:
        text = [[" "]*width for _ in range(height)]
        attr:Dict[Tuple[int,int], Tuple[int, int]] = {}
        for (x,y) in itertools.product(range(x_start, x_start+width), range(y_start, y_start+height)):
            if (x,y) in self._cached_star_layout:
                s,a,c = self._cached_star_layout[x,y]
                text[y-y_start][x-x_start] = s
                attr[x-x_start,y-y_start] = (a,c)

        return core.Sprite("ephemeral_starfield", ["".join(t) for t in text], attr)
