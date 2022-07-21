import logging
import curses
import math
from typing import Any, Tuple, Sequence

import numpy as np
import drawille # type: ignore

from stellarpunk import interface, util, core
from stellarpunk.interface import command_input, sector as sector_interface

class UniverseView(interface.View):
    def __init__(self, gamestate:core.Gamestate, *args:Any, **kwargs:Any):
        super().__init__(*args, **kwargs)

        self.gamestate = gamestate

        # position of the universe sector cursor (in universe-sector coords)
        self.ucursor_x = 0.
        self.ucursor_y = 0.

        self.selected_sector:core.Sector = None # type: ignore[assignment]

        # universe zoom level, expressed in meters to fit on screen
        # this is four times the default sector zoom level, so one sector will
        # fit nicely on screen
        self.uzoom = 20 * 1e5 * 2
        self.meters_per_char_x = 0.
        self.meters_per_char_y = 0.

        # coord bounding box on screen (ul_x, ul_y, lr_x, lr_y)
        self.bbox = (0.,0.,0.,0.)

        self._cached_sector_layout:Tuple[Sequence[str], Sequence[str]] = ([], [])

    def initialize(self) -> None:
        self.logger.info(f'entering universe mode')

        self.selected_sector = next(iter(self.gamestate.sectors.values()))
        self.pan_camera()
        self.update_bbox()
        self.interface.reinitialize_screen(name="Universe Map")

    def focus(self) -> None:
        super().focus()
        self.active = True
        self.interface.reinitialize_screen(name="Universe Map")

    def meters_per_char(self) -> Tuple[float, float]:
        meters_per_char_x = self.uzoom / min(self.interface.viewscreen_width, math.floor(self.interface.viewscreen_height/self.interface.font_width*self.interface.font_height))
        meters_per_char_y = meters_per_char_x / self.interface.font_width * self.interface.font_height

        assert self.uzoom / meters_per_char_y <= self.interface.viewscreen_height
        assert self.uzoom / meters_per_char_x <= self.interface.viewscreen_width

        return (meters_per_char_x, meters_per_char_y)

    def update_bbox(self) -> None:
        self.meters_per_char_x, self.meters_per_char_y = self.meters_per_char()

        vsw = self.interface.viewscreen_width
        vsh = self.interface.viewscreen_height

        ul_x = self.ucursor_x - (vsw/2 * self.meters_per_char_x)
        ul_y = self.ucursor_y - (vsh/2 * self.meters_per_char_y)
        lr_x = self.ucursor_x + (vsw/2 * self.meters_per_char_x)
        lr_y = self.ucursor_y + (vsh/2 * self.meters_per_char_y)

        self.bbox = (ul_x, ul_y, lr_x, lr_y)

        self._cached_sector_layout = self.compute_sector_layout()

    def pan_camera(self) -> None:
        self.ucursor_x, self.ucursor_y = self.selected_sector.loc

    def move_ucursor(self, direction:int) -> None:
        old_x = self.ucursor_x
        old_y = self.ucursor_y

        stepsize = self.uzoom/32.

        if direction == ord('w'):
            self.ucursor_y -= stepsize
        elif direction == ord('a'):
            self.ucursor_x -= stepsize
        elif direction == ord('s'):
            self.ucursor_y += stepsize
        elif direction == ord('d'):
            self.ucursor_x += stepsize
        else:
            raise ValueError(f'unknown direction {direction}')

        self.update_bbox()

    def zoom_ucursor(self, direction:int) -> None:
        if direction == ord('+'):
            self.uzoom *= 0.9
        elif direction == ord('-'):
            self.uzoom *= 1.1
        else:
            raise ValueError(f'unknown direction {direction}')

        self.update_bbox()

    def draw_umap_sector(self, y:int, x:int, sector:core.Sector) -> None:
        """ Draws a single sector to viewscreen starting at position (y,x) """

        #textpad.rectangle(self.viewscreen.viewscreen, y, x, y+interface.Settings.UMAP_SECTOR_HEIGHT-1, x+interface.Settings.UMAP_SECTOR_WIDTH-1)

        if np.all((self.ucursor_x, self.ucursor_y) == sector.loc):
            self.viewscreen.addstr(y+1,x+1, sector.short_id(), curses.A_STANDOUT)
        else:
            self.viewscreen.addstr(y+1,x+1, sector.short_id())

        self.viewscreen.addstr(y+2,x+1, sector.name)

        for resource, asteroids in sector.asteroids.items():
            amount = sum(map(lambda x: x.cargo[x.resource], asteroids))
            icon = interface.Icons.ASTEROID
            icon_attr = curses.color_pair(interface.Icons.RESOURCE_COLORS[resource])
            self.viewscreen.addstr(y+3+resource, x+2, f'{icon} {amount:.2e}', icon_attr)

        self.viewscreen.addstr(y+interface.Settings.UMAP_SECTOR_HEIGHT-2, x+1, f'{len(sector.entities)} objects')

    def add_sector_to_canvas(self, c:drawille.Canvas, sector:core.Sector) -> bool:
        used_canvas = False
        if sector.radius > 0 and self.meters_per_char_x < sector.radius:
            used_canvas = True

            # draw the sector as a circle
            r = sector.radius
            theta = 0.
            step = 1.5/r*self.meters_per_char_x
            while theta < 2*math.pi:
                c_x, c_y = util.polar_to_cartesian(r, theta)
                d_x, d_y = util.sector_to_drawille(c_x+sector.loc[0], c_y+sector.loc[1], self.meters_per_char_x, self.meters_per_char_y)
                c.set(d_x, d_y)
                theta += step

        return used_canvas

    def add_sector_edges_to_canvas(self, c:drawille.Canvas, sector:core.Sector) -> bool:
        used_canvas = False
        # draw edges with neighbors
        sector_idx = self.gamestate.sector_idx[sector.entity_id]
        for i, edge in enumerate(self.gamestate.sector_edges[sector_idx]):
            dest_sector = self.gamestate.sectors[self.gamestate.sector_ids[i]]
            # skip symmetric edges
            if i <= sector_idx:
                continue
            if edge > 0:
                used_canvas = True
                # there's an edge between sector and sector at index i
                # draw a line from the edge of the sector to the edge of the other sector
                r = sector.radius
                distance, theta = util.cartesian_to_polar(*(dest_sector.loc - sector.loc))
                step = 2*self.meters_per_char_x
                r += step
                while r < distance - dest_sector.radius:
                    c_x, c_y = util.polar_to_cartesian(r, theta)
                    d_x, d_y = util.sector_to_drawille(c_x+sector.loc[0], c_y+sector.loc[1], self.meters_per_char_x, self.meters_per_char_y)
                    c.set(d_x, d_y)
                    r += step

        return used_canvas

    def compute_sector_layout(self) -> Tuple[Sequence[str], Sequence[str]]:
        c_sectors = drawille.Canvas()
        c_edges = drawille.Canvas()
        used_canvas_sectors = False
        used_canvas_edges = False

        for sector in self.gamestate.sectors.values():
            # compute a bounding box of interest for this sector
            # that's this sector (including radius) plus all sectors it connects to
            self.gamestate.sector_edges[self.gamestate.sector_idx[sector.entity_id]]
            sector_bbox = (
                    sector.loc[0]-sector.radius-self.gamestate.max_edge_length,
                    sector.loc[1]-sector.radius-self.gamestate.max_edge_length,
                    sector.loc[0]+sector.radius+self.gamestate.max_edge_length,
                    sector.loc[1]+sector.radius+self.gamestate.max_edge_length
            )

            if not util.intersects(self.bbox, sector_bbox):
                continue

            # order here matters because of short circuit evaluation!
            used_canvas_sectors = self.add_sector_to_canvas(c_sectors, sector) or used_canvas_sectors

            used_canvas_edges = self.add_sector_edges_to_canvas(c_edges, sector) or used_canvas_edges

        text_sectors = []
        if used_canvas_sectors:
            (d_x, d_y) = util.sector_to_drawille(
                self.bbox[0], self.bbox[1],
                self.meters_per_char_x, self.meters_per_char_y)
            text_sectors = c_sectors.rows(d_x, d_y)

        text_edges = []
        if used_canvas_edges:
            (d_x, d_y) = util.sector_to_drawille(
                self.bbox[0], self.bbox[1],
                self.meters_per_char_x, self.meters_per_char_y)
            text_edges = c_edges.rows(d_x, d_y)

        return text_sectors, text_edges

    def update_display(self) -> None:
        """ Draws a map of all sectors. """

        self.viewscreen.erase()

        #TODO: these draw_line calls are slow, we should figure out how we can speed things up
        for lineno, line in enumerate(self._cached_sector_layout[1]):
            util.draw_line(lineno, 0, line, self.viewscreen.viewscreen, curses.color_pair(interface.Icons.COLOR_UNIVERSE_EDGE), bounds=self.viewscreen_bounds)

        for lineno, line in enumerate(self._cached_sector_layout[0]):
            util.draw_line(lineno, 0, line, self.viewscreen.viewscreen, curses.color_pair(interface.Icons.COLOR_UNIVERSE_SECTOR), bounds=self.viewscreen_bounds)

        for sector in self.gamestate.sectors.values():
            # compute a bounding box of interest for this sector
            # that's this sector (including radius) plus all sectors it connects to
            self.gamestate.sector_edges[self.gamestate.sector_idx[sector.entity_id]]
            sector_bbox = (
                    sector.loc[0]-sector.radius-self.gamestate.max_edge_length,
                    sector.loc[1]-sector.radius-self.gamestate.max_edge_length,
                    sector.loc[0]+sector.radius+self.gamestate.max_edge_length,
                    sector.loc[1]+sector.radius+self.gamestate.max_edge_length
            )

            if not util.intersects(self.bbox, sector_bbox):
                continue

            s_x, s_y = util.sector_to_screen(
                    sector.loc[0], sector.loc[1],
                    self.bbox[0], self.bbox[1],
                    self.meters_per_char_x, self.meters_per_char_y)

            self.viewscreen.addstr(s_y, s_x, sector.short_id(), 0)

        self.interface.refresh_viewscreen()

    def handle_input(self, key:int, dt:float) -> bool:
        if key in (ord('w'), ord('a'), ord('s'), ord('d')):
            self.move_ucursor(key)
        elif key in (ord("+"), ord("-")):
            self.zoom_ucursor(key)
        elif key in (ord('\n'), ord('\r')):
            sector = next(sector for sector in self.gamestate.sectors.values() if np.all(sector.loc == (self.ucursor_x, self.ucursor_y)))
            sector_view = sector_interface.SectorView(
                    sector, self.interface)
            self.interface.open_view(sector_view)
            # suspend input until we get focus again
            self.active = False
        elif key == ord(":"):
            ci = command_input.CommandInput(self.interface)
            self.interface.open_view(ci)

        return True

