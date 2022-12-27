import logging
import curses
import math
import functools
import uuid
from typing import Any, Tuple, Sequence, Mapping, Optional, Callable, Collection

import numpy as np
import drawille # type: ignore

from stellarpunk import interface, util, core, config
from stellarpunk.interface import command_input, starfield, sector as sector_interface

class UniverseView(interface.View, interface.PerspectiveObserver):
    def __init__(self, gamestate:core.Gamestate, *args:Any, **kwargs:Any):
        super().__init__(*args, **kwargs)

        self.gamestate = gamestate

        # perspective on the universe, zoomed in so the mean sector fits
        # comfortably in 80 characters
        self.perspective = interface.Perspective(
                self.interface,
                zoom=config.Settings.generate.Universe.SECTOR_RADIUS_MEAN/80*16,
                min_zoom=config.Settings.generate.Universe.UNIVERSE_RADIUS/80.,
                max_zoom=config.Settings.generate.Universe.SECTOR_RADIUS_MEAN/80*8,
        )
        self.perspective.observe(self)

        #TODO: what's the right choice for selected sector?
        # start with max entities as the selected sector
        self.selected_sector:core.Sector = functools.reduce(lambda x, y: x if y is None or len(x.entities) > len(y.entities) else y, self.gamestate.sectors.values())

        #self._cached_sector_layout:Tuple[Sequence[str], Sequence[str]] = ([], [])
        self._cached_sector_layout:Tuple[Mapping[Tuple[int, int], str], Mapping[Tuple[int, int], str]] = ({}, {})

        self.starfield = starfield.Starfield(self.interface.gamestate.starfield, self.perspective)

    def initialize(self) -> None:
        self.logger.info(f'entering universe mode')
        self.perspective.update_bbox()
        self.interface.reinitialize_screen(name="Universe Map")

    def focus(self) -> None:
        super().focus()
        self.perspective.update_bbox()
        self.active = True
        self.interface.reinitialize_screen(name="Universe Map")

    def pan_camera(self) -> None:
        self.perspective.cursor = tuple(self.selected_sector.loc)

    def perspective_updated(self, perspective:interface.Perspective) -> None:
        self._cached_sector_layout = self.compute_sector_layout()

    def add_sector_to_canvas(self, c:drawille.Canvas, sector:core.Sector) -> bool:
        used_canvas = False
        if sector.radius > 0 and self.perspective.meters_per_char[0] < sector.radius:
            used_canvas = True

            # draw the sector as a circle
            r = sector.radius
            theta = 0.
            step = 1.5/r*self.perspective.meters_per_char[0]
            while theta < 2*math.pi:
                c_x, c_y = util.polar_to_cartesian(r, theta)
                d_x, d_y = util.sector_to_drawille(c_x+sector.loc[0], c_y+sector.loc[1], *self.perspective.meters_per_char)
                c.set(d_x, d_y)
                theta += step

        return used_canvas

    def add_sector_edges_to_canvas(self, c:drawille.Canvas, sector:core.Sector) -> bool:
        used_canvas = False
        #TODO: how should this be synchronized with the existence of gates between sectors?
        # draw edges with neighbors
        sector_idx = self.gamestate.sector_idx[sector.entity_id]
        for i, edge in enumerate(self.gamestate.sector_edges[sector_idx]):
            dest_sector = self.gamestate.sectors[self.gamestate.sector_ids[i]]
            # skip symmetric edges
            if i <= sector_idx:
                continue
            # skip non-edges
            elif edge == 0:
                continue
            # skip edges that are off screen
            subsegment = util.segment_intersects_rect(
                (sector.loc[0], sector.loc[1], dest_sector.loc[0], dest_sector.loc[1]),
                self.perspective.bbox
            )
            if subsegment is None:
                continue
            else:
                used_canvas = True
                # there's an edge between sector and sector at index i
                # draw a line from the edge of the sector to the edge of the other sector
                #TODO: start/end at the bounds of the rect instead of along the
                # entire edge. see util.segments_intersect where we compute the
                # intersection of two line segments

                r = 0.
                distance, theta = util.cartesian_to_polar(subsegment[0]-subsegment[2], subsegment[1]-subsegment[3])
                step = 2*self.perspective.meters_per_char[0]
                r += step
                while r < distance:
                    c_x, c_y = util.polar_to_cartesian(r, theta)
                    r += step
                    s_x, s_y = (c_x+subsegment[2], c_y+subsegment[3])
                    assert util.point_inside_rect((s_x, s_y), self.perspective.bbox)
                    #TODO: we could optimize this so we start/end at the
                    # boundaries of the sector circles
                    if util.magnitude(s_x-sector.loc[0], s_y-sector.loc[1]) < sector.radius:
                        continue

                    if util.magnitude(s_x-dest_sector.loc[0], s_y-dest_sector.loc[1]) < dest_sector.radius:
                        continue
                    d_x, d_y = util.sector_to_drawille(
                        s_x, s_y,
                        *self.perspective.meters_per_char
                    )
                    c.set(d_x, d_y)

        return used_canvas

    def compute_sector_layout(self) -> Tuple[Mapping[Tuple[int, int], str], Mapping[Tuple[int, int], str]]:
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

            if not util.intersects(self.perspective.bbox, sector_bbox):
                continue

            # order here matters because of short circuit evaluation!
            used_canvas_sectors = self.add_sector_to_canvas(c_sectors, sector) or used_canvas_sectors

            used_canvas_edges = self.add_sector_edges_to_canvas(c_edges, sector) or used_canvas_edges

        (d_min_x, d_min_y) = util.sector_to_drawille(
            self.perspective.bbox[0], self.perspective.bbox[1],
            self.perspective.meters_per_char[0], self.perspective.meters_per_char[1]
        )
        (d_max_x, d_max_y) = util.sector_to_drawille(
            self.perspective.bbox[2], self.perspective.bbox[3],
            self.perspective.meters_per_char[0], self.perspective.meters_per_char[1]
        )

        text_sectors = []
        if used_canvas_sectors:
            text_sectors = c_sectors.rows(d_min_x, d_min_y, d_max_x, d_max_y)

        text_edges = []
        if used_canvas_edges:
            text_edges = c_edges.rows(d_min_x, d_min_y, d_max_x, d_max_y)

        return util.lines_to_dict(text_sectors, bounds=self.viewscreen_bounds), util.lines_to_dict(text_edges, bounds=self.viewscreen_bounds)

    def open_sector_view(self, sector:core.Sector) -> sector_interface.SectorView:
        sector_view = sector_interface.SectorView(self.selected_sector, self.interface)
        self.interface.swap_view(sector_view, self)
        return sector_view

    def update_display(self) -> None:
        """ Draws a map of all sectors. """

        self.viewscreen.erase()

        self.starfield.draw_starfield(self.viewscreen)

        # draw the cached sector/edge geometry
        for (y,x), c in self._cached_sector_layout[1].items():
            self.viewscreen.window.addch(y, x, c, curses.color_pair(interface.Icons.COLOR_UNIVERSE_EDGE))
        for (y,x), c in self._cached_sector_layout[0].items():
            self.viewscreen.window.addch(y, x, c, curses.color_pair(interface.Icons.COLOR_UNIVERSE_SECTOR))

        # draw info for each sector
        for sector in self.gamestate.sectors.values():
            # compute a bounding box of interest for this sector
            # only if the sector is actually on screen
            self.gamestate.sector_edges[self.gamestate.sector_idx[sector.entity_id]]
            sector_bbox = (
                    sector.loc[0]-sector.radius,
                    sector.loc[1]-sector.radius,
                    sector.loc[0]+sector.radius,
                    sector.loc[1]+sector.radius,
            )

            if not util.intersects(self.perspective.bbox, sector_bbox):
                continue

            s_x, s_y = self.perspective.sector_to_screen(sector.loc[0], sector.loc[1])

            name_attr = 0
            if sector == self.selected_sector:
                name_attr = name_attr | curses.A_STANDOUT
            self.viewscreen.addstr(s_y, s_x, sector.short_id(), name_attr)
            self.viewscreen.addstr(s_y+1, s_x, f'[ {sector.loc[0]:.2e} {sector.loc[1]:.2e} ]')
            self.viewscreen.addstr(s_y+2, s_x, f'{len(sector.entities)} objects')

        self.interface.refresh_viewscreen()

    def select_sector(self, sector:core.Sector, focus:bool=False) -> None:
        self.selected_sector = sector
        if focus:
            self.perspective.cursor = tuple(self.selected_sector.loc)

    def command_list(self) -> Collection[interface.CommandBinding]:
        def target(args:Sequence[str])->None:
            if not args:
                raise command_input.UserError("need a valid target")
            try:
                target_id = uuid.UUID(args[0])
            except ValueError:
                raise command_input.UserError("not a valid target id, try tab completion.")
            if target_id not in self.interface.gamestate.sectors:
                raise command_input.UserError("{args[0]} not found among sectors")
            self.select_sector(self.interface.gamestate.sectors[target_id])

        def debug_collision(args:Sequence[str])->None:
            if len(self.interface.collisions) == 0:
                raise command_input.UserError("no collisions to debug")

            collision = self.interface.collisions[-1]
            if isinstance(collision[0], core.Ship):
                ship = collision[0]
            elif isinstance(collision[1], core.Ship):
                ship = collision[1]
            else:
                raise Exception("expected one of colliding objects to be a ship")

            assert ship.sector
            self.open_sector_view(ship.sector).select_target(ship.entity_id, ship)
        return [
            self.bind_command("debug_collision", debug_collision),
            self.bind_command("target", target, util.tab_completer(map(str, self.interface.gamestate.sectors.keys()))),
        ]

    def key_list(self) -> Collection[interface.KeyBinding]:
        def focus_target() -> None:
            if self.selected_sector is None:
                return
            elif not np.all(np.isclose(self.perspective.cursor, self.selected_sector.loc)):
                self.perspective.cursor = tuple(self.selected_sector.loc)
            else:
                self.open_sector_view(self.selected_sector)

        def handle_mouse() -> None:
            m_tuple = curses.getmouse()
            m_id, m_x, m_y, m_z, bstate = m_tuple
            sector_x, sector_y = self.perspective.screen_to_sector(m_x, m_y)

            # select a target within a cell of the mouse click
            bounds = (
                    sector_x-self.perspective.meters_per_char[0], sector_y-self.perspective.meters_per_char[1],
                    sector_x+self.perspective.meters_per_char[0], sector_y+self.perspective.meters_per_char[1]
            )
            hit = next(self.gamestate.spatial_query(bounds), None)
            if hit:
                #TODO: check if the hit is close enough
                self.selected_sector = self.gamestate.sectors[hit]

        key_list = [
            self.bind_key(ord('w'), lambda: self.perspective.move_cursor(ord('w'))),
            self.bind_key(ord('a'), lambda: self.perspective.move_cursor(ord('a'))),
            self.bind_key(ord('s'), lambda: self.perspective.move_cursor(ord('s'))),
            self.bind_key(ord('d'), lambda: self.perspective.move_cursor(ord('d'))),
            self.bind_key(ord("+"), lambda: self.perspective.zoom_cursor(ord("+"))),
            self.bind_key(ord("-"), lambda: self.perspective.zoom_cursor(ord("-"))),
            self.bind_key(ord('\n'), focus_target),
            self.bind_key(ord('\r'), focus_target),
            self.bind_key(curses.KEY_MOUSE, handle_mouse),
        ]

        return key_list

