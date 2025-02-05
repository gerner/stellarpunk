import logging
import curses
import math
import functools
import uuid
from typing import Any, Tuple, Sequence, Mapping, Optional, Callable, Collection

import numpy as np
import drawille # type: ignore

from stellarpunk import interface, util, core, config, intel
from stellarpunk.interface import command_input, starfield, sector as sector_interface

class UniverseView(interface.PerspectiveObserver, interface.View):
    def __init__(self, gamestate:core.Gamestate, *args:Any, **kwargs:Any):
        super().__init__(*args, **kwargs)

        self.gamestate = gamestate

        # perspective on the universe, zoomed in so the mean sector fits
        # comfortably in 80 characters
        self.perspective = interface.Perspective(
                self.interface.viewscreen,
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

        self.starfield = starfield.Starfield(self.gamestate.starfield, self.perspective)

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

    def add_sector_to_canvas(self, c:drawille.Canvas, sector:intel.SectorIntel) -> bool:
        used_canvas = False
        if sector.radius > 0 and self.perspective.meters_per_char[0] < sector.radius:
            used_canvas = True
            loc_x, loc_y = sector.loc
            util.make_circle_canvas(sector.radius, *self.perspective.meters_per_char, bbox=util.translate_rect(self.perspective.bbox, (-loc_x, -loc_y)), offset_x=loc_x, offset_y=loc_y, c=c)
        return used_canvas

    def add_sector_edges_to_canvas(self, c:drawille.Canvas, sector:intel.SectorIntel, universe_view:intel.UniverseView) -> bool:
        used_canvas = False

        # draw edges with neighbors
        sector_idx = universe_view.sector_idx_lookup[sector.intel_entity_id]
        for i, edge in enumerate(universe_view.adj_matrix[sector_idx]):
            # skip symmetric edges
            #if i <= sector_idx:
            #    continue
            # skip non-edges
            if not edge < np.inf:
                continue

            if i >= len(universe_view.sector_intels):
                # we must not have intel on the destination of this edge
                # we'll just draw a stub for this gate
                gate_intel = universe_view.gate_intel_lookup[(sector.intel_entity_id, universe_view.sector_id_lookup[i])]
                stub_start_loc = util.polar_to_cartesian(sector.radius, gate_intel.direction) + sector.loc
                stub_end_loc = util.polar_to_cartesian(sector.radius*15, gate_intel.direction) + sector.loc

                util.drawille_line(stub_start_loc, stub_end_loc, *self.perspective.meters_per_char, canvas=c, bbox=self.perspective.bbox)
            else:
                dest_sector = universe_view.sector_intels[i]
                # there's an edge between sector and sector at index i
                # draw a line from the edge of the sector to the edge of the other sector
                util.drawille_line(sector.loc, dest_sector.loc, *self.perspective.meters_per_char, canvas=c, bbox=self.perspective.bbox)
            used_canvas = True

        return used_canvas

    def compute_sector_layout(self) -> Tuple[Mapping[Tuple[int, int], str], Mapping[Tuple[int, int], str]]:
        c_sectors = drawille.Canvas()
        c_edges = drawille.Canvas()
        used_canvas_sectors = False
        used_canvas_edges = False

        assert self.gamestate.player.character

        # construct the player's view of the universe, depending on intel
        universe_view = intel.UniverseView.create(self.gamestate.player.character)

        for sector in universe_view.sector_intels:
            # compute a bounding box of interest for this sector
            # that's this sector (including radius) plus all sectors it connects to
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

            used_canvas_edges = self.add_sector_edges_to_canvas(c_edges, sector, universe_view) or used_canvas_edges

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
        sector_view = sector_interface.SectorView(self.selected_sector, self.gamestate, self.interface)
        self.interface.swap_view(sector_view, self)
        return sector_view

    def update_display(self) -> None:
        """ Draws a map of all sectors. """

        self.viewscreen.erase()

        self.starfield.draw_starfield(self.viewscreen)

        # draw the cached sector/edge geometry
        for (y,x), c in self._cached_sector_layout[1].items():
            self.viewscreen.addstr(y, x, c, curses.color_pair(interface.Icons.COLOR_UNIVERSE_EDGE))
        for (y,x), c in self._cached_sector_layout[0].items():
            self.viewscreen.addstr(y, x, c, curses.color_pair(interface.Icons.COLOR_UNIVERSE_SECTOR))

        # draw info for each sector
        assert self.gamestate.player.character
        for sector in self.gamestate.player.character.intel_manager.intel(intel.TrivialMatchCriteria(cls=intel.SectorIntel), intel.SectorIntel):
            # compute a bounding box of interest for this sector
            # only if the sector is actually on screen
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
            self.viewscreen.addstr(s_y, s_x+1, sector.intel_entity_short_id, name_attr)
            self.viewscreen.addstr(s_y+1, s_x+1, sector.intel_entity_name)
            self.viewscreen.addstr(s_y+2, s_x+1, sector.culture, interface.Icons.culture_attr(sector.culture))

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
            if target_id not in self.gamestate.sectors:
                raise command_input.UserError("{args[0]} not found among sectors")
            self.select_sector(self.gamestate.sectors[target_id])

        return [
            self.bind_command("target", target, util.tab_completer(map(str, self.gamestate.sectors.keys()))),
        ]

    def handle_mouse(self, m_id: int, m_x: int, m_y: int, m_z: int, bstate: int) -> bool:
        if not bstate & (curses.BUTTON1_CLICKED | curses.BUTTON1_PRESSED):
            return False

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
        return True

    def key_list(self) -> Collection[interface.KeyBinding]:
        def focus_target() -> None:
            if self.selected_sector is None:
                return
            elif not np.all(np.isclose(self.perspective.cursor, self.selected_sector.loc)):
                self.perspective.cursor = tuple(self.selected_sector.loc)
            else:
                self.open_sector_view(self.selected_sector)

        key_list = [
            self.bind_key(ord('w'), lambda: self.perspective.move_cursor(ord('w'))),
            self.bind_key(ord('a'), lambda: self.perspective.move_cursor(ord('a'))),
            self.bind_key(ord('s'), lambda: self.perspective.move_cursor(ord('s'))),
            self.bind_key(ord('d'), lambda: self.perspective.move_cursor(ord('d'))),
            self.bind_key(ord("+"), lambda: self.perspective.zoom_cursor(ord("+"))),
            self.bind_key(ord("-"), lambda: self.perspective.zoom_cursor(ord("-"))),
            self.bind_key(ord('\n'), focus_target),
            self.bind_key(ord('\r'), focus_target),
        ]

        return key_list

