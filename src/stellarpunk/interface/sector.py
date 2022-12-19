""" Interface methods dealing with displaying a single sector map. """

import logging
import math
import bisect
import curses
import curses.textpad
import curses.ascii
import time
import uuid
import re
from typing import Tuple, Optional, Any, Sequence, Dict, Tuple, List, Mapping, Callable, Union

import drawille # type: ignore
import numpy as np

from stellarpunk import util, core, interface, orders, effects, generate
from stellarpunk.interface import command_input, starfield, presenter, pilot as pilot_interface

class SectorView(interface.View, interface.PerspectiveObserver):
    """ Sector mode: interacting with the sector map.

    Draw the contents of the sector: ships, stations, asteroids, etc.
    Player can move around the sector, panning the camera at the edges of
    the viewscreen, search for named entities within the sector (jumping to
    each a la vim search.) Can select a ship to interact with. Can enter
    pilot mode. Can return to universe mode or enter command mode.
    """

    def __init__(self, sector:core.Sector, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        #TODO: this circular dependency feels odd
        #   need to know dimensions of the viewscreen (should pass in?)
        #   need to write messages outside the viewscreen (do we?)
        self.sector = sector

        # perspective on the sector, zoomed in so all of it fits comfortably
        # in 80 characters
        self.perspective = interface.Perspective(
            self.interface,
            zoom=self.sector.radius/80/2,
            min_zoom=(6*generate.Settings.SECTOR_RADIUS_STD+generate.Settings.SECTOR_RADIUS_MEAN)/80,
            max_zoom=25*8*generate.Settings.Ship.RADIUS/80.,
        )
        self.perspective.observe(self)

        # entity id of the currently selected target
        self.selected_target:Optional[uuid.UUID] = None
        self.selected_entity:Optional[core.SectorEntity] = None

        self.selected_character:Optional[core.Character] = None

        self.presenter = presenter.Presenter(self.interface.gamestate, self, self.sector, self.perspective)

        self._cached_grid:Tuple[util.NiceScale, util.NiceScale, util.NiceScale, util.NiceScale, Mapping[Tuple[int, int], str]] = (util.NiceScale(0,0), util.NiceScale(0,0), util.NiceScale(0,0), util.NiceScale(0,0), {})
        self.debug_entity = False
        self.debug_entity_vectors = False

        # the child view we spawn
        # if we receive focus, this should be dead
        self.pilot_view:Optional[pilot_interface.PilotView] = None

        self.starfield = starfield.Starfield(self.interface.gamestate.sector_starfield, self.perspective)

    def initialize(self) -> None:
        self.logger.info(f'entering sector mode for {self.sector.entity_id}')
        self.perspective.update_bbox()
        self.interface.reinitialize_screen(name=f'Sector Map of {self.sector.short_id()}')

    def focus(self) -> None:
        super().focus()
        self.active = True
        self.perspective.update_bbox()
        if self.pilot_view:
            if self.pilot_view.ship.sector and self.pilot_view.ship.sector != self.sector:
                self.logger.info(f'piloted ship in new sector, changing to view {self.pilot_view.ship.sector}')
                self.sector = self.pilot_view.ship.sector
                self.perspective.cursor = tuple(self.pilot_view.ship.loc)
            self.pilot_view = None
        self.interface.reinitialize_screen(name=f'Sector Map of {self.sector.short_id()}')

    def perspective_updated(self, perspective:interface.Perspective) -> None:
        self._compute_grid()
        self.presenter.sector = self.sector

    def select_target(self, target_id:Optional[uuid.UUID], entity:Optional[core.SectorEntity]) -> None:
        if target_id == self.selected_target:
            # no-op selecting the same target
            return
        self.selected_target = target_id
        self.selected_entity = entity

        self.presenter.selected_target = target_id

        self.logger.info(f'selected target {entity}')
        if entity:
            if isinstance(entity, core.Ship):
                self.interface.log_message(f'{entity.short_id()}: {entity.name} order: {entity.current_order()}')
                #TODO: display queued orders?
                #for order in list(entity.orders)[1:]:
                #    self.interface.log_message(f'queued: {order}')
            elif isinstance(entity, core.TravelGate):
                self.interface.log_message(f'{entity.short_id()}: {entity.name} direction: {entity.direction}')
            else:
                self.interface.log_message(f'{entity.short_id()}: {entity.name}')
            for i in range(entity.cargo.shape[0]):
                if entity.cargo[i] > 0.:
                    self.interface.log_message(f'cargo {i}: {entity.cargo[i]}')

    def _compute_grid(self, max_ticks:int=10) -> None:
        self._cached_grid = util.compute_uigrid(self.perspective.bbox, *self.perspective.meters_per_char, bounds=self.viewscreen_bounds, max_ticks=max_ticks)

    def draw_grid(self) -> None:
        """ Draws a grid at tick lines. """

        major_ticks_x, minor_ticks_y, major_ticks_y, minor_ticks_x, grid_content = self._cached_grid

        #for lineno, line in enumerate(text):
        #    self.viewscreen.addstr(lineno, 0, line, curses.color_pair(29))
        for (y,x), c in grid_content.items():
            self.viewscreen.viewscreen.addch(y, x, c, curses.color_pair(29))

        # draw location indicators
        i = major_ticks_x.niceMin
        while i <= major_ticks_x.niceMax:
            s_i, _ = self.perspective.sector_to_screen(i, 0)
            self.viewscreen.addstr(0, s_i, util.human_distance(i), curses.color_pair(29))
            i += major_ticks_x.tickSpacing
        j = major_ticks_y.niceMin
        while j <= major_ticks_y.niceMax:
            _, s_j = self.perspective.sector_to_screen(0, j)
            self.viewscreen.addstr(s_j, 0, util.human_distance(j), curses.color_pair(29))
            j += major_ticks_y.tickSpacing

        # add a scale near corner
        scale_label = f'scale {util.human_distance(major_ticks_x.tickSpacing)}'
        scale_x = self.interface.viewscreen_width - len(scale_label) - 2
        scale_y = self.interface.viewscreen_height - 2
        self.viewscreen.addstr(scale_y, scale_x, scale_label, curses.color_pair(29))

        # add center position near corner
        pos_label = f'({self.perspective.cursor[0]:.0f},{self.perspective.cursor[1]:.0f})'
        pos_x = self.interface.viewscreen_width - len(pos_label) - 2
        pos_y = self.interface.viewscreen_height - 1
        self.viewscreen.addstr(pos_y, pos_x, pos_label, curses.color_pair(29))

    def draw_sector_map(self) -> None:
        """ Draws a map of a sector. """

        self.presenter.draw_shapes()
        self.draw_grid()
        self.presenter.draw_sector_map()

    def _draw_target_info(self) -> None:
        if self.selected_entity is None:
            return

        info_width = 12 + 1 + 16
        status_x = self.interface.viewscreen_width - info_width
        status_y = 1

        self.viewscreen.addstr(status_y, status_x, "Target Info:")

        label_id ="id:"
        label_speed = "speed:"
        label_location = "location:"
        label_owner = "owner:"
        self.viewscreen.addstr(status_y+1, status_x, f'{label_id:>12} {self.selected_entity.short_id()}')
        self.viewscreen.addstr(status_y+2, status_x, f'{label_speed:>12} {self.selected_entity.speed:.0f}m/s')
        self.viewscreen.addstr(status_y+3, status_x, f'{label_location:>12} {self.selected_entity.loc[0]:.0f},{self.selected_entity.loc[1]:.0f}')

        if isinstance(self.selected_entity, core.Asset):
            self.viewscreen.addstr(status_y+4, status_x, f'{label_owner:>12} {self.selected_entity.owner}')

    def _draw_character_info(self) -> None:
        #TODO: not sure we want this at all, but it's a quick way to see some character info
        if self.selected_character is None:
            return

        info_x = 1
        info_y = 1

        lineno = 0
        for row in self.selected_character.portrait.text:
            self.viewscreen.addstr(info_y+lineno, info_x, row)
            lineno += 1

        self.viewscreen.addstr(info_y+lineno+1, info_x, self.selected_character.short_id())
        self.viewscreen.addstr(info_y+lineno+2, info_x, self.selected_character.name)
        self.viewscreen.addstr(info_y+lineno+3, info_x, f'located in: {self.selected_character.location.address_str()}')
        self.viewscreen.addstr(info_y+lineno+4, info_x, f'assets:')
        lineno += 5
        for asset in self.selected_character.assets:
            self.viewscreen.addstr(info_y+lineno, info_x+2, f'{asset.short_id()}')
            lineno+=1

        self.viewscreen.addstr(info_y+lineno, info_x, f'agenda:')
        lineno += 1
        for agendum in self.selected_character.agenda:
            self.viewscreen.addstr(info_y+lineno, info_x+2, f'{agendum}')
            lineno+=1

    def _draw_hud(self) -> None:
        self._draw_target_info()
        self._draw_character_info()

    def open_pilot_view(self, ship:core.Ship) -> pilot_interface.PilotView:
        self.pilot_view = pilot_interface.PilotView(ship, self.interface)
        self.interface.open_view(self.pilot_view)
        # suspend input until we get focus again
        self.active = False

        return self.pilot_view

    def update_display(self) -> None:
        self.viewscreen.erase()
        #TODO: the starfield will get stomped on by the radar unless we switch to a lines_to_dict style approach as in SectorView
        self.starfield.draw_starfield(self.viewscreen)
        self.draw_sector_map()
        self._draw_hud()
        self.interface.refresh_viewscreen()

    def command_list(self) -> Mapping[str, command_input.CommandInput.CommandSig]:
        def target(args:Sequence[str])->None:
            if not args:
                raise command_input.CommandInput.UserError("need a valid target")
            try:
                target_id = uuid.UUID(args[0])
            except ValueError:
                raise command_input.CommandInput.UserError("not a valid target id, try tab completion.")
            if target_id not in self.sector.entities:
                raise command_input.CommandInput.UserError("{args[0]} not found in sector")
            self.select_target(target_id, self.sector.entities[target_id])

        def goto(args:Sequence[str])->None:
            if not self.selected_entity or not isinstance(self.selected_entity, core.Ship):
                raise command_input.CommandInput.UserError(f'order only valid on a ship target')
            if len(args) < 2:
                x,y = self.perspective.cursor
            else:
                try:
                    x,y = int(args[0]), int(args[1])
                except Exception:
                    raise command_input.CommandInput.UserError("need two int args for x,y pos")
            self.selected_entity.clear_orders(self.interface.gamestate)
            order = orders.GoToLocation(np.array((x,y)), self.selected_entity, self.interface.gamestate)
            self.selected_entity.prepend_order(order)

        def wait(args:Sequence[str])->None:
            if not self.selected_entity or not isinstance(self.selected_entity, core.Ship):
                raise command_input.CommandInput.UserError(f'order only valid on a ship target')
            self.selected_entity.clear_orders(self.interface.gamestate)
            order = orders.WaitOrder(self.selected_entity, self.interface.gamestate)
            self.selected_entity.prepend_order(order)

        def debug_entity(args:Sequence[str])->None: self.debug_entity = not self.debug_entity
        def debug_vectors(args:Sequence[str])->None: self.debug_entity_vectors = not self.debug_entity_vectors
        def debug_write_history(args:Sequence[str])->None:
            if not self.selected_entity:
                raise command_input.CommandInput.UserError(f'can only write history for a selected target')
            filename = "/tmp/stellarpunk.history"
            self.logger.info(f'writing history for {self.selected_entity} to {filename}')
            core.write_history_to_file(self.selected_entity, filename, now=self.interface.gamestate.timestamp)

        def debug_write_sector(args:Sequence[str])->None:
            filename = "/tmp/stellarpunk.history.gz"
            self.logger.info(f'writing history for sector {self.sector.short_id()} to {filename}')
            core.write_history_to_file(self.sector, filename, now=self.interface.gamestate.timestamp)

        def spawn_ship(args:Sequence[str])->None:
            if len(args) < 2:
                x,y = self.perspective.cursor
            else:
                try:
                    x,y = int(args[0]), int(args[1])
                except Exception:
                    raise command_input.CommandInput.UserError("need two int args for x,y pos")

            self.interface.generator.spawn_ship(self.sector, x, y, v=np.array((0,0)), w=0)

        def spawn_collision(args:Sequence[str])->None:
            self.interface.generator.spawn_ship(self.sector, 0, 1100, v=np.array((0,0)), w=0)
            self.interface.generator.spawn_ship(self.sector, 0, 2200, v=np.array((0,0)), w=0)

        def spawn_resources(args:Sequence[str])->None:
            x,y = self.perspective.cursor
            self.interface.generator.spawn_resource_field(self.sector, x, y, 0, 1e6)

        def pilot(args:Sequence[str])->None:
            if not self.selected_entity or not isinstance(self.selected_entity, core.Ship):
                raise command_input.CommandInput.UserError(f'can only pilot a selected ship target')
            self.open_pilot_view(self.selected_entity)

        def chr_info(args:Sequence[str])->None:
            if not args:
                raise command_input.CommandInput.UserError("need a valid target")
            try:
                target_id = uuid.UUID(args[0])
            except ValueError:
                raise command_input.CommandInput.UserError("not a valid character id, try tab completion.")
            if target_id not in self.interface.gamestate.characters:
                raise command_input.CommandInput.UserError("{args[0]} not found")
            self.selected_character = self.interface.gamestate.characters[target_id]

        def scursor(args:Sequence[str])->None:
            if not args:
                raise command_input.CommandInput.UserError("need a valid target")
            try:
                m = re.match(r"(?P<x>[+-]?([0-9]*[.])?[0-9]+),(?P<y>[+-]?([0-9]*[.])?[0-9]+)", args[0])
                res = m.groupdict() # type: ignore
                x = float(res["x"])
                y = float(res["y"])
            except:
                raise command_input.CommandInput.UserError("not a valid coordinate")

            self.perspective.cursor = (x,y)

        return {
                "debug_entity": debug_entity,
                "debug_vectors": debug_vectors,
                "debug_write_history": debug_write_history,
                "debug_write_sector": debug_write_sector,
                "target": (target, util.tab_completer(map(str, self.sector.entities.keys()))),
                "spawn_ship": spawn_ship,
                "spawn_collision": spawn_collision,
                "spawn_resources": spawn_resources,
                "goto": goto,
                "wait": wait,
                "pilot": pilot,
                "chr_info": (chr_info, util.tab_completer(map(str, self.interface.gamestate.characters.keys()))),
                "scursor": scursor,
        }

    def handle_input(self, key:int, dt:float) -> bool:
        if key in (ord('w'), ord('a'), ord('s'), ord('d')):
            self.perspective.move_cursor(key)
        elif key in (ord("+"), ord("-")):
            self.perspective.zoom_cursor(key)
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
                self.perspective.cursor = tuple(self.sector.entities[self.selected_target].loc)
        elif key == ord(":"):
            self.interface.open_view(command_input.CommandInput(
                self.interface, commands=self.command_list()))
        elif key == curses.KEY_MOUSE:
            m_tuple = curses.getmouse()
            m_id, m_x, m_y, m_z, bstate = m_tuple
            sector_x, sector_y = self.perspective.screen_to_sector(m_x, m_y)

            # select a target within a cell of the mouse click
            bounds = (
                    sector_x-self.perspective.meters_per_char[0], sector_y-self.perspective.meters_per_char[1],
                    sector_x+self.perspective.meters_per_char[0], sector_y+self.perspective.meters_per_char[1]
            )
            hit = next(self.sector.spatial_query(bounds), None)
            if hit:
                #TODO: check if the hit is close enough
                self.select_target(hit.entity_id, hit)
        elif key == curses.ascii.ESC: #TODO: should handle escape here
            if self.selected_character is not None:
                self.selected_character = None
            elif self.selected_target is not None:
                self.select_target(None, None)
            else:
                return False

        return True
