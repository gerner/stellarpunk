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
from typing import Tuple, Optional, Any, Sequence, Dict, Tuple, List, Mapping, Callable, Union, Collection

import drawille # type: ignore
import numpy as np

from stellarpunk import util, core, interface, orders, effects, config
from stellarpunk.core import sector_entity
from stellarpunk.interface import command_input, starfield, presenter, pilot as pilot_interface

class SectorView(interface.PerspectiveObserver, core.SectorEntityObserver, interface.GameView):
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
            self.interface.viewscreen,
            zoom=self.sector.radius/80/2,
            min_zoom=(6*config.Settings.generate.Universe.SECTOR_RADIUS_STD+config.Settings.generate.Universe.SECTOR_RADIUS_MEAN)/80,
            max_zoom=25*8*config.Settings.generate.SectorEntities.ship.RADIUS/80.,
        )
        self.perspective.observe(self)

        # entity id of the currently selected target
        self.selected_target:Optional[uuid.UUID] = None
        self.selected_entity:Optional[core.SectorEntity] = None

        self.selected_character:Optional[core.Character] = None

        self.presenter = presenter.SectorPresenter(self.gamestate, self, self.sector, self.perspective)

        self._cached_grid:Tuple[util.NiceScale, util.NiceScale, util.NiceScale, util.NiceScale, Mapping[Tuple[int, int], str]] = (util.NiceScale(0,0), util.NiceScale(0,0), util.NiceScale(0,0), util.NiceScale(0,0), {})
        self.debug_entity = False
        self.debug_entity_vectors = False

        # the child view we spawn
        # if we receive focus, this should be dead
        self.pilot_view:Optional[pilot_interface.PilotView] = None

        self.starfield = starfield.Starfield(self.gamestate.sector_starfield, self.perspective)

    def initialize(self) -> None:
        self.logger.info(f'entering sector mode for {self.sector.entity_id}')
        self.perspective.update_bbox()
        self.interface.reinitialize_screen(name=f'Sector Map of {self.sector.short_id()}')

    def terminate(self) -> None:
        self.presenter.terminate()

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

    def select_target(self, target_id:Optional[uuid.UUID], entity:Optional[core.SectorEntity], focus:bool=False) -> None:
        if target_id == self.selected_target:
            # no-op selecting the same target
            return
        if self.selected_entity:
            self.selected_entity.unobserve(self)
        self.selected_target = target_id
        self.selected_entity = entity

        self.presenter.selected_target = target_id

        self.logger.info(f'selected target {entity}')
        if entity:
            entity.observe(self)
            if isinstance(entity, core.Ship):
                self.interface.log_message(f'{entity.short_id()}: {entity.name} order: {entity.current_order()}')
                #TODO: display queued orders?
                #for order in list(entity.orders)[1:]:
                #    self.interface.log_message(f'queued: {order}')
            elif isinstance(entity, sector_entity.TravelGate):
                self.interface.log_message(f'{entity.short_id()}: {entity.name} direction: {entity.direction}')
            else:
                self.interface.log_message(f'{entity.short_id()}: {entity.name}')
            for i in range(entity.cargo.shape[0]):
                if entity.cargo[i] > 0.:
                    self.interface.log_message(f'cargo {i}: {entity.cargo[i]}')

        if focus:
            self.focus_target()

    # core.SectorEntityObserver
    @property
    def observer_id(self) -> uuid.UUID:
        return core.OBSERVER_ID_NULL

    def entity_destroyed(self, entity:core.SectorEntity) -> None:
        if entity == self.selected_entity:
            self.interface.log_message(f'target destroyed')
            self.select_target(None, None)

    def entity_migrated(self, entity:core.SectorEntity, from_sector:core.Sector, to_sector:core.Sector) -> None:
        if entity == self.selected_entity and to_sector != self.sector:
            self.interface.log_message(f'target left sector')
            self.select_target(None, None)

    def _compute_grid(self, max_ticks:int=10) -> None:
        self._cached_grid = util.compute_uigrid(self.perspective.bbox, self.perspective.meters_per_char, bounds=self.viewscreen_bounds, max_ticks=max_ticks)

    def draw_grid(self) -> None:
        """ Draws a grid at tick lines. """

        major_ticks_x, minor_ticks_y, major_ticks_y, minor_ticks_x, grid_content = self._cached_grid

        #for lineno, line in enumerate(text):
        #    self.viewscreen.addstr(lineno, 0, line, curses.color_pair(29))
        for (y,x), c in grid_content.items():
            self.viewscreen.addstr(y, x, c, curses.color_pair(29))

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
        scale_x = self.interface.viewscreen.width - len(scale_label) - 2
        scale_y = self.interface.viewscreen.height - 2
        self.viewscreen.addstr(scale_y, scale_x, scale_label, curses.color_pair(29))

        # add center position near corner
        pos_label = f'({self.perspective.cursor[0]:.0f},{self.perspective.cursor[1]:.0f})'
        pos_x = self.interface.viewscreen.width - len(pos_label) - 2
        pos_y = self.interface.viewscreen.height - 1
        self.viewscreen.addstr(pos_y, pos_x, pos_label, curses.color_pair(29))

    def draw_sector_map(self) -> None:
        """ Draws a map of a sector. """

        self.presenter.draw_weather()
        self.presenter.draw_shapes()
        self.draw_grid()
        if self.selected_entity:
            self.presenter.draw_sensor_rings(self.selected_entity)
            self.presenter.draw_profile_rings(self.selected_entity)
        self.presenter.draw_sector_map()

    def _draw_target_info(self) -> None:
        if self.selected_entity is None:
            return

        info_width = 12 + 1 + 16
        status_x = self.interface.viewscreen.width - info_width
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

        assert self.selected_character.location

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
        self.pilot_view = pilot_interface.PilotView(ship, self.gamestate, self.interface)
        self.interface.close_view(self)
        self.interface.open_view(self.pilot_view)
        return self.pilot_view

    def update_display(self) -> None:
        self.viewscreen.erase()
        #TODO: the starfield will get stomped on by the radar unless we switch to a lines_to_dict style approach as in SectorView
        self.starfield.draw_starfield(self.viewscreen)
        self.draw_sector_map()
        self._draw_hud()
        self.interface.refresh_viewscreen()

    def command_list(self) -> Collection[interface.CommandBinding]:
        def target(args:Sequence[str])->None:
            if not args:
                raise command_input.UserError("need a valid target")
            try:
                target_id = uuid.UUID(args[0])
            except ValueError:
                raise command_input.UserError("not a valid target id, try tab completion.")
            if target_id not in self.sector.entities:
                raise command_input.UserError("{args[0]} not found in sector")
            self.select_target(target_id, self.sector.entities[target_id])

        def goto(args:Sequence[str])->None:
            if not self.selected_entity or not isinstance(self.selected_entity, core.Ship):
                raise command_input.UserError(f'order only valid on a ship target')
            if len(args) < 2:
                x,y = self.perspective.cursor
            else:
                try:
                    x,y = int(args[0]), int(args[1])
                except Exception:
                    raise command_input.UserError("need two int args for x,y pos")
            self.selected_entity.clear_orders()
            order = orders.GoToLocation.create_go_to_location(np.array((x,y)), self.selected_entity, self.gamestate)
            self.selected_entity.prepend_order(order)

        def wait(args:Sequence[str])->None:
            if not self.selected_entity or not isinstance(self.selected_entity, core.Ship):
                raise command_input.UserError(f'order only valid on a ship target')
            self.selected_entity.clear_orders()
            order = orders.WaitOrder.create_wait_order(self.selected_entity, self.gamestate)
            self.selected_entity.prepend_order(order)

        def debug_entity(args:Sequence[str])->None: self.debug_entity = not self.debug_entity
        def debug_vectors(args:Sequence[str])->None: self.debug_entity_vectors = not self.debug_entity_vectors
        def debug_write_history(args:Sequence[str])->None:
            if not self.selected_entity:
                raise command_input.UserError(f'can only write history for a selected target')
            filename = "/tmp/stellarpunk.history"
            self.logger.info(f'writing history for {self.selected_entity} to {filename}')
            core.write_history_to_file(self.selected_entity, filename, now=self.gamestate.timestamp)

        def debug_write_sector(args:Sequence[str])->None:
            filename = "/tmp/stellarpunk.history.gz"
            self.logger.info(f'writing history for sector {self.sector.short_id()} to {filename}')
            core.write_history_to_file(self.sector, filename, now=self.gamestate.timestamp)

        def spawn_ship(args:Sequence[str])->None:
            if len(args) < 2:
                x,y = self.perspective.cursor
            else:
                try:
                    x,y = int(args[0]), int(args[1])
                except Exception:
                    raise command_input.UserError("need two int args for x,y pos")

            self.interface.generator.spawn_ship(self.sector, x, y, v=np.array((0,0)), w=0)

        def spawn_collision(args:Sequence[str])->None:
            pass
            self.interface.generator.spawn_ship(self.sector, 0, 1100, v=np.array((0,0)), w=0)
            self.interface.generator.spawn_ship(self.sector, 0, 2200, v=np.array((0,0)), w=0)

        def spawn_resources(args:Sequence[str])->None:
            x,y = self.perspective.cursor
            self.interface.generator.spawn_resource_field(self.sector, x, y, 0, 1e6)

        def pilot(args:Sequence[str])->None:
            if not self.selected_entity or not isinstance(self.selected_entity, core.Ship):
                raise command_input.UserError(f'can only pilot a selected ship target')
            self.open_pilot_view(self.selected_entity)

        def chr_info(args:Sequence[str])->None:
            if not args:
                raise command_input.UserError("need a valid target")
            try:
                target_id = uuid.UUID(args[0])
            except ValueError:
                raise command_input.UserError("not a valid character id, try tab completion.")
            if target_id not in self.gamestate.characters:
                raise command_input.UserError("{args[0]} not found")
            self.selected_character = self.gamestate.characters[target_id]

        def scursor(args:Sequence[str])->None:
            if not args:
                raise command_input.UserError("need a valid target")
            try:
                m = re.match(r"(?P<x>[+-]?([0-9]*[.])?[0-9]+),(?P<y>[+-]?([0-9]*[.])?[0-9]+)", args[0])
                res = m.groupdict() # type: ignore
                x = float(res["x"])
                y = float(res["y"])
            except:
                raise command_input.UserError("not a valid coordinate")

            self.perspective.cursor = (x,y)

        def show_orders(args:Sequence[str])->None:
            if not self.selected_entity or not isinstance(self.selected_entity, core.Ship):
                raise command_input.UserError(f'can only pilot a selected ship target')
            for o in self.selected_entity._orders:
                self.interface.log_message(f'{o}')

        return [
            self.bind_command("debug_entity", debug_entity),
            self.bind_command("debug_vectors", debug_vectors),
            self.bind_command("debug_write_history", debug_write_history),
            self.bind_command("debug_write_sector", debug_write_sector),
            self.bind_command("debug_pilot", pilot),
            self.bind_command("spawn_ship", spawn_ship),
            self.bind_command("spawn_collision", spawn_collision),
            self.bind_command("spawn_resources", spawn_resources),
            self.bind_command("orders", show_orders),
            self.bind_command("goto", goto),
            self.bind_command("wait", wait),

            self.bind_command("target", target, util.tab_completer(map(str, self.sector.entities.keys()))),
            self.bind_command("chr_info", chr_info, util.tab_completer(map(str, self.gamestate.characters.keys()))),
            self.bind_command("cursor", scursor),
        ]

    def _change_target(self, direction:int) -> None:
        entity_id_list = sorted(self.sector.entities.keys())
        if len(entity_id_list) == 0:
            self.select_target(None, None)
        elif self.selected_target is None:
            self.select_target(entity_id_list[0], self.sector.entities[entity_id_list[0]])
        else:
            if direction >= 0:
                next_index = bisect.bisect_right(entity_id_list, self.selected_target)
            else:
                next_index = bisect.bisect_left(entity_id_list, self.selected_target)
                if entity_id_list[next_index] == self.selected_target:
                    next_index -= 1
            if next_index >= len(entity_id_list):
                next_index = 0
            self.select_target(entity_id_list[next_index], self.sector.entities[entity_id_list[next_index]])

    def focus_target(self) -> None:
        if self.selected_target:
            self.perspective.cursor = tuple(self.sector.entities[self.selected_target].loc)

    def handle_mouse(self, m_id: int, m_x: int, m_y: int, m_z: int, bstate: int) -> bool:
        if not bstate & (curses.BUTTON1_CLICKED | curses.BUTTON1_PRESSED):
            return False

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
        return True

    def _handle_cancel(self) -> None:
        if self.selected_character is not None:
            self.selected_character = None
        elif self.selected_target is not None:
            self.select_target(None, None)

    def key_list(self) -> Collection[interface.KeyBinding]:
        key_list = [
            self.bind_key(ord('w'), lambda: self.perspective.move_cursor(ord('w')), help_key="sector_scroll"),
            self.bind_key(ord('a'), lambda: self.perspective.move_cursor(ord('a')), help_key="sector_scroll"),
            self.bind_key(ord('s'), lambda: self.perspective.move_cursor(ord('s')), help_key="sector_scroll"),
            self.bind_key(ord('d'), lambda: self.perspective.move_cursor(ord('d')), help_key="sector_scroll"),
            self.bind_key(ord("+"), lambda: self.perspective.zoom_cursor(ord("+"))),
            self.bind_key(ord("-"), lambda: self.perspective.zoom_cursor(ord("-"))),
            self.bind_key(ord("t"), lambda: self._change_target(1)),
            self.bind_key(ord("r"), lambda: self._change_target(-1)),
            self.bind_key(ord('\n'), self.focus_target),
            self.bind_key(ord('\r'), self.focus_target),
            self.bind_key(curses.ascii.ESC, self._handle_cancel),
        ]

        return key_list
