""" Interface methods dealing with displaying a single sector map. """

import logging
import math
import bisect
import curses
import curses.textpad
import curses.ascii
import time
import uuid
from typing import Tuple, Optional, Any, Sequence, Dict, Tuple, List, Mapping, Callable, Union

import drawille # type: ignore
import numpy as np

from stellarpunk import util, core, interface, orders, effects
from stellarpunk.interface import command_input, presenter, pilot as pilot_interface

class SectorView(interface.View):
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

        # where the sector map is centered in sector coordinates
        self.scursor_x = 0.
        self.scursor_y = 0.

        # entity id of the currently selected target
        self.selected_target:Optional[uuid.UUID] = None
        self.selected_entity:Optional[core.SectorEntity] = None

        # sector zoom level, expressed in meters to fit on screen
        self.szoom = self.sector.radius*2
        self.meters_per_char_x = 0.
        self.meters_per_char_y = 0.

        # sector coord bounding box (ul_x, ul_y, lr_x, lr_y)
        self.bbox = (0.,0.,0.,0.)

        self.presenter = presenter.Presenter(self.interface.gamestate, self, self.sector, self.bbox, self.meters_per_char_x, self.meters_per_char_y)

        self._cached_grid = (util.NiceScale(0,0), util.NiceScale(0,0), util.NiceScale(0,0), util.NiceScale(0,0), "")
        self.debug_entity = False
        self.debug_entity_vectors = False

    def initialize(self) -> None:
        self.logger.info(f'entering sector mode for {self.sector.entity_id}')
        self.interface.camera_x = 0
        self.interface.camera_y = 0
        self.update_bbox()
        self.interface.reinitialize_screen(name="Sector Map")

    def focus(self) -> None:
        super().focus()
        self.active = True
        self.interface.camera_x = 0
        self.interface.camera_y = 0
        self.interface.reinitialize_screen(name="Sector Map")

    def update_bbox(self) -> None:
        self.meters_per_char_x, self.meters_per_char_y = self.meters_per_char()

        vsw = self.interface.viewscreen_width
        vsh = self.interface.viewscreen_height

        ul_x = self.scursor_x - (vsw/2 * self.meters_per_char_x)
        ul_y = self.scursor_y - (vsh/2 * self.meters_per_char_y)
        lr_x = self.scursor_x + (vsw/2 * self.meters_per_char_x)
        lr_y = self.scursor_y + (vsh/2 * self.meters_per_char_y)

        self.bbox = (ul_x, ul_y, lr_x, lr_y)

        self._compute_grid()

        self.presenter.bbox = self.bbox
        self.presenter.meters_per_char_x = self.meters_per_char_x
        self.presenter.meters_per_char_y = self.meters_per_char_y

    def set_scursor(self, x:float, y:float) -> None:
        self.scursor_x = x
        self.scursor_y = y

        self.update_bbox()

    def move_scursor(self, direction:int) -> None:
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

    def zoom_scursor(self, direction:int) -> None:
        if direction == ord('+'):
            self.szoom *= 0.9
        elif direction == ord('-'):
            self.szoom *= 1.1
        else:
            raise ValueError(f'unknown direction {direction}')

        self.update_bbox()

    def meters_per_char(self) -> Tuple[float, float]:
        meters_per_char_x = self.szoom / min(self.interface.viewscreen_width, math.floor(self.interface.viewscreen_height/self.interface.font_width*self.interface.font_height))
        meters_per_char_y = meters_per_char_x / self.interface.font_width * self.interface.font_height

        assert self.szoom / meters_per_char_y <= self.interface.viewscreen_height
        assert self.szoom / meters_per_char_x <= self.interface.viewscreen_width

        return (meters_per_char_x, meters_per_char_y)

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
                self.interface.log_message(f'{entity.short_id()}: {entity.name} order: {entity.orders[0]}')
                for order in list(entity.orders)[1:]:
                    self.interface.log_message(f'queued: {order}')
            else:
                self.interface.log_message(f'{entity.short_id()}: {entity.name}')
            for i in range(entity.cargo.shape[0]):
                if entity.cargo[i] > 0.:
                    self.interface.log_message(f'cargo {i}: {entity.cargo[i]}')

    def _compute_grid(self, max_ticks:int=10) -> None:
        self._cached_grid = util.compute_uigrid(
                self.bbox, self.meters_per_char_x, self.meters_per_char_y)

    def draw_grid(self) -> None:
        """ Draws a grid at tick lines. """

        major_ticks_x, minor_ticks_y, major_ticks_y, minor_ticks_x, text = self._cached_grid

        for lineno, line in enumerate(text):
            self.viewscreen.addstr(lineno, 0, line, curses.color_pair(29))

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

    def draw_sector_map(self) -> None:
        """ Draws a map of a sector. """

        self.viewscreen.erase()
        self.draw_grid()
        self.presenter.draw_sector_map()
        self.interface.refresh_viewscreen()

    def update_display(self) -> None:
        self.interface.camera_x = 0
        self.interface.camera_y = 0
        self.draw_sector_map()
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
                x,y = self.scursor_x, self.scursor_y
            else:
                try:
                    x,y = int(args[0]), int(args[1])
                except Exception:
                    raise command_input.CommandInput.UserError("need two int args for x,y pos")
            self.selected_entity.orders.clear()
            self.selected_entity.orders.append(orders.GoToLocation(np.array((x,y)), self.selected_entity, self.interface.gamestate))

        def wait(args:Sequence[str])->None:
            if not self.selected_entity or not isinstance(self.selected_entity, core.Ship):
                raise command_input.CommandInput.UserError(f'order only valid on a ship target')
            self.selected_entity.orders.clear()
            self.selected_entity.orders.append(orders.WaitOrder(self.selected_entity, self.interface.gamestate))

        def harvest(args:Sequence[str])->None:
            if not self.selected_entity or not isinstance(self.selected_entity, core.Ship):
                raise command_input.CommandInput.UserError(f'order only valid on a ship target')
            self.logger.info('adding harvest order to {self.selected_entity}')
            base = self.sector.stations[0]
            self.selected_entity.orders.clear()
            self.selected_entity.orders.append(orders.HarvestOrder(base, 0, self.selected_entity, self.interface.gamestate))

        def debug_entity(args:Sequence[str])->None: self.debug_entity = not self.debug_entity
        def debug_vectors(args:Sequence[str])->None: self.debug_entity_vectors = not self.debug_entity_vectors
        def debug_write_history(args:Sequence[str])->None:
            if not self.selected_entity:
                raise command_input.CommandInput.UserError(f'can only write history for a selected target')
            filename = "/tmp/stellarpunk.history"
            self.logger.info(f'writing history for {self.selected_entity} to {filename}')
            core.write_history_to_file(self.selected_entity, filename)

        def debug_write_sector(args:Sequence[str])->None:
            filename = "/tmp/stellarpunk.history.gz"
            self.logger.info(f'writing history for sector {self.sector.short_id()} to {filename}')
            core.write_history_to_file(self.sector, filename)

        def spawn_ship(args:Sequence[str])->None:
            if len(args) < 2:
                x,y = self.scursor_x, self.scursor_y
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
            x,y = self.scursor_x, self.scursor_y
            self.interface.generator.spawn_resource_field(self.sector, x, y, 0, 1e6)

        def pilot(args:Sequence[str])->None:
            if not self.selected_entity or not isinstance(self.selected_entity, core.Ship):
                raise command_input.CommandInput.UserError(f'can only pilot a selected ship target')
            pilot_view = pilot_interface.PilotView(self.selected_entity, self.interface)
            self.interface.open_view(pilot_view)
            # suspend input until we get focus again
            self.active = False

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
                "harvest": harvest,
                "pilot": pilot,
        }

    def handle_input(self, key:int, dt:float) -> bool:
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
                        self.sector.entities[self.selected_target].loc[0],
                        self.sector.entities[self.selected_target].loc[1]
                )
        elif key == ord("k"):
            if not self.selected_entity or not isinstance(self.selected_entity, core.Ship):
                self.interface.status_message(f'order only valid on a ship target', curses.color_pair(1))
            else:
                self.selected_entity.orders.clear()
                self.selected_entity.orders.append(orders.KillRotationOrder(self.selected_entity, self.interface.gamestate))
        elif key == ord("r"):
            if not self.selected_entity or not isinstance(self.selected_entity, core.Ship):
                self.interface.status_message(f'order only valid on a ship target', curses.color_pair(1))
            else:
                self.selected_entity.orders.clear()
                self.selected_entity.orders.append(orders.RotateOrder(0, self.selected_entity, self.interface.gamestate))
        elif key == ord("x"):
            if not self.selected_entity or not isinstance(self.selected_entity, core.Ship):
                self.interface.status_message(f'order only valid on a ship target', curses.color_pair(1))
            else:
                self.selected_entity.orders.clear()
                self.selected_entity.orders.append(orders.KillVelocityOrder(self.selected_entity, self.interface.gamestate))
        elif key == ord("g"):
            if not self.selected_entity or not isinstance(self.selected_entity, core.Ship):
                self.interface.status_message(f'order only valid on a ship target', curses.color_pair(1))
            else:
                self.selected_entity.orders.clear()
                self.selected_entity.orders.append(orders.GoToLocation(np.array((0,0)), self.selected_entity, self.interface.gamestate))
        elif key == ord("o"):
            for ship in self.sector.ships:
                station = self.interface.generator.r.choice(np.array((self.sector.stations)))
                ship.orders.append(orders.GoToLocation(np.array((station.loc[0], station.loc[1])), ship, self.interface.gamestate))
        elif key == ord(":"):
            self.interface.open_view(command_input.CommandInput(
                self.interface, commands=self.command_list()))
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
            bounds = (
                    sector_x-self.meters_per_char_x, sector_y-self.meters_per_char_y,
                    sector_x+self.meters_per_char_x, sector_y+self.meters_per_char_y
            )
            hit = next(self.sector.spatial_query(bounds), None)
            if hit:
                #TODO: check if the hit is close enough
                self.select_target(hit.entity_id, hit)
        elif key == curses.ascii.ESC: #TODO: should handle escape here
            if self.selected_target is not None:
                self.select_target(None, None)
            else:
                return False

        return True
