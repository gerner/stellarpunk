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

        self.cached_grid = None

        self.debug_entity = False
        self.debug_entity_vectors = False

    @property
    def viewscreen(self) -> curses.window:
        return self.interface.viewscreen

    def initialize(self) -> None:
        self.logger.info(f'entering sector mode for {self.sector.entity_id}')
        self.interface.camera_x = 0
        self.interface.camera_y = 0
        self.update_bbox()
        self.interface.reinitialize_screen(name="Sector Map")

    def focus(self) -> None:
        super().focus()
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

        #self.logger.debug(f'viewing sector {self.sector.entity_id} with bounding box ({(ul_x, ul_y)}, {(lr_x, lr_y)}) with per {self.meters_per_char_x:.0f}m x {self.meters_per_char_y:.0f}m char')

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
        self.logger.info(f'selected target {entity}')
        if entity:
            if isinstance(entity, core.Ship):
                self.interface.log_message(f'{entity.short_id()}: {entity.name} order: {entity.orders[0]}')
                for order in list(entity.orders)[1:]:
                    self.interface.log_message(f'queued: {order}')
            else:
                self.interface.log_message(f'{entity.short_id()}: {entity.name}')

    def _compute_grid(self, max_ticks:int=10) -> None:
        # choose ticks
        #TODO: should choose maxTicks based on resolution

        major_ticks_x = util.NiceScale(
                self.bbox[0], self.bbox[2],
                maxTicks=max_ticks, constrain_to_range=True)
        minor_ticks_y = util.NiceScale(
                self.bbox[1], self.bbox[3],
                maxTicks=max_ticks*4, constrain_to_range=True)
        major_ticks_y = util.NiceScale(
                self.bbox[1], self.bbox[3],
                maxTicks=max_ticks, constrain_to_range=True,
                tickSpacing=major_ticks_x.tickSpacing)
        minor_ticks_x = util.NiceScale(
                self.bbox[0], self.bbox[2],
                maxTicks=max_ticks*4, constrain_to_range=True,
                tickSpacing=minor_ticks_y.tickSpacing)

        c = drawille.Canvas()

        # draw the vertical lines
        i = major_ticks_x.niceMin
        while i < self.bbox[2]:
            j = minor_ticks_y.niceMin
            while j < self.bbox[3]:
                d_x, d_y = util.sector_to_drawille(
                        i, j,
                        self.meters_per_char_x, self.meters_per_char_y)
                c.set(d_x, d_y)
                j += minor_ticks_y.tickSpacing
            i += major_ticks_x.tickSpacing

        # draw the horizonal lines
        j = major_ticks_y.niceMin
        while j < self.bbox[3]:
            i = minor_ticks_x.niceMin
            while i < self.bbox[2]:
                d_x, d_y = util.sector_to_drawille(
                        i, j,
                        self.meters_per_char_x, self.meters_per_char_y)
                c.set(d_x, d_y)
                i += minor_ticks_x.tickSpacing
            j += major_ticks_y.tickSpacing

        # get upper left corner position so drawille canvas fills the screen
        (d_x, d_y) = util.sector_to_drawille(
                self.bbox[0], self.bbox[1],
                self.meters_per_char_x, self.meters_per_char_y)
        # draw the grid to the screen
        text = c.rows(d_x, d_y)

        self._cached_grid = (
                major_ticks_x,
                minor_ticks_y,
                major_ticks_y,
                minor_ticks_x,
                text
        )

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

    def draw_radar(self, y:int, x:int, radius:float) -> None:
        """ Draws a radar graphic to get sense of scale centered at y, x. """

        # choose ticks
        ticks = util.NiceScale(-1*radius, radius, constrain_to_range=True)
        stepsize = ticks.tickSpacing

        c = drawille.Canvas()

        # draw a cross
        i = 0.
        while i < radius:
            drawille_x,_ = util.sector_to_drawille(
                    i, 0, self.meters_per_char_x, self.meters_per_char_y)
            _,drawille_y = util.sector_to_drawille(
                    0, i, self.meters_per_char_x, self.meters_per_char_y)
            c.set(drawille_x, 0)
            c.set(-1*drawille_x, 0)
            c.set(0, drawille_y)
            c.set(0, -1*drawille_y)
            i += stepsize/2

        # draw rings to fill up the square with sides 2*radius
        r = stepsize
        theta_step = math.pi/16
        while r < math.sqrt(2*radius*radius):
            theta = 0.
            while theta < 2*math.pi:
                s_x, s_y = util.polar_to_cartesian(r, theta)
                if abs(s_x) < radius and abs(s_y) < radius:
                    d_x, d_y = util.sector_to_drawille(
                            s_x, s_y, self.meters_per_char_x, self.meters_per_char_y)
                    c.set(d_x, d_y)
                theta += theta_step
            r += stepsize

        # get upper left corner position so drawille canvas fills the screen
        (d_x, d_y) = util.sector_to_drawille(
                self.bbox[0], self.bbox[1],
                self.meters_per_char_x, self.meters_per_char_y)
        # draw the grid to the screen
        text = c.rows(d_x, d_y)

        for i, line in enumerate(text):
            self.viewscreen.addstr(i, 0, line, curses.color_pair(29))

        # draw distance indicators
        for r in range(int(stepsize), int(ticks.niceMax), int(stepsize)):
            self.viewscreen.addstr(y+int(r/self.meters_per_char_y), x, util.human_distance(r), curses.color_pair(29))

    def draw_entity_vectors(self, y:int, x:int, entity:core.SectorEntity) -> None:
        """ Draws heading, velocity, force vectors for the entity. """

        if not isinstance(entity, core.Ship):
            return

        heading_x, heading_y = util.polar_to_cartesian(self.meters_per_char_y*5, entity.angle)
        d_x, d_y = util.sector_to_drawille(heading_x, heading_y, self.meters_per_char_x, self.meters_per_char_y)
        c = util.drawille_vector(d_x, d_y)

        velocity_x, velocity_y = entity.velocity
        d_x, d_y = util.sector_to_drawille(velocity_x, velocity_y, self.meters_per_char_x, self.meters_per_char_y)
        util.drawille_vector(d_x, d_y, canvas=c)

        accel_x, accel_y = entity.phys.force / entity.mass
        d_x, d_y = util.sector_to_drawille(accel_x, accel_y, self.meters_per_char_x, self.meters_per_char_y)
        util.draw_canvas_at(util.drawille_vector(d_x, d_y, canvas=c), self.viewscreen, y, x, bounds=self.interface.viewscreen_bounds)

    def draw_entity_debug_info(self, y:int, x:int, entity:core.SectorEntity, description_attr:int) -> None:
        if isinstance(entity, core.Ship):
            self.viewscreen.addstr(y, x+1, f' s: {entity.loc[0]:.0f},{entity.loc[1]:.0f}', description_attr)
            self.viewscreen.addstr(y+1, x+1, f' v: {entity.velocity[0]:.0f},{entity.velocity[1]:.0f}', description_attr)
            self.viewscreen.addstr(y+2, x+1, f' 𝜔: {entity.angular_velocity:.2f}', description_attr)
            self.viewscreen.addstr(y+3, x+1, f' 𝜃: {entity.angle:.2f}', description_attr)
            if isinstance(entity, core.Ship) and entity.collision_threat:
                self.viewscreen.addstr(y+4, x+1, f' c: {entity.collision_threat.short_id()}', description_attr)
        else:
            self.viewscreen.addstr(y, x+1, f' s: {entity.loc[0]:.0f},{entity.loc[1]:.0f}', description_attr)

    def draw_entity_info(self, y:int, x:int, entity:core.SectorEntity, description_attr:int) -> None:
        self.viewscreen.addstr(y, x+1, ' cargo:', description_attr)
        non_zero_cargo = 0
        for i in range(len(entity.cargo)):
            if entity.cargo[i] > 0.:
                self.viewscreen.addstr(y, x+1+non_zero_cargo, f' {i}: {entity.cargo[i]:.0f}', description_attr)
                non_zero_cargo+=1

    def draw_entity(self, y:int, x:int, entity:core.SectorEntity, icon_attr:int=0) -> None:
        """ Draws a single sector entity at screen position (y,x) """

        #TODO: handle shapes not circles?
        #TODO: better handle drawing entity shapes: refactor into own method
        if entity.radius > 0 and self.meters_per_char_x < entity.radius:
            c = drawille.Canvas()
            r = entity.radius
            theta = 0.
            step = 2/r*self.meters_per_char_x
            while theta < 2*math.pi:
                c_x, c_y = util.polar_to_cartesian(r, theta)
                d_x, d_y = util.sector_to_drawille(c_x, c_y, self.meters_per_char_x, self.meters_per_char_y)
                c.set(d_x, d_y)
                theta += step
            util.draw_canvas_at(c, self.viewscreen, y, x, bounds=self.interface.viewscreen_bounds)

        icon = interface.Icons.sector_entity_icon(entity)
        icon_attr |= interface.Icons.sector_entity_attr(entity)

        description_attr = interface.Icons.sector_entity_attr(entity)
        if entity.entity_id == self.selected_target:
            icon_attr |= curses.A_STANDOUT
        else:
            description_attr |= curses.A_DIM

        self.viewscreen.addstr(y, x, icon, icon_attr)

        # draw a "tail" from the entity's history
        i=len(entity.history)-1
        cutoff_time = self.interface.gamestate.timestamp - 5.
        last_x = -1
        last_y = -1
        while i > 0:
            entry = entity.history[i]
            if entry.ts < cutoff_time:
                break
            hist_x, hist_y = util.sector_to_screen(
                    entry.loc[0], entry.loc[1], self.bbox[0], self.bbox[1],
                    self.meters_per_char_x, self.meters_per_char_y)
            if (hist_x != x or hist_y != y) and (hist_x != last_x or hist_y != last_y) and hist_x >= 0 and hist_y >= 0:
                self.viewscreen.addstr(hist_y, hist_x, interface.Icons.sector_entity_icon(entity, angle=entry.angle), (icon_attr | curses.A_DIM) & (~curses.A_STANDOUT))
            last_x = hist_x
            last_y = hist_y
            i-=1

        if not isinstance(entity, core.Asteroid):
            speed = entity.speed()
            if speed > 0.:
                name_tag = f' {entity.short_id()} {speed:.0f}'
            else:
                name_tag = f' {entity.short_id()}'
            self.viewscreen.addstr(y+1, x+1, name_tag, description_attr)

        if self.debug_entity:
            self.draw_entity_debug_info(y+2, x, entity, description_attr)
        elif entity.entity_id == self.selected_target:
            self.draw_entity_info(y+2, x, entity, description_attr)

    def draw_multiple_entities(self, y:int, x:int, entities:Sequence[core.SectorEntity]) -> None:

        icons = set(map(interface.Icons.sector_entity_icon, entities))
        icon = icons.pop() if len(icons) == 1 else interface.Icons.MULTIPLE

        icon_attrs = set(map(interface.Icons.sector_entity_attr, entities))
        icon_attr = icon_attrs.pop() if len(icon_attrs) == 1 else 0

        prefixes = set(map(lambda x: x.id_prefix, entities))
        prefix = prefixes.pop() if len(prefixes) == 1 else "entities"

        self.viewscreen.addstr(y, x, icon, icon_attr)
        self.viewscreen.addstr(y, x+1, f' {len(entities)} {prefix}', icon_attr | curses.A_DIM)

    def draw_cells(self, occupied:Mapping[Tuple[int,int], Sequence[core.SectorEntity]], collision_threats:Sequence[core.SectorEntity]) -> None:
        for loc, entities in occupied.items():
            if len(entities) > 1:
                self.draw_multiple_entities(
                        loc[1], loc[0], entities)
            else:
                icon_attr = 0
                if entities[0] in collision_threats:
                    icon_attr = curses.color_pair(1)

                self.draw_entity(loc[1], loc[0], entities[0], icon_attr=icon_attr)

    def draw_effect(self, effect:core.Effect) -> None:
        """ Draws an effect (if visible) on the map. """
        for effect in self.sector.effects:
            if isinstance(effect, effects.MiningEffect):
                #TODO: draw some stuff coming from the asteroid to the ship
                icon = interface.Icons.EFFECT_MINING
                icon_attr = curses.color_pair(interface.Icons.RESOURCE_COLORS[effect.source.resource])
                s_x, s_y = util.sector_to_screen(
                        effect.source.loc[0], effect.source.loc[1],
                        self.bbox[0], self.bbox[1],
                        self.meters_per_char_x, self.meters_per_char_y)
                d_x, d_y = util.sector_to_screen(
                        effect.destination.loc[0], effect.destination.loc[1],
                        self.bbox[0], self.bbox[1],
                        self.meters_per_char_x, self.meters_per_char_y)

                if s_x != d_x or s_y != d_y:
                    for y,x in np.linspace((s_y,s_x), (d_y,d_x), 10, dtype=int):
                        if (y == s_y and x == s_x) or (y == d_y and x == d_x):
                            continue
                        if y < 0 or x < 0 or y > self.interface.viewscreen_height or x > self.interface.viewscreen_width:
                            continue
                        self.viewscreen.addstr(y, x, icon, icon_attr)
            elif isinstance(effect, effects.TransferCargoEffect):
                #TODO: draw some stuff transferring between the two
                pass

    def draw_sector_map(self) -> None:
        """ Draws a map of a sector. """

        self.viewscreen.erase()

        #self.draw_radar(
        #        int(self.viewscreen_height/2), int(self.viewscreen_width/2),
        #        meters_per_char_x, meters_per_char_y,
        #        self.szoom/2,
        #        self.viewscreen
        #)
        self.draw_grid()

        # list x,y coords of center of screen
        #self.viewscreen.addstr(1,3, f'{self.scursor_x:.0f},{self.scursor_y:.0f}', curses.color_pair(29))

        collision_threats = []
        for ship in self.sector.ships:
            if ship.collision_threat:
                collision_threats.append(ship.collision_threat)

        last_loc = None
        occupied:Dict[Tuple[int,int], List[core.SectorEntity]] = {}

        for effect in self.sector.effects:
            if util.intersects(effect.bbox(), self.bbox):
                self.draw_effect(effect)

        for entity in self.sector.spatial_query(self.bbox):
            screen_x, screen_y = util.sector_to_screen(
                    entity.loc[0], entity.loc[1], self.bbox[0], self.bbox[1],
                    self.meters_per_char_x, self.meters_per_char_y)
            if screen_x < 0 or screen_y < 0:
                continue

            last_loc = (screen_x, screen_y)
            if last_loc in occupied:
                entities = occupied[last_loc]
                entities.append(entity)
            else:
                occupied[last_loc] = [entity]

        self.draw_cells(occupied, collision_threats)

        #TODO: draw an indicator for off-screen targeted entities

        if self.debug_entity_vectors and self.selected_entity:
            entity = self.selected_entity
            screen_x, screen_y = util.sector_to_screen(
                    entity.loc[0], entity.loc[1], self.bbox[0], self.bbox[1],
                    self.meters_per_char_x, self.meters_per_char_y)

            self.draw_entity_vectors(screen_y, screen_x, entity)

        self.interface.refresh_viewscreen()

    def update_display(self) -> None:
        self.interface.camera_x = 0
        self.interface.camera_y = 0
        self.draw_sector_map()
        self.interface.refresh_viewscreen()

    def command_list(self) -> Mapping[str, interface.CommandInput.CommandSig]:
        def target(args:Sequence[str])->None:
            if not args:
                raise interface.CommandInput.UserError("need a valid target")
            try:
                target_id = uuid.UUID(args[0])
            except ValueError:
                raise interface.CommandInput.UserError("not a valid target id, try tab completion.")
            if target_id not in self.sector.entities:
                raise interface.CommandInput.UserError("{args[0]} not found in sector")
            self.select_target(target_id, self.sector.entities[target_id])

        def goto(args:Sequence[str])->None:
            if not self.selected_entity or not isinstance(self.selected_entity, core.Ship):
                raise interface.CommandInput.UserError(f'order only valid on a ship target')
            if len(args) < 2:
                x,y = self.scursor_x, self.scursor_y
            else:
                try:
                    x,y = int(args[0]), int(args[1])
                except Exception:
                    raise interface.CommandInput.UserError("need two int args for x,y pos")
            self.selected_entity.orders.clear()
            self.selected_entity.orders.append(orders.GoToLocation(np.array((x,y)), self.selected_entity, self.interface.gamestate))

        def wait(args:Sequence[str])->None:
            if not self.selected_entity or not isinstance(self.selected_entity, core.Ship):
                raise interface.CommandInput.UserError(f'order only valid on a ship target')
            self.selected_entity.orders.clear()
            self.selected_entity.orders.append(orders.WaitOrder(self.selected_entity, self.interface.gamestate))

        def harvest(args:Sequence[str])->None:
            if not self.selected_entity or not isinstance(self.selected_entity, core.Ship):
                raise interface.CommandInput.UserError(f'order only valid on a ship target')
            self.logger.info('adding harvest order to {self.selected_entity}')
            base = self.sector.stations[0]
            self.selected_entity.orders.clear()
            self.selected_entity.orders.append(orders.HarvestOrder(base, 0, self.selected_entity, self.interface.gamestate))

        def debug_entity(args:Sequence[str])->None: self.debug_entity = not self.debug_entity
        def debug_vectors(args:Sequence[str])->None: self.debug_entity_vectors = not self.debug_entity_vectors
        def debug_write_history(args:Sequence[str])->None:
            if not self.selected_entity:
                raise interface.CommandInput.UserError(f'can only write history for a selected target')
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
                    raise interface.CommandInput.UserError("need two int args for x,y pos")

            self.interface.generator.spawn_ship(self.sector, x, y, v=np.array((0,0)), w=0)

        def spawn_collision(args:Sequence[str])->None:
            self.interface.generator.spawn_ship(self.sector, 0, 1100, v=np.array((0,0)), w=0)
            self.interface.generator.spawn_ship(self.sector, 0, 2200, v=np.array((0,0)), w=0)

        def spawn_resources(args:Sequence[str])->None:
            x,y = self.scursor_x, self.scursor_y
            self.interface.generator.spawn_resource_field(self.sector, x, y, 0, 1e6)

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
        }

    def handle_input(self, key:int) -> bool:
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
            self.interface.open_view(interface.CommandInput(
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
