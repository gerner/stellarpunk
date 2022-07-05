import curses
import uuid
import math
from typing import Tuple, Optional, Any, Sequence, Dict, Tuple, List, Mapping, Callable, Union

import drawille # type: ignore
import numpy as np

from stellarpunk import core, interface, util, effects

class Presenter:
    """ Prsents entities in a sector. """

    def __init__(self, interface, viewscreen, sector, bbox, meters_per_char_x:float, meters_per_char_y:float) -> None:
        self.interface = interface
        self.viewscreen = viewscreen
        self.sector = sector
        self.bbox = bbox
        self.meters_per_char_x = meters_per_char_x
        self.meters_per_char_y = meters_per_char_y

        self.debug_entity = False
        self.selected_target:Optional[uuid.UUID] = None

    def draw_effect(self, effect:core.Effect) -> None:
        """ Draws an effect (if visible) on the map. """
        for effect in self.sector.effects:
            if isinstance(effect, effects.MiningEffect):
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
                icon = interface.Icons.EFFECT_TRANSFER
                icon_attr = curses.color_pair(interface.Icons.COLOR_CARGO)
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
                self.viewscreen.addstr(y+1+non_zero_cargo, x, f' {i}: {entity.cargo[i]:.0f}', description_attr)
                non_zero_cargo += 1

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

    def draw_sector_map(self) -> None:
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

