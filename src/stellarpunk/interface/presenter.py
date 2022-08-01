import curses
import uuid
import math
from typing import Tuple, Optional, Any, Sequence, Dict, Tuple, List, Mapping, Callable, Union

import drawille # type: ignore
import numpy as np

from stellarpunk import core, interface, util, effects

class Presenter:
    """ Prsents entities in a sector. """

    def __init__(self,
            gamestate:core.Gamestate, view:interface.View,
            sector:core.Sector, bbox:Tuple[float, float, float, float],
            meters_per_char_x:float, meters_per_char_y:float) -> None:
        self.gamestate = gamestate
        self.view = view
        self.sector = sector
        self.bbox = bbox
        self.meters_per_char_x = meters_per_char_x
        self.meters_per_char_y = meters_per_char_y

        self.debug_entity = False
        self.selected_target:Optional[uuid.UUID] = None
        self.debug_entity_vectors = False

    def draw_effect(self, effect:core.Effect) -> None:
        """ Draws an effect (if visible) on the map. """
        for effect in self.sector.effects:
            if isinstance(effect, effects.MiningEffect):
                if not isinstance(effect.source, core.Asteroid):
                    raise Exception("expected mining effect source to be an asteroid")
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
                        if y < 0 or x < 0 or y > self.view.viewscreen_dimensions[1] or x > self.view.viewscreen_dimensions[0]:
                            continue
                        self.view.viewscreen.addstr(y, x, icon, icon_attr)
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
                        if y < 0 or x < 0 or y > self.view.viewscreen_dimensions[1] or x > self.view.viewscreen_dimensions[0]:
                            continue
                        self.view.viewscreen.addstr(y, x, icon, icon_attr)
            elif isinstance(effect, effects.WarpOutEffect):
                # circle grows outward
                r = util.interpolate(effect.started_at, effect.radius, effect.expiration_time, 0., self.gamestate.timestamp)
                c = util.make_circle_canvas(r, self.meters_per_char_x, self.meters_per_char_y)
                util.draw_canvas_at(c, self.view.viewscreen.viewscreen, effect.loc[1], effect.loc[0], bounds=self.view.viewscreen_bounds)
            elif isinstance(effect, effects.WarpInEffect):
                #circle shrinks inward
                r = util.interpolate(effect.started_at, 0., effect.expiration_time, effect.radius, self.gamestate.timestamp)
                c = util.make_circle_canvas(r, self.meters_per_char_x, self.meters_per_char_y)
                util.draw_canvas_at(c, self.view.viewscreen.viewscreen, effect.loc[1], effect.loc[0], bounds=self.view.viewscreen_bounds)
            else:
                e_bbox = effect.bbox()
                loc = ((e_bbox[2] - e_bbox[0])/2, (e_bbox[3] - e_bbox[1])/2)
                icon = interface.Icons.EFFECT_UNKNOWN
                icon_attr = curses.color_pair(1)
                s_x, s_y = util.sector_to_screen(
                        loc[0], loc[1],
                        self.bbox[0], self.bbox[1],
                        self.meters_per_char_x, self.meters_per_char_y)
                self.view.viewscreen.addstr(s_y, s_x, icon, icon_attr)

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
        util.draw_canvas_at(util.drawille_vector(d_x, d_y, canvas=c), self.view.viewscreen.viewscreen, y, x, bounds=self.view.viewscreen_bounds)

    def draw_entity_debug_info(self, y:int, x:int, entity:core.SectorEntity, description_attr:int) -> None:
        if isinstance(entity, core.Ship):
            self.view.viewscreen.addstr(y, x+1, f' s: {entity.loc[0]:.0f},{entity.loc[1]:.0f}', description_attr)
            self.view.viewscreen.addstr(y+1, x+1, f' v: {entity.velocity[0]:.0f},{entity.velocity[1]:.0f}', description_attr)
            self.view.viewscreen.addstr(y+2, x+1, f' 𝜔: {entity.angular_velocity:.2f}', description_attr)
            self.view.viewscreen.addstr(y+3, x+1, f' 𝜃: {entity.angle:.2f}', description_attr)
            if isinstance(entity, core.Ship) and entity.collision_threat:
                self.view.viewscreen.addstr(y+4, x+1, f' c: {entity.collision_threat.short_id()}', description_attr)
        else:
            self.view.viewscreen.addstr(y, x+1, f' s: {entity.loc[0]:.0f},{entity.loc[1]:.0f}', description_attr)

    def draw_entity_info(self, y:int, x:int, entity:core.SectorEntity, description_attr:int) -> None:
        self.view.viewscreen.addstr(y, x+1, ' cargo:', description_attr)
        non_zero_cargo = 0
        for i in range(len(entity.cargo)):
            if entity.cargo[i] > 0.:
                self.view.viewscreen.addstr(y+1+non_zero_cargo, x, f' {i}: {entity.cargo[i]:.0f}', description_attr)
                non_zero_cargo += 1

    def draw_entity(self, y:int, x:int, entity:core.SectorEntity, icon_attr:int=0) -> None:
        """ Draws a single sector entity at screen position (y,x) """

        #TODO: handle shapes not circles?
        #TODO: better handle drawing entity shapes: refactor into own method
        if entity.radius > 0 and self.meters_per_char_x < entity.radius:
            c = util.make_circle_canvas(entity.radius, self.meters_per_char_x, self.meters_per_char_y)
            util.draw_canvas_at(c, self.view.viewscreen.viewscreen, y, x, bounds=self.view.viewscreen_bounds)

        icon = interface.Icons.sector_entity_icon(entity)
        icon_attr |= interface.Icons.sector_entity_attr(entity)

        description_attr = interface.Icons.sector_entity_attr(entity)
        if entity.entity_id == self.selected_target:
            icon_attr |= curses.A_STANDOUT
        else:
            description_attr |= curses.A_DIM

        self.view.viewscreen.addstr(y, x, icon, icon_attr)

        # draw a "tail" from the entity's history
        i=len(entity.history)-1
        cutoff_time = self.gamestate.timestamp - 5.
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
                self.view.viewscreen.addstr(hist_y, hist_x, interface.Icons.sector_entity_icon(entity, angle=entry.angle), (icon_attr | curses.A_DIM) & (~curses.A_STANDOUT))
            last_x = hist_x
            last_y = hist_y
            i-=1

        if not isinstance(entity, core.Asteroid):
            speed = entity.speed()
            if speed > 0.:
                name_tag = f' {entity.short_id()} {speed:.0f}'
            else:
                name_tag = f' {entity.short_id()}'
            self.view.viewscreen.addstr(y+1, x+1, name_tag, description_attr)

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

        self.view.viewscreen.addstr(y, x, icon, icon_attr)
        self.view.viewscreen.addstr(y, x+1, f' {len(entities)} {prefix}', icon_attr | curses.A_DIM)

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

        #TODO: draw an indicator for off-screen targeted entities

        if self.debug_entity_vectors and self.selected_target:
            entity = self.sector.entities[self.selected_target]
            screen_x, screen_y = util.sector_to_screen(
                    entity.loc[0], entity.loc[1], self.bbox[0], self.bbox[1],
                    self.meters_per_char_x, self.meters_per_char_y)

            self.draw_entity_vectors(screen_y, screen_x, entity)

