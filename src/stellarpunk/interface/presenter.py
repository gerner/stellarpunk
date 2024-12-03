import logging
import curses
import uuid
import math
import functools
import abc
from typing import Tuple, Optional, Any, Sequence, Dict, Tuple, List, Mapping, Callable, Union, Iterable, Set

import cymunk # type: ignore
import drawille # type: ignore
import numpy as np
import numpy.typing as npt
from numba import jit # type: ignore
import rtree.index # type: ignore

from stellarpunk import core, interface, util, effects, config, sensors
from stellarpunk.core import combat
from stellarpunk.orders import steering, collision

SENSOR_ANGLE_BINS = np.linspace(0, 2*np.pi, 64)
#COLLISION_MARGIN_BINS = np.linspace(0, 5e3, 128)
#NEIGHBORHOOD_RADIUS_BINS = np.linspace(100., 1e3, 64)

#@jit(cache=True, nopython=True, fastmath=True)
def quantize(a:float, error:float=0.1) -> float:
    exp = math.floor(math.log(a) / math.log(2))
    return (a/(2** exp) // error * error ) * 2 ** exp

@functools.lru_cache(maxsize=256)
def compute_sensor_cone_memoize(stopped:bool, angle:float, neighborhood_radius:float, collision_margin:float, radius:float, offset_x:float, offset_y:float, bounds:Tuple[int, int, int, int], meters_per_char:Tuple[float, float]) -> Mapping[Tuple[int, int], str]:
    sensor_cone = collision.compute_sensor_cone(cymunk.Vec2d(*util.polar_to_cartesian(0.0 if stopped else 1.0, angle)), neighborhood_radius, collision_margin, steering.CYZERO_VECTOR, radius)

    c = util.make_circle_canvas(neighborhood_radius, *meters_per_char)
    c = util.drawille_line(sensor_cone[0], sensor_cone[1], *meters_per_char, canvas = c)
    c = util.drawille_line(sensor_cone[1], sensor_cone[2], *meters_per_char, canvas = c)
    c = util.drawille_line(sensor_cone[2], sensor_cone[3], *meters_per_char, canvas = c)
    c = util.drawille_line(sensor_cone[3], sensor_cone[0], *meters_per_char, canvas = c)

    # appears to fill the whole screen
    (d_x, d_y) = util.sector_to_drawille(
        offset_x,
        offset_y,
        *meters_per_char)
    content = util.lines_to_dict(c.rows(d_x, d_y), bounds=bounds)

    return content

@functools.lru_cache
def compute_sensor_rings_memoize(radii:Iterable[float], width:float, height:float, bounds:Tuple[int, int, int, int], meters_per_char:Tuple[float, float]) -> Mapping[Tuple[int, int], str]:
    canvas = drawille.Canvas()
    for r in radii:
        util.drawille_circle(
            r,
            meters_per_char[0]*4,
            width,
            height,
            *meters_per_char,
            canvas=canvas
        )

    # get upper left corner position so drawille canvas fills the screen
    (d_x, d_y) = util.sector_to_drawille(
            -width/2, -height/2,
            *meters_per_char)

    # draw the grid to the screen
    text = canvas.rows(d_x, d_y)

    content = util.lines_to_dict(text, bounds=bounds)

    return content

class Presenter:
    """ Prsents entities in a sector. """

    def __init__(self,
            gamestate:core.Gamestate,
            view:interface.View,
            sector:core.Sector,
            perspective:interface.Perspective
            ) -> None:
        self.logger = logging.getLogger(util.fullname(self))
        self.gamestate = gamestate
        self.view = view
        self.sector = sector

        self.perspective = perspective

        self.debug_entity = False
        self.selected_target:Optional[uuid.UUID] = None
        self.debug_entity_vectors = False
        self.show_sensor_cone = False

        #self.sensor_image_ttl = 120.
        #self._cached_entities:Dict[uuid.UUID, core.AbstractSensorImage] = {}
        #self._cached_entities_ts = -1.
        #self._sensor_loc_index = rtree.index.Index()

    @property
    @abc.abstractmethod
    def selected_target_image(self) -> core.AbstractSensorImage: ...

    @abc.abstractmethod
    def visible_entities(self) -> Iterable[core.AbstractSensorImage]: ...

    def compute_sensor_cone(self, ship:core.Ship, neighborhood_radius:float, collision_margin:float) -> Mapping[Tuple[int, int], str]:
        # quantize parameters for caching
        if ship.phys.velocity.get_length_sqrd() == 0.:
            stopped = True
            quantized_theta = 0.0
        else:
            stopped = False
            theta = util.normalize_angle(ship.phys.velocity.get_angle())
            quantized_theta = SENSOR_ANGLE_BINS[np.digitize(theta, SENSOR_ANGLE_BINS)-1]

        neighborhood_radius = quantize(neighborhood_radius, 0.1)#NEIGHBORHOOD_RADIUS_BINS[np.digitize(neigborhood_radius, NEIGHBORHOOD_RADIUS_BINS)]
        collision_margin = quantize(collision_margin, 0.1)#COLLISION_MARGIN_BINS[np.digitize(collision_margin, COLLISION_MARGIN_BINS)]
        radius = quantize(ship.radius, 0.1)

        return compute_sensor_cone_memoize(stopped, quantized_theta, neighborhood_radius, collision_margin, ship.radius, -(self.perspective.bbox[2]-self.perspective.bbox[0])/2, -(self.perspective.bbox[3]-self.perspective.bbox[1])/2, self.view.viewscreen_bounds, self.perspective.meters_per_char)


    def draw_effect(self, effect:core.Effect) -> None:
        """ Draws an effect (if visible) on the map. """

        assert isinstance(self.view.viewscreen, interface.Canvas)
        window = self.view.viewscreen.window
        if isinstance(effect, effects.MiningEffect):
            if not isinstance(effect.source, core.Asteroid):
                raise Exception("expected mining effect source to be an asteroid")
            icon = interface.Icons.EFFECT_MINING
            icon_attr = curses.color_pair(interface.Icons.RESOURCE_COLORS[effect.source.resource])
            s_x, s_y = self.perspective.sector_to_screen(
                    effect.source.loc[0], effect.source.loc[1],
            )
            d_x, d_y = self.perspective.sector_to_screen(
                    effect.destination.loc[0], effect.destination.loc[1],
            )

            if abs(s_x - d_x) > 1 or abs(s_y - d_y) > 1:
                for y,x in np.linspace((s_y,s_x), (d_y,d_x), 10, dtype=int):
                    if (y == s_y and x == s_x) or (y == d_y and x == d_x):
                        continue
                    if y < 0 or x < 0 or y > self.view.viewscreen_dimensions[1] or x > self.view.viewscreen_dimensions[0]:
                        continue
                    self.view.viewscreen.addstr(y, x, icon, icon_attr)
        elif isinstance(effect, effects.TransferCargoEffect):
            icon = interface.Icons.EFFECT_TRANSFER
            icon_attr = curses.color_pair(interface.Icons.COLOR_CARGO)
            s_x, s_y = self.perspective.sector_to_screen(
                    effect.source.loc[0], effect.source.loc[1],
            )
            d_x, d_y = self.perspective.sector_to_screen(
                    effect.destination.loc[0], effect.destination.loc[1],
            )

            if abs(s_x - d_x) > 1 or abs(s_y - d_y) > 1:
                for y,x in np.linspace((s_y,s_x), (d_y,d_x), 10, dtype=int):
                    if (y == s_y and x == s_x) or (y == d_y and x == d_x):
                        continue
                    if y < 0 or x < 0 or y > self.view.viewscreen_dimensions[1] or x > self.view.viewscreen_dimensions[0]:
                        continue
                    self.view.viewscreen.addstr(y, x, icon, icon_attr)
        elif isinstance(effect, effects.WarpOutEffect):
            # circle grows outward
            s_x, s_y = self.perspective.sector_to_screen(
                    effect.loc[0], effect.loc[1],
            )
            r = util.interpolate(effect.started_at, effect.radius, effect.expiration_time, 0., self.gamestate.timestamp)
            c = util.make_circle_canvas(r, *self.perspective.meters_per_char)
            util.draw_canvas_at(c, window, s_y, s_x, bounds=self.view.viewscreen_bounds)
        elif isinstance(effect, effects.WarpInEffect):
            #circle shrinks inward
            s_x, s_y = self.perspective.sector_to_screen(
                    effect.loc[0], effect.loc[1],
            )
            r = util.interpolate(effect.started_at, 0., effect.expiration_time, effect.radius, self.gamestate.timestamp)
            c = util.make_circle_canvas(r, *self.perspective.meters_per_char)
            util.draw_canvas_at(c, window, s_y, s_x, bounds=self.view.viewscreen_bounds)
        elif isinstance(effect, combat.PointDefenseEffect):
            if effect.state == combat.PointDefenseEffect.State.ACTIVE:
                assert effect._pd_shape
                loc_x, loc_y = effect._pd_shape.body.position[0], effect._pd_shape.body.position[1]
                s_x, s_y = self.perspective.sector_to_screen(loc_x, loc_y)
                c = util.make_polygon_canvas(effect._pd_shape.get_vertices(), *self.perspective.meters_per_char, offset_x=-loc_x, offset_y=-loc_y)
                util.draw_canvas_at(c, window, s_y, s_x, bounds=self.view.viewscreen_bounds)
        else:
            e_bbox = effect.bbox()
            loc = ((e_bbox[2] - e_bbox[0])/2, (e_bbox[3] - e_bbox[1])/2)
            icon = interface.Icons.EFFECT_UNKNOWN
            icon_attr = curses.color_pair(1)
            s_x, s_y = self.perspective.sector_to_screen(
                    loc[0], loc[1],
            )
            self.view.viewscreen.addstr(s_y, s_x, icon, icon_attr)

    def draw_entity_vectors(self, y:int, x:int, entity:core.SectorEntity) -> None:
        """ Draws heading, velocity, force vectors for the entity. """

        if not isinstance(entity, core.Ship):
            return

        assert isinstance(self.view.viewscreen, interface.Canvas)
        window = self.view.viewscreen.window

        heading_x, heading_y = util.polar_to_cartesian(self.perspective.meters_per_char[1]*5, entity.angle)
        d_x, d_y = util.sector_to_drawille(heading_x, heading_y, *self.perspective.meters_per_char)
        c = util.drawille_vector(d_x, d_y)

        velocity_x, velocity_y = entity.velocity
        d_x, d_y = util.sector_to_drawille(velocity_x, velocity_y, *self.perspective.meters_per_char)
        util.drawille_vector(d_x, d_y, canvas=c)

        accel_x, accel_y = entity.phys.force / entity.mass
        d_x, d_y = util.sector_to_drawille(accel_x, accel_y, *self.perspective.meters_per_char)
        util.draw_canvas_at(util.drawille_vector(d_x, d_y, canvas=c), window, y, x, bounds=self.view.viewscreen_bounds)

    def draw_entity_debug_info(self, y:int, x:int, entity:core.SectorEntity, description_attr:int) -> None:
        if isinstance(entity, core.Ship):
            self.view.viewscreen.addstr(y, x+1, f' s: {entity.loc[0]:.0f},{entity.loc[1]:.0f}', description_attr)
            self.view.viewscreen.addstr(y+1, x+1, f' v: {entity.velocity[0]:.0f},{entity.velocity[1]:.0f}', description_attr)
            self.view.viewscreen.addstr(y+2, x+1, f' 𝜔: {entity.angular_velocity:.2f}', description_attr)
            self.view.viewscreen.addstr(y+3, x+1, f' 𝜃: {entity.angle:.2f}', description_attr)
        else:
            self.view.viewscreen.addstr(y, x+1, f' s: {entity.loc[0]:.0f},{entity.loc[1]:.0f}', description_attr)

    def draw_entity_info(self, y:int, x:int, entity:core.SectorEntity, description_attr:int) -> None:
        self.view.viewscreen.addstr(y, x+1, ' cargo:', description_attr)
        non_zero_cargo = 0
        for i in range(len(entity.cargo)):
            if entity.cargo[i] > 0.:
                self.view.viewscreen.addstr(y+1+non_zero_cargo, x, f' {i}: {entity.cargo[i]:.0f}', description_attr)
                non_zero_cargo += 1

    def draw_entity_shape(self, entity:core.AbstractSensorImage) -> None:
        #TODO: handle shapes not circles?
        #TODO: better handle drawing entity shapes: refactor into own method

        # clear out the interior of the entity circle
        loc_x, loc_y = entity.loc
        s_x_min, s_y_min = self.perspective.sector_to_screen(loc_x-entity.identity.radius, loc_y-entity.identity.radius)
        s_x_max, s_y_max = self.perspective.sector_to_screen(loc_x+entity.identity.radius, loc_y+entity.identity.radius)
        for s_x in range(s_x_min, s_x_max+1):
            for s_y in range(s_y_min, s_y_max+1):
                e_x, e_y = self.perspective.screen_to_sector(s_x, s_y)
                dist2 = (e_x - loc_x)**2.+(e_y - loc_y)**2
                if dist2 < entity.identity.radius**2.:
                    self.view.viewscreen.addstr(s_y, s_x, " ", 0)


        #TODO: should not do this
        assert isinstance(self.view.viewscreen, interface.Canvas)
        window = self.view.viewscreen.window

        # actually draw the circle
        screen_x, screen_y = self.perspective.sector_to_screen(loc_x, loc_y)
        c = util.make_circle_canvas(entity.identity.radius, *self.perspective.meters_per_char, bbox=util.translate_rect(self.perspective.bbox, (-loc_x, -loc_y)))
        util.draw_canvas_at(c, window, screen_y, screen_x, bounds=self.view.viewscreen_bounds)

    def draw_entity(self, y:int, x:int, entity:core.AbstractSensorImage, icon_attr:int=0) -> None:
        """ Draws a single sector entity at screen position (y,x) """

        #DEBUG: draw a circle representing uncertainty of position
        #if entity._target is not None:
        #    ptr = self.sector.sensor_manager.compute_target_profile(entity._target, entity._ship) / self.sector.sensor_manager.compute_effective_threshold(entity._ship)
        #    sensor_radius = config.Settings.sensors.COEFF_BIAS_LOC / ptr
        #    screen_x, screen_y = self.perspective.sector_to_screen(entity._target.loc[0], entity._target.loc[1])
        #    c = util.make_circle_canvas(sensor_radius, *self.perspective.meters_per_char)
        #    util.draw_canvas_at(c, self.view.viewscreen.window, screen_y, screen_x, bounds=self.view.viewscreen_bounds)

        #    s_x, s_y = self.perspective.sector_to_screen(*entity._target.loc)
        #    self.view.viewscreen.addstr(s_y, s_x, interface.Icons.TARGET_INDICATOR, curses.color_pair(interface.Icons.COLOR_TARGET_IMAGE_INDICATOR))

        #TODO: icon and text info below should depend on whether we have fully
        #resolved the sensor reading
        icon = interface.Icons.sensor_image_icon(entity.identity) if entity.identified else interface.Icons.UNKNOWN
        icon_attr |= interface.Icons.sensor_image_attr(entity.identity) if entity.identified else curses.color_pair(interface.Icons.COLOR_UNKNOWN)

        description_attr = interface.Icons.sensor_image_attr(entity.identity) if entity.identified else curses.color_pair(interface.Icons.COLOR_UNKNOWN)
        if entity.identity.entity_id == self.selected_target:
            icon_attr |= curses.A_STANDOUT
        else:
            description_attr |= curses.A_DIM

        self.view.viewscreen.addstr(y, x, icon, icon_attr)

        """
        # draw a "tail" from the entity's history
        i=len(entity.history)-1
        cutoff_time = self.gamestate.timestamp - 5.
        sampling_period = 0.5
        last_ts = np.inf
        last_x = -1
        last_y = -1
        while i > 0:
            entry = entity.history[i]
            i-=1
            if entry.ts < cutoff_time:
                break

            if entry.ts - sampling_period < sampling_period:
                continue
            last_ts = entry.ts

            hist_x, hist_y = self.perspective.sector_to_screen(entry.loc[0], entry.loc[1])
            if (hist_x != x or hist_y != y) and (hist_x != last_x or hist_y != last_y) and hist_x >= 0 and hist_y >= 0:
                self.view.viewscreen.addstr(hist_y, hist_x, interface.Icons.sector_entity_icon(entity, angle=entry.angle), (icon_attr | curses.A_DIM) & (~curses.A_STANDOUT))
            last_x = hist_x
            last_y = hist_y
        """

        if entity.identified and entity.identity.object_type not in (core.ObjectType.ASTEROID, core.ObjectType.PROJECTILE):
            speed = util.magnitude(*entity.velocity)
            if speed > 0.:
                name_tag = f' {entity.identity.short_id} {speed:.0f}'
            else:
                name_tag = f' {entity.identity.short_id}'
            self.view.viewscreen.addstr(y+1, x+1, name_tag, description_attr)

        #if self.debug_entity:
        #    self.draw_entity_debug_info(y+2, x, entity, description_attr)
        # it's not clear what entity info we'd draw. this is just cargo as of
        # 2024-10-29
        #elif entity.target_entity_id == self.selected_target:
        #    self.draw_entity_info(y+2, x, entity, description_attr)

    def draw_multiple_entities(self, y:int, x:int, entities:Sequence[core.AbstractSensorImage]) -> None:

        only_projectile = all(entity.identified and entity.identity.object_type == core.ObjectType.PROJECTILE for entity in entities)

        icons = set(interface.Icons.sensor_image_icon(x.identity) if x.identified else interface.Icons.UNKNOWN for x in entities)
        icon = icons.pop() if len(icons) == 1 else interface.Icons.MULTIPLE

        icon_attrs = set(interface.Icons.sensor_image_attr(x.identity) if x.identified else curses.color_pair(interface.Icons.COLOR_UNKNOWN) for x in entities)
        icon_attr = icon_attrs.pop() if len(icon_attrs) == 1 else 0

        prefixes = set(x.identity.id_prefix if x.identified else "???" for x in entities)

        self.view.viewscreen.addstr(y, x, icon, icon_attr)

        if not only_projectile:
            prefix = prefixes.pop() if len(prefixes) == 1 else "entities"
            self.view.viewscreen.addstr(y, x+1, f' {len(entities)} {prefix}', icon_attr | curses.A_DIM)

    def draw_cells(self, occupied:Mapping[Tuple[int,int], Sequence[core.AbstractSensorImage]]) -> None:
        for loc, entities in occupied.items():
            if len(entities) > 1:
                self.draw_multiple_entities(
                        loc[1], loc[0], entities)
            else:
                icon_attr = 0

                self.draw_entity(loc[1], loc[0], entities[0], icon_attr=icon_attr)

    def draw_sector_map(self) -> None:
        last_loc = None
        occupied:Dict[Tuple[int,int], List[core.AbstractSensorImage]] = {}

        for effect in self.sector.current_effects():
            if util.intersects(effect.bbox(), self.perspective.bbox):
                self.draw_effect(effect)

        for entity in self.visible_entities():
            screen_x, screen_y = self.perspective.sector_to_screen(entity.loc[0], entity.loc[1])
            last_loc = (screen_x, screen_y)
            if last_loc in occupied:
                if isinstance(entity, core.Projectile):
                        continue
                entities = occupied[last_loc]
                if isinstance(entities[0], core.Projectile):
                    occupied[last_loc] = [entity]
                else:
                    entities.append(entity)
            else:
                occupied[last_loc] = [entity]

        self.draw_cells(occupied)

        #DEBUG: show collision avoidance sensor cone for selected target
        if self.show_sensor_cone and self.selected_target:
            selected_entity = self.sector.entities[self.selected_target]
            if isinstance(selected_entity, core.Ship):
                self.draw_sensor_cone(selected_entity)

        #TODO: draw an indicator for off-screen targeted entities

        if self.debug_entity_vectors and self.selected_target:
            underlying_entity = self.sector.entities[self.selected_target]
            screen_x, screen_y = self.perspective.sector_to_screen(underlying_entity.loc[0], underlying_entity.loc[1])

            self.draw_entity_vectors(screen_y, screen_x, underlying_entity)

    def draw_shapes(self) -> None:
        for entity in self.visible_entities():
            if entity.identity.radius > 0 and self.perspective.meters_per_char[0] < entity.identity.radius:
                self.draw_entity_shape(entity)

    def draw_weather(self) -> None:
        for weather in self.sector.region_query(self.perspective.bbox):
            screen_x, screen_y = self.perspective.sector_to_screen(*weather.loc)
            c = util.make_circle_canvas(weather.radius, *self.perspective.meters_per_char, bbox=util.translate_rect(self.perspective.bbox, -weather.loc))
            #TODO: should not need to do this
            assert isinstance(self.view.viewscreen, interface.Canvas)
            window = self.view.viewscreen.window
            util.draw_canvas_at(c, window, screen_y, screen_x, bounds=self.view.viewscreen_bounds)


    def draw_sensor_cone(self, ship:core.Ship) -> None:
        """ Visualize sensor cone used, e.g. in navigation/collision avoid """
        # these are upper bounds on what gets used in collision avoidance
        neighborhood_radius = 8.5e3
        collision_margin = 1e3

        current_order = ship.current_order()
        if isinstance(current_order, steering.AbstractSteeringOrder):
            neighborhood_radius = current_order.computed_neighborhood_radius
            collision_margin = current_order.collision_margin

        neighborhood_loc = collision.compute_neighborhood_center(ship.phys, neighborhood_radius, collision_margin)

        content = self.compute_sensor_cone(ship, neighborhood_radius, collision_margin)

        s_x, s_y = (neighborhood_loc - self.perspective.cursor) / self.perspective.meters_per_char
        s_x = int(round(s_x))
        s_y = int(round(s_y))
        for (y,x), c in content.items():
            self.view.viewscreen.addstr(y+s_y, x+s_x, c)
        #assert isinstance(self.view.viewscreen, interface.Canvas)
        #window = self.view.viewscreen.window
        #util.draw_canvas_at(c, window, s_y, s_x, bounds=self.view.viewscreen_bounds)

    def draw_sensor_rings(self, ship:core.SectorEntity) -> None:
        radii = self.sector.sensor_manager.sensor_ranges(ship)
        content = compute_sensor_rings_memoize(
            tuple(map(quantize, radii)),
            self.perspective.bbox[2] - self.perspective.bbox[0],
            self.perspective.bbox[3] - self.perspective.bbox[1],
            self.view.viewscreen_bounds,
            self.perspective.meters_per_char
        )

        sensor_color = self.view.interface.get_color(interface.Color.SENSOR_RING)
        for (y,x), c in content.items():
            self.view.viewscreen.addstr(y, x, c, sensor_color)

        s_x, s_y = self.perspective.sector_to_screen(ship.loc[0]+radii[0], ship.loc[1])
        self.view.viewscreen.addstr(s_y, s_x, "A", sensor_color)
        s_x, s_y = self.perspective.sector_to_screen(ship.loc[0]+radii[1], ship.loc[1])
        self.view.viewscreen.addstr(s_y, s_x, "B", sensor_color)
        s_x, s_y = self.perspective.sector_to_screen(ship.loc[0]+radii[2], ship.loc[1])
        self.view.viewscreen.addstr(s_y, s_x, "C", sensor_color)

    def draw_profile_rings(self, ship:core.SectorEntity) -> None:
        radii = self.sector.sensor_manager.profile_ranges(ship)
        content = compute_sensor_rings_memoize(
            tuple(map(quantize, radii)),
            self.perspective.bbox[2] - self.perspective.bbox[0],
            self.perspective.bbox[3] - self.perspective.bbox[1],
            self.view.viewscreen_bounds,
            self.perspective.meters_per_char
        )

        sensor_color = self.view.interface.get_color(interface.Color.PROFILE_RING)
        for (y,x), c in content.items():
            self.view.viewscreen.addstr(y, x, c, sensor_color)

        s_x, s_y = self.perspective.sector_to_screen(ship.loc[0]+radii[0], ship.loc[1])
        self.view.viewscreen.addstr(s_y+1, s_x, "A", sensor_color)
        s_x, s_y = self.perspective.sector_to_screen(ship.loc[0]+radii[1], ship.loc[1])
        self.view.viewscreen.addstr(s_y+1, s_x, "B", sensor_color)

    def draw_cymunk_shapes(self) -> None:
        """ Debug visualization of cymunk shapes. """
        assert isinstance(self.view.viewscreen, interface.Canvas)
        window = self.view.viewscreen.window
        for shape in self.sector.space.bb_query(cymunk.BB(*self.view.viewscreen_bounds)):
            if isinstance(shape, cymunk.Circle):
                loc_x, loc_y = shape.body.position[0], shape.body.position[1]
                screen_x, screen_y = self.perspective.sector_to_screen(loc_x, loc_y)
                c = util.make_circle_canvas(shape.radius, *self.perspective.meters_per_char)
                util.draw_canvas_at(c, window, screen_y, screen_x, bounds=self.view.viewscreen_bounds)
            elif isinstance(shape, cymunk.Poly):
                loc_x, loc_y = shape.body.position[0], shape.body.position[1]
                screen_x, screen_y = self.perspective.sector_to_screen(loc_x, loc_y)
                c = util.make_polygon_canvas(shape.get_vertices(), *self.perspective.meters_per_char, offset_x=-loc_x, offset_y=-loc_y)
                util.draw_canvas_at(c, window, screen_y, screen_x, bounds=self.view.viewscreen_bounds)
            else:
                raise ValueError(f'do not know how to draw {type(shape)}')

    def update(self) -> None:
        pass

class SectorPresenter(Presenter):
    #TODO: presenter for sector wide view
    def visible_entities(self) -> Iterable[core.AbstractSensorImage]:
        return []

    @property
    def selected_target_image(self) -> core.AbstractSensorImage:
        assert self.selected_target
        raise NotImplementedError("ohnoes")

class PilotPresenter(Presenter):
    def __init__(self, ship:core.Ship, *args:Any, **kwargs:Any):
        super().__init__(*args, **kwargs)
        self.sensor_image_manager = sensors.SensorImageManager(ship, 120.0)

    def update(self) -> None:
        self.sensor_image_manager.update()

    def visible_entities(self) -> Iterable[core.AbstractSensorImage]:
        for image in self.sensor_image_manager.spatial_query(self.perspective.bbox):
            yield image

    @property
    def selected_target_image(self) -> core.AbstractSensorImage:
        assert self.selected_target
        return self.sensor_image_manager.sensor_contacts[self.selected_target]
