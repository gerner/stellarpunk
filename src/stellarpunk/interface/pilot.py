""" Ship piloting interface.

Sits within a sector view.
"""

import math
import curses
import enum
from typing import Tuple, Optional, Any, Callable, Mapping, MutableMapping, Sequence

import numpy as np

from stellarpunk import core, interface, util, orders
from stellarpunk.interface import presenter, command_input
from stellarpunk.orders import steering, movement

DRIVE_KEYS = tuple(map(lambda x: ord(x), "wasdijkl"))
TRANSLATE_KEYS = tuple(map(lambda x: ord(x), "ijkl"))
TRANSLATE_DIRECTIONS = {
    ord("i"): 0.,
    ord("l"): np.pi/2.,
    ord("k"): np.pi,
    ord("j"): np.pi/-2.,
}

class PlayerControlOrder(steering.AbstractSteeringOrder):
    """ Order indicating the player is in direct control.

    This order does nothing (at all). It's important that this is cleaned up if
    the player at any time relinquishes control.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.has_command = False

    def _clip_force_to_max_speed(self, force:Tuple[float, float], dt:float, max_thrust:float) -> Tuple[float, float]:
        # clip force s.t. resulting speed (after dt) is at most max_speed
        # we do this by computing what the resulting speed will be after dt
        # and checking if that's ok and if not, finding a (positive) scale for
        # 
        r_v, theta_v = util.cartesian_to_polar(*(self.ship.velocity))
        r_h, theta_h = util.cartesian_to_polar(force[0], force[1])
        r_h = r_h / self.ship.mass * dt
        expected_speed = np.sqrt(r_v**2 + r_h**2 + 2 * r_v * r_h * np.cos(theta_h - theta_v))

        if expected_speed > self.ship.max_speed():
            # this equation solves for exactly hitting the max speed:
            # r_h**2 + 2*r_v*cos(theta_h - theta_v) * r_h + r_v**2 - self.ship.max_speed(**2) = 0
            # find roots of that equation
            a = 1.
            b = 2*r_v*np.cos(theta_v - theta_h)
            c = r_v**2 - self.ship.max_speed()**2

            # check for no solution? (e.g. perpendicular thrust)
            # and just do no thrust
            if b**2 - 4*a*c < 0.:
                r_desired = 0.
            else:
                r_1 = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
                r_2 = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)

                # pick the positive one, if one exists
                r_desired = 0.
                if r_1 >= 0:
                    r_desired = r_1
                elif r_2 >= 0:
                    r_desired = r_2

            # convert back to a force and clip to max thrust

            if r_desired <= 0.:
                return (0., 0.)

            r_desired = np.clip(r_desired * self.ship.mass / dt, 0., max_thrust)
            return util.polar_to_cartesian(r_desired, theta_h)
        else:
            return force


    def act(self, dt:float) -> None:
        # if the player is controlling, do nothing and wait until next tick
        if self.has_command:
            self.has_command = False
            return

        # otherwise try to kill rotation
        if self.ship.phys.angular_velocity == 0.:
            return

        t = self.ship.moment * -1 * self.ship.angular_velocity / dt
        if t == 0:
            self.ship.phys.angular_velocity = 0.
        else:
            self.ship.apply_torque(np.clip(t, -1*self.ship.max_torque, self.ship.max_torque))

    # action functions, imply player direct input

    def accelerate(self, dt:float) -> None:
        self.has_command = True
        #TODO: up to max speed?
        force = util.polar_to_cartesian(self.ship.max_thrust, self.ship.angle)

        force = self._clip_force_to_max_speed(force, dt, self.ship.max_thrust)

        if not np.allclose(force, steering.ZERO_VECTOR):
            self.ship.apply_force(force)

    def kill_velocity(self, dt:float) -> None:
        self.has_command = True
        if self.ship.angular_velocity == 0 and np.allclose(self.ship.velocity, steering.ZERO_VECTOR):
            return
        self._accelerate_to(steering.ZERO_VECTOR, dt)

    def rotate(self, scale:float, dt:float) -> None:
        """ Rotates the ship in desired direction

        scale is the direction 1 is clockwise,-1 is counter clockwise
        """

        self.has_command = True
        #TODO: up to max angular acceleration?
        self.ship.apply_torque(self.ship.max_torque * scale)

    def translate(self, direction:float, dt:float) -> None:
        """ Translates the ship in the desired direction

        direction is an angle relative to heading
        """

        #TODO: up to max speed?
        force = util.polar_to_cartesian(self.ship.max_fine_thrust, self.ship.angle + direction)

        force = self._clip_force_to_max_speed(force, dt, self.ship.max_fine_thrust)

        self.ship.apply_force(force)

class MouseState(enum.Enum):
    """ States to interpret mouse clicks.

    e.g. maybe a click means select a target, maybe it means go to position.
    """

    CLEAR_INTERVAL = 5

    EMPTY = enum.auto()
    GOTO = enum.auto()

class PilotView(interface.View):
    """ Piloting mode: direct command of a ship. """

    def __init__(self, ship:core.Ship, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        self.ship = ship
        if self.ship.sector is None:
            raise ValueError("ship must be in a sector to pilot")

        # where the sector map is centered in sector coordinates
        self.scursor_x = 0.
        self.scursor_y = 0.

        # sector zoom level, expressed in meters to fit on screen
        self.szoom = 40.*1000.
        self.meters_per_char_x = 0.
        self.meters_per_char_y = 0.

        self.direction_indicator_radius = 15
        self.heading_indicator_radius = 12
        self.target_indicator_radius = 13
        self.velocity_indicator_radius_min = 2
        self.velocity_indicator_radius_max = 14

        # sector coord bounding box (ul_x, ul_y, lr_x, lr_y)
        self.bbox = (0.,0.,0.,0.)

        self._cached_radar = (util.NiceScale(0,0), util.NiceScale(0,0), util.NiceScale(0,0), util.NiceScale(0,0), "")

        self.presenter = presenter.Presenter(self.interface.gamestate, self, self.ship.sector, self.bbox, self.meters_per_char_x, self.meters_per_char_y)

        # indicates if the ship should follow its orders, or direct player
        # control
        self.autopilot_on = False
        self.control_order:Optional[PlayerControlOrder] = None

        self.goto_order:Optional[movement.GoToLocation] = None

        self.selected_entity:Optional[core.SectorEntity] = None

        self.mouse_state = MouseState.EMPTY
        self.mouse_state_clear_time = np.inf


    def _command_list(self) -> Mapping[str, command_input.CommandInput.CommandSig]:

        def order_jump(args:Sequence[str]) -> None:
            if self.selected_entity is None or not isinstance(self.selected_entity, core.TravelGate):
                raise command_input.CommandInput.UserError("can only jump through travel gates as selected target")
            order = orders.TravelThroughGate(self.selected_entity, self.ship, self.interface.gamestate)
            self.ship.prepend_order(order)

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
            self.selected_entity.clear_orders()
            order = orders.GoToLocation(np.array((x,y)), self.selected_entity, self.interface.gamestate)
            self.selected_entity.prepend_order(order)

        return {
            "clear_orders": lambda x: self.ship.clear_orders(),
            "jump": order_jump,
        }

    def _open_command_prompt(self) -> bool:
        self.interface.open_view(command_input.CommandInput(
            self.interface, commands=self._command_list()))
        return True

    def _select_target(self, entity:Optional[core.SectorEntity]) -> None:
        if entity is None:
            self.selected_entity = None
            self.presenter.selected_target = None
        else:
            self.selected_entity = entity
            self.presenter.selected_target = entity.entity_id

    def _toggle_autopilot(self) -> bool:
        """ Toggles autopilot state.

        If autopilot is on, we follow normal orders for the ship. Otherwise we
        suspend the current order queue and follow the user's input directly.
        """

        if self.autopilot_on:
            self.logger.info("entering autopilot")

            if self.control_order is None:
                raise ValueError("autopilot on, but no control order while toggling autopilot")
            self.control_order.cancel_order()
            self.control_order = None
            self.autopilot_on = False
        else:
            self.logger.info("exiting autopilot")

            if self.control_order is not None:
                raise ValueError("autopilot off, but has control order while toggling autopilot")
            control_order = PlayerControlOrder(self.ship, self.interface.gamestate)
            self.ship.prepend_order(control_order)
            self.control_order = control_order
            self.autopilot_on = True

        return True

    def _drive(self, key:int, dt:float) -> bool:
        """ Inputs a direct navigation control for the ship.

        "w" increases velocity, "a" rotates left, "d" rotates right,
        "s" attempts to kill velocity. All of these give one step of the desired
        action.
        """

        if not self.autopilot_on:
            return True
        elif self.control_order is None:
            raise ValueError("autopilot on, but no control order set")

        if key == ord("w"):
            self.control_order.accelerate(dt)
        elif key == ord("s"):
            self.control_order.kill_velocity(dt)
        elif key in (ord("a"), ord("d")):
            self.control_order.rotate(1. if key == ord("d") else -1., dt)
        elif key in TRANSLATE_KEYS:
            self.control_order.translate(TRANSLATE_DIRECTIONS[key], dt)

        return True

    def _zoom_scursor(self, key:int) -> bool:
        if key == ord("+"):
            self.szoom *= 0.9
        elif key == ord("-"):
            self.szoom *= 1.1
        else:
            raise ValueError("can only zoom + or -")
        self._update_bbox()
        return True

    def _handle_cancel(self) -> bool:
        """ Cancels current operation (e.g. deselect target) """
        if self.selected_entity is not None:
            self._select_target(None)
            return True
        else:
            return False

    def _handle_mouse(self) -> bool:
        """ Orders trip to go to location via autopilot. """
        m_tuple = curses.getmouse()
        m_id, m_x, m_y, m_z, bstate = m_tuple

        ul_x = self.scursor_x - (self.interface.viewscreen_width/2 * self.meters_per_char_x)
        ul_y = self.scursor_y - (self.interface.viewscreen_height/2 * self.meters_per_char_y)
        sector_x, sector_y = util.screen_to_sector(
                m_x, m_y, ul_x, ul_y,
                self.meters_per_char_x, self.meters_per_char_y,
                self.interface.viewscreen_x, self.interface.viewscreen_y)


        if self.mouse_state == MouseState.EMPTY:
            # select a target within a cell of the mouse click
            if self.ship.sector is None:
                raise ValueError("ship must be in a sector to select a target")

            hit = next(self.ship.sector.spatial_point(np.array((sector_x, sector_y)), self.meters_per_char_y), None)
            if hit:
                #TODO: check if the hit is close enough
                self._select_target(hit)

            return True
        elif self.mouse_state == MouseState.GOTO:
            if self.autopilot_on:
                self._toggle_autopilot()

            if self.goto_order is not None:
                self.goto_order.cancel_order()

            goto_order = movement.GoToLocation(np.array((sector_x, sector_y)), self.ship, self.interface.gamestate, arrival_distance=5e2)
            self.ship.prepend_order(goto_order)
            self.goto_order = goto_order

            self.mouse_state = MouseState.EMPTY

            return True
        else:
            raise ValueError(f'unknown mouse state {self.mouse_state}')

    def _next_target(self, direction:int) -> bool:
        """ Selects the next or previous target from a sorted list. """

        if self.ship.sector is None:
            raise ValueError("ship must be in a sector to select a target")

        if self.selected_entity is None:
            self._select_target(next(self.ship.sector.spatial_point(self.ship.loc), None))

        return True

    def handle_input(self, key:int, dt:float) -> bool:

        if self.interface.gamestate.timestamp > self.mouse_state_clear_time:
            self.mouse_state = MouseState.EMPTY

        if key == curses.ascii.ESC: return self._handle_cancel()
        elif key == ord(":"): return self._open_command_prompt()
        elif key in (ord("+"), ord("-")): return self._zoom_scursor(key)
        elif key == ord("p"): return self._toggle_autopilot()
        elif key in DRIVE_KEYS: return self._drive(key, dt)
        elif key == curses.KEY_MOUSE: return self._handle_mouse()
        elif key == ord("g"): self.mouse_state = MouseState.GOTO
        elif key == ord("t"): return self._next_target(1)
        elif key == ord("r"): return self._next_target(-1)

        return True

    def _compute_radar(self, max_ticks:int=10) -> None:
        self._cached_radar = util.compute_uiradar(
                (self.scursor_x, self.scursor_y),
                self.bbox, self.meters_per_char_x, self.meters_per_char_y)

    def _update_bbox(self, recompute_radar:bool=True) -> None:
        if self.ship.sector is None:
            raise ValueError(f'ship {self.ship} is not in any sector')

        self.meters_per_char_x, self.meters_per_char_y = self._meters_per_char()

        vsw = self.interface.viewscreen_width
        vsh = self.interface.viewscreen_height

        ul_x = self.scursor_x - (vsw/2 * self.meters_per_char_x)
        ul_y = self.scursor_y - (vsh/2 * self.meters_per_char_y)
        lr_x = self.scursor_x + (vsw/2 * self.meters_per_char_x)
        lr_y = self.scursor_y + (vsh/2 * self.meters_per_char_y)

        self.bbox = (ul_x, ul_y, lr_x, lr_y)

        if recompute_radar:
            self._compute_radar()

        self.presenter.sector = self.ship.sector
        self.presenter.bbox = self.bbox
        self.presenter.meters_per_char_x = self.meters_per_char_x
        self.presenter.meters_per_char_y = self.meters_per_char_y

    def _auto_pan(self) -> None:
        """ Pans the viewscreen to center on the ship, but only if the ship has
        moved enough. Avoid's updating the bbox on every tick. """

        # exit early if recentering won't move by half a display cell
        if (abs(self.ship.loc[0] - self.scursor_x) < self.meters_per_char_x/2.
            and abs(self.ship.loc[1] - self.scursor_y) < self.meters_per_char_y/2.):
            return

        self.scursor_x = self.ship.loc[0]
        self.scursor_y = self.ship.loc[1]
        self._update_bbox(recompute_radar=False)

    def _draw_radar(self) -> None:
        """ Draws a grid at tick lines. """

        major_ticks_x, minor_ticks_y, major_ticks_y, minor_ticks_x, text = self._cached_radar

        for lineno, line in enumerate(text):
            self.viewscreen.viewscreen.addstr(lineno, 0, line, curses.color_pair(29))

        # draw location indicators
        i = major_ticks_x.niceMin
        while i <= major_ticks_x.niceMax:
            s_x, s_y = util.sector_to_screen(
                    i+self.scursor_x, self.scursor_y,
                    self.bbox[0], self.bbox[1],
                    self.meters_per_char_x, self.meters_per_char_y)
            self.viewscreen.addstr(s_y, s_x, util.human_distance(i), curses.color_pair(29))
            i += major_ticks_x.tickSpacing
        j = major_ticks_y.niceMin
        while j <= major_ticks_y.niceMax:
            s_x, s_y = util.sector_to_screen(
                    self.scursor_x, j+self.scursor_y,
                    self.bbox[0], self.bbox[1],
                    self.meters_per_char_x, self.meters_per_char_y)
            self.viewscreen.addstr(s_y, s_x, util.human_distance(j), curses.color_pair(29))
            j += major_ticks_y.tickSpacing

        # draw degree indicators
        r = self.meters_per_char_y * self.direction_indicator_radius
        for theta in np.linspace(0., 2*np.pi, 12, endpoint=False):
            # convert to 0 at positive x to get the right display
            x,y = util.polar_to_cartesian(r, theta-np.pi/2)
            s_x, s_y = util.sector_to_screen(self.scursor_x+x, self.scursor_y+y, self.bbox[0], self.bbox[1], self.meters_per_char_x, self.meters_per_char_y)
            self.viewscreen.addstr(s_y, s_x, f'{math.degrees(theta):.0f}°', curses.color_pair(29))

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

    def _draw_target_indicators(self) -> None:
        if self.goto_order is not None:
            if self.goto_order.is_complete():
                self.goto_order = None
            else:
                s_x, s_y = util.sector_to_screen(self.goto_order.target_location[0], self.goto_order.target_location[1], self.bbox[0], self.bbox[1], self.meters_per_char_x, self.meters_per_char_y)

                self.viewscreen.addstr(s_y, s_x, interface.Icons.LOCATION_INDICATOR, curses.color_pair(interface.Icons.COLOR_LOCATION_INDICATOR))

    def _draw_nav_indicators(self) -> None:
        """ Draws navigational indicators on the display.

        Includes velocity and heading.
        """

        # heading, on a circle at a fixed distance from the center
        x,y = util.polar_to_cartesian(self.meters_per_char_y * self.heading_indicator_radius, self.ship.angle)
        s_x, s_y = util.sector_to_screen(self.scursor_x+x, self.scursor_y+y, self.bbox[0], self.bbox[1], self.meters_per_char_x, self.meters_per_char_y)

        self.viewscreen.addstr(s_y, s_x, interface.Icons.HEADING_INDICATOR, curses.color_pair(interface.Icons.COLOR_HEADING_INDICATOR))

        if self.selected_entity is not None:
            distance, bearing = util.cartesian_to_polar(*(self.selected_entity.loc - self.ship.loc))
            x,y = util.polar_to_cartesian(self.meters_per_char_y * self.target_indicator_radius, bearing)
            s_x, s_y = util.sector_to_screen(self.scursor_x+x, self.scursor_y+y, self.bbox[0], self.bbox[1], self.meters_per_char_x, self.meters_per_char_y)

            self.viewscreen.addstr(s_y, s_x, interface.Icons.TARGET_DIRECTION_INDICATOR, curses.color_pair(interface.Icons.COLOR_TARGET_DIRECTION_INDICATOR))

        # velocity, on a circle, radius expands with velocity from 0 to max
        vmag, vangle = util.cartesian_to_polar(*self.ship.velocity)
        if not util.isclose(vmag, 0.):
            r = vmag / self.ship.max_speed() * (self.velocity_indicator_radius_max - self.velocity_indicator_radius_min) + self.velocity_indicator_radius_min
            x,y = util.polar_to_cartesian(self.meters_per_char_y * r, vangle)
            s_x, s_y = util.sector_to_screen(self.scursor_x+x, self.scursor_y+y, self.bbox[0], self.bbox[1], self.meters_per_char_x, self.meters_per_char_y)

            self.viewscreen.addstr(s_y, s_x, interface.Icons.VELOCITY_INDICATOR, curses.color_pair(interface.Icons.COLOR_VELOCITY_INDICATOR))

    def _meters_per_char(self) -> Tuple[float, float]:
        meters_per_char_x = self.szoom / min(self.interface.viewscreen_width, math.floor(self.interface.viewscreen_height/self.interface.font_width*self.interface.font_height))
        meters_per_char_y = meters_per_char_x / self.interface.font_width * self.interface.font_height

        assert self.szoom / meters_per_char_y <= self.interface.viewscreen_height
        assert self.szoom / meters_per_char_x <= self.interface.viewscreen_width

        return (meters_per_char_x, meters_per_char_y)

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
        label_bearing = "bearing:"
        label_distance = "distance:"
        distance, bearing = util.cartesian_to_polar(*(self.selected_entity.loc - self.ship.loc))
        rel_bearing = bearing - self.ship.angle
        # convert bearing so 0, North is negative y, instead of positive x
        bearing += np.pi/2
        self.viewscreen.addstr(status_y+1, status_x, f'{label_id:>12} {self.selected_entity.short_id()}')
        self.viewscreen.addstr(status_y+2, status_x, f'{label_speed:>12} {self.selected_entity.speed():.0f}m/s')
        self.viewscreen.addstr(status_y+3, status_x, f'{label_location:>12} {self.selected_entity.loc[0]:.0f},{self.selected_entity.loc[1]:.0f}')
        self.viewscreen.addstr(status_y+4, status_x, f'{label_bearing:>12} {math.degrees(util.normalize_angle(bearing)):.0f}° ({math.degrees(util.normalize_angle(rel_bearing, shortest=True)):.0f}°)')
        self.viewscreen.addstr(status_y+5, status_x, f'{label_distance:>12} {util.human_distance(distance)}')

    def _draw_status(self) -> None:
        current_order = self.ship.current_order()

        status_x = 1
        status_y = 1
        self.viewscreen.addstr(status_y, status_x, "Status:")

        label_speed = "speed:"
        label_location = "location:"
        label_heading = "heading:"
        label_order = "order:"
        # convert heading so 0, North is negative y, instead of positive x
        heading = self.ship.angle + np.pi/2
        self.viewscreen.addstr(status_y+1, status_x, f'{label_speed:>12} {self.ship.speed():.0f}m/s')
        self.viewscreen.addstr(status_y+2, status_x, f'{label_location:>12} {self.ship.loc[0]:.0f},{self.ship.loc[1]:.0f}')
        self.viewscreen.addstr(status_y+3, status_x, f'{label_heading:>12} {math.degrees(util.normalize_angle(heading)):.0f}°')
        self.viewscreen.addstr(status_y+4, status_x, f'{label_order:>12} {current_order}')

        status_y += 5

        if isinstance(current_order, movement.GoToLocation):
            # distance
            # neighborhood_density
            # nearest_neighbor_dist
            label_distance = "distance:"
            distance = util.distance(self.ship.loc, current_order.target_location)
            label_ndensity = "ndensity:"
            label_nndist = "nndist:"

            self.viewscreen.addstr(status_y, status_x, f'{label_distance:>12} {distance}')
            self.viewscreen.addstr(status_y+1, status_x, f'{label_ndensity:>12} {current_order.neighborhood_density}')
            self.viewscreen.addstr(status_y+2, status_x, f'{label_nndist:>12} {current_order.nearest_neighbor_dist}')

    def _draw_command_state(self) -> None:
        status_x = 1
        status_y = self.interface.viewscreen_height - 2
        if self.mouse_state == MouseState.GOTO:
            self.viewscreen.addstr(status_y, status_x, f'Go To')

    def _draw_hud(self) -> None:
        self._draw_target_indicators()
        self._draw_nav_indicators()
        self._draw_target_info()
        self._draw_status()
        self._draw_command_state()

    def initialize(self) -> None:
        self.logger.info(f'entering pilot mode for {self.ship.entity_id}')
        self.meters_per_char_x = 0.
        self.meters_per_char_y = 0.

        self._update_bbox()
        self.interface.reinitialize_screen(name="Pilot's Seat")

    def terminate(self) -> None:
        if self.autopilot_on:
            if self.control_order is None:
                raise ValueError("autopilot on, but no control order while terminating pilot view")
            self.control_order.cancel_order()

    def focus(self) -> None:
        self.interface.reinitialize_screen(name="Pilot's Seat")

    def update_display(self) -> None:
        self._auto_pan()

        #TODO: would be great not to erase the screen on every tick
        self.viewscreen.erase()
        self._draw_radar()
        self.presenter.draw_sector_map()

        # draw hud overlay on top of everything else
        self._draw_hud()

        self.interface.refresh_viewscreen()

