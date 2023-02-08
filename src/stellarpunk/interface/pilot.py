""" Ship piloting interface.

Sits within a sector view.
"""

import math
import curses
import enum
from typing import Tuple, Optional, Any, Callable, Mapping, Sequence, Collection, Dict

import numpy as np
import cymunk # type: ignore

from stellarpunk import core, interface, util, orders, config
from stellarpunk.interface import presenter, command_input, starfield, ui_util
from stellarpunk.interface import station as v_station
from stellarpunk.orders import steering, movement, collision

DRIVE_KEYS = tuple(map(lambda x: ord(x), "wasdijkl"))
TRANSLATE_KEYS = tuple(map(lambda x: ord(x), "ijkl"))
TRANSLATE_DIRECTIONS = {
    ord("i"): 0.,
    ord("l"): np.pi/2.,
    ord("k"): np.pi,
    ord("j"): np.pi/-2.,
}

class Settings:
    MAX_ANGULAR_VELOCITY = 2. # about 115 degrees per second

class LambdaOrderObserver(core.OrderObserver):
    def __init__(
        self,
        begin: Optional[Callable[[core.Order], None]] = None,
        complete: Optional[Callable[[core.Order], None]] = None,
        cancel: Optional[Callable[[core.Order], None]] = None,
    ):
        self.begin = begin
        self.complete = complete
        self.cancel = cancel

    def order_begin(self, order: core.Order) -> None:
        if self.begin:
            self.begin(order)

    def order_complete(self, order: core.Order) -> None:
        if self.complete:
            self.complete(order)

    def order_cancel(self, order: core.Order) -> None:
        if self.cancel:
            self.cancel(order)

class PlayerControlOrder(steering.AbstractSteeringOrder):
    """ Order indicating the player is in direct control.

    This order does nothing (at all). It's important that this is cleaned up if
    the player at any time relinquishes control.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.has_command = False

    def _clip_force_to_max_speed(self, force:Tuple[float, float], max_thrust:float) -> Tuple[float, float]:
        # clip force s.t. resulting speed (after dt) is at most max_speed
        # we do this by computing what the resulting speed will be after dt
        # and checking if that's ok and if not, finding a (positive) scale for
        # 
        r_v, theta_v = util.cartesian_to_polar(*(self.ship.velocity))
        r_h, theta_h = util.cartesian_to_polar(force[0], force[1])
        r_h = r_h / self.ship.mass * self.gamestate.dt
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

            r_desired = np.clip(r_desired * self.ship.mass / self.gamestate.dt, 0., max_thrust)
            return util.polar_to_cartesian(r_desired, theta_h)
        else:
            return force

    def _clip_torque_to_max_angular_velocity(self, torque:float, max_torque:float, max_angular_velocity:float) -> float:
        # clip torque s.t. resulting angular velocity (after dt) is at most
        # max_angular_velocity
        # w = w_0 + torque * dt

        expected_w = self.ship.phys.angular_velocity + torque/self.ship.moment * self.gamestate.dt
        if abs(expected_w) > max_angular_velocity:
            if expected_w < 0:
                return (-max_angular_velocity - self.ship.phys.angular_velocity)/self.gamestate.dt*self.ship.moment
            else:
                return (max_angular_velocity - self.ship.phys.angular_velocity)/self.gamestate.dt*self.ship.moment
        return torque

    def act(self, dt:float) -> None:
        # if the player is controlling, do nothing and wait until next tick
        if self.has_command:
            self.has_command = False
            self.gamestate.schedule_order(self.gamestate.timestamp + 1/10, self)
            return

        self.ship.apply_force(steering.ZERO_VECTOR, False)
        # otherwise try to kill rotation
        # apply torque up to max torque to kill angular velocity
        # torque = moment * angular_acceleration
        # the perfect acceleration would be -1 * angular_velocity / timestep
        # implies torque = moment * -1 * angular_velocity / timestep
        t = np.clip(self.ship.moment * -1 * self.ship.angular_velocity / dt, -self.ship.max_torque, self.ship.max_torque)
        if t == 0:
            self.ship.phys.angular_velocity = 0
            # schedule again to get cleaned up on next tick
            self.ship.apply_torque(0., False)
            self.gamestate.schedule_order(self.gamestate.timestamp + 1/10, self)
        else:
            self.ship.apply_torque(t, True)

            self.gamestate.schedule_order_immediate(self)

    # action functions, imply player direct input

    def accelerate(self) -> None:
        self.has_command = True
        #TODO: up to max speed?
        force = util.polar_to_cartesian(self.ship.max_thrust, self.ship.angle)

        force = self._clip_force_to_max_speed(force, self.ship.max_thrust)

        self.ship.apply_torque(0., False)
        if not np.allclose(force, steering.ZERO_VECTOR):
            self.ship.apply_force(force, False)
        else:
            self.ship.apply_force(steering.ZERO_VECTOR, False)

    def kill_velocity(self) -> None:
        self.has_command = True
        if self.ship.angular_velocity == 0 and np.allclose(self.ship.velocity, steering.ZERO_VECTOR):
            return
        #TODO: handle continuous force/torque
        period = collision.accelerate_to(
                self.ship.phys, cymunk.Vec2d(0,0), self.gamestate.dt,
                self.ship.max_speed(), self.ship.max_torque,
                self.ship.max_thrust, self.ship.max_fine_thrust)

    def rotate(self, scale:float) -> None:
        """ Rotates the ship in desired direction

        scale is the direction 1 is clockwise,-1 is counter clockwise
        """

        self.has_command = True
        #TODO: up to max angular acceleration?
        self.ship.apply_force(steering.ZERO_VECTOR, False)
        self.ship.apply_torque(
            self._clip_torque_to_max_angular_velocity(
                self.ship.max_torque * scale,
                self.ship.max_torque,
                Settings.MAX_ANGULAR_VELOCITY
            ),
            False
        )

    def translate(self, direction:float) -> None:
        """ Translates the ship in the desired direction

        direction is an angle relative to heading
        """

        self.has_command = True
        #TODO: up to max speed?
        force = util.polar_to_cartesian(self.ship.max_fine_thrust, self.ship.angle + direction)

        force = self._clip_force_to_max_speed(force, self.ship.max_fine_thrust)

        self.ship.apply_force(force, False)

class MouseState(enum.Enum):
    """ States to interpret mouse clicks.

    e.g. maybe a click means select a target, maybe it means go to position.
    """

    CLEAR_INTERVAL = 5

    EMPTY = enum.auto()
    GOTO = enum.auto()

class PilotView(interface.View, interface.PerspectiveObserver):
    """ Piloting mode: direct command of a ship. """

    def __init__(self, ship:core.Ship, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        self.ship = ship
        if self.ship.sector is None:
            raise ValueError("ship must be in a sector to pilot")

        # perspective on the sector, zoomed so the player ship's shape is
        # barely visible on screen
        self.perspective = interface.Perspective(
            self.interface.viewscreen,
            zoom=self.ship.radius,
            min_zoom=(6*config.Settings.generate.Universe.SECTOR_RADIUS_STD+config.Settings.generate.Universe.SECTOR_RADIUS_MEAN)/80,
            max_zoom=2*8*config.Settings.generate.SectorEntities.SHIP_RADIUS/80.,
        )
        self.perspective.observe(self)

        self.direction_indicator_radius = 15
        self.heading_indicator_radius = 12
        self.target_indicator_radius = 13
        self.velocity_indicator_radius_min = 2
        self.velocity_indicator_radius_max = 14

        # cache the radar, we'll regenerate it when the zoom changes
        self._cached_radar_zoom = 0.
        self._cached_radar:Tuple[util.NiceScale, util.NiceScale, util.NiceScale, util.NiceScale, Mapping[Tuple[int, int], str]] = (util.NiceScale(0,0), util.NiceScale(0,0), util.NiceScale(0,0), util.NiceScale(0,0), {})

        self.presenter = presenter.Presenter(self.gamestate, self, self.ship.sector, self.perspective)

        # indicates if the ship should follow its orders, or direct player
        # control
        self.autopilot_on = False
        self.control_order:Optional[PlayerControlOrder] = None

        self.selected_entity:Optional[core.SectorEntity] = None

        self.mouse_state = MouseState.EMPTY
        self.mouse_state_clear_time = np.inf

        self.starfield = starfield.Starfield(self.gamestate.sector_starfield, self.perspective)

    def open_station_view(self, dock_station: core.Station) -> None:
        # TODO: make sure we're within docking range?
        station_view = v_station.StationView(dock_station, self.ship, self.interface)
        self.interface.open_view(station_view, deactivate_views=True)

    def command_list(self) -> Collection[interface.CommandBinding]:

        def order_jump(args:Sequence[str]) -> None:
            if self.selected_entity is None or not isinstance(self.selected_entity, core.TravelGate):
                raise command_input.UserError("can only jump through travel gates as selected target")
            order = orders.TravelThroughGate(self.selected_entity, self.ship, self.gamestate)
            self.ship.clear_orders(self.gamestate)
            self.ship.prepend_order(order)

        def order_mine(args:Sequence[str]) -> None:
            if self.selected_entity is None or not isinstance(self.selected_entity, core.Asteroid):
                raise command_input.UserError("can only mine asteroids")
            order = orders.MineOrder(self.selected_entity, math.inf, self.ship, self.gamestate)
            self.ship.clear_orders(self.gamestate)
            self.ship.prepend_order(order)

        def order_dock(args:Sequence[str]) -> None:
            if not isinstance(self.selected_entity, core.Station):
                raise command_input.UserError("can only dock at stations")
            order = orders.DockingOrder(self.selected_entity, self.ship, self.gamestate)
            dock_station = self.selected_entity

            def complete_docking(order: core.Order) -> None:
                self.open_station_view(dock_station)

            order.observe(LambdaOrderObserver(complete=complete_docking))
            self.ship.clear_orders(self.gamestate)
            self.ship.prepend_order(order)

        def log_cargo(args:Sequence[str]) -> None:
            if np.sum(self.ship.cargo) == 0.:
                self.interface.log_message("No cargo on ship")
                return

            cargo:Dict[str,float] = {}
            resource_width = 0
            for resource, amount in enumerate(self.ship.cargo):
                if amount > 0:
                    cargo[ui_util.product_name(self.gamestate.production_chain, resource)] = amount

            amount_width = int(math.ceil(np.max(self.ship.cargo)/10.))
            cargo_lines = ["Ship cargo:"]
            for name, amount in cargo.items():
                cargo_lines.append(f'\t{name:>{resource_width}}: {amount:>{amount_width}.2f}')
            self.interface.log_message("\n".join(cargo_lines))

        return [
            self.bind_command("clear_orders", lambda x: self.ship.clear_orders(self.gamestate)),
            self.bind_command("jump", order_jump),
            self.bind_command("mine", order_mine),
            self.bind_command("dock", order_dock),
            self.bind_command("cargo", log_cargo),
        ]

    def _select_target(self, entity:Optional[core.SectorEntity]) -> None:
        if entity is None:
            self.selected_entity = None
            self.presenter.selected_target = None
        else:
            self.selected_entity = entity
            self.presenter.selected_target = entity.entity_id

    def _toggle_autopilot(self) -> None:
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
            self.ship.clear_orders(self.gamestate)

            if self.control_order is not None:
                raise ValueError("autopilot off, but has control order while toggling autopilot")
            control_order = PlayerControlOrder(self.ship, self.gamestate)
            self.ship.prepend_order(control_order)
            self.control_order = control_order
            self.autopilot_on = True

    def _drive(self, key:int) -> None:
        """ Inputs a direct navigation control for the ship.

        "w" increases velocity, "a" rotates left, "d" rotates right,
        "s" attempts to kill velocity. All of these give one step of the desired
        action.
        """

        if not self.autopilot_on:
            return
        elif self.control_order is None:
            raise ValueError("autopilot on, but no control order set")

        if key == ord("w"):
            self.control_order.accelerate()
        elif key == ord("s"):
            self.control_order.kill_velocity()
        elif key in (ord("a"), ord("d")):
            self.control_order.rotate(1. if key == ord("d") else -1.)
        elif key in TRANSLATE_KEYS:
            self.control_order.translate(TRANSLATE_DIRECTIONS[key])

    def _handle_cancel(self) -> None:
        """ Cancels current operation (e.g. deselect target) """
        if self.mouse_state != MouseState.EMPTY:
            self._cancel_mouse()
        elif self.selected_entity is not None:
            self._select_target(None)
        elif self.autopilot_on:
            self._toggle_autopilot()

    def _cancel_mouse(self) -> None:
            self.mouse_state = MouseState.EMPTY

    def handle_mouse(self, m_id: int, m_x: int, m_y: int, m_z: int, bstate: int) -> bool:
        """ Handle mouse input according to MouseState """
        if not bstate & (curses.BUTTON1_CLICKED | curses.BUTTON1_PRESSED):
            return False

        sector_x, sector_y = self.perspective.screen_to_sector(m_x, m_y)
        self.logger.debug(f'clicked {(m_x,m_y)} translates to {(sector_x,sector_y)}')

        if self.mouse_state == MouseState.EMPTY:
            # select a target within a cell of the mouse click
            if self.ship.sector is None:
                raise ValueError("ship must be in a sector to select a target")

            hit = next(self.ship.sector.spatial_point(np.array((sector_x, sector_y)), self.perspective.meters_per_char[1]), None)
            if hit:
                #TODO: check if the hit is close enough
                self._select_target(hit)

        elif self.mouse_state == MouseState.GOTO:
            if self.autopilot_on:
                self._toggle_autopilot()

            self.ship.clear_orders(self.gamestate)

            goto_order = movement.GoToLocation(np.array((sector_x, sector_y)), self.ship, self.gamestate, arrival_distance=5e2)
            self.ship.prepend_order(goto_order)

            self.mouse_state = MouseState.EMPTY

        else:
            raise ValueError(f'unknown mouse state {self.mouse_state}')
        return True

    def _next_target(self, direction:int) -> None:
        """ Selects the next or previous target from a sorted list. """

        if self.ship.sector is None:
            raise ValueError("ship must be in a sector to select a target")

        potential_targets = sorted(
            (x for x in self.ship.sector.spatial_point(self.ship.loc) if x != self.ship),
            key=lambda x: util.distance(self.ship.loc, x.loc)
        )

        if len(potential_targets) == 0:
            self._select_target(None)
            return

        if self.selected_entity is None:
            self._select_target(potential_targets[0])
            return

        try:
            idx = potential_targets.index(self.selected_entity)
            if direction > 0:
                self._select_target(potential_targets[(idx+1)%len(potential_targets)])
            else:
                self._select_target(potential_targets[(idx-1)%len(potential_targets)])

        except ValueError:
            self._select_target(potential_targets[0])

    def _start_goto(self) -> None:
        self.mouse_state = MouseState.GOTO

    def key_list(self) -> Collection[interface.KeyBinding]:
        key_list = [
            self.bind_key(curses.ascii.ESC, self._handle_cancel),
            self.bind_key(ord("+"), lambda: self.perspective.zoom_cursor(ord("+"))),
            self.bind_key(ord("-"), lambda: self.perspective.zoom_cursor(ord("-"))),
            self.bind_key(ord("p"), self._toggle_autopilot),
            self.bind_key(ord("g"), self._start_goto),
            self.bind_key(ord("t"), lambda: self._next_target(1), help_key="pilot_targt_cycle"),
            self.bind_key(ord("r"), lambda: self._next_target(-1), help_key="pilot_targt_cycle"),
            self.bind_key(ord("w"), lambda: self._drive(ord("w"))),
            self.bind_key(ord("a"), lambda: self._drive(ord("a")), help_key="pilot_rotate"),
            self.bind_key(ord("s"), lambda: self._drive(ord("s"))),
            self.bind_key(ord("d"), lambda: self._drive(ord("d")), help_key="pilot_rotate"),
            self.bind_key(ord("i"), lambda: self._drive(ord("i")), help_key="pilot_translate"),
            self.bind_key(ord("j"), lambda: self._drive(ord("j")), help_key="pilot_translate"),
            self.bind_key(ord("k"), lambda: self._drive(ord("k")), help_key="pilot_translate"),
            self.bind_key(ord("l"), lambda: self._drive(ord("l")), help_key="pilot_translate"),
        ]
        return key_list

    def _compute_radar(self, max_ticks:int=10) -> None:
        self._cached_radar_zoom = self.perspective.zoom
        self._cached_radar = util.compute_uiradar(
                self.perspective.cursor,
                self.perspective.bbox,
                *self.perspective.meters_per_char,
                bounds=self.viewscreen_bounds
                )

    def perspective_updated(self, perspective:interface.Perspective) -> None:
        if self.perspective.zoom != self._cached_radar_zoom:
            self._compute_radar()

        assert self.ship.sector

        self.presenter.sector = self.ship.sector

    def _auto_pan(self) -> None:
        """ Pans the viewscreen to center on the ship, but only if the ship has
        moved enough. Avoid's updating the bbox on every tick. """

        # exit early if recentering won't move by half a display cell
        if (abs(self.ship.loc[0] - self.perspective.cursor[0]) < self.perspective.meters_per_char[0]/2.
            and abs(self.ship.loc[1] - self.perspective.cursor[1]) < self.perspective.meters_per_char[1]/2.):
            return

        self.perspective.cursor = tuple(self.ship.loc)

    def _draw_radar(self) -> None:
        """ Draws a grid at tick lines. """

        major_ticks_x, minor_ticks_y, major_ticks_y, minor_ticks_x, radar_content = self._cached_radar

        for (y,x), c in radar_content.items():
            self.viewscreen.addstr(y, x, c, curses.color_pair(29))

        # draw location indicators
        i = major_ticks_x.niceMin
        while i <= major_ticks_x.niceMax:
            s_x, s_y = self.perspective.sector_to_screen(
                    i+self.perspective.cursor[0], self.perspective.cursor[1]
            )
            self.viewscreen.addstr(s_y, s_x, util.human_distance(i), curses.color_pair(29))
            i += major_ticks_x.tickSpacing
        j = major_ticks_y.niceMin
        while j <= major_ticks_y.niceMax:
            s_x, s_y = self.perspective.sector_to_screen(
                    self.perspective.cursor[0], j+self.perspective.cursor[1]
            )
            self.viewscreen.addstr(s_y, s_x, util.human_distance(j), curses.color_pair(29))
            j += major_ticks_y.tickSpacing

        # draw degree indicators
        r = self.perspective.meters_per_char[1] * self.direction_indicator_radius
        for theta in np.linspace(0., 2*np.pi, 12, endpoint=False):
            # convert to 0 at positive x to get the right display
            x,y = util.polar_to_cartesian(r, theta-np.pi/2)
            s_x, s_y = self.perspective.sector_to_screen(
                    self.perspective.cursor[0]+x, self.perspective.cursor[1]+y
            )
            self.viewscreen.addstr(s_y, s_x, f'{math.degrees(theta):.0f}°', curses.color_pair(29))

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

    def _draw_target_indicators(self) -> None:
        current_order = self.ship.current_order()
        if isinstance(current_order, movement.GoToLocation):
            s_x, s_y = self.perspective.sector_to_screen(*current_order._target_location)

            self.viewscreen.addstr(s_y, s_x, interface.Icons.LOCATION_INDICATOR, curses.color_pair(interface.Icons.COLOR_LOCATION_INDICATOR))

    def _draw_nav_indicators(self) -> None:
        """ Draws navigational indicators on the display.

        Includes velocity and heading.
        """

        # heading, on a circle at a fixed distance from the center
        x,y = util.polar_to_cartesian(self.perspective.meters_per_char[1] * self.heading_indicator_radius, self.ship.angle)
        s_x, s_y = self.perspective.sector_to_screen(self.perspective.cursor[0]+x, self.perspective.cursor[1]+y)

        self.viewscreen.addstr(s_y, s_x, interface.Icons.HEADING_INDICATOR, curses.color_pair(interface.Icons.COLOR_HEADING_INDICATOR))

        if self.selected_entity is not None:
            distance, bearing = util.cartesian_to_polar(*(self.selected_entity.loc - self.ship.loc))
            x,y = util.polar_to_cartesian(self.perspective.meters_per_char[1] * self.target_indicator_radius, bearing)
            s_x, s_y = self.perspective.sector_to_screen(self.perspective.cursor[0]+x, self.perspective.cursor[1]+y)

            self.viewscreen.addstr(s_y, s_x, interface.Icons.TARGET_DIRECTION_INDICATOR, curses.color_pair(interface.Icons.COLOR_TARGET_DIRECTION_INDICATOR))

        # velocity, on a circle, radius expands with velocity from 0 to max
        vmag, vangle = util.cartesian_to_polar(*self.ship.velocity)
        if not util.isclose(vmag, 0.):
            r = vmag / self.ship.max_speed() * (self.velocity_indicator_radius_max - self.velocity_indicator_radius_min) + self.velocity_indicator_radius_min
            x,y = util.polar_to_cartesian(self.perspective.meters_per_char[1] * r, vangle)
            s_x, s_y = self.perspective.sector_to_screen(self.perspective.cursor[0]+x, self.perspective.cursor[1]+y)

            self.viewscreen.addstr(s_y, s_x, interface.Icons.VELOCITY_INDICATOR, curses.color_pair(interface.Icons.COLOR_VELOCITY_INDICATOR))

    def _draw_target_info(self) -> None:
        if self.selected_entity is None:
            return

        info_width = 12 + 1 + 24
        status_x = self.interface.viewscreen.width - info_width
        status_y = 1

        self.viewscreen.addstr(status_y, status_x, "Target Info:")

        label_id = "id:"
        label_speed = "speed:"
        label_location = "location:"
        label_bearing = "bearing:"
        label_distance = "distance:"
        label_rel_speed = "rel speed:"
        label_eta = "eta:"

        rel_pos = self.selected_entity.loc - self.ship.loc
        rel_vel = self.selected_entity.velocity - self.ship.velocity
        rel_speed = util.magnitude(*rel_vel)
        distance, bearing = util.cartesian_to_polar(*rel_pos)
        rel_bearing = bearing - self.ship.angle
        # convert bearing so 0, North is negative y, instead of positive x
        bearing += np.pi/2
        # rel speed (toward and perpendicular to target)
        if distance > 0.:
            vel_toward = np.dot(rel_vel, rel_pos)/distance
            vel_perpendicular = util.magnitude(*(rel_vel - vel_toward * rel_pos / distance))
        else:
            vel_toward = 0.
            vel_perpendicular = 0.
        # eta to closest approach
        if rel_speed == 0.:
            approach_t = math.inf
        else:
            approach_t = -1 * np.dot(rel_vel / rel_speed, rel_pos) / rel_speed
        if approach_t < 0.:
            approach_t = math.inf
        if approach_t < math.inf:
            closest_approach = util.magnitude(*(rel_pos + rel_vel * approach_t))
        else:
            closest_approach = math.inf

        self.viewscreen.addstr(status_y+1, status_x, f'{label_id:>12} {self.selected_entity.short_id()}')
        self.viewscreen.addstr(status_y+2, status_x, f'{label_speed:>12} {util.human_speed(self.selected_entity.speed)}')
        self.viewscreen.addstr(status_y+3, status_x, f'{label_location:>12} {self.selected_entity.loc[0]:.0f},{self.selected_entity.loc[1]:.0f}')
        self.viewscreen.addstr(status_y+4, status_x, f'{label_bearing:>12} {math.degrees(util.normalize_angle(bearing)):.0f}° ({math.degrees(util.normalize_angle(rel_bearing, shortest=True)):.0f}°)')
        self.viewscreen.addstr(status_y+5, status_x, f'{label_distance:>12} {util.human_distance(distance)}')
        self.viewscreen.addstr(status_y+6, status_x, f'{label_rel_speed:>12} {util.human_speed(vel_toward)} ({util.human_speed(vel_perpendicular)})')
        if approach_t < 60*60:
            self.viewscreen.addstr(status_y+7, status_x, f'{label_eta:>12} {approach_t:.0f}s ({util.human_distance(closest_approach)})')

        status_y += 8

        if isinstance(self.selected_entity, core.Station):
            assert self.selected_entity.resource is not None
            label_product = "product:"
            self.viewscreen.addstr(status_y, status_x, f'{label_product:>12} {ui_util.product_name(self.gamestate.production_chain, self.selected_entity.resource, 20)}')
        elif isinstance(self.selected_entity, core.Asteroid):
            assert self.selected_entity.resource is not None
            label_ore = "ore:"
            self.viewscreen.addstr(status_y, status_x, f'{label_ore:>12} {ui_util.product_name(self.gamestate.production_chain, self.selected_entity.resource, 20)}')

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
        self.viewscreen.addstr(status_y+1, status_x, f'{label_speed:>12} {util.human_speed(self.ship.speed)} ({self.ship.phys.force.length}N)')
        self.viewscreen.addstr(status_y+2, status_x, f'{label_location:>12} {self.ship.loc[0]:.0f},{self.ship.loc[1]:.0f}')
        self.viewscreen.addstr(status_y+3, status_x, f'{label_heading:>12} {math.degrees(util.normalize_angle(heading)):.0f}° ({math.degrees(self.ship.phys.angular_velocity):.0f}°/s) ({self.ship.phys.torque:.2}N-m))')
        self.viewscreen.addstr(status_y+4, status_x, f'{label_order:>12} {current_order}')

        status_y += 5

        if isinstance(current_order, movement.GoToLocation):
            # distance
            # neighborhood_density
            # nearest_neighbor_dist
            label_distance = "distance:"
            distance = self.ship.phys.position.get_distance(current_order._target_location)
            label_ndensity = "ndensity:"
            label_nndist = "nndist:"

            self.viewscreen.addstr(status_y, status_x, f'{label_distance:>12} {distance}')
            self.viewscreen.addstr(status_y+1, status_x, f'{label_ndensity:>12} {current_order.num_neighbors}')
            self.viewscreen.addstr(status_y+2, status_x, f'{label_nndist:>12} {current_order.nearest_neighbor_dist}')

            status_y += 3

        if np.any(self.ship.cargo > 0.):
            label_cargo = "Cargo:"
            self.viewscreen.addstr(status_y, status_x, f'{label_cargo:>12} (capacity {self.ship.cargo_capacity:.0f})')
            status_y += 1
            for i in range(len(self.ship.cargo)):
                if self.ship.cargo[i] == 0.:
                    continue
                label = ui_util.product_name(self.gamestate.production_chain, i, 16)
                self.viewscreen.addstr(status_y, status_x, f'{label:>16}: {math.floor(self.ship.cargo[i])}')
                status_y += 1
        else:
            label_cargo = "No Cargo"
            self.viewscreen.addstr(status_y, status_x, f'{label_cargo:>12} (capacity {self.ship.cargo_capacity:.0f})')
            status_y += 1


    def _draw_command_state(self) -> None:
        status_x = 1
        status_y = self.interface.viewscreen.height - 2
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
        self.perspective.update_bbox()
        self.interface.reinitialize_screen(name="Pilot's Seat")

    def terminate(self) -> None:
        if self.autopilot_on:
            if self.control_order is None:
                raise ValueError("autopilot on, but no control order while terminating pilot view")
            self.control_order.cancel_order()

    def focus(self) -> None:
        super().focus()
        self.interface.reinitialize_screen(name="Pilot's Seat")
        self.active=True

    def update_display(self) -> None:
        if self.gamestate.timestamp > self.mouse_state_clear_time:
            self.mouse_state = MouseState.EMPTY

        self._auto_pan()

        self.viewscreen.erase()
        self.starfield.draw_starfield(self.viewscreen)
        self.presenter.draw_shapes()
        self._draw_radar()
        self.presenter.draw_sector_map()

        # draw hud overlay on top of everything else
        self._draw_hud()

        self.interface.refresh_viewscreen()

