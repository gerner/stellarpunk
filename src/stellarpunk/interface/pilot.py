""" Ship piloting interface.

Sits within a sector view.
"""

import re
import math
import curses
import enum
import collections
import uuid
from typing import Tuple, Optional, Any, Callable, Mapping, Sequence, Collection, Dict, MutableMapping, Set, Type

import numpy as np
import cymunk # type: ignore

from stellarpunk import core, interface, util, orders, config, intel
from stellarpunk.core import combat, sector_entity
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
        lifetime_collection:Set,
        *args:Any,
        begin: Optional[Callable[[core.Order], None]] = None,
        complete: Optional[Callable[[core.Order], None]] = None,
        cancel: Optional[Callable[[core.Order], None]] = None,
        **kwargs:Any
    ):
        super().__init__(*args, **kwargs)
        # we must handle at least one event
        assert begin or complete or cancel

        self.begin = begin
        self.complete = complete
        self.cancel = cancel


        # observers are held in weak references, so we need some other
        # way to keep this observer alive. the rule is that after we handle
        # one event, we go away
        lifetime_collection.add(self)
        self.lifetime_collection = lifetime_collection

    @property
    def observer_id(self) -> uuid.UUID:
        return core.OBSERVER_ID_NULL

    def order_begin(self, order: core.Order) -> None:
        if self.begin:
            self.begin(order)
            self.lifetime_collection.remove(self)

    def order_complete(self, order: core.Order) -> None:
        if self.complete:
            self.complete(order)
            self.lifetime_collection.remove(self)

    def order_cancel(self, order: core.Order) -> None:
        if self.cancel:
            self.cancel(order)
            self.lifetime_collection.remove(self)

class PlayerControlOrder(steering.AbstractSteeringOrder):
    """ Order indicating the player is in direct control.

    This order does nothing (at all). It's important that this is cleaned up if
    the player at any time relinquishes control.
    """

    @classmethod
    def create_player_control_order[T:"PlayerControlOrder"](cls:Type[T], dt:float, *args: Any, **kwargs: Any) -> T:
        return cls.create_abstract_steering_order(*args, dt, **kwargs)

    def __init__(self, dt:float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.dt = dt
        self.has_thrust_command = False
        self.has_torque_command = False

    def _clip_force_to_max_speed(self, force:Tuple[float, float], max_thrust:float) -> Tuple[float, float]:
        # clip force s.t. resulting speed (after dt) is at most max_speed
        # we do this by computing what the resulting speed will be after dt
        # and checking if that's ok and if not, finding a (positive) scale for
        # 
        r_v, theta_v = util.cartesian_to_polar(*(self.ship.velocity))
        r_h, theta_h = util.cartesian_to_polar(force[0], force[1])
        r_h = r_h / self.ship.mass * self.dt
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

            r_desired = np.clip(r_desired * self.ship.mass / self.dt, 0., max_thrust)
            return util.polar_to_cartesian(r_desired, theta_h)
        else:
            return force

    def _clip_torque_to_max_angular_velocity(self, torque:float, max_torque:float, max_angular_velocity:float) -> float:
        # clip torque s.t. resulting angular velocity (after dt) is at most
        # max_angular_velocity
        # w = w_0 + torque * dt

        expected_w = self.ship.phys.angular_velocity + torque/self.ship.moment * self.dt
        if abs(expected_w) > max_angular_velocity:
            if expected_w < 0:
                return (-max_angular_velocity - self.ship.phys.angular_velocity)/self.dt*self.ship.moment
            else:
                return (max_angular_velocity - self.ship.phys.angular_velocity)/self.dt*self.ship.moment
        return torque

    def act(self, dt:float) -> None:
        self.dt = dt
        # if the player is controlling, do nothing and wait until next tick
        if self.has_thrust_command:
            self.has_thrust_command = False
        else:
            self.ship.apply_force(steering.ZERO_VECTOR, False)

        rotate_time = 1/15

        if self.has_torque_command:
            self.has_torque_command = False
        else:
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
            else:
                self.ship.apply_torque(t, True)
                rotate_time = 1/60

        self.gamestate.schedule_order(self.gamestate.timestamp + min(rotate_time, 1/15), self)


    # action functions, imply player direct input

    def accelerate(self) -> None:
        self.has_thrust_command = True
        #TODO: up to max speed?
        force = util.polar_to_cartesian(self.ship.max_thrust, self.ship.angle)

        force = self._clip_force_to_max_speed(force, self.ship.max_thrust)

        if not np.allclose(force, steering.ZERO_VECTOR):
            self.ship.apply_force(force, False)
        else:
            self.ship.apply_force(steering.ZERO_VECTOR, False)

    def kill_velocity(self) -> None:
        self.has_thrust_command = True
        self.has_torque_command = True
        if self.ship.angular_velocity == 0 and np.allclose(self.ship.velocity, steering.ZERO_VECTOR):
            return
        #TODO: handle continuous force/torque
        period = collision.accelerate_to(
                self.ship.phys, cymunk.Vec2d(0,0), self.dt,
                self.ship.max_speed(), self.ship.max_torque,
                self.ship.max_thrust, self.ship.max_fine_thrust,
                self.ship.sensor_settings)

    def rotate(self, scale:float) -> None:
        """ Rotates the ship in desired direction

        scale is the direction 1 is clockwise,-1 is counter clockwise
        """

        self.has_torque_command = True
        #TODO: up to max angular acceleration?
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

        self.has_thrust_command = True
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

class PilotView(interface.PerspectiveObserver, core.SectorEntityObserver, interface.GameView):
    """ Piloting mode: direct command of a ship. """

    def __init__(self, ship:core.Ship, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        self.ship = ship
        if self.ship.sector is None:
            raise ValueError("ship must be in a sector to pilot")
        self.sector = self.ship.sector

        # perspective on the sector, zoomed so the player ship's shape is
        # barely visible on screen
        self.perspective = interface.Perspective(
            self.interface.viewscreen,
            zoom=self.ship.radius,
            min_zoom=(12*config.Settings.generate.Universe.SECTOR_RADIUS_STD+config.Settings.generate.Universe.SECTOR_RADIUS_MEAN)/80,
            max_zoom=2*8*config.Settings.generate.SectorEntities.ship.RADIUS/80.,
        )
        self.perspective.observe(self)

        self.m_sector_x, self.m_sector_y = (0.0, 0.0)

        self.direction_indicator_radius = 15
        self.heading_indicator_radius = 12
        self.target_indicator_radius = 13
        self.velocity_indicator_radius_min = 2
        self.velocity_indicator_radius_max = 14

        # cache the radar, we'll regenerate it when the zoom changes
        self._cached_radar_zoom = 0.
        self._cached_radar:Tuple[util.NiceScale, util.NiceScale, util.NiceScale, util.NiceScale, Mapping[Tuple[int, int], str]] = (util.NiceScale(0,0), util.NiceScale(0,0), util.NiceScale(0,0), util.NiceScale(0,0), {})

        self.presenter = presenter.PilotPresenter(self.ship, self.gamestate, self, self.sector, self.perspective)

        # indicates if the ship should follow its orders, or direct player
        # control
        self.control_order:Optional[PlayerControlOrder] = None

        self.mouse_state = MouseState.EMPTY
        self.mouse_state_clear_time = np.inf

        self.starfield = starfield.Starfield(self.gamestate.sector_starfield, self.perspective)

        self.targeted_notification_backoff = 30.
        self.targeted_ts:MutableMapping[uuid.UUID, float] = collections.defaultdict(lambda: -self.targeted_notification_backoff)

        self.point_defense:Optional[combat.PointDefenseEffect] = None

        # keeps these order observers alive while giving the illusion that
        # there aren't any other references to them
        self.order_observers:Set[LambdaOrderObserver] = set()

        # should only draw cymunk shapes, not the stellarpunk visualization of
        # stuff in the sector (for debugging)
        self.draw_cymunk_shapes = False

        self.draw_hexes = False

    def make_order_observer(self,
        begin: Optional[Callable[[core.Order], None]] = None,
        complete: Optional[Callable[[core.Order], None]] = None,
        cancel: Optional[Callable[[core.Order], None]] = None,
    ) -> LambdaOrderObserver:
        observer = LambdaOrderObserver(self.order_observers, begin=begin, complete=complete, cancel=cancel)
        return observer


    def open_station_view(self, dock_station: sector_entity.Station) -> None:
        # TODO: make sure we're within docking range?
        station_view = v_station.StationView(dock_station, self.ship, self.gamestate, self.interface)
        self.interface.open_view(station_view, deactivate_views=True)

    def command_list(self) -> Collection[interface.CommandBinding]:

        def show_orders(args:Sequence[str]) -> None:
            for order in self.ship._orders:
                self.interface.log_message(f'{order}')

        def target_entity(args:Sequence[str]) -> None:
            if len(args) == 0:
                raise command_input.UserError("must provide an entity long or short id")
            arg_id = args[0]
            if re.match(r'^[A-Z]{3}-[a-z0-9]{8}', args[0]):
                # provided a short id
                try:
                    image = next(x for x in self.presenter.sensor_image_manager.sensor_contacts.values() if x.identified and x.identity.short_id == args[0])
                except StopIteration:
                    raise command_input.UserError("target not found")
            else:
                # try as uuid
                try:
                    entity_id = uuid.UUID(args[0])
                except ValueError:
                    raise command_input.UserError("bad entity id format")

                if entity_id not in self.presenter.sensor_image_manager.sensor_contacts:
                    raise command_input.UserError("target not found")
                image = self.presenter.sensor_image_manager.sensor_contacts[entity_id]
                if not image.identified:
                    raise command_input.UserError("target not found")

            self._select_target(image)
            self.interface.log_message(f'{image.identity.short_id} targted')

        def order_jump(args:Sequence[str]) -> None:
            if self.presenter.selected_target is None:
                raise command_input.UserError("no target for jump")
            if not self.presenter.selected_target_image.identified or not issubclass(self.presenter.selected_target_image.identity.object_type, sector_entity.TravelGate):
                raise command_input.UserError("target is not identified as a travel gate")

            if self.presenter.selected_target_image.identity.entity_id not in self.sector.entities:
                raise command_input.UserError("cannot reach the travel gate")
            selected_entity = self.sector.entities[self.presenter.selected_target_image.identity.entity_id]
            assert isinstance(selected_entity, sector_entity.TravelGate)
            order = orders.TravelThroughGate.create_travel_through_gate(selected_entity, self.ship, self.gamestate)
            self.ship.clear_orders()
            self.ship.prepend_order(order)

        def order_mine(args:Sequence[str]) -> None:
            if self.presenter.selected_target is None:
                raise command_input.UserError("no target for mining")
            if not self.presenter.selected_target_image.identified or not issubclass(self.presenter.selected_target_image.identity.object_type, sector_entity.Asteroid):
                raise command_input.UserError("target is not identified as an asteroid")

            if self.presenter.selected_target_image.identity.entity_id not in self.sector.entities:
                raise command_input.UserError("cannot reach the asteroid")

            assert self.interface.player.character
            asteroid_intel = self.interface.player.character.intel_manager.get_intel(intel.EntityIntelMatchCriteria(self.presenter.selected_target), intel.AsteroidIntel)
            if not asteroid_intel:
                raise command_input.UserError(f'no intel for {self.presenter.selected_target}')

            order = orders.MineOrder.create_mine_order(asteroid_intel, math.inf, self.ship, self.gamestate)
            self.ship.clear_orders()
            self.ship.prepend_order(order)

        def order_dock(args:Sequence[str]) -> None:
            if self.presenter.selected_target is None:
                raise command_input.UserError("no target for docking")
            if not self.presenter.selected_target_image.identified or not issubclass(self.presenter.selected_target_image.identity.object_type, sector_entity.Station):
                raise command_input.UserError("target is not identified as a station")

            order = orders.DockingOrder.create_docking_order(self.presenter.selected_target_image, self.ship, self.gamestate)

            station_id = self.presenter.selected_target
            def complete_docking(order: core.Order) -> None:
                dock_station = self.gamestate.get_entity(station_id, sector_entity.Station)
                self.open_station_view(dock_station)

            order.observe(self.make_order_observer(complete=complete_docking))
            self.ship.clear_orders()
            self.ship.prepend_order(order)

        def order_pursue(args:Sequence[str]) -> None:
            if self.presenter.selected_target is None:
                raise command_input.UserError("no target")
            order = movement.PursueOrder.create_pursue_order(self.presenter.selected_target_image, self.ship, self.gamestate)
            self.ship.clear_orders()
            self.ship.prepend_order(order)

        def order_evade(args:Sequence[str]) -> None:
            if self.presenter.selected_target is None:
                raise command_input.UserError("no target")

            order = movement.EvadeOrder.create_evade_order(self.presenter.selected_target_image, self.ship, self.gamestate)
            self.ship.clear_orders()
            self.ship.prepend_order(order)

        def order_attack(args:Sequence[str]) -> None:
            if self.presenter.selected_target is None:
                raise command_input.UserError("no target")

            order = combat.AttackOrder.create_attack_order(self.presenter.selected_target_image, self.ship, self.gamestate)
            self.ship.clear_orders()
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

        def spawn_missile(args:Sequence[str]) -> None:
            if self.presenter.selected_target is None:
                raise command_input.UserError("no target")
            missile = combat.MissileOrder.spawn_missile(self.ship, self.gamestate, self.presenter.selected_target_image, spawn_radius=10000)

        def toggle_sensor_cone(args:Sequence[str]) -> None:
            self.presenter.show_sensor_cone = not self.presenter.show_sensor_cone
            cone_txt_state = "on" if self.presenter.show_sensor_cone else "off"
            self.interface.log_message(f'sensor_cone {cone_txt_state}')

        def toggle_sensors(args:Sequence[str]) -> None:
            if self.ship.sensor_settings.sensor_power > 0.:
                self.ship.sensor_settings.set_sensors(0.)
            else:
                self.ship.sensor_settings.set_sensors(1.)

        def toggle_transponder(args:Sequence[str]) -> None:
            self.ship.sensor_settings.set_transponder(not self.ship.sensor_settings.transponder)

        def cache_stats(args:Sequence[str]) -> None:
            self.logger.info(presenter.compute_sensor_rings_memoize.cache_info())

        def toggle_point_defense(args:Sequence[str]) -> None:
            assert self.ship.sector
            if self.point_defense is None:
                self.point_defense = combat.PointDefenseEffect.create_point_defense_effect(self.ship, self.ship.sector, self.gamestate)
                self.ship.sector.add_effect(self.point_defense)
            else:
                self.point_defense.cancel_effect()
                self.point_defense = None

        def max_thrust(args:Sequence[str]) -> None:
            if len(args) > 0:
                try:
                    thrust_ratio = float(args[0])
                except ValueError:
                    raise command_input.UserError("must provide a valid ratio to set max thrust to")
                if thrust_ratio < 0.0 or thrust_ratio > 1.0:
                    raise command_input.UserError("must choose a ratio between 0 and 1")
                #TODO: we should really not monkey with the ship's max thrust
                # there is nothing that will set it back to the real max
                self.ship.max_thrust = self.ship.max_base_thrust * thrust_ratio
            self.interface.log_message(f'max thrust: {util.human_si_scale(self.ship.max_thrust, "N")}')

        def mouse_pos(args:Sequence[str]) -> None:
            self.interface.log_message(f'{self.m_sector_x},{self.m_sector_y}')

        detected_short_ids = (x.identity.short_id() for x in self.presenter.sensor_image_manager.sensor_contacts.values() if x.identified)

        def only_draw_cymunk(args:Sequence[str]) -> None:
            self.draw_cymunk_shapes = not self.draw_cymunk_shapes

        def draw_hexes(args:Sequence[str]) -> None:
            self.draw_hexes = not self.draw_hexes

        return [
            self.bind_command("orders", show_orders),
            self.bind_command("clear_orders", lambda x: self.ship.clear_orders()),
            self.bind_command("target", target_entity, util.tab_completer(detected_short_ids)),
            self.bind_command("jump", order_jump),
            self.bind_command("mine", order_mine),
            self.bind_command("dock", order_dock),
            self.bind_command("pursue", order_pursue),
            self.bind_command("evade", order_evade),
            self.bind_command("attack", order_attack),
            self.bind_command("cargo", log_cargo),
            self.bind_command("spawn_missile", spawn_missile),
            self.bind_command("toggle_sensor_cone", toggle_sensor_cone),
            self.bind_command("toggle_sensors", toggle_sensors),
            self.bind_command("toggle_transponder", toggle_transponder),
            self.bind_command("point_defense", toggle_point_defense),
            self.bind_command("cache_stats", cache_stats),
            self.bind_command("max_thrust", max_thrust),
            self.bind_command("mouse_pos", mouse_pos),
            self.bind_command("cymunk_shapes", only_draw_cymunk),
            self.bind_command("draw_hexes", draw_hexes),
        ]

    def _select_target(self, target:Optional[core.AbstractSensorImage]) -> None:
        # observe target in case it is destroyed or migrates so we deselect it
        if target is None:
            self.presenter.selected_target = None
        else:
            self.presenter.selected_target = target.identity.entity_id

    # core.SectorEntityObserver
    @property
    def observer_id(self) -> uuid.UUID:
        return core.OBSERVER_ID_NULL

    def entity_migrated(self, entity:core.SectorEntity, from_sector:core.Sector, to_sector:core.Sector) -> None:
        if entity != self.ship:
            raise ValueError(f'got unexpected entity in migration {entity} migrating {from_sector} to {to_sector}')
        if from_sector != self.sector:
            raise ValueError(f'got unexpected from sector in migration {entity} migrating {from_sector} to {to_sector}')

        assert(self.interface.player.character and isinstance(self.interface.player.character.location, core.Ship))
        self.interface.swap_view(
                PilotView(self.interface.player.character.location, self.gamestate, self.interface),
                self
        )

    def entity_targeted(self, entity:core.SectorEntity, threat:core.SectorEntity) -> None:
        if entity == self.ship:
            if self.gamestate.timestamp - self.targeted_ts[threat.entity_id] > self.targeted_notification_backoff:
                self.interface.log_message(f'you have been targed by {threat}')
            self.targeted_ts[threat.entity_id] = self.gamestate.timestamp
        else:
            raise ValueError("got unexpected entity_targeted event")


    def _clear_control_order(self, order: core.Order) -> None:
        self.logger.debug("clearing pilot control order")
        assert order == self.control_order
        self.control_order = None

    def _toggle_player_control(self) -> None:
        """ Toggles player control state.

        When player control is off, we follow normal orders for the ship.
        When it's on, we cancel the current order queue and follow the user's
        input directly.
        """

        if self.control_order is not None:
            self.logger.info("exiting player control")

            self.control_order.cancel_order()
            # clearing control_order is handled by the order observer callback
            assert self.control_order is None
        else:
            self.logger.info("entering player control")
            self.ship.clear_orders()

            control_order = PlayerControlOrder.create_player_control_order(
                self.interface.runtime.get_dt(),
                self.ship,
                self.gamestate,
                observer=self.make_order_observer(
                    complete=self._clear_control_order,
                    cancel=self._clear_control_order
                )
            )
            self.ship.prepend_order(control_order)
            self.control_order = control_order

    def _drive(self, key:int) -> None:
        """ Inputs a direct navigation control for the ship.

        "w" increases velocity, "a" rotates left, "d" rotates right,
        "s" attempts to kill velocity. All of these give one step of the desired
        action.
        """

        if not self.control_order:
            return

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
        elif self.presenter.selected_target is not None:
            self._select_target(None)
        elif self.control_order:
            self._toggle_player_control()

    def _cancel_mouse(self) -> None:
            self.mouse_state = MouseState.EMPTY

    def handle_mouse(self, m_id: int, m_x: int, m_y: int, m_z: int, bstate: int) -> bool:
        """ Handle mouse input according to MouseState """

        self.m_sector_x, self.m_sector_y = self.perspective.screen_to_sector(m_x, m_y)
        if not bstate & (curses.BUTTON1_CLICKED | curses.BUTTON1_PRESSED):
            return False

        sector_x, sector_y = self.m_sector_x, self.m_sector_y
        self.logger.debug(f'clicked {(m_x,m_y)} translates to {(sector_x,sector_y)}')

        if self.mouse_state == MouseState.EMPTY:
            # select a target within a cell of the mouse click
            if self.ship.sector is None:
                raise ValueError("ship must be in a sector to select a target")

            r_x = self.perspective.meters_per_char[0]
            r_y = self.perspective.meters_per_char[1]
            hit = next((x for x in self.presenter.sensor_image_manager.spatial_query((sector_x-r_x, sector_y-r_y, sector_x+r_x, sector_y+r_y)) if not issubclass(x.identity.object_type, sector_entity.Projectile)), None)
            if hit:
                #TODO: check if the hit is close enough
                self._select_target(hit)

        elif self.mouse_state == MouseState.GOTO:
            if self.control_order:
                self._toggle_player_control()

            self.ship.clear_orders()

            goto_order = movement.GoToLocation.create_go_to_location(np.array((sector_x, sector_y)), self.ship, self.gamestate, arrival_distance=5e2)
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
            (x for x in self.presenter.sensor_image_manager.spatial_point(self.ship.loc) if x.identity.entity_id != self.ship.entity_id and not issubclass(x.identity.object_type, sector_entity.Projectile)),
            key=lambda x: util.distance(self.ship.loc, x.loc)
        )

        if len(potential_targets) == 0:
            self._select_target(None)
            return

        if self.presenter.selected_target is None:
            self._select_target(potential_targets[0])
            return

        try:
            idx = potential_targets.index(self.presenter.selected_target_image)
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
            self.bind_key(ord("p"), self._toggle_player_control),
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

        radar_color = self.interface.get_color(interface.Color.RADAR_RING)
        for (y,x), c in radar_content.items():
            self.viewscreen.addstr(y, x, c, radar_color)

        # draw location indicators
        i = major_ticks_x.niceMin
        while i <= major_ticks_x.niceMax:
            s_x, s_y = self.perspective.sector_to_screen(
                    i+self.perspective.cursor[0], self.perspective.cursor[1]
            )
            self.viewscreen.addstr(s_y, s_x, util.human_distance(i), radar_color)
            i += major_ticks_x.tickSpacing
        j = major_ticks_y.niceMin
        while j <= major_ticks_y.niceMax:
            s_x, s_y = self.perspective.sector_to_screen(
                    self.perspective.cursor[0], j+self.perspective.cursor[1]
            )
            self.viewscreen.addstr(s_y, s_x, util.human_distance(j), radar_color)
            j += major_ticks_y.tickSpacing

        # draw degree indicators
        r = self.perspective.meters_per_char[1] * self.direction_indicator_radius
        for theta in np.linspace(0., 2*np.pi, 12, endpoint=False):
            # convert to 0 at positive x to get the right display
            x,y = util.polar_to_cartesian(r, theta-np.pi/2)
            s_x, s_y = self.perspective.sector_to_screen(
                    self.perspective.cursor[0]+x, self.perspective.cursor[1]+y
            )
            self.viewscreen.addstr(s_y, s_x, f'{math.degrees(theta):.0f}°', radar_color)

        # add a scale near corner
        scale_label = f'scale {util.human_distance(major_ticks_x.tickSpacing)}'
        scale_x = self.interface.viewscreen.width - len(scale_label) - 2
        scale_y = self.interface.viewscreen.height - 2
        self.viewscreen.addstr(scale_y, scale_x, scale_label, radar_color)

        # add center position near corner
        pos_label = f'({self.perspective.cursor[0]:.0f},{self.perspective.cursor[1]:.0f})'
        pos_x = self.interface.viewscreen.width - len(pos_label) - 2
        pos_y = self.interface.viewscreen.height - 1
        self.viewscreen.addstr(pos_y, pos_x, pos_label, radar_color)

    def _draw_target_indicators(self) -> None:
        top_order = self.ship.top_order()
        current_order = self.ship.current_order()
        if isinstance(current_order, movement.GoToLocation):
            s_x, s_y = self.perspective.sector_to_screen(*current_order._target_location)

            self.viewscreen.addstr(s_y, s_x, interface.Icons.LOCATION_INDICATOR, curses.color_pair(interface.Icons.COLOR_LOCATION_INDICATOR))
        elif isinstance(current_order, movement.EvadeOrder) or isinstance(current_order, movement.PursueOrder):
            s_x, s_y = self.perspective.sector_to_screen(*current_order.intercept_location)

            self.viewscreen.addstr(s_y, s_x, interface.Icons.LOCATION_INDICATOR, curses.color_pair(interface.Icons.COLOR_LOCATION_INDICATOR))

        if isinstance(top_order, combat.AttackOrder):
            s_x, s_y = self.perspective.sector_to_screen(*top_order.target.loc)

            self.viewscreen.addstr(s_y, s_x, interface.Icons.TARGET_INDICATOR, curses.color_pair(interface.Icons.COLOR_TARGET_INDICATOR))

            #self.ship.sensor_settings._ignore_bias=False
            #s_x, s_y = self.perspective.sector_to_screen(*top_order.target.loc)
            #self.viewscreen.addstr(s_y, s_x, interface.Icons.TARGET_INDICATOR, curses.color_pair(100))
            #self.ship.sensor_settings._ignore_bias=True

        #DEBUG: draw selected entity's notion of where their target is
        #if self.presenter.selected_target and self.presenter.selected_target_image.identity.object_type == core.ObjectType.SHIP:
        #    selected_entity = self.presenter.selected_target_image._target
        #    target_order:Optional[core.Order] = selected_entity.top_order()
        #    if isinstance(target_order, combat.HuntOrder):
        #        target_order = target_order.attack_order
        #    if isinstance(target_order, combat.AttackOrder) or isinstance(target_order, combat.MissileOrder):
        #        s_x, s_y = self.perspective.sector_to_screen(*target_order.target.loc)
        #        self.viewscreen.addstr(s_y, s_x, interface.Icons.TARGET_INDICATOR, curses.color_pair(interface.Icons.COLOR_TARGET_IMAGE_INDICATOR))

    def _draw_nav_indicators(self) -> None:
        """ Draws navigational indicators on the display.

        Includes velocity and heading.
        """

        # heading, on a circle at a fixed distance from the center
        x,y = util.polar_to_cartesian(self.perspective.meters_per_char[1] * self.heading_indicator_radius, self.ship.angle)
        s_x, s_y = self.perspective.sector_to_screen(self.perspective.cursor[0]+x, self.perspective.cursor[1]+y)

        self.viewscreen.addstr(s_y, s_x, interface.Icons.HEADING_INDICATOR, curses.color_pair(interface.Icons.COLOR_HEADING_INDICATOR))

        if self.presenter.selected_target is not None:
            distance, bearing = util.cartesian_to_polar(*(self.presenter.selected_target_image.loc - self.ship.loc))
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
        if self.presenter.selected_target is None:
            return

        info_width = 12 + 1 + 24
        status_x = self.interface.viewscreen.width - info_width
        status_y = 1

        self.viewscreen.addstr(status_y, status_x, "Target Info:")

        label_id = "id:"
        label_sensor_profile = "s profile:"
        label_speed = "speed:"
        label_location = "location:"
        label_bearing = "bearing:"
        label_distance = "distance:"
        label_rel_speed = "rel speed:"
        label_eta = "eta:"

        rel_pos = self.presenter.selected_target_image.loc - self.ship.loc
        rel_vel = self.presenter.selected_target_image.velocity - self.ship.velocity
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

        self.viewscreen.addstr(status_y+1, status_x, f'{label_id:>12} {self.presenter.selected_target_image.identity.short_id}')
        self.viewscreen.addstr(status_y+2, status_x, f'{label_sensor_profile:>12} {self.presenter.selected_target_image.profile}')
        self.viewscreen.addstr(status_y+3, status_x, f'{label_speed:>12} {util.human_speed(util.magnitude(*self.presenter.selected_target_image.velocity))}')
        self.viewscreen.addstr(status_y+4, status_x, f'{label_location:>12} {self.presenter.selected_target_image.loc[0]:.0f},{self.presenter.selected_target_image.loc[1]:.0f}')
        self.viewscreen.addstr(status_y+5, status_x, f'{label_bearing:>12} {math.degrees(util.normalize_angle(bearing)):.0f}° ({math.degrees(util.normalize_angle(rel_bearing, shortest=True)):.0f}°)')
        self.viewscreen.addstr(status_y+6, status_x, f'{label_distance:>12} {util.human_distance(distance)}')
        self.viewscreen.addstr(status_y+7, status_x, f'{label_rel_speed:>12} {util.human_speed(vel_toward)} ({util.human_speed(vel_perpendicular)})')
        if approach_t < 60*60:
            self.viewscreen.addstr(status_y+8, status_x, f'{label_eta:>12} {approach_t:.0f}s ({util.human_distance(closest_approach)})')

        #DEBUG:
        #label_bias_mag = "bias:"
        #label_thrust = "thrust:"
        #bias_mag = util.magnitude(*(self.presenter.selected_target_image._loc_bias))
        #thrust = self.presenter.selected_target_image._target.sensor_settings.effective_thrust()
        #self.viewscreen.addstr(status_y+9, status_x, f'{label_bias_mag:>12} {util.human_distance(bias_mag)}')
        #self.viewscreen.addstr(status_y+10, status_x, f'{label_thrust:>12} {thrust}')

        status_y += 9

        if self.presenter.selected_target_image.identified and issubclass(self.presenter.selected_target_image.identity.object_type, sector_entity.Station):
            #TODO: how do we get the product type?
            entity_id = self.presenter.selected_target_image.identity.entity_id
            if entity_id in self.sector.entities:
                selected_entity = self.sector.entities[entity_id]
                assert isinstance(selected_entity, sector_entity.Station)
                assert selected_entity.resource is not None
                label_product = "product:"
                self.viewscreen.addstr(status_y, status_x, f'{label_product:>12} {ui_util.product_name(self.gamestate.production_chain, selected_entity.resource, 20)}')
        elif self.presenter.selected_target_image.identified and issubclass(self.presenter.selected_target_image.identity.object_type, sector_entity.Asteroid):
            entity_id = self.presenter.selected_target_image.identity.entity_id
            if entity_id in self.sector.entities:
                selected_entity = self.sector.entities[entity_id]
                assert isinstance(selected_entity, sector_entity.Asteroid)
                #TODO: how do we get the resource type?
                assert selected_entity.resource is not None
                label_ore = "ore:"
                self.viewscreen.addstr(status_y, status_x, f'{label_ore:>12} {ui_util.product_name(self.gamestate.production_chain, selected_entity.resource, 20)}')

    def _draw_status(self) -> None:
        current_order = self.ship.current_order()

        status_x = 1
        status_y = 1
        self.viewscreen.addstr(status_y, status_x, "Status:")

        label_sensor_profile = "s profile:"
        label_sensor_threshold = "s threshold:"
        label_speed = "speed:"
        label_location = "location:"
        label_heading = "heading:"
        label_course = "course:"
        label_fuel = "propellant:"
        label_top_order = "top order:"
        label_order = "order:"
        label_eta = "eta:"
        label_pd = "pd:"
        # convert heading so 0, North is negative y, instead of positive x
        heading = self.ship.angle + np.pi/2
        course = self.ship.phys.velocity.get_angle() + np.pi/2
        pd_status = "off" if self.point_defense is None else "on"

        self.viewscreen.addstr(status_y+1, status_x, f'{label_sensor_profile:>12} {self.sector.sensor_manager.compute_effective_profile(self.ship)}')
        self.viewscreen.addstr(status_y+2, status_x, f'{label_sensor_threshold:>12} {self.sector.sensor_manager.compute_sensor_threshold(self.ship)}')
        self.viewscreen.addstr(status_y+3, status_x, f'{label_speed:>12} {util.human_speed(self.ship.speed)} ({self.ship.phys.force.length}N)')
        self.viewscreen.addstr(status_y+4, status_x, f'{label_location:>12} {self.ship.loc[0]:.0f},{self.ship.loc[1]:.0f}')
        self.viewscreen.addstr(status_y+5, status_x, f'{label_heading:>12} {math.degrees(util.normalize_angle(heading)):.0f}° ({math.degrees(self.ship.phys.angular_velocity):.0f}°/s) ({self.ship.phys.torque:.2}N-m))')
        self.viewscreen.addstr(status_y+6, status_x, f'{label_course:>12} {math.degrees(util.normalize_angle(course)):.0f}°')
        self.viewscreen.addstr(status_y+7, status_x, f'{label_fuel:>12} {self.ship.sensor_settings.thrust_seconds / 4435.:.0f}')
        self.viewscreen.addstr(status_y+8, status_x, f'{label_pd:>12} {pd_status}')
        status_y += 9

        if current_order is not None:
            ancestor_order = current_order
            while ancestor_order.parent_order is not None:
                ancestor_order = ancestor_order.parent_order
            if ancestor_order != current_order:
                self.viewscreen.addstr(status_y, status_x, f'{label_top_order:>12} {ancestor_order}')
                status_y += 1

            self.viewscreen.addstr(status_y, status_x, f'{label_order:>12} {current_order}')
            eta = current_order.estimate_eta()
            self.viewscreen.addstr(status_y+1, status_x, f'{label_eta:>12} {eta:.1f}s')
            status_y += 2
            if not self.gamestate.is_order_scheduled(current_order):
                self.viewscreen.addstr(status_y, status_x, f'order is not scheduled!', self.interface.get_color(interface.Color.ERROR))
                status_y +=1

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

        weather_x = 1
        weather_y = self.interface.viewscreen.height - 4
        sensor_factor = self.sector.weather((self.m_sector_x, self.m_sector_y)).sensor_factor
        self.viewscreen.addstr(weather_y, weather_x, f'{(int(self.m_sector_x), int(self.m_sector_y))} sensor factor: {sensor_factor}')

    def initialize(self) -> None:
        self.logger.info(f'entering pilot mode for {self.ship.entity_id}')
        self.perspective.update_bbox()
        self.interface.reinitialize_screen(name="Pilot's Seat")
        self.ship.observe(self)

    def terminate(self) -> None:
        if self.ship:
            self.ship.unobserve(self)
        if self.control_order:
            self.control_order.cancel_order()
        if self.point_defense:
            self.point_defense.cancel_effect()
        self.presenter.terminate()

    def focus(self) -> None:
        super().focus()
        self.interface.reinitialize_screen(name="Pilot's Seat")
        self.active=True


    def update_display(self) -> None:
        assert self.sector == self.ship.sector
        if self.gamestate.timestamp > self.mouse_state_clear_time:
            self.mouse_state = MouseState.EMPTY

        self._auto_pan()

        self.viewscreen.erase()

        # we need to check here because the presenter is about to drop it
        if self.presenter.selected_target:
            if not self.presenter.selected_target_image.is_active():
                reason = self.presenter.selected_target_image.inactive_reason
                if reason == core.SensorImageInactiveReason.DESTROYED:
                    self.interface.log_message("target destroyed")
                elif reason == core.SensorImageInactiveReason.MIGRATED:
                    self.interface.log_message("target moved to another sector")
                elif reason == core.SensorImageInactiveReason.OTHER:
                    self.interface.log_message("target missing")
            elif self.presenter.selected_target_image.age > 120.0:
                self.interface.log_message("target sensor image lost")
        self.presenter.update()

        if self.draw_cymunk_shapes:
            self.presenter.draw_cymunk_shapes()
        if self.draw_hexes:
            self.presenter.draw_hexes()
        else:
            self.starfield.draw_starfield(self.viewscreen)
            self.presenter.draw_hexes()
            self.presenter.draw_weather()
            self.presenter.draw_shapes()
            self._draw_radar()
            self.presenter.draw_sensor_rings(self.ship)
            self.presenter.draw_profile_rings(self.ship)
            self.presenter.draw_sector_map()

        # draw hud overlay on top of everything else
        self._draw_hud()

        self.interface.refresh_viewscreen()

