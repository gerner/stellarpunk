import io
import uuid
import abc
import uuid
from typing import Any

import numpy as np
import cymunk # type: ignore

from stellarpunk import util, core
from stellarpunk.orders import collision, steering, movement
from stellarpunk.serialization import util as s_util, save_game, order as s_order

class AbstractSteeringOrderSaver[T:steering.AbstractSteeringOrder](s_order.OrderSaver[T]):

    @abc.abstractmethod
    def _save_steering_order(self, obj:T, f:io.IOBase) -> int: ...
    @abc.abstractmethod
    def _load_steering_order(self, f:io.IOBase, load_context:save_game.LoadContext, order_id:uuid.UUID) -> tuple[T, Any]: ...
    def _post_load_steering_order(self, obj:T, load_context:save_game.LoadContext, extra_context:Any) -> None:
        pass

    def _save_navigator_params(self, params:collision.NavigatorParameters, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += self.save_game.debug_string_w("navigator basic params", f)

        bytes_written += s_util.float_to_f(params.radius, f)
        bytes_written += s_util.float_to_f(params.max_thrust, f)
        bytes_written += s_util.float_to_f(params.max_torque, f)
        bytes_written += s_util.float_to_f(params.max_acceleration, f)
        bytes_written += s_util.float_to_f(params.max_angular_acceleration, f)
        bytes_written += s_util.float_to_f(params.worst_case_rot_time, f)

        bytes_written += s_util.float_to_f(params.base_neighborhood_radius, f)
        bytes_written += s_util.float_to_f(params.neighborhood_radius, f)
        bytes_written += s_util.float_to_f(params.full_neighborhood_radius_period, f)
        bytes_written += s_util.float_to_f(params.full_neighborhood_radius_ts, f)

        bytes_written += s_util.float_to_f(params.base_max_speed, f)
        bytes_written += s_util.float_to_f(params.max_speed, f)

        bytes_written += s_util.float_to_f(params.max_speed_cap, f)
        bytes_written += s_util.float_to_f(params.max_speed_cap_ts, f)
        bytes_written += s_util.float_to_f(params.max_speed_cap_alpha, f)
        bytes_written += s_util.float_to_f(params.min_max_speed, f)
        bytes_written += s_util.float_to_f(params.max_speed_cap_max_expiration, f)

        bytes_written += s_util.float_to_f(params.base_margin, f)
        bytes_written += s_util.float_to_f(params.margin, f)

        bytes_written += s_util.float_pair_to_f(params.target_location, f)
        bytes_written += s_util.float_to_f(params.arrival_radius, f)
        bytes_written += s_util.float_to_f(params.min_radius, f)

        bytes_written += s_util.float_to_f(params.last_threat_id, f)
        bytes_written += s_util.float_to_f(params.collision_margin_histeresis, f)
        bytes_written += s_util.bool_to_f(params.cannot_avoid_collision_hold, f)
        bytes_written += s_util.bool_to_f(params.collision_cbdr, f)

        # analysis params
        bytes_written += self.save_game.debug_string_w("navigator analysis params", f)
        bytes_written += s_util.float_to_f(params.analysis.neighborhood_radius, f)
        bytes_written += s_util.int_to_f(params.analysis.threat_count, f)
        bytes_written += s_util.int_to_f(params.analysis.neighborhood_size, f)
        bytes_written += s_util.float_to_f(params.analysis.nearest_neighborhood_dist, f)
        bytes_written += s_util.bool_to_f(params.analysis.cannot_avoid_collision, f)
        bytes_written += s_util.int_to_f(params.analysis.coalesced_threat_count, f)

        # prior threats
        bytes_written += self.save_game.debug_string_w("navigator prior threats", f)
        bytes_written += s_util.uuids_to_f(list(x.body.data.entity_id for x in params.prior_threats), f)

        return bytes_written

    def _load_navigator_params(self, f:io.IOBase, order:T, load_context:save_game.LoadContext) -> tuple[collision.NavigatorParameters, Any]:
        params = collision.NavigatorParameters()

        load_context.debug_string_r("navigator basic params", f)

        params.radius = s_util.float_from_f(f)
        params.max_thrust = s_util.float_from_f(f)
        params.max_torque = s_util.float_from_f(f)
        params.max_acceleration = s_util.float_from_f(f)
        params.max_angular_acceleration = s_util.float_from_f(f)
        params.worst_case_rot_time = s_util.float_from_f(f)

        params.base_neighborhood_radius = s_util.float_from_f(f)
        params.neighborhood_radius = s_util.float_from_f(f)
        params.full_neighborhood_radius_period = s_util.float_from_f(f)
        params.full_neighborhood_radius_ts = s_util.float_from_f(f)

        params.base_max_speed = s_util.float_from_f(f)
        params.max_speed = s_util.float_from_f(f)

        params.max_speed_cap = s_util.float_from_f(f)
        params.max_speed_cap_ts = s_util.float_from_f(f)
        params.max_speed_cap_alpha = s_util.float_from_f(f)
        params.min_max_speed = s_util.float_from_f(f)
        params.max_speed_cap_max_expiration = s_util.float_from_f(f)

        params.base_margin = s_util.float_from_f(f)
        params.margin = s_util.float_from_f(f)

        target_location = s_util.float_pair_from_f(f)
        params.target_location = (float(target_location[0]), float(target_location[1]))
        params.arrival_radius = s_util.float_from_f(f)
        params.min_radius = s_util.float_from_f(f)

        params.last_threat_id = s_util.int_from_f(f)
        params.collision_margin_histeresis = s_util.float_from_f(f)
        params.cannot_avoid_collision_hold = s_util.bool_from_f(f)
        params.collision_cbdr = s_util.bool_from_f(f)

        # analysis params
        load_context.debug_string_r("navigator analysis params", f)
        params.analysis.neighborhood_radius = s_util.float_from_f(f)
        params.analysis.threat_count = s_util.int_from_f(f)
        params.analysis.neighborhood_size = s_util.int_from_f(f)
        params.analysis.nearest_neighborhood_dist = s_util.float_from_f(f)
        params.analysis.cannot_avoid_collision = s_util.bool_from_f(f)
        params.analysis.coalesced_threat_count = s_util.int_from_f(f)

        # prior threats, we'll fully set this up in post load
        load_context.debug_string_r("navigator prior threats", f)
        prior_threat_ids = s_util.uuids_from_f(f)

        load_context.register_custom_post_load(self._post_load_navigator_params, order, (params, prior_threat_ids))

        return params, prior_threat_ids

    def _post_load_navigator_params(self, order:T, load_context:save_game.LoadContext, context:Any) -> None:
        context_data:tuple[collision.NavigatorParameters, list[uuid.UUID]] = context

        params, prior_threat_ids = context_data

        # we assume that the sector has been completely post loaded by the time
        # we are. this should happen because orders are loaded after entities.
        assert(order.ship.sector)
        space = order.ship.sector.space
        body = order.ship.phys

        for prior_threat_id in prior_threat_ids:
            sector_entity = load_context.gamestate.get_entity(prior_threat_id, core.SectorEntity)
            params.prior_threats.append(sector_entity.phys_shape)

        order.neighbor_analyzer = collision.Navigator(space, body, params.radius, params.max_thrust, params.max_torque, params.max_speed, params.base_margin, params.base_neighborhood_radius)
        order.neighbor_analyzer.set_navigator_parameters(params)

    def _save_order(self, obj:T, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.float_to_f(obj.safety_factor, f)
        bytes_written += s_util.float_to_f(obj.collision_dv[0], f)
        bytes_written += s_util.float_to_f(obj.collision_dv[1], f)
        bytes_written += self._save_steering_order(obj, f)
        bytes_written += self._save_navigator_params(obj.neighbor_analyzer.get_navigator_parameters(), f)
        return bytes_written

    def _load_order(self, f:io.IOBase, load_context:save_game.LoadContext, order_id:uuid.UUID) -> tuple[T, Any]:
        safety_factor = s_util.float_from_f(f)
        cdv_x = s_util.float_from_f(f)
        cdv_y = s_util.float_from_f(f)

        steering_order, extra_context = self._load_steering_order(f, load_context, order_id)
        self._load_navigator_params(f, steering_order, load_context)
        steering_order.safety_factor = safety_factor
        if util.isclose(cdv_x, 0.) and util.isclose(cdv_y, 0.):
            steering_order.collision_dv = steering.ZERO_VECTOR
        else:
            steering_order.collision_dv = np.array((cdv_x, cdv_y))

        return steering_order, extra_context

    def _post_load_order(self, order:T, load_context:save_game.LoadContext, context:Any) -> None:
        context_data:tuple[collision.NavigatorParameters, Any] = context
        extra_context = context_data

        self._post_load_steering_order(order, load_context, extra_context)

class KillRotationOrderSaver(s_order.OrderSaver[movement.KillRotationOrder]):
    def _save_order(self, order:movement.KillRotationOrder, f:io.IOBase) -> int:
        return 0
    def _load_order(self, f:io.IOBase, load_context:save_game.LoadContext, order_id:uuid.UUID) -> tuple[movement.KillRotationOrder, Any]:
        order = movement.KillRotationOrder(load_context.gamestate, _check_flag=True, order_id=order_id)
        return (order, None)

class RotateOrderSaver(AbstractSteeringOrderSaver[movement.RotateOrder]):
    def _save_steering_order(self, obj:movement.RotateOrder, f:io.IOBase) -> int:
        return s_util.float_to_f(obj.target_angle, f)

    def _load_steering_order(self, f:io.IOBase, load_context:save_game.LoadContext, order_id:uuid.UUID) -> tuple[movement.RotateOrder, Any]:
        target_angle = s_util.float_from_f(f)
        order = movement.RotateOrder(target_angle, load_context.gamestate, _check_flag=True, order_id=order_id)
        return (order, None)

class KillVelocityOrderSaver(AbstractSteeringOrderSaver[movement.KillVelocityOrder]):
    def _save_steering_order(self, obj:movement.KillVelocityOrder, f:io.IOBase) -> int:
        return 0

    def _load_steering_order(self, f:io.IOBase, load_context:save_game.LoadContext, order_id:uuid.UUID) -> tuple[movement.KillVelocityOrder, Any]:
        order = movement.KillVelocityOrder(load_context.gamestate, _check_flag=True, order_id=order_id)
        return (order, None)

class GoToLocationSaver(AbstractSteeringOrderSaver[movement.GoToLocation]):
    def _save_steering_order(self, obj:movement.GoToLocation, f:io.IOBase) -> int:
        bytes_written = 0

        bytes_written += s_util.uuid_to_f(obj.target_sector.entity_id, f)
        bytes_written += s_util.float_pair_to_f(obj.target_location, f)
        bytes_written += s_util.float_to_f(obj.arrival_distance, f)
        bytes_written += s_util.float_to_f(obj.min_distance, f)
        bytes_written += s_util.float_pair_to_f(np.array(obj.target_v), f)
        bytes_written += s_util.bool_to_f(obj.cannot_stop, f)
        bytes_written += s_util.float_to_f(obj.distance_estimate, f)
        bytes_written += s_util.float_pair_to_f(np.array(obj._desired_velocity), f)
        bytes_written += s_util.float_to_f(obj._next_compute_ts, f)
        bytes_written += s_util.float_to_f(obj._nts, f)

        return bytes_written

    def _load_steering_order(self, f:io.IOBase, load_context:save_game.LoadContext, order_id:uuid.UUID) -> tuple[movement.GoToLocation, Any]:
        target_sector_id = s_util.uuid_from_f(f)
        target_location = s_util.float_pair_from_f(f)
        arrival_distance = s_util.float_from_f(f)
        min_distance = s_util.float_from_f(f)
        target_v = s_util.float_pair_from_f(f)
        cannot_stop = s_util.bool_from_f(f)
        distance_estimate = s_util.float_from_f(f)
        desired_velocity = s_util.float_pair_from_f(f)
        next_compute_ts = s_util.float_from_f(f)
        nts = s_util.float_from_f(f)

        order = movement.GoToLocation(target_location, load_context.gamestate, arrival_distance=arrival_distance, min_distance=min_distance, _check_flag=True, order_id=order_id)
        order.target_v = cymunk.Vec2d(*target_v)
        order.cannot_stop = cannot_stop
        order.distance_estimate = distance_estimate
        order._desired_velocity = cymunk.Vec2d(*desired_velocity)
        order._next_compute_ts = next_compute_ts
        order._nts = nts

        return (order, target_sector_id)

    def _post_load_steering_order(self, obj:movement.GoToLocation, load_context:save_game.LoadContext, extra_context:Any) -> None:
        target_sector_id:uuid.UUID = extra_context
        target_sector = load_context.gamestate.get_entity(target_sector_id, klass=core.Sector)
        obj.target_sector = target_sector

#TODO: EvadeOrder
class EvadeOrderSaver(AbstractSteeringOrderSaver[movement.EvadeOrder]):
    def _save_steering_order(self, obj:movement.EvadeOrder, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.uuid_to_f(obj.target.identity.entity_id, f)
        bytes_written += s_util.float_pair_to_f(obj.intercept_location, f)
        bytes_written += s_util.float_to_f(obj.intercept_time, f)
        bytes_written += s_util.float_to_f(obj.escape_distance, f)
        bytes_written += s_util.float_to_f(obj.max_thrust, f)
        bytes_written += s_util.float_to_f(obj.max_fine_thrust, f)
        return bytes_written

    def _load_steering_order(self, f:io.IOBase, load_context:save_game.LoadContext, order_id:uuid.UUID) -> tuple[movement.EvadeOrder, Any]:
        target_id = s_util.uuid_from_f(f)
        intercept_location = s_util.float_pair_from_f(f)
        intercept_time = s_util.float_from_f(f)
        escape_distance = s_util.float_from_f(f)
        max_thrust = s_util.float_from_f(f)
        max_fine_thrust = s_util.float_from_f(f)

        order = movement.EvadeOrder(load_context.gamestate, escape_distance=escape_distance, _check_flag=True, order_id=order_id)
        order.intercept_location = intercept_location
        order.intercept_time = intercept_time
        order.escape_distance = escape_distance
        order.max_thrust = max_thrust
        order.max_fine_thrust = max_fine_thrust
        return (order, target_id)

    def _post_load_steering_order(self, obj:movement.EvadeOrder, load_context:save_game.LoadContext, extra_context:Any) -> None:
        target_id:uuid.UUID = extra_context
        target = obj.ship.sensor_settings.get_image(target_id)
        obj.target = target

#TODO: PursueOrder (this is subclassed)
class PursueOrderSaver[T:movement.PursueOrder](AbstractSteeringOrderSaver[T]):
    def _save_pursue_order(self, order:T, f:io.IOBase) -> int:
        return 0
    def _load_pursue_order(self, f:io.IOBase, load_context:save_game.LoadContext, arrival_distance:float, avoid_collisions:bool, order_id:uuid.UUID) -> tuple[T, Any]:
        order = movement.PursueOrder(load_context.gamestate, arrival_distance=arrival_distance, avoid_collisions=avoid_collisions, _check_flag=True, order_id=order_id)
        return order, None # type: ignore
    def _post_load_pursue_order(self, obj:T, load_context:save_game.LoadContext, extra_context:Any) -> None:
        pass

    def _save_steering_order(self, obj:T, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.uuid_to_f(obj.target.identity.entity_id, f)
        bytes_written += s_util.float_pair_to_f(obj.intercept_location, f)
        bytes_written += s_util.float_to_f(obj.intercept_time, f)
        bytes_written += s_util.float_to_f(obj.arrival_distance, f)
        bytes_written += s_util.bool_to_f(obj.avoid_collisions, f)
        bytes_written += s_util.float_to_f(obj.max_speed, f)
        bytes_written += s_util.float_to_f(obj.max_thrust, f)
        bytes_written += s_util.float_to_f(obj.max_fine_thrust, f)
        bytes_written += s_util.float_to_f(obj.final_speed, f)

        bytes_written += self._save_pursue_order(obj, f)
        return bytes_written

    def _load_steering_order(self, f:io.IOBase, load_context:save_game.LoadContext, order_id:uuid.UUID) -> tuple[T, Any]:
        target_id = s_util.uuid_from_f(f)
        intercept_location = s_util.float_pair_from_f(f)
        intercept_time = s_util.float_from_f(f)
        arrival_distance = s_util.float_from_f(f)
        avoid_collisions = s_util.bool_from_f(f)
        max_speed = s_util.float_from_f(f)
        max_thrust = s_util.float_from_f(f)
        max_fine_thrust = s_util.float_from_f(f)
        final_speed = s_util.float_from_f(f)

        order, extra_context = self._load_pursue_order(f, load_context, arrival_distance, avoid_collisions, order_id)
        order.intercept_location = intercept_location
        order.intercept_time = intercept_time
        order.max_speed = max_speed
        order.max_thrust = max_thrust
        order.max_fine_thrust = max_fine_thrust
        order.final_speed = final_speed
        return (order, (target_id, extra_context))

    def _post_load_steering_order(self, obj:T, load_context:save_game.LoadContext, context:Any) -> None:
        context_data:tuple[uuid.UUID, Any] = context
        target_id, extra_context = context_data
        target = obj.ship.sensor_settings.get_image(target_id)
        obj.target = target

        self._post_load_pursue_order(obj, load_context, extra_context)

class WaitOrderSaver(AbstractSteeringOrderSaver[movement.WaitOrder]):
    def _save_steering_order(self, obj:movement.WaitOrder, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.float_to_f(obj.wait_wakeup_period, f)
        return bytes_written

    def _load_steering_order(self, f:io.IOBase, load_context:save_game.LoadContext, order_id:uuid.UUID) -> tuple[movement.WaitOrder, Any]:
        wait_wakeup_period = s_util.float_from_f(f)
        order = movement.WaitOrder(load_context.gamestate, _check_flag=True, order_id=order_id)
        order.wait_wakeup_period = wait_wakeup_period
        return (order, None)
