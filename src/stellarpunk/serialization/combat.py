import io
import uuid
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
import cymunk # type: ignore

from stellarpunk import core
from stellarpunk.core import combat
from stellarpunk.serialization import save_game, util as s_util, movement as s_movement, order as s_order, effect as s_effect

class TimedOrderTaskSaver(save_game.Saver[combat.TimedOrderTask]):
    def save(self, obj:combat.TimedOrderTask, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.uuid_to_f(obj.task_id, f)
        bytes_written += s_util.uuid_to_f(obj.order.order_id, f)
        return bytes_written
    def load(self, f:io.IOBase, load_context:save_game.LoadContext) -> combat.TimedOrderTask:
        task_id = s_util.uuid_from_f(f)
        order_id = s_util.uuid_from_f(f)
        tot = combat.TimedOrderTask(task_id=task_id)
        load_context.register_post_load(tot, order_id)
        return tot
    def post_load(self, obj:combat.TimedOrderTask, load_context:save_game.LoadContext, context:Any) -> None:
        order_id:uuid.UUID = context
        order = load_context.gamestate.orders[order_id]
        assert(isinstance(order, core.Order))
        obj.order = order

class ThreatTrackerSaver(save_game.Saver[combat.ThreatTracker]):
    def save(self, obj:combat.ThreatTracker, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.uuid_to_f(obj.craft.entity_id, f)
        bytes_written += s_util.uuids_to_f(obj.threat_ids, f)
        bytes_written += s_util.float_to_f(obj.last_target_ts, f)
        bytes_written += s_util.optional_uuid_to_f(obj.closest_threat.identity.entity_id if obj.closest_threat else None, f)
        bytes_written += s_util.float_to_f(obj.threat_ttl, f)
        return bytes_written

    def load(self, f:io.IOBase, load_context:save_game.LoadContext) -> combat.ThreatTracker:
        craft_id = s_util.uuid_from_f(f)
        threat_ids = set(s_util.uuids_from_f(f))
        last_target_ts = s_util.float_from_f(f)
        closest_threat_id = s_util.optional_uuid_from_f(f)
        threat_ttl = s_util.float_from_f(f)

        threat_tracker = combat.ThreatTracker(threat_ttl=threat_ttl)
        threat_tracker.threat_ids = threat_ids
        threat_tracker.last_target_ts = last_target_ts
        threat_tracker.threat_ttl = threat_ttl

        load_context.register_post_load(threat_tracker, (craft_id, threat_ids, closest_threat_id))

        return threat_tracker

    def post_load(self, threat_tracker:combat.ThreatTracker, load_context:save_game.LoadContext, context:Any) -> None:
        context_data:tuple[uuid.UUID, set[uuid.UUID], Optional[uuid.UUID]] = context
        craft_id, threat_ids, closest_threat_id = context_data

        craft = load_context.gamestate.get_entity(craft_id, core.SectorEntity)
        threat_tracker.craft = craft

        for threat_id in threat_ids:
            threat_tracker.threats.add(craft.sensor_settings.get_image(threat_id))

        if closest_threat_id:
            threat_tracker.closest_threat = craft.sensor_settings.get_image(threat_id)
#TODO: PointDefenseEffect

class MissileOrderSaver(s_movement.PursueOrderSaver[combat.MissileOrder]):
    def _save_pursue_order(self, order:combat.MissileOrder, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.float_to_f(order.ttl, f)
        bytes_written += s_util.float_to_f(order.expiration_time, f)
        return bytes_written

    def _load_pursue_order(self, f:io.IOBase, load_context:save_game.LoadContext, arrival_distance:float, avoid_collisions:bool, order_id:uuid.UUID) -> tuple[combat.MissileOrder, Any]:
        ttl = s_util.float_from_f(f)
        expiration_time = s_util.float_from_f(f)

        order = combat.MissileOrder(load_context.gamestate, arrival_distance=arrival_distance, avoid_collisions=avoid_collisions, _check_flag=True, order_id=order_id)
        order.ttl = ttl
        order.expiration_time = expiration_time
        return order, None

    def _post_load_pursue_order(self, order:combat.MissileOrder, load_context:save_game.LoadContext, context:Any) -> None:
        if order.started_at >= 0. and order.completed_at < 0.:
            assert(order.ship.sector)
            order.ship.sector.register_collision_observer(order.ship.entity_id, order)

class AttackOrderSaver(s_movement.AbstractSteeringOrderSaver[combat.AttackOrder]):

    def _save_steering_order(self, obj:combat.AttackOrder, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.uuid_to_f(obj.target.identity.entity_id, f)
        bytes_written += s_util.float_to_f(obj.distance_min, f)
        bytes_written += s_util.float_to_f(obj.distance_max, f)
        bytes_written += s_util.float_to_f(obj.max_active_age, f)
        bytes_written += s_util.float_to_f(obj.max_passive_age, f)
        bytes_written += s_util.float_to_f(obj.search_distance, f)
        bytes_written += s_util.float_to_f(obj.max_fire_age, f)
        bytes_written += s_util.float_to_f(obj.min_profile_to_threshold, f)
        bytes_written += s_util.float_to_f(obj.fire_backoff_time, f)
        bytes_written += s_util.float_to_f(obj.max_fire_rel_bearing, f)
        bytes_written += s_util.float_to_f(obj.ttl_order_time, f)
        bytes_written += s_util.int_to_f(obj.max_missiles, f)

        bytes_written += s_util.int_to_f(obj.state, f)

        bytes_written += s_util.float_to_f(obj.last_fire_ts, f)
        #TODO: fire period is a temporary hack to limit firing
        bytes_written += s_util.float_to_f(obj.fire_period, f)
        bytes_written += s_util.int_to_f(obj.missiles_fired, f)
        return bytes_written

    def _load_steering_order(self, f:io.IOBase, load_context:save_game.LoadContext, order_id:uuid.UUID) -> tuple[combat.AttackOrder, Any]:
        target_id = s_util.uuid_from_f(f)
        distance_min = s_util.float_from_f(f)
        distance_max = s_util.float_from_f(f)
        max_active_age = s_util.float_from_f(f)
        max_passive_age = s_util.float_from_f(f)
        search_distance = s_util.float_from_f(f)
        max_fire_age = s_util.float_from_f(f)
        min_profile_to_threshold = s_util.float_from_f(f)
        fire_backoff_time = s_util.float_from_f(f)
        max_fire_rel_bearing = s_util.float_from_f(f)
        ttl_order_time = s_util.float_from_f(f)
        max_missiles = s_util.int_from_f(f)

        state = s_util.int_from_f(f)

        last_fire_ts = s_util.float_from_f(f)
        #TODO: fire period is a temporary hack to limit firing
        fire_period = s_util.float_from_f(f)
        missiles_fired = s_util.int_from_f(f)

        order = combat.AttackOrder(load_context.gamestate, distance_min=distance_min, distance_max=distance_max, max_active_age=max_active_age, max_passive_age=max_passive_age, search_distance=search_distance, max_fire_rel_bearing=max_fire_rel_bearing, max_missiles=max_missiles, _check_flag=True, order_id=order_id)
        order.max_fire_age = max_fire_age
        order.min_profile_to_threshold = min_profile_to_threshold
        order.fire_backoff_time = fire_backoff_time
        order.ttl_order_time = ttl_order_time
        order.state = combat.AttackOrder.State(state)
        order.last_fire_ts = last_fire_ts
        order.fire_period = fire_period
        order.missiles_fired = missiles_fired

        return order, target_id

    def _post_load_steering_order(self, order:combat.AttackOrder, load_context:save_game.LoadContext, extra_context:Any) -> None:
        target_id:uuid.UUID = extra_context

        order.target = order.ship.sensor_settings.get_image(target_id)

class HuntOrderSaver(s_order.OrderSaver[combat.HuntOrder]):
    def _save_order(self, order:combat.HuntOrder, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.uuid_to_f(order.target_id, f)
        bytes_written += s_util.optional_uuid_to_f(order.attack_order.order_id if order.attack_order else None, f)
        bytes_written += s_util.float_pair_to_f(order.start_loc, f)
        bytes_written += s_util.float_to_f(order.ttl_order_time, f)
        return bytes_written

    def _load_order(self, f:io.IOBase, load_context:save_game.LoadContext, order_id:uuid.UUID) -> tuple[combat.HuntOrder, Any]:
        target_id = s_util.uuid_from_f(f)
        attack_order_id = s_util.optional_uuid_from_f(f)
        start_loc = s_util.float_pair_from_f(f)
        ttl_order_time = s_util.float_from_f(f)

        order = combat.HuntOrder(target_id, load_context.gamestate, _check_flag=True, order_id=order_id)
        order.start_loc = start_loc
        order.ttl_order_time = ttl_order_time

        return order, attack_order_id

    def _post_load_order(self, order:combat.HuntOrder, load_context:save_game.LoadContext, extra_context:Any) -> None:
        attack_order_id:Optional[uuid.UUID] = extra_context
        if attack_order_id:
            order.attack_order = load_context.gamestate.get_order(attack_order_id, combat.AttackOrder)

class FleeOrderSaver(s_order.OrderSaver[combat.FleeOrder]):
    def _save_order(self, order:combat.FleeOrder, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.float_to_f(order.ttl_order_time, f)
        bytes_written += s_util.float_to_f(order.last_target_ttl, f)
        bytes_written += self.save_game.save_object(order.threat_tracker, f)
        bytes_written += s_util.uuid_to_f(order.point_defense.effect_id, f)
        bytes_written += s_util.float_to_f(order.max_thrust, f)
        return bytes_written

    def _load_order(self, f:io.IOBase, load_context:save_game.LoadContext, order_id:uuid.UUID) -> tuple[combat.FleeOrder, Any]:
        ttl_order_time = s_util.float_from_f(f)
        last_target_ttl = s_util.float_from_f(f)
        threat_tracker = self.save_game.load_object(combat.ThreatTracker, f, load_context)
        point_defense_id = s_util.uuid_from_f(f)
        max_thrust = s_util.float_from_f(f)

        order = combat.FleeOrder(load_context.gamestate, _check_flag=True, order_id=order_id)
        order.ttl_order_time = ttl_order_time
        order.last_target_ttl = last_target_ttl
        order.threat_tracker = threat_tracker
        order.max_thrust = max_thrust

        return order, point_defense_id

    def _post_load_order(self, order:combat.FleeOrder, load_context:save_game.LoadContext, extra_context:Any) -> None:
        point_defense_id:uuid.UUID = extra_context
        order.point_defense = load_context.gamestate.get_effect(point_defense_id, combat.PointDefenseEffect)

class PointDefenseEffectSaver(s_effect.EffectSaver[combat.PointDefenseEffect]):
    def _save_effect(self, effect:combat.PointDefenseEffect, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.uuid_to_f(effect.craft.entity_id, f)
        bytes_written += self.save_game.save_object(effect.threat_tracker, f)
        bytes_written += s_util.int_to_f(effect.state, f)
        bytes_written += s_util.optional_uuid_to_f(effect.current_target.identity.entity_id if effect.current_target else None, f)
        bytes_written += s_util.int_to_f(effect.targets_destroyed, f)

        bytes_written += s_util.float_to_f(effect.idle_interval, f)
        bytes_written += s_util.float_to_f(effect.active_interval, f)
        bytes_written += s_util.float_to_f(effect.pdtarget_expiration, f)

        bytes_written += s_util.float_to_f(effect.muzzle_velocity, f)
        bytes_written += s_util.float_to_f(effect.projectile_ttl, f)
        bytes_written += s_util.float_to_f(effect.cone_half_angle, f)

        if effect._pd_shape:
            bytes_written += s_util.bool_to_f(True, f)
            bytes_written += s_util.float_to_f(effect._pd_shape.body.angle, f)
            bytes_written += s_util.float_pair_to_f(np.array(effect._pd_shape.body.velocity), f)
        else:
            bytes_written += s_util.bool_to_f(False, f)

        bytes_written += s_util.bool_to_f(effect._collided, f)

        return bytes_written

    def _load_effect(self, f:io.IOBase, load_context:save_game.LoadContext, effect_id:uuid.UUID) -> tuple[combat.PointDefenseEffect, Any]:

        craft_id = s_util.uuid_from_f(f)
        threat_tracker = self.save_game.load_object(combat.ThreatTracker, f, load_context)
        state = s_util.int_from_f(f)
        current_target_id = s_util.optional_uuid_from_f(f)
        targets_destroyed = s_util.int_from_f(f)

        idle_interval = s_util.float_from_f(f)
        active_interval = s_util.float_from_f(f)
        pdtarget_expiration = s_util.float_from_f(f)

        muzzle_velocity = s_util.float_from_f(f)
        projectile_ttl = s_util.float_from_f(f)
        cone_half_angle = s_util.float_from_f(f)

        has_pd_shape = s_util.bool_from_f(f)
        if has_pd_shape:
            pd_angle = s_util.float_from_f(f)
            pd_velocity = s_util.float_pair_from_f(f)
            pd_shape_params = (pd_angle, pd_velocity)
        else:
            pd_shape_params = None
        collided = s_util.bool_from_f(f)

        effect = combat.PointDefenseEffect(load_context.gamestate, _check_flag=True, effect_id=effect_id)
        effect.threat_tracker = threat_tracker
        effect.targets_destroyed = targets_destroyed

        effect.idle_interval = idle_interval
        effect.active_interval = active_interval
        effect.pdtarget_expiration = pdtarget_expiration

        effect.muzzle_velocity = muzzle_velocity
        effect.projectile_ttl = projectile_ttl
        effect.cone_half_angle = cone_half_angle

        effect._collided = collided

        return effect, (craft_id, current_target_id, pd_shape_params)

    def _post_load_effect(self, effect:combat.PointDefenseEffect, load_context:save_game.LoadContext, context:Any) -> None:
        context_data:tuple[uuid.UUID, Optional[uuid.UUID], Optional[tuple[float, npt.NDArray[np.float64]]]] = context
        craft_id, current_target_id, pd_shape_params = context_data
        effect.craft = load_context.gamestate.get_entity(craft_id, core.SectorEntity)

        if current_target_id:
            effect.current_target = effect.craft.sensor_settings.get_image(current_target_id)

        if pd_shape_params:
            pd_angle, pd_velocity = pd_shape_params
            shape = effect._setup_shape()
            shape.body.angle = pd_angle
            shape.body.velocity = cymunk.Vec2d(pd_velocity)

