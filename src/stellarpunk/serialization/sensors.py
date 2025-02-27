import io
import uuid
import pydoc
from typing import Any, Optional

from stellarpunk import core, sensors, util
from stellarpunk.serialization import save_game, util as s_util, order as s_order

class SensorSettingsSaver(save_game.Saver[sensors.SensorSettings]):
    def save(self, sensor_settings:sensors.SensorSettings, f:io.IOBase) -> int:
        bytes_written = 0

        # basic fields
        bytes_written += self.save_game.debug_string_w("basic fields", f)
        bytes_written += s_util.float_to_f(sensor_settings._max_sensor_power, f)
        bytes_written += s_util.float_to_f(sensor_settings._sensor_intercept, f)
        bytes_written += s_util.float_to_f(sensor_settings._sensor_power, f)
        bytes_written += s_util.bool_to_f(sensor_settings._transponder_on, f)
        bytes_written += s_util.float_to_f(sensor_settings._thrust, f)

        bytes_written += s_util.float_to_f(sensor_settings._last_sensor_power, f)
        bytes_written += s_util.float_to_f(sensor_settings._last_sensor_power_ts, f)
        bytes_written += s_util.float_to_f(sensor_settings._last_transponder_ts, f)
        bytes_written += s_util.float_to_f(sensor_settings._last_thrust, f)
        bytes_written += s_util.float_to_f(sensor_settings._last_thrust_ts, f)

        bytes_written += s_util.float_to_f(sensor_settings._thrust_seconds, f)
        bytes_written += s_util.bool_to_f(sensor_settings._ignore_bias, f)

        # images
        bytes_written += self.save_game.debug_string_w("images", f)
        bytes_written += s_util.size_to_f(len(sensor_settings.images), f)
        for image in sensor_settings.images:
            bytes_written += self.save_game.save_object(image, f)

        return bytes_written

    def load(self, f:io.IOBase, load_context:save_game.LoadContext) -> sensors.SensorSettings:
        # basic fields
        load_context.debug_string_r("basic fields", f)
        max_sensor_power = s_util.float_from_f(f)
        sensor_intercept = s_util.float_from_f(f)

        sensor_power = s_util.float_from_f(f)
        transponder_on = s_util.bool_from_f(f)
        thrust = s_util.float_from_f(f)

        last_sensor_power = s_util.float_from_f(f)
        last_sensor_power_ts = s_util.float_from_f(f)
        last_transponder_ts = s_util.float_from_f(f)
        last_thrust = s_util.float_from_f(f)
        last_thrust_ts = s_util.float_from_f(f)

        thrust_seconds = s_util.float_from_f(f)
        ignore_bias = s_util.bool_from_f(f)

        sensor_settings = sensors.SensorSettings(max_sensor_power, sensor_intercept)

        sensor_settings._sensor_power = sensor_power
        sensor_settings._transponder_on = transponder_on
        sensor_settings._thrust = thrust

        sensor_settings._last_sensor_power = last_sensor_power
        sensor_settings._last_sensor_power_ts = last_sensor_power_ts
        sensor_settings._last_transponder_ts = last_transponder_ts
        sensor_settings._last_thrust = last_thrust
        sensor_settings._last_thrust_ts = last_thrust_ts

        sensor_settings._thrust_seconds = thrust_seconds
        sensor_settings._ignore_bias = ignore_bias

        # images
        load_context.debug_string_r("images", f)
        count = s_util.size_from_f(f)
        for i in range(count):
            sensor_image = self.save_game.load_object(sensors.SensorImage, f, load_context)
            sensor_settings.register_image(sensor_image)
            # load_context will hold a strong reference to sensor_image because
            # otherwise the only reference is a weak one in sensor_settings
            load_context.reference(sensor_image)

        return sensor_settings

class SensorImageSaver(save_game.Saver[sensors.SensorImage]):
    def save(self, sensor_image:sensors.SensorImage, f:io.IOBase) -> int:
        # the sensor image must be valid (i.e. _ship is non-None, in a sector)
        assert(sensor_image._ship)
        assert(sensor_image._ship.sector)

        bytes_written = 0

        bytes_written += self.save_game.debug_string_w("identity", f)
        bytes_written += s_util.uuid_to_f(sensor_image._identity.entity_id, f)
        bytes_written += s_util.to_len_pre_f(util.fullname(sensor_image._identity.object_type), f)
        bytes_written += s_util.to_len_pre_f(sensor_image._identity.id_prefix, f)
        bytes_written += s_util.float_to_f(sensor_image._identity.mass, f)
        bytes_written += s_util.float_to_f(sensor_image._identity.radius, f)
        bytes_written += s_util.bool_to_f(sensor_image._identity.is_static, f)
        bytes_written += s_util.uuid_to_f(sensor_image._identity.sector_id, f)
        bytes_written += s_util.float_to_f(sensor_image._identity.angle, f)

        bytes_written += self.save_game.debug_string_w("basic fields", f)
        bytes_written += s_util.uuid_to_f(sensor_image._detector_id, f)
        bytes_written += s_util.float_to_f(sensor_image._last_update, f)
        bytes_written += s_util.float_to_f(sensor_image._prior_fidelity, f)
        bytes_written += s_util.float_to_f(sensor_image._last_profile, f)
        bytes_written += s_util.float_pair_to_f(sensor_image._loc, f)
        bytes_written += s_util.float_pair_to_f(sensor_image._velocity, f)
        bytes_written += s_util.float_pair_to_f(sensor_image._acceleration, f)
        bytes_written += s_util.bool_to_f(sensor_image._is_active, f)
        bytes_written += s_util.int_to_f(sensor_image._inactive_reason, f, blen=1)
        bytes_written += s_util.bool_to_f(sensor_image._identified, f)
        bytes_written += s_util.uuid_to_f(sensor_image._ship.sector.entity_id, f)

        bytes_written += self.save_game.debug_string_w("noise", f)
        bytes_written += s_util.float_pair_to_f(sensor_image._loc_bias_direction, f)
        bytes_written += s_util.float_pair_to_f(sensor_image._loc_bias, f)
        bytes_written += s_util.float_pair_to_f(sensor_image._velocity_bias, f)
        bytes_written += s_util.float_to_f(sensor_image._last_bias_update_ts, f)


        return bytes_written

    def load(self, f:io.IOBase, load_context:save_game.LoadContext) -> sensors.SensorImage:

        load_context.debug_string_r("identity", f)
        target_id = s_util.uuid_from_f(f)
        object_type_str = s_util.from_len_pre_f(f)
        id_prefix = s_util.from_len_pre_f(f)
        mass = s_util.float_from_f(f)
        radius = s_util.float_from_f(f)
        is_static = s_util.bool_from_f(f)
        sector_id = s_util.uuid_from_f(f)
        angle = s_util.float_from_f(f)

        load_context.debug_string_r("basic fields", f)
        detector_id = s_util.uuid_from_f(f)
        last_update = s_util.float_from_f(f)
        prior_fidelity = s_util.float_from_f(f)
        last_profile = s_util.float_from_f(f)
        loc = s_util.float_pair_from_f(f)
        velocity = s_util.float_pair_from_f(f)
        acceleration = s_util.float_pair_from_f(f)
        is_active = s_util.bool_from_f(f)
        inactive_reason = s_util.int_from_f(f, blen=1)
        identified = s_util.bool_from_f(f)
        sector_id = s_util.uuid_from_f(f)

        load_context.debug_string_r("noise", f)
        loc_bias_direction = s_util.float_pair_from_f(f)
        loc_bias = s_util.float_pair_from_f(f)
        velocity_bias = s_util.float_pair_from_f(f)
        last_bias_update_ts = s_util.float_from_f(f)

        object_type = pydoc.locate(object_type_str)
        assert(isinstance(object_type, type))
        assert(issubclass(object_type, core.SectorEntity))
        identity = core.SensorIdentity(None, object_type, id_prefix, target_id, mass, radius, is_static, sector_id)
        identity.angle = angle
        sensor_image = sensors.SensorImage(identity)

        sensor_image._detector_id = detector_id
        sensor_image._last_update = last_update
        sensor_image._prior_fidelity = prior_fidelity
        sensor_image._last_profile = last_profile
        sensor_image._loc = loc
        sensor_image._velocity = velocity
        sensor_image._acceleration = acceleration
        sensor_image._is_active = is_active
        sensor_image._inactive_reason = core.SensorImageInactiveReason(inactive_reason)
        sensor_image._identified = identified
        sensor_image._loc_bias_direction = loc_bias_direction
        sensor_image._loc_bias = loc_bias
        sensor_image._velocity_bias = velocity_bias
        sensor_image._last_bias_update_ts = last_bias_update_ts

        load_context.register_post_load(sensor_image, (target_id, detector_id, sector_id))
        return sensor_image

    def post_load(self, sensor_image:sensors.SensorImage, load_context:save_game.LoadContext, context:Any) -> None:
        context_data:tuple[uuid.UUID, uuid.UUID, uuid.UUID] = context
        target_id, detector_id, sector_id = context_data

        sensor_image._ship = load_context.gamestate.get_entity(detector_id, core.SectorEntity)

        sector = load_context.gamestate.get_entity(sector_id, core.Sector)
        sensor_image._sensor_manager = sector.sensor_manager

class SensorScanOrderSaver(s_order.OrderSaver[sensors.SensorScanOrder]):
    def _save_order(self, order:sensors.SensorScanOrder, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.float_to_f(order.images_ttl, f)
        bytes_written += s_util.float_to_f(order.init_sensor_power_ratio, f)
        bytes_written += s_util.float_to_f(order.sensor_power_ratio, f)
        if order.images is None:
            bytes_written += s_util.bool_to_f(False, f)
        else:
            bytes_written += s_util.bool_to_f(True, f)
            bytes_written += s_util.uuids_to_f(list(x.identity.entity_id for x in order.images), f)
        return bytes_written

    def _load_order(self, f:io.IOBase, load_context:save_game.LoadContext, order_id:uuid.UUID) -> tuple[sensors.SensorScanOrder, Any]:
        images_ttl = s_util.float_from_f(f)
        init_sensor_power_ratio = s_util.float_from_f(f)
        sensor_power_ratio = s_util.float_from_f(f)
        has_image_ids = s_util.bool_from_f(f)
        if has_image_ids:
            image_ids = s_util.uuids_from_f(f)
        else:
            image_ids = None

        order = sensors.SensorScanOrder(load_context.gamestate, images_ttl=images_ttl, _check_flag=True, order_id=order_id)
        order.init_sensor_power_ratio = init_sensor_power_ratio
        order.sensor_power_ratio = sensor_power_ratio

        return order, image_ids

    def _post_load_order(self, order:sensors.SensorScanOrder, load_context:save_game.LoadContext, extra_context:Any) -> None:
        image_ids:Optional[list[uuid.UUID]] = extra_context
        if image_ids is None:
            return
        order.images = list(order.ship.sensor_settings.get_image(image_id) for image_id in image_ids)
