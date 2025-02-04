import io
import uuid
import pydoc
import collections
from typing import Any

import cymunk # type: ignore
import numpy as np
import numpy.typing as npt

from stellarpunk import core, sensors, util
from . import save_game, util as s_util, gamestate as s_gamestate

PHYS_ID_NULL = uuid.UUID(hex="deadbeefdeadbeefdeadbeefdeadbeef")
class BodyParams:
    @classmethod
    def from_phys(cls, phys:cymunk.Body) -> "BodyParams":
        if isinstance(phys.data, core.Entity):
            entity_id = phys.data.entity_id
        else:
            entity_id = PHYS_ID_NULL
        return BodyParams(
                entity_id,
                type(phys.data),
                phys.mass,
                phys.moment,
                phys.angle,
                phys.angular_velocity,
                np.array(phys.position),
                np.array(phys.velocity),
                np.array(phys.force),
                phys.torque,
                phys.is_static,
        )

    def __init__(self, entity_id:uuid.UUID, klass:type, mass:float, moment:float, angle:float, angular_velocity:float, position:npt.NDArray[np.float64], velocity:npt.NDArray[np.float64], force:npt.NDArray[np.float64], torque:float, is_static:int) -> None:
        self.entity_id = entity_id
        self.klass = klass
        self.mass = mass
        self.moment = moment
        self.angle = angle
        self.angular_velocity = angular_velocity
        self.position = position
        self.velocity = velocity
        self.force = force
        self.torque = torque
        self.is_static = is_static

    def __eq__(self, other:Any) -> bool:
        if not isinstance(other, BodyParams):
            return False
        if self.entity_id != other.entity_id:
            return False
        if self.klass != other.klass:
            return False
        if not util.inf_nan_isclose(self.mass, other.mass):
            return False
        if not util.inf_nan_isclose(self.moment, other.moment):
            return False
        if not util.inf_nan_isclose(self.angle, other.angle):
            return False
        if not util.inf_nan_isclose(self.angular_velocity, other.angular_velocity):
            return False
        if not util.both_isclose(self.position, other.position):
            return False
        if not util.both_isclose(self.velocity, other.velocity):
            return False
        if not util.both_isclose(self.force, other.force):
            return False
        if not util.inf_nan_isclose(self.torque, other.torque):
            return False
        if self.is_static != other.is_static:
            return False
        return True

class SectorSaver(s_gamestate.EntitySaver[core.Sector]):
    def _save_entity(self, sector:core.Sector, f:io.IOBase) -> int:
        bytes_written = 0

        # basic fields: loc, radius, culture
        bytes_written += self.save_game.debug_string_w("basic fields", f)
        bytes_written += s_util.matrix_to_f(sector.loc, f)
        bytes_written += s_util.float_to_f(sector.radius, f)
        bytes_written += s_util.float_to_f(sector.hex_size, f)
        bytes_written += s_util.to_len_pre_f(sector.culture, f)

        # entities. we'll reconstruct planets, etc. from entities
        bytes_written += self.save_game.debug_string_w("sector entities", f)
        bytes_written += s_util.uuids_to_f(sector.entities.keys(), f)

        # effects
        bytes_written += self.save_game.debug_string_w("effects", f)
        bytes_written += s_util.size_to_f(len(sector._effects), f)
        for effect in sector._effects:
            bytes_written += s_util.uuid_to_f(effect.effect_id, f)


        # weather. we'll reconstruct weather index on load from weather regions
        bytes_written += self.save_game.debug_string_w("weather regions", f)
        bytes_written += s_util.size_to_f(len(sector._weathers), f)
        # we assume weathers are in compact index order (so first element is
        # index 0, second is 1, and so on)
        for weather in sector._weathers.values():
            bytes_written += self.save_game.save_object(weather, f)

        if self.save_game.debug:
            # collision observers for sanity checking
            bytes_written += self.save_game.debug_string_w("collision observers", f)
            bytes_written += s_util.str_uuids_to_f(list((util.fullname(v), k) for k,v in sector.entities.items()), f)
            bytes_written += s_util.str_uuids_to_f(list((util.fullname(v), v.effect_id) for v in sector._effects), f)
            observer_info:list[tuple[str, uuid.UUID]] = []
            for u, observers in sector.collision_observers.items():
                for observer in observers:
                    observer_info.append((util.fullname(observer), u))
            bytes_written += s_util.str_uuids_to_f(observer_info, f)

            # phys object params for sanity checking
            bytes_written += self.save_game.debug_string_w("phys object params", f)
            bodies = sector.space.bodies
            bytes_written += s_util.size_to_f(len(bodies), f)
            for body in bodies:
                if isinstance(body.data, core.Entity):
                    entity_id = body.data.entity_id
                else:
                    entity_id = PHYS_ID_NULL
                bytes_written += s_util.uuid_to_f(entity_id, f)
                bytes_written += s_util.to_len_pre_f(util.fullname(body.data), f)
                bytes_written += s_util.float_to_f(body.mass, f)
                bytes_written += s_util.float_to_f(body.moment, f)
                bytes_written += s_util.float_to_f(body.angle, f)
                bytes_written += s_util.float_to_f(body.angular_velocity, f)
                bytes_written += s_util.float_pair_to_f(np.array(body.position), f)
                bytes_written += s_util.float_pair_to_f(np.array(body.velocity), f)
                bytes_written += s_util.float_pair_to_f(body.force, f)
                bytes_written += s_util.float_to_f(body.torque, f)
                bytes_written += s_util.int_to_f(body.is_static, f)

        return bytes_written

    def _load_entity(self, f:io.IOBase, load_context:save_game.LoadContext, entity_id:uuid.UUID) -> core.Sector:
        #we can't fully load the sector until all entities have been loaded.

        # basic fields: loc, radius, culture
        load_context.debug_string_r("basic fields", f)
        loc = s_util.matrix_from_f(f)
        radius = s_util.float_from_f(f)
        hex_size = s_util.float_from_f(f)
        culture = s_util.from_len_pre_f(f)

        # we can create a blank space here and let it be populated by
        # SectorEntity objects in post_load.
        # SectorEntity is responsible for saving/loading its own physics body
        # and shape
        sector = core.Sector(loc, radius, hex_size, cymunk.Space(), load_context.gamestate, entity_id=entity_id, culture=culture)

        # entities. we'll reconstruct these in post load
        load_context.debug_string_r("sector entities", f)
        entities:list[uuid.UUID] = s_util.uuids_from_f(f)

        # effects
        load_context.debug_string_r("effects", f)
        effect_ids:list[uuid.UUID] = s_util.uuids_from_f(f)

        # weather
        load_context.debug_string_r("weather regions", f)
        count = s_util.size_from_f(f)
        for _ in range(count):
            sector.add_region(self.save_game.load_object(core.SectorWeatherRegion, f, load_context))

        if load_context.debug:
            # collision observers for sanity checking
            load_context.debug_string_r("collision observers", f)
            entity_info = s_util.str_uuids_from_f(f)
            effect_info = s_util.str_uuids_from_f(f)
            observer_info = s_util.str_uuids_from_f(f)

            load_context.debug_string_r("phys object params", f)
            body_count = s_util.size_from_f(f)
            body_params:dict[uuid.UUID, BodyParams] = {}
            for i in range(body_count):
                entity_id = s_util.uuid_from_f(f)
                klass:type = pydoc.locate(s_util.from_len_pre_f(f)) # type: ignore
                mass = s_util.float_from_f(f)
                moment = s_util.float_from_f(f)
                angle = s_util.float_from_f(f)
                angular_velocity = s_util.float_from_f(f)
                position = s_util.float_pair_from_f(f)
                velocity = s_util.float_pair_from_f(f)
                force = s_util.float_pair_from_f(f)
                torque = s_util.float_from_f(f)
                phys_is_static = s_util.int_from_f(f)

                assert(entity_id not in body_params)
                body_params[entity_id] = BodyParams(entity_id, klass, mass, moment, angle, angular_velocity, position, velocity, force, torque, phys_is_static)

            load_context.register_sanity_check(sector, (entity_info, effect_info, observer_info, body_params))

        sector.sensor_manager = sensors.SensorManager(sector)

        load_context.register_post_load(sector, (entities, effect_ids))
        return sector

    def post_load(self, sector:core.Sector, load_context:save_game.LoadContext, context:Any) -> None:
        context_data:tuple[list[uuid.UUID], list[uuid.UUID]] = context
        entities, effect_ids = context_data
        for entity_id in entities:
            entity = load_context.gamestate.get_entity(entity_id, core.SectorEntity)
            sector.add_entity(entity)
        for effect_id in effect_ids:
            effect = load_context.gamestate.get_effect(effect_id, core.Effect) # type: ignore
            # can't use add_effect because it calls begin effect
            sector._effects.append(effect)

    def sanity_check(self, sector:core.Sector, load_context:save_game.LoadContext, context:Any) -> None:
        context_data:tuple[list[tuple[str, uuid.UUID]], list[tuple[str, uuid.UUID]], list[tuple[str, uuid.UUID]], dict[uuid.UUID, BodyParams]] = context
        entity_info, effect_info, observer_info, saved_bodies = context_data
        saved_observers:collections.Counter[tuple[str, uuid.UUID]] = collections.Counter()
        for klass, u in entity_info:
            assert(util.fullname(sector.entities[u]) == klass)
        for klass, u in effect_info:
            effect = next(x for x in sector._effects if x.effect_id == u)
            assert(util.fullname(effect) == klass)

        for klass, u in observer_info:
            saved_observers[(klass, u)] += 1
        loaded_observers:collections.Counter[tuple[str, uuid.UUID]] = collections.Counter()
        for u, observers in sector.collision_observers.items():
            for observer in observers:
                loaded_observers[(util.fullname(observer), u)] += 1

        saved_observers.subtract(loaded_observers)
        non_zero_observers = {observer_id: count for observer_id, count in saved_observers.items() if count != 0}
        assert(non_zero_observers == {})

        bodies = sector.space.bodies
        extra_loaded_bodies:dict[uuid.UUID, BodyParams] = {}
        for body in bodies:
            body_params = BodyParams.from_phys(body)
            if body_params.entity_id in saved_bodies and saved_bodies[body_params.entity_id] == body_params:
                del saved_bodies[body_params.entity_id]
            else:
                extra_loaded_bodies[body_params.entity_id] = body_params

        assert(saved_bodies == {})
        assert(extra_loaded_bodies == {})

class SectorWeatherRegionSaver(save_game.Saver[core.SectorWeatherRegion]):
    def save(self, weather:core.SectorWeatherRegion, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.matrix_to_f(weather.loc, f)
        bytes_written += s_util.float_to_f(weather.radius, f)
        bytes_written += s_util.float_to_f(weather.sensor_factor, f)
        return bytes_written

    def load(self, f:io.IOBase, load_context:save_game.LoadContext) -> core.SectorWeatherRegion:
        loc = s_util.matrix_from_f(f)
        radius = s_util.float_from_f(f)
        sensor_factor = s_util.float_from_f(f)
        weather = core.SectorWeatherRegion(loc, radius, sensor_factor)
        return weather

