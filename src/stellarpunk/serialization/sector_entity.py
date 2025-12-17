import io
import uuid
import abc
import math
import json
import collections
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from stellarpunk import core, util
from stellarpunk.core import combat, sector_entity

import cymunk # type: ignore

from . import save_game, util as s_util, gamestate as s_gamestate

class SectorEntitySaver[SectorEntity: core.SectorEntity](s_gamestate.EntitySaver[SectorEntity], abc.ABC):
    @abc.abstractmethod
    def _save_sector_entity(self, sector_entity:SectorEntity, f:io.IOBase) -> int: ...
    @abc.abstractmethod
    def _load_sector_entity(self, f:io.IOBase, load_context:save_game.LoadContext, entity_id:uuid.UUID, loc:npt.NDArray[np.float64], phys_body:cymunk.Body, sensor_settings:core.AbstractSensorSettings) -> tuple[SectorEntity, Any]: ...

    def _post_load_sector_entity(self, sector_entity:SectorEntity, load_context:save_game.LoadContext, extra_context:Any) -> None:
        # SectorEntity types can override this to get post load action
        pass

    def _phys_body(self, mass:float, radius:float) -> cymunk.Body:
        # SectorEntity types with non-static bodies should override this method
        return self.save_game.generator.phys_body()

    # EntitySaver
    def _save_entity(self, sector_entity:SectorEntity, f:io.IOBase) -> int:

        assert(not np.isnan(sector_entity.loc[0]))
        assert(not np.isnan(sector_entity.loc[1]))
        assert(not np.isnan(sector_entity.velocity[0]))
        assert(not np.isnan(sector_entity.velocity[1]))

        bytes_written = 0

        # all the physical properties needed for the phys body/shape
        bytes_written += self.save_game.debug_string_w("phys props", f)
        bytes_written += s_util.float_to_f(sector_entity.mass, f)
        bytes_written += s_util.float_to_f(sector_entity.moment, f)
        bytes_written += s_util.float_to_f(sector_entity.radius, f)
        bytes_written += s_util.float_to_f(sector_entity.loc[0], f)
        bytes_written += s_util.float_to_f(sector_entity.loc[1], f)
        bytes_written += s_util.float_to_f(sector_entity.velocity[0], f)
        bytes_written += s_util.float_to_f(sector_entity.velocity[1], f)
        bytes_written += s_util.float_to_f(sector_entity.angle, f)
        bytes_written += s_util.float_to_f(sector_entity.angular_velocity, f)
        bytes_written += s_util.float_pair_to_f(np.array(sector_entity.phys.force), f)
        bytes_written += s_util.float_to_f(sector_entity.phys.torque, f)

        # other fields
        bytes_written += self.save_game.debug_string_w("others", f)
        bytes_written += s_util.float_to_f(sector_entity.cargo_capacity, f)
        bytes_written += s_util.matrix_to_f(sector_entity.cargo, f)
        bytes_written += s_util.bool_to_f(sector_entity.is_static, f)
        if isinstance(sector_entity, core.CrewedSectorEntity) and sector_entity.captain:
            bytes_written += s_util.int_to_f(1, f, blen=1)
            bytes_written += s_util.uuid_to_f(sector_entity.captain.entity_id, f)
        else:
            bytes_written += s_util.int_to_f(0, f, blen=1)

        # sensor settings
        bytes_written += self.save_game.debug_string_w("sensor settings", f)
        bytes_written += self.save_game.save_object(sector_entity.sensor_settings, f, klass=core.AbstractSensorSettings)

        if self.save_game.debug:
            bytes_written += self.save_game.debug_string_w("history", f)
            history_json = json.dumps(list(x.to_dict() for x in sector_entity.get_history()))
            bytes_written += s_util.to_len_pre_f(history_json, f, blen=4)

        bytes_written += self.save_game.debug_string_w("type specific", f)
        bytes_written += self._save_sector_entity(sector_entity, f)

        return bytes_written

    def _load_entity(self, f:io.IOBase, load_context:save_game.LoadContext, entity_id:uuid.UUID) -> SectorEntity:
        # physical properties
        load_context.debug_string_r("phys props", f)
        mass = s_util.float_from_f(f)
        moment = s_util.float_from_f(f)
        radius = s_util.float_from_f(f)
        loc_x = s_util.float_from_f(f)
        loc_y = s_util.float_from_f(f)
        velocity_x = s_util.float_from_f(f)
        velocity_y = s_util.float_from_f(f)
        angle = s_util.float_from_f(f)
        angular_velocity = s_util.float_from_f(f)
        force = s_util.float_pair_from_f(f)
        torque = s_util.float_from_f(f)

        # other fields
        load_context.debug_string_r("others", f)
        cargo_capacity = s_util.float_from_f(f)
        cargo = s_util.matrix_from_f(f)
        is_static = s_util.bool_from_f(f)
        has_captain = s_util.int_from_f(f, blen=1)
        captain_id:Optional[uuid.UUID] = None
        if has_captain:
            captain_id = s_util.uuid_from_f(f)

        load_context.debug_string_r("sensor settings", f)
        sensor_settings = self.save_game.load_object(core.AbstractSensorSettings, f, load_context)

        phys_body = self._phys_body(mass, radius)
        # location gets set in SectorEntity
        phys_body.velocity = (velocity_x, velocity_y)
        phys_body.angle = angle
        phys_body.angular_velocity = angular_velocity
        phys_body.force = cymunk.Vec2d(force)
        phys_body.torque = torque
        assert(util.isclose(phys_body.moment, moment) or (math.isinf(phys_body.mass) and math.isinf(phys_body.moment)))

        if self.save_game.debug:
            load_context.debug_string_r("history", f)
            history_json = s_util.from_len_pre_f(f, blen=4)
            history_list = json.loads(history_json)
            history = list(core.HistoryEntry.from_dict(h) for h in history_list)


        load_context.debug_string_r("type specific", f)
        sector_entity, extra_context = self._load_sector_entity(f, load_context, entity_id, np.array((loc_x, loc_y)), phys_body, sensor_settings)
        sector_entity.is_static = is_static

        if self.save_game.debug:
            sector_entity.set_history(history)

        # phys_shape sets the shape and radius on the sector entity
        self.save_game.generator.phys_shape(phys_body, sector_entity, radius)
        sector_entity.cargo_capacity = cargo_capacity
        sector_entity.cargo = cargo

        assert(not np.isnan(sector_entity.loc[0]))
        assert(not np.isnan(sector_entity.loc[1]))
        assert(not np.isnan(sector_entity.velocity[0]))
        assert(not np.isnan(sector_entity.velocity[1]))

        if has_captain or extra_context is not None:
            load_context.register_post_load(sector_entity, (captain_id, extra_context))
        load_context.register_sanity_check(sector_entity, None)

        return sector_entity

    def post_load(self, sector_entity:SectorEntity, load_context:save_game.LoadContext, context:Any) -> None:
        context_data:tuple[Optional[uuid.UUID], Any] = context
        captain_id, extra_context = context_data

        if captain_id is not None:
            assert(isinstance(sector_entity, core.CrewedSectorEntity))
            captain = load_context.gamestate.entities[captain_id]
            assert(isinstance(captain, core.Character))
            sector_entity.captain = captain

        self._post_load_sector_entity(sector_entity, load_context, extra_context)

    def sanity_check(self, sector_entity:SectorEntity, load_context:save_game.LoadContext, context:Any) -> None:

        assert(not np.isnan(sector_entity.loc[0]))
        assert(not np.isnan(sector_entity.loc[1]))
        assert(not np.isnan(sector_entity.velocity[0]))
        assert(not np.isnan(sector_entity.velocity[1]))

class ShipSaver(SectorEntitySaver[core.Ship]):
    def _save_sector_entity(self, ship:core.Ship, f:io.IOBase) -> int:
        bytes_written = 0

        # basic fields
        bytes_written += self.save_game.debug_string_w("basic fields", f)
        bytes_written += s_util.float_to_f(ship.max_base_thrust, f)
        bytes_written += s_util.float_to_f(ship.max_thrust, f)
        bytes_written += s_util.float_to_f(ship.max_fine_thrust, f)
        bytes_written += s_util.float_to_f(ship.max_torque, f)

        # rocket model
        bytes_written += self.save_game.debug_string_w("rocket model", f)
        bytes_written += s_util.float_to_f(ship.rocket_model.get_i_sp(), f)
        bytes_written += s_util.float_to_f(ship.rocket_model.get_thrust(), f)
        bytes_written += s_util.float_to_f(ship.rocket_model.get_propellant(), f)

        # orders
        bytes_written += self.save_game.debug_string_w("orders", f)
        bytes_written += s_util.size_to_f(len(ship._orders), f)
        for order in ship._orders:
            bytes_written += s_util.uuid_to_f(order.order_id, f)

        #TODO: default_order_fn

        return bytes_written

    def _load_sector_entity(self, f:io.IOBase, load_context:save_game.LoadContext, entity_id:uuid.UUID, loc:npt.NDArray[np.float64], phys_body:cymunk.Body, sensor_settings:core.AbstractSensorSettings) -> tuple[core.Ship, Any]:
        num_products = load_context.gamestate.production_chain.shape[0]
        ship = core.Ship(loc, phys_body, num_products, sensor_settings, load_context.gamestate, entity_id=entity_id)

        # basic fields
        load_context.debug_string_r("basic fields", f)
        ship.max_base_thrust = s_util.float_from_f(f)
        ship.max_thrust = s_util.float_from_f(f)
        ship.max_fine_thrust = s_util.float_from_f(f)
        ship.max_torque = s_util.float_from_f(f)

        # rocket model
        load_context.debug_string_r("rocket model", f)
        ship.rocket_model.set_i_sp(s_util.float_from_f(f))
        ship.rocket_model.set_thrust(s_util.float_from_f(f))
        ship.rocket_model.set_propellant(s_util.float_from_f(f))

        # orders
        load_context.debug_string_r("orders", f)
        order_ids:list[uuid.UUID] = []
        count = s_util.size_from_f(f)
        for i in range(count):
            order_id = s_util.uuid_from_f(f)
            order_ids.append(order_id)

        #TODO: default_order_fn

        return (ship, order_ids)

    def _phys_body(self, mass:float, radius:float) -> cymunk.Body:
        # ship is a non-static body so we override the default implementation
        # which creates static bodies
        return self.save_game.generator.phys_body(mass, radius)

    def _post_load_sector_entity(self, ship:core.Ship, load_context:save_game.LoadContext, extra_context:Any) -> None:
        order_ids:list[uuid.UUID] = extra_context
        for order_id in order_ids:
            order = load_context.gamestate.orders[order_id]
            assert(isinstance(order, core.Order))
            ship._orders.append(order)

#TODO: should we inherit from ShipSaver?
class MissileSaver(SectorEntitySaver[combat.Missile]):
    def _save_sector_entity(self, ship:combat.Missile, f:io.IOBase) -> int:
        bytes_written = 0

        # basic fields
        bytes_written += self.save_game.debug_string_w("basic fields", f)
        bytes_written += s_util.float_to_f(ship.max_base_thrust, f)
        bytes_written += s_util.float_to_f(ship.max_thrust, f)
        bytes_written += s_util.float_to_f(ship.max_fine_thrust, f)
        bytes_written += s_util.float_to_f(ship.max_torque, f)
        if ship.firer:
            bytes_written += s_util.int_to_f(1, f, blen=1)
            bytes_written += s_util.uuid_to_f(ship.firer.entity_id, f)
        else:
            bytes_written += s_util.int_to_f(0, f, blen=1)

        # rocket model
        bytes_written += self.save_game.debug_string_w("rocket model", f)
        bytes_written += s_util.float_to_f(ship.rocket_model.get_i_sp(), f)
        bytes_written += s_util.float_to_f(ship.rocket_model.get_thrust(), f)
        bytes_written += s_util.float_to_f(ship.rocket_model.get_propellant(), f)

        # orders
        bytes_written += self.save_game.debug_string_w("orders", f)
        bytes_written += s_util.size_to_f(len(ship._orders), f)
        for order in ship._orders:
            bytes_written += s_util.uuid_to_f(order.order_id, f)

        #TODO: default_order_fn

        return bytes_written

    def _load_sector_entity(self, f:io.IOBase, load_context:save_game.LoadContext, entity_id:uuid.UUID, loc:npt.NDArray[np.float64], phys_body:cymunk.Body, sensor_settings:core.AbstractSensorSettings) -> tuple[combat.Missile, Any]:
        num_products = load_context.gamestate.production_chain.shape[0]
        ship = combat.Missile(loc, phys_body, num_products, sensor_settings, load_context.gamestate, entity_id=entity_id)

        # basic fields
        load_context.debug_string_r("basic fields", f)
        ship.max_base_thrust = s_util.float_from_f(f)
        ship.max_thrust = s_util.float_from_f(f)
        ship.max_fine_thrust = s_util.float_from_f(f)
        ship.max_torque = s_util.float_from_f(f)
        has_firer = s_util.int_from_f(f, blen=1)
        firer_id:Optional[uuid.UUID] = None
        if has_firer:
            firer_id = s_util.uuid_from_f(f)

        # rocket model
        load_context.debug_string_r("rocket model", f)
        ship.rocket_model.set_i_sp(s_util.float_from_f(f))
        ship.rocket_model.set_thrust(s_util.float_from_f(f))
        ship.rocket_model.set_propellant(s_util.float_from_f(f))

        # orders
        load_context.debug_string_r("orders", f)
        order_ids:list[uuid.UUID] = []
        count = s_util.size_from_f(f)
        for i in range(count):
            order_id = s_util.uuid_from_f(f)
            order_ids.append(order_id)

        #TODO: default_order_fn

        return ship, (firer_id, order_ids)

    def _phys_body(self, mass:float, radius:float) -> cymunk.Body:
        # ship is a non-static body so we override the default implementation
        # which creates static bodies
        return self.save_game.generator.phys_body(mass, radius)

    def _post_load_sector_entity(self, ship:combat.Missile, load_context:save_game.LoadContext, extra_context:Any) -> None:
        context_data:tuple[uuid.UUID, list[uuid.UUID]] = extra_context
        firer_id, order_ids = context_data
        firer = load_context.gamestate.entities[firer_id]
        assert(isinstance(firer, core.SectorEntity))
        ship.firer = firer

        for order_id in order_ids:
            order = load_context.gamestate.orders[order_id]
            assert(isinstance(order, core.Order))
            ship._orders.append(order)

class PlanetSaver(SectorEntitySaver[sector_entity.Planet]):
    def _save_sector_entity(self, sector_entity:sector_entity.Planet, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.float_to_f(sector_entity.population, f)
        return bytes_written
    def _load_sector_entity(self, f:io.IOBase, load_context:save_game.LoadContext, entity_id:uuid.UUID, loc:npt.NDArray[np.float64], phys_body:cymunk.Body, sensor_settings:core.AbstractSensorSettings) -> tuple[sector_entity.Planet, Any]:
        population = s_util.float_from_f(f)
        num_products = load_context.gamestate.production_chain.shape[0]
        planet = sector_entity.Planet(loc, phys_body, num_products, sensor_settings, load_context.gamestate, entity_id=entity_id)
        planet.population = population
        return (planet, None)

class StationSaver(SectorEntitySaver[sector_entity.Station]):
    def _save_sector_entity(self, sector_entity:sector_entity.Station, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.int_to_f(int(sector_entity.resource), f)
        bytes_written += s_util.float_to_f(sector_entity.next_batch_time, f)
        bytes_written += s_util.float_to_f(sector_entity.next_production_time, f)
        bytes_written += s_util.to_len_pre_f(sector_entity.sprite.sprite_id, f)
        return bytes_written
    def _load_sector_entity(self, f:io.IOBase, load_context:save_game.LoadContext, entity_id:uuid.UUID, loc:npt.NDArray[np.float64], phys_body:cymunk.Body, sensor_settings:core.AbstractSensorSettings) -> tuple[sector_entity.Station, Any]:

        resource = s_util.int_from_f(f)
        next_batch_time = s_util.float_from_f(f)
        next_production_time = s_util.float_from_f(f)
        sprite_id = s_util.from_len_pre_f(f)
        sprite = load_context.generator.sprite_store[sprite_id]

        num_products = load_context.gamestate.production_chain.shape[0]
        station = sector_entity.Station(sprite, loc, phys_body, num_products, sensor_settings, load_context.gamestate, entity_id=entity_id)
        station.resource = resource
        station.next_batch_time = next_batch_time
        station.next_production_time = next_production_time
        return (station, None)

class AsteroidSaver(SectorEntitySaver[sector_entity.Asteroid]):
    def _save_sector_entity(self, asteroid:sector_entity.Asteroid, f:io.IOBase) -> int:
        return s_util.int_to_f(int(asteroid.resource), f)

    def _load_sector_entity(self, f:io.IOBase, load_context:save_game.LoadContext, entity_id:uuid.UUID, loc:npt.NDArray[np.float64], phys_body:cymunk.Body, sensor_settings:core.AbstractSensorSettings) -> tuple[sector_entity.Asteroid, Any]:
        resource = s_util.int_from_f(f)
        num_products = load_context.gamestate.production_chain.shape[0]
        asteroid = sector_entity.Asteroid(resource, 0.0, loc, phys_body, num_products, sensor_settings, load_context.gamestate, entity_id=entity_id)
        return (asteroid, None)

class TravelGateSaver(SectorEntitySaver[sector_entity.TravelGate]):
    def _save_sector_entity(self, sector_entity:sector_entity.TravelGate, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.uuid_to_f(sector_entity.destination.entity_id, f)
        bytes_written += s_util.float_to_f(sector_entity.direction, f)
        return bytes_written
    def _load_sector_entity(self, f:io.IOBase, load_context:save_game.LoadContext, entity_id:uuid.UUID, loc:npt.NDArray[np.float64], phys_body:cymunk.Body, sensor_settings:core.AbstractSensorSettings) -> tuple[sector_entity.TravelGate, Any]:
        destination_id = s_util.uuid_from_f(f)
        direction = s_util.float_from_f(f)
        num_products = load_context.gamestate.production_chain.shape[0]
        gate = sector_entity.TravelGate(direction, loc, phys_body, num_products, sensor_settings, load_context.gamestate, entity_id=entity_id)
        return (gate, destination_id)

    def _post_load_sector_entity(self, sector_entity:sector_entity.TravelGate, load_context:save_game.LoadContext, extra_context:Any) -> None:
        destination_id:uuid.UUID = extra_context
        destination = load_context.gamestate.entities[destination_id]
        assert(isinstance(destination, core.Sector))
        sector_entity.destination = destination

class ProjectileSaver(SectorEntitySaver[sector_entity.Projectile]):
    def _save_sector_entity(self, sector_entity:sector_entity.Projectile, f:io.IOBase) -> int:
        bytes_written = 0
        return bytes_written
    def _load_sector_entity(self, f:io.IOBase, load_context:save_game.LoadContext, entity_id:uuid.UUID, loc:npt.NDArray[np.float64], phys_body:cymunk.Body, sensor_settings:core.AbstractSensorSettings) -> tuple[sector_entity.Projectile, Any]:
        num_products = load_context.gamestate.production_chain.shape[0]
        projectile = sector_entity.Projectile(loc, phys_body, num_products, sensor_settings, load_context.gamestate, entity_id=entity_id)
        return (projectile, None)
