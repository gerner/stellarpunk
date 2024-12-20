import io
import uuid
import abc
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from stellarpunk import core

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
        bytes_written = 0

        # all the physical properties needed for the phys body/shape
        bytes_written += s_util.debug_string_w("phys props", f)
        bytes_written += s_util.float_to_f(sector_entity.mass, f)
        bytes_written += s_util.float_to_f(sector_entity.moment, f)
        bytes_written += s_util.float_to_f(sector_entity.radius, f)
        bytes_written += s_util.float_to_f(sector_entity.loc[0], f)
        bytes_written += s_util.float_to_f(sector_entity.loc[1], f)
        bytes_written += s_util.float_to_f(sector_entity.velocity[0], f)
        bytes_written += s_util.float_to_f(sector_entity.velocity[1], f)
        bytes_written += s_util.float_to_f(sector_entity.angle, f)
        bytes_written += s_util.float_to_f(sector_entity.angular_velocity, f)

        # other fields
        bytes_written += s_util.debug_string_w("others", f)
        bytes_written += s_util.float_to_f(sector_entity.cargo_capacity, f)
        bytes_written += s_util.matrix_to_f(sector_entity.cargo, f)
        if sector_entity.captain:
            bytes_written += s_util.int_to_f(1, f, blen=1)
            bytes_written += s_util.uuid_to_f(sector_entity.captain.entity_id, f)
        else:
            bytes_written += s_util.int_to_f(0, f, blen=1)

        # sensor settings
        bytes_written += s_util.debug_string_w("sensor settings", f)
        bytes_written += self.save_game.save_object(sector_entity.sensor_settings, f, klass=core.AbstractSensorSettings)

        bytes_written += s_util.debug_string_w("type specific", f)
        bytes_written += self._save_sector_entity(sector_entity, f)

        return bytes_written

    def _load_entity(self, f:io.IOBase, load_context:save_game.LoadContext, entity_id:uuid.UUID) -> SectorEntity:
        # physical properties
        s_util.debug_string_r("phys props", f)
        mass = s_util.float_from_f(f)
        moment = s_util.float_from_f(f)
        radius = s_util.float_from_f(f)
        loc_x = s_util.float_from_f(f)
        loc_y = s_util.float_from_f(f)
        velocity_x = s_util.float_from_f(f)
        velocity_y = s_util.float_from_f(f)
        angle = s_util.float_from_f(f)
        angular_velocity = s_util.float_from_f(f)

        # other fields
        s_util.debug_string_r("others", f)
        cargo_capacity = s_util.float_from_f(f)
        cargo = s_util.matrix_from_f(f)
        has_captain = s_util.int_from_f(f, blen=1)
        captain_id:Optional[uuid.UUID] = None
        if has_captain:
            captain_id = s_util.uuid_from_f(f)

        s_util.debug_string_r("sensor settings", f)
        sensor_settings = self.save_game.load_object(core.AbstractSensorSettings, f, load_context)

        phys_body = self._phys_body(mass, radius)
        # location gets set in SectorEntity
        phys_body.velocity = (velocity_x, velocity_y)
        phys_body.angle = angle
        phys_body.angular_velocity = angular_velocity
        assert(phys_body.moment == moment)

        s_util.debug_string_r("type specific", f)
        sector_entity, extra_context = self._load_sector_entity(f, load_context, entity_id, np.array((loc_x, loc_y)), phys_body, sensor_settings)

        # phys_shape sets the shape and radius on the sector entity
        self.save_game.generator.phys_shape(phys_body, sector_entity, radius)
        sector_entity.mass = mass
        sector_entity.moment = moment
        sector_entity.cargo_capacity = cargo_capacity
        sector_entity.cargo = cargo

        if has_captain or extra_context is not None:
            load_context.register_post_load(sector_entity, (captain_id, extra_context))

        return sector_entity

    def post_load(self, sector_entity:SectorEntity, load_context:save_game.LoadContext, context:Any) -> None:
        context_data:tuple[Optional[uuid.UUID], Any] = context
        captain_id, extra_context = context_data

        if captain_id is not None:
            captain = load_context.gamestate.entities[captain_id]
            assert(isinstance(captain, core.Character))
            sector_entity.captain = captain

        self._post_load_sector_entity(sector_entity, load_context, extra_context)

class ShipSaver(SectorEntitySaver[core.Ship]):
    def _save_sector_entity(self, ship:core.Ship, f:io.IOBase) -> int:
        bytes_written = 0

        # basic fields
        bytes_written += s_util.debug_string_w("basic fields", f)
        bytes_written += s_util.float_to_f(ship.max_base_thrust, f)
        bytes_written += s_util.float_to_f(ship.max_thrust, f)
        bytes_written += s_util.float_to_f(ship.max_fine_thrust, f)
        bytes_written += s_util.float_to_f(ship.max_torque, f)

        # orders
        bytes_written += s_util.debug_string_w("orders", f)
        bytes_written += s_util.size_to_f(len(ship._orders), f)
        for order in ship._orders:
            bytes_written += self.save_game.save_object(order, f, klass=core.Order)

        #TODO: default_order_fn

        return bytes_written

    def _load_sector_entity(self, f:io.IOBase, load_context:save_game.LoadContext, entity_id:uuid.UUID, loc:npt.NDArray[np.float64], phys_body:cymunk.Body, sensor_settings:core.AbstractSensorSettings) -> tuple[core.Ship, Any]:
        num_products = load_context.gamestate.production_chain.shape[0]
        ship = core.Ship(loc, phys_body, num_products, sensor_settings, load_context.gamestate, entity_id=entity_id)

        #TODO: basic fields
        s_util.debug_string_r("basic fields", f)
        ship.max_base_thrust = s_util.float_from_f(f)
        ship.max_thrust = s_util.float_from_f(f)
        ship.max_fine_thrust = s_util.float_from_f(f)
        ship.max_torque = s_util.float_from_f(f)

        # orders
        s_util.debug_string_r("orders", f)
        count = s_util.size_from_f(f)
        for i in range(count):
            ship._orders.append(self.save_game.load_object(core.Order, f, load_context))

        #TODO: default_order_fn

        return (ship, None)

    def _phys_body(self, mass:float, radius:float) -> cymunk.Body:
        # ship is a non-static body so we override the default implementation
        # which creates static bodies
        return self.save_game.generator.phys_body(mass, radius)

#TODO: should we inherit from ShipSaver?
class MissileSaver(SectorEntitySaver[core.Missile]):
    def _save_sector_entity(self, ship:core.Missile, f:io.IOBase) -> int:
        bytes_written = 0

        # basic fields
        bytes_written += s_util.debug_string_w("basic fields", f)
        bytes_written += s_util.float_to_f(ship.max_base_thrust, f)
        bytes_written += s_util.float_to_f(ship.max_thrust, f)
        bytes_written += s_util.float_to_f(ship.max_fine_thrust, f)
        bytes_written += s_util.float_to_f(ship.max_torque, f)
        if ship.firer:
            bytes_written += s_util.int_to_f(1, f, blen=1)
            bytes_written += s_util.uuid_to_f(ship.firer.entity_id, f)
        else:
            bytes_written += s_util.int_to_f(0, f, blen=1)

        # orders
        bytes_written += s_util.debug_string_w("orders", f)
        for order in ship._orders:
            bytes_written += self.save_game.save_object(order, f, klass=core.Order)

        #TODO: default_order_fn

        return bytes_written

    def _load_sector_entity(self, f:io.IOBase, load_context:save_game.LoadContext, entity_id:uuid.UUID, loc:npt.NDArray[np.float64], phys_body:cymunk.Body, sensor_settings:core.AbstractSensorSettings) -> tuple[core.Missile, Any]:
        num_products = load_context.gamestate.production_chain.shape[0]
        ship = core.Missile(loc, phys_body, num_products, sensor_settings, load_context.gamestate, entity_id=entity_id)

        #TODO: basic fields
        s_util.debug_string_r("basic fields", f)
        ship.max_base_thrust = s_util.float_from_f(f)
        ship.max_thrust = s_util.float_from_f(f)
        ship.max_fine_thrust = s_util.float_from_f(f)
        ship.max_torque = s_util.float_from_f(f)
        has_firer = s_util.int_from_f(f, blen=1)
        firer_id:Optional[uuid.UUID] = None
        if has_firer:
            firer_id = s_util.uuid_from_f(f)

        # orders
        s_util.debug_string_r("orders", f)
        count = s_util.size_from_f(f)
        for i in range(count):
            ship._orders.append(self.save_game.load_object(core.Order, f, load_context))

        #TODO: default_order_fn

        return (ship, firer_id)

    def _phys_body(self, mass:float, radius:float) -> cymunk.Body:
        # ship is a non-static body so we override the default implementation
        # which creates static bodies
        return self.save_game.generator.phys_body(mass, radius)

    def _post_load_sector_entity(self, ship:core.Missile, load_context:save_game.LoadContext, extra_context:Any) -> None:
        firer_id:uuid.UUID = extra_context
        firer = load_context.gamestate.entities[firer_id]
        assert(isinstance(firer, core.SectorEntity))
        ship.firer = firer
