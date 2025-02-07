import io
import uuid
import abc
import uuid
from typing import Any, Optional

from stellarpunk import core, effects, sensors, intel
from stellarpunk.core import sector_entity
from stellarpunk.orders import core as ocore, movement
from stellarpunk.serialization import util as s_util, save_game, order as s_order

class MineOrderSaver(s_order.OrderSaver[ocore.MineOrder]):
    def _save_order(self, order:ocore.MineOrder, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.uuid_to_f(order.target_intel.entity_id, f)
        bytes_written += s_util.bool_to_f(True if order.target_image else False, f)
        bytes_written += s_util.float_to_f(order.max_dist, f)
        bytes_written += s_util.float_to_f(order.amount, f)
        if order.mining_effect:
            bytes_written += s_util.bool_to_f(True, f)
            bytes_written += s_util.uuid_to_f(order.mining_effect.effect_id, f)
        else:
            bytes_written += s_util.bool_to_f(False, f)
        bytes_written += s_util.float_to_f(order.mining_rate, f)
        return bytes_written

    def _load_order(self, f:io.IOBase, load_context:save_game.LoadContext, order_id:uuid.UUID) -> tuple[ocore.MineOrder, Any]:
        asteroid_id = s_util.uuid_from_f(f)
        has_image = s_util.bool_from_f(f)
        max_dist = s_util.float_from_f(f)
        amount = s_util.float_from_f(f)
        has_effect = s_util.bool_from_f(f)
        effect_id:Optional[uuid.UUID] = None
        if has_effect:
            effect_id = s_util.uuid_from_f(f)
        mining_rate = s_util.float_from_f(f)
        order = ocore.MineOrder(load_context.gamestate, max_dist=max_dist, _check_flag=True, order_id=order_id)
        order.amount = amount
        order.mining_rate = mining_rate

        return order, (asteroid_id, has_image, effect_id)

    def _post_load_order(self, order:ocore.MineOrder, load_context:save_game.LoadContext, extra_context:Any) -> None:
        context_data:tuple[uuid.UUID, bool, Optional[uuid.UUID]] = extra_context
        asteroid_id, has_image, effect_id = context_data

        asteroid_intel = load_context.gamestate.get_entity(asteroid_id, intel.AsteroidIntel)
        order.target_intel = asteroid_intel

        if has_image:
            order.target_image = order.ship.sensor_settings.get_image(asteroid_intel.intel_entity_id)

        if effect_id:
            effect = load_context.gamestate.get_effect(effect_id, effects.MiningEffect)
            order.mining_effect=effect

class TransferCargoSaver[T:ocore.TransferCargo](s_order.OrderSaver[T]):
    def _save_transfer_cargo(self, order:T, f:io.IOBase) -> int:
        return 0
    def _load_transfer_cargo(self, f:io.IOBase, load_context:save_game.LoadContext, order_id:uuid.UUID, resource:int, amount:float, max_dist:float) -> tuple[T, Any]:
        order = ocore.TransferCargo(resource, amount, load_context.gamestate, max_dist=max_dist, _check_flag=True, order_id=order_id)
        #TODO: not sure why this is a type error. maybe it has to do with the constraint and this being the super class in the constraint?
        return order, None # type: ignore
    def _post_load_transfer_cargo(self, order:T, load_context:save_game.LoadContext, context:Any) -> None:
        pass

    def _save_order(self, order:T, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.uuid_to_f(order.target_image.identity.entity_id, f)
        bytes_written += s_util.int_to_f(order.resource, f)
        bytes_written += s_util.float_to_f(order.amount, f)
        bytes_written += s_util.float_to_f(order.transferred, f)
        bytes_written += s_util.float_to_f(order.max_dist, f)
        if order.transfer_effect:
            bytes_written += s_util.bool_to_f(True, f)
            bytes_written += s_util.uuid_to_f(order.transfer_effect.effect_id, f)
        else:
            bytes_written += s_util.bool_to_f(False, f)
        bytes_written += self._save_transfer_cargo(order, f)
        return bytes_written

    def _load_order(self, f:io.IOBase, load_context:save_game.LoadContext, order_id:uuid.UUID) -> tuple[T, Any]:

        target_id = s_util.uuid_from_f(f)
        resource = s_util.int_from_f(f)
        amount = s_util.float_from_f(f)
        transferred = s_util.float_from_f(f)
        max_dist = s_util.float_from_f(f)
        has_effect = s_util.bool_from_f(f)
        effect_id:Optional[uuid.UUID] = None
        if has_effect:
            effect_id = s_util.uuid_from_f(f)

        order, extra_context = self._load_transfer_cargo(f, load_context, order_id, resource, amount, max_dist)
        order.transferred = transferred

        return order, (target_id, effect_id, extra_context)


    def _post_load_order(self, order:T, load_context:save_game.LoadContext, context:Any) -> None:
        context_data:tuple[uuid.UUID, uuid.UUID, Any] = context
        target_id, effect_id, extra_context = context_data

        target_image = order.ship.sensor_settings.get_image(target_id)
        order.target_image = target_image

        if effect_id:
            effect = load_context.gamestate.get_effect(effect_id, effects.TransferCargoEffect)
            order.transfer_effect = effect

        self._post_load_transfer_cargo(order, load_context, extra_context)

class TradeCargoToStationSaver(TransferCargoSaver[ocore.TradeCargoToStation]):
    def _save_transfer_cargo(self, order:ocore.TradeCargoToStation, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.uuid_to_f(order.buyer.entity_id, f)
        bytes_written += s_util.uuid_to_f(order.seller.entity_id, f)
        bytes_written += s_util.float_to_f(order.floor_price, f)
        return bytes_written
    def _load_transfer_cargo(self, f:io.IOBase, load_context:save_game.LoadContext, order_id:uuid.UUID, resource:int, amount:float, max_dist:float) -> tuple[ocore.TradeCargoToStation, Any]:
        buyer_id = s_util.uuid_from_f(f)
        seller_id = s_util.uuid_from_f(f)
        floor_price = s_util.float_from_f(f)
        order = ocore.TradeCargoToStation(floor_price, resource, amount, load_context.gamestate, max_dist=max_dist, _check_flag=True, order_id=order_id)
        return order, (buyer_id, seller_id)
    def _post_load_transfer_cargo(self, order:ocore.TradeCargoToStation, load_context:save_game.LoadContext, context:Any) -> None:
        context_data:tuple[uuid.UUID, uuid.UUID] = context
        buyer_id, seller_id = context_data

        #TODO: I'm not sure why passing core.EconAgent (an abstract class) in here causes mypy issues
        buyer = load_context.gamestate.get_entity(buyer_id, core.EconAgent) # type: ignore
        seller = load_context.gamestate.get_entity(seller_id, core.EconAgent) # type: ignore
        order.buyer = buyer
        order.seller = seller

class TradeCargoFromStationSaver(TransferCargoSaver[ocore.TradeCargoFromStation]):
    def _save_transfer_cargo(self, order:ocore.TradeCargoFromStation, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.uuid_to_f(order.buyer.entity_id, f)
        bytes_written += s_util.uuid_to_f(order.seller.entity_id, f)
        bytes_written += s_util.float_to_f(order.ceiling_price, f)
        return bytes_written
    def _load_transfer_cargo(self, f:io.IOBase, load_context:save_game.LoadContext, order_id:uuid.UUID, resource:int, amount:float, max_dist:float) -> tuple[ocore.TradeCargoFromStation, Any]:
        buyer_id = s_util.uuid_from_f(f)
        seller_id = s_util.uuid_from_f(f)
        ceiling_price = s_util.float_from_f(f)
        order = ocore.TradeCargoFromStation(ceiling_price, resource, amount, load_context.gamestate, max_dist=max_dist, _check_flag=True, order_id=order_id)
        return order, (buyer_id, seller_id)
    def _post_load_transfer_cargo(self, order:ocore.TradeCargoFromStation, load_context:save_game.LoadContext, context:Any) -> None:
        context_data:tuple[uuid.UUID, uuid.UUID] = context
        buyer_id, seller_id = context_data

        #TODO: I'm not sure why passing core.EconAgent (an abstract class) in here causes mypy issues
        buyer = load_context.gamestate.get_entity(buyer_id, core.EconAgent) # type: ignore
        seller = load_context.gamestate.get_entity(seller_id, core.EconAgent) # type: ignore
        order.buyer = buyer
        order.seller = seller

"""
class DisembarkToEntitySaver(s_order.OrderSaver[ocore.DisembarkToEntity]):
    def _save_order(self, order:ocore.DisembarkToEntity, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.float_to_f(order.disembark_dist, f)
        bytes_written += s_util.float_to_f(order.disembark_margin, f)
        bytes_written += s_util.optional_uuid_to_f(order.disembark_from.entity_id if order.disembark_from else None, f)
        bytes_written += s_util.uuid_to_f(order.embark_to.entity_id, f)
        bytes_written += s_util.optional_uuid_to_f(order.disembark_order.order_id if order.disembark_order else None, f)
        bytes_written += s_util.optional_uuid_to_f(order.embark_order.order_id if order.embark_order else None, f)
        return bytes_written
    def _load_order(self, f:io.IOBase, load_context:save_game.LoadContext, order_id:uuid.UUID) -> tuple[ocore.DisembarkToEntity, Any]:
        disembark_dist = s_util.float_from_f(f)
        disembark_margin = s_util.float_from_f(f)
        disembark_from_id = s_util.optional_uuid_from_f(f)
        embark_to_id = s_util.uuid_from_f(f)
        disembark_order_id = s_util.optional_uuid_from_f(f)
        embark_order_id = s_util.optional_uuid_from_f(f)

        order = ocore.DisembarkToEntity(load_context.gamestate, disembark_dist=disembark_dist, disembark_margin=disembark_margin)
        return order, (disembark_from_id, embark_to_id, disembark_order_id, embark_order_id)
    def _post_load_order(self, order:ocore.DisembarkToEntity, load_context:save_game.LoadContext, context:Any) -> None:
        context_data:tuple[uuid.UUID, uuid.UUID, Optional[uuid.UUID], Optional[uuid.UUID]] = context
        disembark_from_id, embark_to_id, disembark_order_id, embark_order_id = context_data
        if disembark_from_id:
            order.disembark_from = load_context.gamestate.get_entity(disembark_from_id, core.SectorEntity)
        order.embark_to = load_context.gamestate.get_entity(embark_to_id, core.SectorEntity)
        if disembark_order_id:
            order.disembark_order = load_context.gamestate.get_order(disembark_order_id, movement.GoToLocation)
        if embark_order_id:
            order.embark_order = load_context.gamestate.get_order(embark_order_id, movement.GoToLocation)
"""

class TravelThroughGateSaver(s_order.OrderSaver[ocore.TravelThroughGate], abc.ABC):
    def _save_order(self, order:ocore.TravelThroughGate, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.uuid_to_f(order.target_gate.entity_id, f)
        bytes_written += s_util.float_to_f(order.position_margin, f)
        bytes_written += s_util.float_to_f(order.travel_time, f)
        bytes_written += s_util.float_to_f(order.travel_thrust, f)
        bytes_written += s_util.float_to_f(order.max_gate_dist, f)

        bytes_written += s_util.int_to_f(order.phase, f)
        bytes_written += s_util.float_to_f(order.travel_start_time, f)
        bytes_written += s_util.optional_uuid_to_f(order.rotate_order.order_id if order.rotate_order else None, f)

        bytes_written += s_util.optional_uuid_to_f(order.warp_out.effect_id if order.warp_out else None, f)
        bytes_written += s_util.optional_uuid_to_f(order.warp_in.effect_id if order.warp_in else None, f)
        return bytes_written

    def _load_order(self, f:io.IOBase, load_context:save_game.LoadContext, order_id:uuid.UUID) -> tuple[ocore.TravelThroughGate, Any]:
        target_gate_id = s_util.uuid_from_f(f)
        position_margin = s_util.float_from_f(f)
        travel_time = s_util.float_from_f(f)
        travel_thrust = s_util.float_from_f(f)
        max_gate_dist = s_util.float_from_f(f)

        phase = s_util.int_from_f(f)
        travel_start_time = s_util.float_from_f(f)
        rotate_order_id = s_util.optional_uuid_from_f(f)

        warp_out_id = s_util.optional_uuid_from_f(f)
        warp_in_id = s_util.optional_uuid_from_f(f)

        order = ocore.TravelThroughGate(load_context.gamestate, position_margin=position_margin, travel_time=travel_time, travel_thrust=travel_thrust, max_gate_dist=max_gate_dist, _check_flag=True, order_id=order_id)
        order.phase = ocore.TravelThroughGate.Phase(phase)
        order.travel_start_time = travel_start_time

        return order, (target_gate_id, rotate_order_id, warp_out_id, warp_in_id)

    def _post_load_order(self, order:ocore.TravelThroughGate, load_context:save_game.LoadContext, context:Any) -> None:
        context_data:tuple[uuid.UUID, Optional[uuid.UUID], Optional[uuid.UUID], Optional[uuid.UUID]] = context
        target_gate_id, rotate_order_id, warp_out_id, warp_in_id = context_data

        target_gate = load_context.gamestate.get_entity(target_gate_id, sector_entity.TravelGate)
        order.target_gate = target_gate

        order.eow = core.EntityOrderWatch(order, target_gate)

        if rotate_order_id:
            order.rotate_order = load_context.gamestate.get_order(rotate_order_id, movement.RotateOrder)
        if warp_out_id:
            order.warp_out = load_context.gamestate.get_effect(warp_out_id, effects.WarpOutEffect)
        if warp_in_id:
            order.warp_in = load_context.gamestate.get_effect(warp_in_id, effects.WarpInEffect)

class DockingOrderSaver(s_order.OrderSaver[ocore.DockingOrder], abc.ABC):
    def _save_order(self, order:ocore.DockingOrder, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.uuid_to_f(order.target_image.identity.entity_id, f)
        bytes_written += s_util.float_to_f(order.surface_distance, f)
        bytes_written += s_util.float_to_f(order.approach_distance, f)
        bytes_written += s_util.float_to_f(order.wait_time, f)
        bytes_written += s_util.float_to_f(order.next_arrival_attempt_time, f)
        bytes_written += s_util.float_to_f(order.started_waiting, f)
        return bytes_written

    def _load_order(self, f:io.IOBase, load_context:save_game.LoadContext, order_id:uuid.UUID) -> tuple[ocore.DockingOrder, Any]:
        target_entity_id = s_util.uuid_from_f(f)
        surface_distance = s_util.float_from_f(f)
        approach_distance = s_util.float_from_f(f)
        wait_time = s_util.float_from_f(f)
        next_arrival_attempt_time = s_util.float_from_f(f)
        started_waiting = s_util.float_from_f(f)

        order = ocore.DockingOrder(load_context.gamestate, surface_distance=surface_distance, approach_distance=approach_distance, wait_time=wait_time, _check_flag=True, order_id=order_id)

        return order, target_entity_id

    def _post_load_order(self, order:ocore.DockingOrder, load_context:save_game.LoadContext, context:Any) -> None:
        target_entity_id:uuid.UUID = context

        target_image = order.ship.sensor_settings.get_image(target_entity_id)
        order.target_image = target_image

class LocationExploreOrderSaver(s_order.OrderSaver[ocore.LocationExploreOrder], abc.ABC):
    def _save_order(self, order:ocore.LocationExploreOrder, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.uuid_to_f(order.sector_id, f)
        bytes_written += s_util.float_pair_to_f(order.loc, f)
        bytes_written += s_util.optional_uuid_to_f(order.goto_order.order_id if order.goto_order else None, f)
        bytes_written += s_util.optional_uuid_to_f(order.scan_order.order_id if order.scan_order else None, f)
        return bytes_written

    def _load_order(self, f:io.IOBase, load_context:save_game.LoadContext, order_id:uuid.UUID) -> tuple[ocore.LocationExploreOrder, Any]:
        sector_id = s_util.uuid_from_f(f)
        loc = s_util.float_pair_from_f(f)
        goto_location_id = s_util.optional_uuid_from_f(f)
        sensor_scan_id = s_util.optional_uuid_from_f(f)

        order = ocore.LocationExploreOrder(sector_id, loc, load_context.gamestate, _check_flag=True, order_id=order_id)

        return order, (goto_location_id, sensor_scan_id)

    def _post_load_order(self, order:ocore.LocationExploreOrder, load_context:save_game.LoadContext, context:Any) -> None:
        context_data:tuple[Optional[uuid.UUID], Optional[uuid.UUID]] = context
        goto_location_id, sensor_scan_id = context_data
        if goto_location_id:
            order.goto_order = load_context.gamestate.get_order(goto_location_id, movement.GoToLocation)
        if sensor_scan_id:
            order.scan_order = load_context.gamestate.get_order(sensor_scan_id, sensors.SensorScanOrder)

