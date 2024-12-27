import io
import abc
import uuid
from typing import Any, Optional

from stellarpunk import core, agenda, econ
from stellarpunk.core import sector_entity
from stellarpunk.orders import core as ocore
from stellarpunk.serialization import save_game, util as s_util

class AgendumSaver[T:agenda.Agendum](save_game.Saver[T], abc.ABC):
    @abc.abstractmethod
    def _save_agendum(self, obj:T, f:io.IOBase) -> int: ...
    @abc.abstractmethod
    def _load_agendum(self, f:io.IOBase, load_context:save_game.LoadContext, agenda_id:uuid.UUID) -> tuple[T, Any]: ...
    def _post_load_agendum(self, obj:T, load_context:save_game.LoadContext, context:Any) -> None: ...

    def save(self, obj:T, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.debug_string_w("basic fields", f)
        bytes_written += s_util.uuid_to_f(obj.agenda_id, f)
        bytes_written += s_util.uuid_to_f(obj.character.entity_id, f)
        bytes_written += s_util.float_to_f(obj.started_at, f)
        bytes_written += s_util.float_to_f(obj.stopped_at, f)
        bytes_written += s_util.debug_string_w("type specific", f)
        bytes_written += self._save_agendum(obj, f)
        return bytes_written

    def load(self, f:io.IOBase, load_context:save_game.LoadContext) -> T:
        s_util.debug_string_r("basic fields", f)
        agenda_id = s_util.uuid_from_f(f)
        character_id = s_util.uuid_from_f(f)
        started_at = s_util.float_from_f(f)
        stopped_at = s_util.float_from_f(f)
        s_util.debug_string_r("type specific", f)
        (agendum, extra_context) = self._load_agendum(f, load_context, agenda_id)
        agendum.started_at = started_at
        agendum.stopped_at = stopped_at
        load_context.gamestate.register_agendum(agendum)

        load_context.register_post_load(agendum, (character_id, extra_context))

        return agendum

    def post_load(self, agendum:T, load_context:save_game.LoadContext, context:Any) -> None:
        context_data:tuple[uuid.UUID, Any] = context
        character_id, extra_context = context_data
        character = load_context.gamestate.entities[character_id]
        assert(isinstance(character, core.Character))
        agendum.initialize_agendum(character)
        self._post_load_agendum(agendum, load_context, extra_context)

class CaptainAgendumSaver(AgendumSaver[agenda.CaptainAgendum]):
    def _save_agendum(self, obj:agenda.CaptainAgendum, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.uuid_to_f(obj.craft.entity_id, f)
        bytes_written += s_util.bool_to_f(obj.enable_threat_response, f)
        if obj.threat_response:
            bytes_written += s_util.bool_to_f(True, f)
            bytes_written += s_util.uuid_to_f(obj.threat_response.order_id, f)
        else:
            bytes_written += s_util.bool_to_f(False, f)
        bytes_written += s_util.bool_to_f(obj._start_transponder, f)
        return bytes_written

    def _load_agendum(self, f:io.IOBase, load_context:save_game.LoadContext, agenda_id:uuid.UUID) -> tuple[agenda.CaptainAgendum, Any]:
        ship_id = s_util.uuid_from_f(f)
        enable_threat_response = s_util.bool_from_f(f)
        has_threat_response = s_util.bool_from_f(f)
        order_id:Optional[uuid.UUID] = None
        if has_threat_response:
            order_id = s_util.uuid_from_f(f)
        start_transponder = s_util.bool_from_f(f)

        captain_agendum = agenda.CaptainAgendum(load_context.gamestate, enable_threat_response=enable_threat_response, start_transponder=start_transponder, agenda_id=agenda_id, _check_flag=True)

        return captain_agendum, ship_id

    def _post_load_agendum(self, obj:agenda.CaptainAgendum, load_context:save_game.LoadContext, context:Any) -> None:
        ship_id:uuid.UUID = context
        craft = load_context.gamestate.entities[ship_id]
        assert(isinstance(craft, core.Ship))
        obj.craft = craft

        if obj.started_at >= 0 and obj.stopped_at < 0:
            obj.craft.observe(obj)

class MiningAgendumSaver(AgendumSaver[agenda.MiningAgendum]):
    def _save_agendum(self, obj:agenda.MiningAgendum, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.uuid_to_f(obj.craft.entity_id, f)
        bytes_written += s_util.uuid_to_f(obj.agent.entity_id, f)
        bytes_written += s_util.ints_to_f(obj.allowed_resources, f)

        if obj.allowed_stations:
            bytes_written += s_util.bool_to_f(True, f)
            bytes_written += s_util.uuids_to_f(list(x.entity_id for x in obj.allowed_stations), f)
        else:
            bytes_written += s_util.bool_to_f(False, f)

        bytes_written += s_util.int_to_f(obj.state, f, blen=1)

        if obj.mining_order:
            bytes_written += s_util.bool_to_f(True, f)
            bytes_written += s_util.uuid_to_f(obj.mining_order.order_id, f)
        else:
            bytes_written += s_util.bool_to_f(False, f)

        if obj.transfer_order:
            bytes_written += s_util.bool_to_f(True, f)
            bytes_written += s_util.uuid_to_f(obj.transfer_order.order_id, f)
        else:
            bytes_written += s_util.bool_to_f(False, f)

        bytes_written += s_util.int_to_f(obj.round_trips, f)
        bytes_written += s_util.int_to_f(obj.max_trips, f, signed=True)
        return bytes_written

    def _load_agendum(self, f:io.IOBase, load_context:save_game.LoadContext, agenda_id:uuid.UUID) -> tuple[agenda.MiningAgendum, Any]:
        ship_id = s_util.uuid_from_f(f)
        agent_id = s_util.uuid_from_f(f)
        allowed_resources = s_util.ints_from_f(f)
        has_allowed_stations = s_util.bool_from_f(f)
        allowed_stations:Optional[list[uuid.UUID]] = None
        if has_allowed_stations:
            allowed_stations = s_util.uuids_from_f(f)
        state = s_util.int_from_f(f, blen=1)
        has_mining_order = s_util.bool_from_f(f)
        mining_order_id:Optional[uuid.UUID] = None
        if has_mining_order:
            mining_order_id = s_util.uuid_from_f(f)
        has_transfer_order = s_util.bool_from_f(f)
        transfer_order_id:Optional[uuid.UUID] = None
        if has_transfer_order:
            transfer_order_id = s_util.uuid_from_f(f)
        round_trips = s_util.int_from_f(f)
        max_trips = s_util.int_from_f(f, signed=True)

        a = agenda.MiningAgendum(load_context.gamestate, allowed_resources=allowed_resources, agenda_id=agenda_id, _check_flag=True)
        a.state = agenda.MiningAgendum.State(state)
        a.round_trips = round_trips
        a.max_trips = max_trips

        return a, (ship_id, agent_id, allowed_stations, mining_order_id, transfer_order_id)

    def _post_load_agendum(self, obj:agenda.MiningAgendum, load_context:save_game.LoadContext, context:Any) -> None:
        context_data:tuple[uuid.UUID, uuid.UUID, Optional[list[uuid.UUID]], Optional[uuid.UUID], Optional[uuid.UUID]] = context
        ship_id, agent_id, allowed_station_ids, mining_order_id, transfer_order_id = context_data
        obj.craft = load_context.gamestate.get_entity(ship_id, core.Ship)

        if obj.started_at >= 0 and obj.stopped_at < 0:
            obj.craft.observe(obj)

        obj.agent = load_context.gamestate.get_entity(agent_id, econ.ShipTraderAgent)

        if allowed_station_ids:
            allowed_stations = list(load_context.gamestate.get_entity(x, core.SectorEntity) for x in allowed_station_ids)
            obj.allowed_stations = allowed_stations

        if mining_order_id:
            mining_order = load_context.gamestate.orders[mining_order_id]
            assert(isinstance(mining_order, ocore.MineOrder))
            obj.mining_order = mining_order

        if transfer_order_id:
            transfer_order = load_context.gamestate.orders[transfer_order_id]
            assert(isinstance(transfer_order, ocore.TradeCargoToStation))
            obj.transfer_order = transfer_order


class TradingAgendumSaver(AgendumSaver[agenda.TradingAgendum]):
    def _save_agendum(self, obj:agenda.TradingAgendum, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.uuid_to_f(obj.craft.entity_id, f)
        bytes_written += s_util.uuid_to_f(obj.agent.entity_id, f)
        bytes_written += s_util.ints_to_f(obj.allowed_goods, f)

        if obj.buy_from_stations:
            bytes_written += s_util.bool_to_f(True, f)
            bytes_written += s_util.uuids_to_f(list(x.entity_id for x in obj.buy_from_stations), f)
        else:
            bytes_written += s_util.bool_to_f(False, f)

        if obj.sell_to_stations:
            bytes_written += s_util.bool_to_f(True, f)
            bytes_written += s_util.uuids_to_f(list(x.entity_id for x in obj.sell_to_stations), f)
        else:
            bytes_written += s_util.bool_to_f(False, f)

        bytes_written += s_util.int_to_f(obj.state, f, blen=1)

        if obj.buy_order:
            bytes_written += s_util.bool_to_f(True, f)
            bytes_written += s_util.uuid_to_f(obj.buy_order.order_id, f)
        else:
            bytes_written += s_util.bool_to_f(False, f)

        if obj.sell_order:
            bytes_written += s_util.bool_to_f(True, f)
            bytes_written += s_util.uuid_to_f(obj.sell_order.order_id, f)
        else:
            bytes_written += s_util.bool_to_f(False, f)

        bytes_written += s_util.int_to_f(obj.trade_trips, f)
        bytes_written += s_util.int_to_f(obj.max_trips, f, signed=True)
        return bytes_written

    def _load_agendum(self, f:io.IOBase, load_context:save_game.LoadContext, agenda_id:uuid.UUID) -> tuple[agenda.TradingAgendum, Any]:
        ship_id = s_util.uuid_from_f(f)
        agent_id = s_util.uuid_from_f(f)
        allowed_goods = s_util.ints_from_f(f)
        has_buy_from_stations = s_util.bool_from_f(f)
        buy_from_stations:Optional[list[uuid.UUID]] = None
        if has_buy_from_stations:
            buy_from_stations = s_util.uuids_from_f(f)
        has_sell_to_stations = s_util.bool_from_f(f)
        sell_to_stations:Optional[list[uuid.UUID]] = None
        if has_sell_to_stations:
            sell_to_stations = s_util.uuids_from_f(f)
        state = s_util.int_from_f(f, blen=1)
        has_buy_order = s_util.bool_from_f(f)
        buy_order_id:Optional[uuid.UUID] = None
        if has_buy_order:
            buy_order_id = s_util.uuid_from_f(f)
        has_sell_order = s_util.bool_from_f(f)
        sell_order_id:Optional[uuid.UUID] = None
        if has_sell_order:
            sell_order_id = s_util.uuid_from_f(f)
        trade_trips = s_util.int_from_f(f)
        max_trips = s_util.int_from_f(f, signed=True)

        a = agenda.TradingAgendum(load_context.gamestate, allowed_goods=allowed_goods, agenda_id=agenda_id, _check_flag=True)
        a.state = agenda.TradingAgendum.State(state)
        a.trade_trips = trade_trips
        a.max_trips = max_trips

        return a, (ship_id, agent_id, buy_from_stations, sell_to_stations, buy_order_id, sell_order_id)

    def _post_load_agendum(self, obj:agenda.TradingAgendum, load_context:save_game.LoadContext, context:Any) -> None:
        context_data:tuple[uuid.UUID, uuid.UUID, Optional[list[uuid.UUID]], Optional[list[uuid.UUID]], Optional[uuid.UUID], Optional[uuid.UUID]] = context
        ship_id, agent_id, buy_from_station_ids, sell_to_station_ids, buy_order_id, sell_order_id = context_data
        obj.craft = load_context.gamestate.get_entity(ship_id, core.Ship)

        if obj.started_at >= 0 and obj.stopped_at < 0:
            obj.craft.observe(obj)

        obj.agent = load_context.gamestate.get_entity(agent_id, econ.ShipTraderAgent)

        if buy_from_station_ids:
            buy_from_stations = list(load_context.gamestate.get_entity(x, core.SectorEntity) for x in buy_from_station_ids)
            obj.buy_from_stations = buy_from_stations

        if sell_to_station_ids:
            sell_to_stations = list(load_context.gamestate.get_entity(x, core.SectorEntity) for x in sell_to_station_ids)
            obj.sell_to_stations = sell_to_stations

        if buy_order_id:
            buy_order = load_context.gamestate.orders[buy_order_id]
            assert(isinstance(buy_order, ocore.TradeCargoFromStation))
            obj.buy_order = buy_order

        if sell_order_id:
            sell_order = load_context.gamestate.orders[sell_order_id]
            assert(isinstance(sell_order, ocore.TradeCargoToStation))
            obj.sell_order = sell_order

class StationManagerSaver(AgendumSaver[agenda.StationManager]):
    def _save_agendum(self, obj:agenda.StationManager, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.uuid_to_f(obj.craft.entity_id, f)
        bytes_written += s_util.uuid_to_f(obj.agent.entity_id, f)
        bytes_written += s_util.int_to_f(obj.produced_batches, f)
        return bytes_written
    def _load_agendum(self, f:io.IOBase, load_context:save_game.LoadContext, agenda_id:uuid.UUID) -> tuple[agenda.StationManager, Any]:
        craft_id = s_util.uuid_from_f(f)
        agent_id = s_util.uuid_from_f(f)
        produced_batches = s_util.int_from_f(f)

        station_manager = agenda.StationManager(load_context.gamestate, agenda_id=agenda_id, _check_flag=True)
        station_manager.produced_batches = produced_batches
        return station_manager, (craft_id, agent_id)

    def _post_load_agendum(self, obj:agenda.StationManager, load_context:save_game.LoadContext, context:Any) -> None:
        context_data:tuple[uuid.UUID, uuid.UUID] = context
        ship_id, agent_id = context_data
        obj.craft = load_context.gamestate.get_entity(ship_id, sector_entity.Station)

        if obj.started_at >= 0 and obj.stopped_at < 0:
            obj.craft.observe(obj)

        obj.agent = load_context.gamestate.get_entity(agent_id, econ.StationAgent)

class PlanetManagerSaver(AgendumSaver[agenda.PlanetManager]):
    def _save_agendum(self, obj:agenda.PlanetManager, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.uuid_to_f(obj.craft.entity_id, f)
        bytes_written += s_util.uuid_to_f(obj.agent.entity_id, f)
        return bytes_written
    def _load_agendum(self, f:io.IOBase, load_context:save_game.LoadContext, agenda_id:uuid.UUID) -> tuple[agenda.PlanetManager, Any]:
        craft_id = s_util.uuid_from_f(f)
        agent_id = s_util.uuid_from_f(f)

        planet_manager = agenda.PlanetManager(load_context.gamestate, agenda_id=agenda_id, _check_flag=True)
        return planet_manager, (craft_id, agent_id)

    def _post_load_agendum(self, obj:agenda.PlanetManager, load_context:save_game.LoadContext, context:Any) -> None:
        context_data:tuple[uuid.UUID, uuid.UUID] = context
        ship_id, agent_id = context_data
        obj.craft = load_context.gamestate.get_entity(ship_id, sector_entity.Planet)

        if obj.started_at >= 0 and obj.stopped_at < 0:
            obj.craft.observe(obj)

        obj.agent = load_context.gamestate.get_entity(agent_id, econ.StationAgent)
