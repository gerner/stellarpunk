import io
import abc
import uuid
import collections
from collections.abc import Collection
from typing import Any, Optional

from stellarpunk import core, agenda, econ
from stellarpunk.agenda import intel as aintel
from stellarpunk.core import sector_entity, combat
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
        bytes_written += s_util.bool_to_f(obj.paused, f)
        bytes_written += s_util.bool_to_f(obj._is_primary, f)
        bytes_written += s_util.debug_string_w("type specific", f)
        bytes_written += self._save_agendum(obj, f)
        return bytes_written

    def load(self, f:io.IOBase, load_context:save_game.LoadContext) -> T:
        s_util.debug_string_r("basic fields", f)
        agenda_id = s_util.uuid_from_f(f)
        character_id = s_util.uuid_from_f(f)
        started_at = s_util.float_from_f(f)
        stopped_at = s_util.float_from_f(f)
        paused = s_util.bool_from_f(f)
        is_primary = s_util.bool_from_f(f)
        s_util.debug_string_r("type specific", f)
        (agendum, extra_context) = self._load_agendum(f, load_context, agenda_id)
        agendum.started_at = started_at
        agendum.stopped_at = stopped_at
        agendum.paused = paused
        agendum._is_primary = is_primary
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
        bytes_written += s_util.optional_uuid_to_f(obj._preempted_primary.agenda_id if obj._preempted_primary else None, f)
        return bytes_written

    def _load_agendum(self, f:io.IOBase, load_context:save_game.LoadContext, agenda_id:uuid.UUID) -> tuple[agenda.CaptainAgendum, Any]:
        ship_id = s_util.uuid_from_f(f)
        enable_threat_response = s_util.bool_from_f(f)
        has_threat_response = s_util.bool_from_f(f)
        threat_response_id:Optional[uuid.UUID] = None
        if has_threat_response:
            threat_response_id = s_util.uuid_from_f(f)
        start_transponder = s_util.bool_from_f(f)
        preempted_primary_id = s_util.optional_uuid_from_f(f)

        captain_agendum = agenda.CaptainAgendum(load_context.gamestate, enable_threat_response=enable_threat_response, start_transponder=start_transponder, agenda_id=agenda_id, _check_flag=True)

        return captain_agendum, (ship_id, threat_response_id, preempted_primary_id)

    def _post_load_agendum(self, obj:agenda.CaptainAgendum, load_context:save_game.LoadContext, context:Any) -> None:
        context_data:tuple[uuid.UUID, Optional[uuid.UUID], Optional[uuid.UUID]] = context
        ship_id, threat_response_id, preempted_primary_id = context_data
        craft = load_context.gamestate.entities[ship_id]
        assert(isinstance(craft, core.Ship))
        obj.craft = craft

        if threat_response_id:
            obj.threat_response = load_context.gamestate.get_order(threat_response_id, combat.FleeOrder)

        if preempted_primary_id:
            obj._preempted_primary = load_context.gamestate.get_agendum(preempted_primary_id, core.AbstractAgendum)

class MiningAgendumSaver(AgendumSaver[agenda.MiningAgendum]):
    def _save_agendum(self, obj:agenda.MiningAgendum, f:io.IOBase) -> int:
        bytes_written = 0

        bytes_written += s_util.debug_string_w("basic fields", f)
        bytes_written += s_util.uuid_to_f(obj.craft.entity_id, f)
        bytes_written += s_util.uuid_to_f(obj.agent.entity_id, f)
        bytes_written += s_util.ints_to_f(obj.allowed_resources, f)
        bytes_written += s_util.int_to_f(obj.state, f, blen=1)
        bytes_written += s_util.int_to_f(obj.round_trips, f)
        bytes_written += s_util.int_to_f(obj.max_trips, f, signed=True)
        bytes_written += s_util.optional_obj_to_f(obj.allowed_stations, f, s_util.uuids_to_f)

        bytes_written += s_util.debug_string_w("orders", f)
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

        bytes_written += s_util.debug_string_w("pending interest", f)
        if obj._pending_intel_interest:
            bytes_written += s_util.bool_to_f(True, f)
            bytes_written += self.save_game.save_object(obj._pending_intel_interest, f, core.IntelMatchCriteria)
        else:
            bytes_written += s_util.bool_to_f(False, f)

        return bytes_written

    def _load_agendum(self, f:io.IOBase, load_context:save_game.LoadContext, agenda_id:uuid.UUID) -> tuple[agenda.MiningAgendum, Any]:
        s_util.debug_string_r("basic fields", f)
        ship_id = s_util.uuid_from_f(f)
        agent_id = s_util.uuid_from_f(f)
        allowed_resources = s_util.ints_from_f(f)
        state = s_util.int_from_f(f, blen=1)
        round_trips = s_util.int_from_f(f)
        max_trips = s_util.int_from_f(f, signed=True)
        allowed_stations = s_util.optional_obj_from_f(f, s_util.uuids_from_f)

        s_util.debug_string_r("orders", f)
        has_mining_order = s_util.bool_from_f(f)
        mining_order_id:Optional[uuid.UUID] = None
        if has_mining_order:
            mining_order_id = s_util.uuid_from_f(f)
        has_transfer_order = s_util.bool_from_f(f)
        transfer_order_id:Optional[uuid.UUID] = None
        if has_transfer_order:
            transfer_order_id = s_util.uuid_from_f(f)

        s_util.debug_string_r("pending interest", f)
        has_pending_interest = s_util.bool_from_f(f)
        pending_interest:Optional[core.IntelMatchCriteria] = None
        if has_pending_interest:
            pending_interest = self.save_game.load_object(core.IntelMatchCriteria, f, load_context)

        a = agenda.MiningAgendum(load_context.gamestate, allowed_resources=allowed_resources, agenda_id=agenda_id, allowed_stations=allowed_stations, _check_flag=True)
        a.state = agenda.MiningAgendum.State(state)
        a.round_trips = round_trips
        a.max_trips = max_trips
        a._pending_intel_interest = pending_interest


        return a, (ship_id, agent_id, mining_order_id, transfer_order_id)

    def _post_load_agendum(self, obj:agenda.MiningAgendum, load_context:save_game.LoadContext, context:Any) -> None:
        context_data:tuple[uuid.UUID, uuid.UUID, Optional[uuid.UUID], Optional[uuid.UUID]] = context
        ship_id, agent_id, mining_order_id, transfer_order_id = context_data
        obj.craft = load_context.gamestate.get_entity(ship_id, core.Ship)

        obj.agent = load_context.gamestate.get_entity(agent_id, econ.ShipTraderAgent)

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

        bytes_written += s_util.debug_string_w("basic fields", f)
        bytes_written += s_util.uuid_to_f(obj.craft.entity_id, f)
        bytes_written += s_util.uuid_to_f(obj.agent.entity_id, f)
        bytes_written += s_util.ints_to_f(obj.allowed_goods, f)
        bytes_written += s_util.int_to_f(obj.state, f, blen=1)
        bytes_written += s_util.ints_to_f(obj._known_sold_resources, f)
        bytes_written += s_util.ints_to_f(obj._known_bought_resources, f)
        bytes_written += s_util.int_to_f(obj.trade_trips, f)
        bytes_written += s_util.int_to_f(obj.max_trips, f, signed=True)
        bytes_written += s_util.optional_obj_to_f(obj.buy_from_stations, f, s_util.uuids_to_f)
        bytes_written += s_util.optional_obj_to_f(obj.sell_to_stations, f, s_util.uuids_to_f)

        bytes_written += s_util.debug_string_w("orders", f)
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

        bytes_written += s_util.debug_string_w("pending interests", f)
        bytes_written += s_util.size_to_f(len(obj._pending_intel_interests), f)
        for interest in obj._pending_intel_interests:
            bytes_written += self.save_game.save_object(interest, f, core.IntelMatchCriteria)
        return bytes_written

    def _load_agendum(self, f:io.IOBase, load_context:save_game.LoadContext, agenda_id:uuid.UUID) -> tuple[agenda.TradingAgendum, Any]:
        s_util.debug_string_r("basic fields", f)
        ship_id = s_util.uuid_from_f(f)
        agent_id = s_util.uuid_from_f(f)
        allowed_goods = s_util.ints_from_f(f)
        state = s_util.int_from_f(f, blen=1)
        known_sold_resources = frozenset(s_util.ints_from_f(f))
        known_bought_resources = frozenset(s_util.ints_from_f(f))
        trade_trips = s_util.int_from_f(f)
        max_trips = s_util.int_from_f(f, signed=True)
        buy_from_stations = s_util.optional_obj_from_f(f, s_util.uuids_from_f)
        sell_to_stations = s_util.optional_obj_from_f(f, s_util.uuids_from_f)

        s_util.debug_string_r("orders", f)
        has_buy_order = s_util.bool_from_f(f)
        buy_order_id:Optional[uuid.UUID] = None
        if has_buy_order:
            buy_order_id = s_util.uuid_from_f(f)
        has_sell_order = s_util.bool_from_f(f)
        sell_order_id:Optional[uuid.UUID] = None
        if has_sell_order:
            sell_order_id = s_util.uuid_from_f(f)

        s_util.debug_string_r("pending interests", f)
        count = s_util.size_from_f(f)
        pending_interests:set[core.IntelMatchCriteria] = set()
        for i in range(count):
            pending_interests.add(self.save_game.load_object(core.IntelMatchCriteria, f, load_context))

        a = agenda.TradingAgendum(load_context.gamestate, allowed_goods=allowed_goods, agenda_id=agenda_id, buy_from_stations=buy_from_stations, sell_to_stations=sell_to_stations, _check_flag=True)
        a.state = agenda.TradingAgendum.State(state)

        a._pending_intel_interests = pending_interests
        a._known_sold_resources = known_sold_resources
        a._known_bought_resources = known_bought_resources

        a.trade_trips = trade_trips
        a.max_trips = max_trips

        return a, (ship_id, agent_id, buy_order_id, sell_order_id)

    def _post_load_agendum(self, obj:agenda.TradingAgendum, load_context:save_game.LoadContext, context:Any) -> None:
        context_data:tuple[uuid.UUID, uuid.UUID, Optional[uuid.UUID], Optional[uuid.UUID]] = context
        ship_id, agent_id, buy_order_id, sell_order_id = context_data
        obj.craft = load_context.gamestate.get_entity(ship_id, core.Ship)

        obj.agent = load_context.gamestate.get_entity(agent_id, econ.ShipTraderAgent)

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

        obj.agent = load_context.gamestate.get_entity(agent_id, econ.StationAgent)

class IntelCollectionAgendumSaver(AgendumSaver[aintel.IntelCollectionAgendum]):
    def _save_agendum(self, obj:aintel.IntelCollectionAgendum, f:io.IOBase) -> int:
        bytes_written = 0

        bytes_written += s_util.float_to_f(obj._idle_period, f)
        bytes_written += s_util.int_to_f(obj._state, f)

        bytes_written += s_util.int_to_f(obj._interests_advertised, f)
        bytes_written += s_util.int_to_f(obj._interests_satisfied, f)
        bytes_written += s_util.int_to_f(obj._immediate_interest_count, f)
        bytes_written += s_util.int_to_f(obj._immediate_interests_satisfied, f)

        def save_imc(x:core.IntelMatchCriteria, f:io.IOBase) -> int:
            return self.save_game.save_object(x, f, core.IntelMatchCriteria)

        def save_imcs(x:Collection[core.IntelMatchCriteria], f:io.IOBase) -> int:
            return s_util.objs_to_f(x, f, save_imc)

        def save_optional_imc(x:Optional[core.IntelMatchCriteria], f:io.IOBase) -> int:
            bytes_written = 0
            if x:
                bytes_written += s_util.bool_to_f(True, f)
                bytes_written += self.save_game.save_object(x, f, core.IntelMatchCriteria)
            else:
                bytes_written += s_util.bool_to_f(False, f)
            return bytes_written

        def save_optional_imcs(x:Collection[Optional[core.IntelMatchCriteria]], f:io.IOBase) -> int:
            return s_util.objs_to_f(x, f, save_optional_imc)

        bytes_written += s_util.objs_to_f(obj._interests, f, save_imc)

        bytes_written += save_optional_imc(obj._immediate_interest, f)

        bytes_written += s_util.fancy_dict_to_f(
                obj._source_interests_by_dependency,
                f,
                save_imc,
                save_optional_imcs
        )
        bytes_written += s_util.fancy_dict_to_f(
                obj._source_interests_by_source,
                f,
                save_optional_imc,
                save_imcs
        )

        bytes_written += s_util.optional_uuid_to_f(obj._preempted_primary.agenda_id if obj._preempted_primary else None, f)

        return bytes_written

    def _load_agendum(self, f:io.IOBase, load_context:save_game.LoadContext, agenda_id:uuid.UUID) -> tuple[aintel.IntelCollectionAgendum, Any]:
        idle_period = s_util.float_from_f(f)
        state = s_util.int_from_f(f)

        interests_advertised = s_util.int_from_f(f)
        interests_satisfied = s_util.int_from_f(f)
        immediate_interest_count = s_util.int_from_f(f)
        immediate_interests_satisfied = s_util.int_from_f(f)

        def load_imc(f:io.IOBase) -> core.IntelMatchCriteria:
            return self.save_game.load_object(core.IntelMatchCriteria, f, load_context)

        def load_imcs(f:io.IOBase) -> set[core.IntelMatchCriteria]:
            return set(s_util.objs_from_f(f, load_imc))

        def load_optional_imc(f:io.IOBase) -> Optional[core.IntelMatchCriteria]:
            has_imc = s_util.bool_from_f(f)
            if has_imc:
                return self.save_game.load_object(core.IntelMatchCriteria, f, load_context)
            else:
                return None

        def load_optional_imcs(f:io.IOBase) -> set[Optional[core.IntelMatchCriteria]]:
            return set(s_util.objs_from_f(f, load_optional_imc))

        interests = s_util.objs_from_f(f, load_imc)

        immediate_interest = load_optional_imc(f)

        source_interests_by_dependency = collections.defaultdict(set, s_util.fancy_dict_from_f(
            f,
            load_imc,
            load_optional_imcs,
        ))

        source_interests_by_source = collections.defaultdict(set, s_util.fancy_dict_from_f(
            f,
            load_optional_imc,
            load_imcs,
        ))

        preempted_primary_id = s_util.optional_uuid_from_f(f)

        intel_agendum = aintel.IntelCollectionAgendum(load_context.generator.intel_director, load_context.gamestate, idle_period=idle_period, _check_flag=True, agenda_id=agenda_id)
        intel_agendum._state = aintel.IntelCollectionAgendum.State(state)
        intel_agendum._interests_advertised = interests_advertised
        intel_agendum._interests_satisfied = interests_satisfied
        intel_agendum._immediate_interest_count = immediate_interest_count
        intel_agendum._immediate_interests_satisfied = immediate_interests_satisfied

        intel_agendum._interests = set(interests)
        intel_agendum._immediate_interest = immediate_interest

        intel_agendum._source_interests_by_dependency = source_interests_by_dependency
        intel_agendum._source_interests_by_source = source_interests_by_source

        return intel_agendum, preempted_primary_id

    def _post_load_agendum(self, obj:aintel.IntelCollectionAgendum, load_context:save_game.LoadContext, context:Any) -> None:
        preempted_primary_id:Optional[uuid.UUID] = context
        if preempted_primary_id:
            preempted_primary = load_context.gamestate.agenda[preempted_primary_id]
            obj._preempted_primary = preempted_primary
