import io
import abc
import uuid
import pydoc
import types
import collections
from typing import Any, Optional, Type

from stellarpunk import core, effects, util, econ

from . import save_game, util as s_util, gamestate as s_gamestate

class EffectSaver[Effect: core.Effect](save_game.Saver[Effect], abc.ABC):
    @abc.abstractmethod
    def _save_effect(self, effect:Effect, f:io.IOBase) -> int: ...
    @abc.abstractmethod
    def _load_effect(self, f:io.IOBase, load_context:save_game.LoadContext, effect_id:uuid.UUID) -> tuple[Effect, Any]: ...
    def _post_load_effect(self, effect:Effect, load_context:save_game.LoadContext, extra_context:Any) -> None:
        pass

    def save(self, effect:Effect, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += self.save_game.debug_string_w("basic fields", f)
        bytes_written += s_util.uuid_to_f(effect.effect_id, f)
        bytes_written += s_util.uuid_to_f(effect.sector.entity_id, f)
        bytes_written += s_util.float_to_f(effect.started_at, f)
        bytes_written += s_util.float_to_f(effect.completed_at, f)

        bytes_written += self.save_game.debug_string_w("type specific", f)
        bytes_written += self._save_effect(effect, f)

        return bytes_written

    def load(self, f:io.IOBase, load_context:save_game.LoadContext) -> Effect:
        load_context.debug_string_r("basic fields", f)
        effect_id = s_util.uuid_from_f(f)
        sector_id = s_util.uuid_from_f(f)
        started_at = s_util.float_from_f(f)
        completed_at = s_util.float_from_f(f)

        load_context.debug_string_r("type specific", f)
        effect, extra_context = self._load_effect(f, load_context, effect_id)
        assert effect.effect_id == effect_id
        effect.started_at = started_at
        effect.completed_at = completed_at

        load_context.gamestate.register_effect(effect)
        load_context.register_post_load(effect, (sector_id, extra_context))

        return effect

    def fetch(self, klass:Type[Effect], effect_id:uuid.UUID, load_context:save_game.LoadContext) -> Effect:
        return load_context.gamestate.get_effect(effect_id, klass)

    def post_load(self, effect:Effect, load_context:save_game.LoadContext, context:Any) -> None:
        context_data:tuple[uuid.UUID, Any]
        sector_id, extra_context = context
        effect.sector = load_context.gamestate.get_entity(sector_id, core.Sector)
        self._post_load_effect(effect, load_context, extra_context)

class TransferCargoEffectSaver[T:effects.TransferCargoEffect](EffectSaver[T]):
    def _save_transfer_cargo_effect(self, effect:T, f:io.IOBase) -> int:
        return 0
    def _load_transfer_cargo_effect(self, f:io.IOBase, load_context:save_game.LoadContext, resource:int, amount:float, transfer_rate:float, max_distance:float, effect_id:uuid.UUID) -> tuple[T, Any]:
        effect = effects.TransferCargoEffect(resource, amount, load_context.gamestate, transfer_rate=transfer_rate, max_distance=max_distance, _check_flag=True, effect_id=effect_id)
        return effect, None # type: ignore
    def _post_load_transfer_cargo_effect(self, effect:T, load_context:save_game.LoadContext, context:Any) -> None:
        pass

    def _save_effect(self, effect:T, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.uuid_to_f(effect.source.entity_id, f)
        bytes_written += s_util.uuid_to_f(effect.destination.entity_id, f)
        bytes_written += s_util.int_to_f(int(effect.resource), f)
        bytes_written += s_util.float_to_f(effect.amount, f)
        bytes_written += s_util.float_to_f(effect.sofar, f)
        bytes_written += s_util.float_to_f(effect.transfer_rate, f)
        bytes_written += s_util.float_to_f(effect.max_distance, f)
        bytes_written += s_util.bool_to_f(effect._completed_transfer, f)
        bytes_written += self._save_transfer_cargo_effect(effect, f)
        return bytes_written
    def _load_effect(self, f:io.IOBase, load_context:save_game.LoadContext, effect_id:uuid.UUID) -> tuple[T, Any]:
        source_id = s_util.uuid_from_f(f)
        destination_id = s_util.uuid_from_f(f)
        resource = s_util.int_from_f(f)
        amount = s_util.float_from_f(f)
        sofar = s_util.float_from_f(f)
        transfer_rate = s_util.float_from_f(f)
        max_distance = s_util.float_from_f(f)
        completed_transfer = s_util.bool_from_f(f)

        effect, extra_context = self._load_transfer_cargo_effect(f, load_context, resource, amount, transfer_rate, max_distance, effect_id)
        effect.sofar = sofar
        effect._completed_transfer = completed_transfer
        return effect, (source_id, destination_id, extra_context)

    def _post_load_effect(self, effect:T, load_context:save_game.LoadContext, context:Any) -> None:
        context_data:tuple[uuid.UUID, uuid.UUID, Any] = context
        source_id, destination_id, extra_context = context_data

        effect.source = load_context.gamestate.get_entity(source_id, core.SectorEntity)
        effect.destination = load_context.gamestate.get_entity(destination_id, core.SectorEntity)

        self._post_load_transfer_cargo_effect(effect, load_context, extra_context)

class TradeTransferEffectSaver(TransferCargoEffectSaver[effects.TradeTransferEffect]):
    def _save_transfer_cargo_effect(self, effect:effects.TradeTransferEffect, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.uuid_to_f(effect.buyer.entity_id, f)
        bytes_written += s_util.uuid_to_f(effect.seller.entity_id, f)
        bytes_written += s_util.float_to_f(effect.floor_price, f)
        bytes_written += s_util.float_to_f(effect.ceiling_price, f)
        bytes_written += s_util.to_len_pre_f(util.fullname(effect.current_price), f)
        return bytes_written

    def _load_transfer_cargo_effect(self, f:io.IOBase, load_context:save_game.LoadContext, resource:int, amount:float, transfer_rate:float, max_distance:float, effect_id:uuid.UUID) -> tuple[effects.TradeTransferEffect, Any]:
        buyer_id = s_util.uuid_from_f(f)
        seller_id = s_util.uuid_from_f(f)
        floor_price = s_util.float_from_f(f)
        ceiling_price = s_util.float_from_f(f)
        current_price = s_util.from_len_pre_f(f)
        current_price_fn:econ.PriceFn = pydoc.locate(current_price) # type: ignore
        assert(isinstance(current_price_fn, types.FunctionType))
        effect = effects.TradeTransferEffect(current_price_fn, resource, amount, load_context.gamestate, floor_price=floor_price, ceiling_price=ceiling_price, transfer_rate=transfer_rate, max_distance=max_distance, _check_flag=True, effect_id=effect_id)
        return effect, (buyer_id, seller_id)

    def _post_load_transfer_cargo_effect(self, effect:effects.TradeTransferEffect, load_context:save_game.LoadContext, context:Any) -> None:
        context_data:tuple[uuid.UUID, uuid.UUID] = context
        buyer_id, seller_id = context_data
        effect.buyer = load_context.gamestate.get_entity(buyer_id, core.EconAgent) # type: ignore
        effect.seller = load_context.gamestate.get_entity(seller_id, core.EconAgent) # type: ignore

class MiningEffectSaver(TransferCargoEffectSaver[effects.MiningEffect]):
    def _save_transfer_cargo_effect(self, effect:effects.MiningEffect, f:io.IOBase) -> int:
        return 0
    def _load_transfer_cargo_effect(self, f:io.IOBase, load_context:save_game.LoadContext, resource:int, amount:float, transfer_rate:float, max_distance:float, effect_id:uuid.UUID) -> tuple[effects.MiningEffect, Any]:
        effect = effects.MiningEffect(resource, amount, load_context.gamestate, transfer_rate=transfer_rate, max_distance=max_distance, _check_flag=True, effect_id=effect_id)
        return effect, None # type: ignore

class WarpOutEffectSaver(EffectSaver[effects.WarpOutEffect]):
    def _save_effect(self, effect:effects.WarpOutEffect, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.float_pair_to_f(effect.loc, f)
        bytes_written += s_util.float_to_f(effect.radius, f)
        bytes_written += s_util.float_to_f(effect.ttl, f)
        bytes_written += s_util.float_to_f(effect.expiration_time, f)
        return bytes_written

    def _load_effect(self, f:io.IOBase, load_context:save_game.LoadContext, effect_id:uuid.UUID) -> tuple[effects.WarpOutEffect, Any]:
        loc = s_util.float_pair_from_f(f)
        radius = s_util.float_from_f(f)
        ttl = s_util.float_from_f(f)
        expiration_time = s_util.float_from_f(f)

        effect = effects.WarpOutEffect(loc, load_context.gamestate, radius=radius, ttl=ttl, _check_flag=True, effect_id=effect_id)
        effect.expiration_time = expiration_time

        return effect, None

class WarpInEffectSaver(EffectSaver[effects.WarpInEffect]):
    def _save_effect(self, effect:effects.WarpInEffect, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.float_pair_to_f(effect.loc, f)
        bytes_written += s_util.float_to_f(effect.radius, f)
        bytes_written += s_util.float_to_f(effect.ttl, f)
        bytes_written += s_util.float_to_f(effect.expiration_time, f)
        return bytes_written

    def _load_effect(self, f:io.IOBase, load_context:save_game.LoadContext, effect_id:uuid.UUID) -> tuple[effects.WarpInEffect, Any]:
        loc = s_util.float_pair_from_f(f)
        radius = s_util.float_from_f(f)
        ttl = s_util.float_from_f(f)
        expiration_time = s_util.float_from_f(f)

        effect = effects.WarpInEffect(loc, load_context.gamestate, radius=radius, ttl=ttl, _check_flag=True, effect_id=effect_id)
        effect.expiration_time = expiration_time

        return effect, None

