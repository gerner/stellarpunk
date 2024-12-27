import io
import abc
import uuid
import collections
from typing import Any, Optional

from stellarpunk import core, effects, util

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
        bytes_written += s_util.debug_string_w("basic fields", f)
        bytes_written += s_util.uuid_to_f(effect.effect_id, f)
        bytes_written += s_util.uuid_to_f(effect.sector.entity_id, f)
        bytes_written += s_util.float_to_f(effect.started_at, f)
        bytes_written += s_util.float_to_f(effect.completed_at, f)

        if self.save_game.debug:
            bytes_written += s_util.debug_string_w("observers", f)
            bytes_written += s_util.str_uuids_to_f(list((util.fullname(x), x.observer_id) for x in effect.observers), f)

        bytes_written += s_util.debug_string_w("type specific", f)
        bytes_written += self._save_effect(effect, f)

        return bytes_written

    def load(self, f:io.IOBase, load_context:save_game.LoadContext) -> Effect:
        s_util.debug_string_r("basic fields", f)
        effect_id = s_util.uuid_from_f(f)
        sector_id = s_util.uuid_from_f(f)
        started_at = s_util.float_from_f(f)
        completed_at = s_util.float_from_f(f)

        observer_ids:list[tuple[str, uuid.UUID]] = []
        if load_context.debug:
            s_util.debug_string_r("observers", f)
            observer_ids = s_util.str_uuids_from_f(f)

        s_util.debug_string_r("type specific", f)
        effect, extra_context = self._load_effect(f, load_context, effect_id)
        load_context.gamestate.register_effect(effect)
        load_context.register_post_load(effect, (sector_id, extra_context))

        if load_context.debug:
            load_context.register_sanity_check(effect, observer_ids)

        return effect

    def post_load(self, effect:Effect, load_context:save_game.LoadContext, context:Any) -> None:
        context_data:tuple[uuid.UUID, Any]
        sector_id, extra_context = context
        effect.sector = load_context.gamestate.get_entity(sector_id, core.Sector)
        self._post_load_effect(effect, load_context, extra_context)

    def sanity_check(self, effect:Effect, load_context:save_game.LoadContext, context:Any) -> None:
        observer_ids:list[tuple[str, uuid.UUID]] = context

        # make sure all the observers we had when saving are back
        saved_observer_counts:collections.Counter[tuple[str, uuid.UUID]] = collections.Counter()
        for observer_id in observer_ids:
            saved_observer_counts[observer_id] += 1
        loaded_observer_counts:collections.Counter[tuple[str, uuid.UUID]] = collections.Counter()
        for observer in effect.observers:
            loaded_observer_counts[(util.fullname(observer), observer.observer_id)] += 1
        saved_observer_counts.subtract(loaded_observer_counts)
        non_zero_observers = {observer_id: count for observer_id, count in saved_observer_counts.items() if count != 0}
        assert(non_zero_observers == {})

class TransferCargoEffectSaver[T:effects.TransferCargoEffect](EffectSaver[T]):
    def _save_transfer_cargo_effect(self, effect:T, f:io.IOBase) -> int:
        return 0
    def _load_transfer_cargo_effect(self, f:io.IOBase, load_context:save_game.LoadContext, resource:int, amount:float, transfer_rate:float, max_distance:float, effect_id:uuid.UUID) -> tuple[T, Any]:
        return effects.TransferCargoEffect(resource, amount, load_context.gamestate, transfer_rate=transfer_rate, max_distance=max_distance), None # type: ignore
    def _post_load_transfer_cargo_effect(self, effect:T, load_context:save_game.LoadContext, context:Any) -> None:
        pass

    def _save_effect(self, effect:T, f:io.IOBase) -> int:
        bytes_written = 0
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
        return effect, (source_id, destination_id, extra_context)
    def _post_load_effect(self, effect:T, load_context:save_game.LoadContext, extra_context:Any) -> None:
        pass

