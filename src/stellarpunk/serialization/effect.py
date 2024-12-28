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

        bytes_written += s_util.debug_string_w("type specific", f)
        bytes_written += self._save_effect(effect, f)

        bytes_written += self.save_observers(effect, f)

        return bytes_written

    def load(self, f:io.IOBase, load_context:save_game.LoadContext) -> Effect:
        s_util.debug_string_r("basic fields", f)
        effect_id = s_util.uuid_from_f(f)
        sector_id = s_util.uuid_from_f(f)
        started_at = s_util.float_from_f(f)
        completed_at = s_util.float_from_f(f)

        s_util.debug_string_r("type specific", f)
        effect, extra_context = self._load_effect(f, load_context, effect_id)
        effect.started_at = started_at
        effect.completed_at = completed_at

        load_context.gamestate.register_effect(effect)
        load_context.register_post_load(effect, (sector_id, extra_context))

        self.load_observers(effect, f, load_context)

        return effect

    def post_load(self, effect:Effect, load_context:save_game.LoadContext, context:Any) -> None:
        context_data:tuple[uuid.UUID, Any]
        sector_id, extra_context = context
        effect.sector = load_context.gamestate.get_entity(sector_id, core.Sector)
        self._post_load_effect(effect, load_context, extra_context)

class TransferCargoEffectSaver[T:effects.TransferCargoEffect](EffectSaver[T]):
    def _save_transfer_cargo_effect(self, effect:T, f:io.IOBase) -> int:
        return 0
    def _load_transfer_cargo_effect(self, f:io.IOBase, load_context:save_game.LoadContext, resource:int, amount:float, transfer_rate:float, max_distance:float, effect_id:uuid.UUID) -> tuple[T, Any]:
        #TODO: is this correct?
        return effects.TransferCargoEffect(resource, amount, load_context.gamestate, transfer_rate=transfer_rate, max_distance=max_distance, _check_flag=True, effect_id=effect_id), None # type: ignore
    def _post_load_transfer_cargo_effect(self, effect:T, load_context:save_game.LoadContext, context:Any) -> None:
        pass

    def _save_effect(self, effect:T, f:io.IOBase) -> int:
        bytes_written = 0
        #TODO: lots of stuff
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

