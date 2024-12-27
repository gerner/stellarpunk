import io
import abc
import uuid
from typing import Any, Optional

from stellarpunk import core, effects

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
        bytes_written += s_util.uuid_to_f(effect.effect_id, f)
        bytes_written += s_util.uuid_to_f(effect.sector.entity_id, f)
        bytes_written += s_util.float_to_f(effect.started_at, f)
        bytes_written += s_util.float_to_f(effect.completed_at, f)

        bytes_written += self._save_effect(effect, f)

        return bytes_written

    def load(self, f:io.IOBase, load_context:save_game.LoadContext) -> Effect:
        effect_id = s_util.uuid_from_f(f)
        sector_id = s_util.uuid_from_f(f)
        started_at = s_util.float_from_f(f)
        completed_at = s_util.float_from_f(f)

        effect, extra_context = self._load_effect(f, load_context, effect_id)
        load_context.gamestate.register_effect(effect)
        load_context.register_post_load(effect, (sector_id, extra_context))
        return effect

    def post_load(self, effect:Effect, load_context:save_game.LoadContext, context:Any) -> None:
        context_data:tuple[uuid.UUID, Any]
        sector_id, extra_context = context
        effect.sector = load_context.gamestate.get_entity(sector_id, core.Sector)
        self._post_load_effect(effect, load_context, extra_context)

#class TransferCargoEffectSaver[T:effects.TransferCargoEffect](EffectSaver[T]):
#    def _save_transfer_cargo_effect(self, effect:T, f:io.IOBase) -> int:
#        return 0
#    def _load_transfer_cargo_effect(self, f:io.IOBase, load_context:save_game.LoadContext, resource:int, amount:int, effect_id:uuid.UUID) -> tuple[T, Any]:
#        return effects.TransferCargoEffect()

