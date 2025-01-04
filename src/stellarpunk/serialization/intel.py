import io
import uuid
import abc
import pydoc
from typing import Any

import numpy as np
import numpy.typing as npt

from . import save_game, util as s_util, gamestate as s_gamestate

from stellarpunk import core, intel, util

class IntelManagerSaver(save_game.Saver[intel.IntelManager]):
    def save(self, intel_manager:intel.IntelManager, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.uuids_to_f(intel_manager._intel, f)
        return bytes_written

    def load(self, f:io.IOBase, load_context:save_game.LoadContext) -> intel.IntelManager:
        intel_ids = s_util.uuids_from_f(f)
        intel_manager = intel.IntelManager(load_context.gamestate)
        load_context.register_post_load(intel_manager, intel_ids)
        return intel_manager

    def post_load(self, intel_manager:intel.IntelManager, load_context:save_game.LoadContext, context:Any) -> None:
        intel_ids:list[uuid.UUID] = context
        for intel_id in intel_ids:
            intel = load_context.gamestate.get_entity(intel_id, core.Intel)
            intel_manager._add_intel(intel)

class ExpireIntelTaskSaver(save_game.Saver[intel.ExpireIntelTask]):
    def save(self, obj:intel.ExpireIntelTask, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.uuid_to_f(obj.task_id, f)
        bytes_written += s_util.uuid_to_f(obj.intel.entity_id, f)
        return bytes_written
    def load(self, f:io.IOBase, load_context:save_game.LoadContext) -> intel.ExpireIntelTask:
        task_id = s_util.uuid_from_f(f)
        intel_id = s_util.uuid_from_f(f)
        task = intel.ExpireIntelTask(task_id=task_id)
        load_context.register_post_load(task, intel_id)
        return task
    def post_load(self, obj:intel.ExpireIntelTask, load_context:save_game.LoadContext, context:Any) -> None:
        intel_id:uuid.UUID = context
        intel = load_context.gamestate.get_entity(intel_id, core.Intel)
        obj.intel = intel

class IntelSaver[T: core.Intel](s_gamestate.EntitySaver[T]):
    @abc.abstractmethod
    def _save_intel(self, intel:T, f:io.IOBase) -> int: ...
    @abc.abstractmethod
    def _load_intel(self, f:io.IOBase, load_context:save_game.LoadContext, entity_id:uuid.UUID) -> T: ...

    def _save_entity(self, intel:T, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.float_to_f(intel.fresh_until, f)
        bytes_written += s_util.float_to_f(intel.expires_at, f)
        bytes_written += self._save_intel(intel, f)
        return bytes_written
    def _load_entity(self, f:io.IOBase, load_context:save_game.LoadContext, entity_id:uuid.UUID) -> T:
        fresh_until = s_util.float_from_f(f)
        expires_at = s_util.float_from_f(f)
        intel = self._load_intel(f, load_context, entity_id)
        intel.fresh_until = fresh_until
        intel.expires_at = expires_at
        return intel

class EntityIntelSaver[T: core.EntityIntel](IntelSaver[T]):
    @abc.abstractmethod
    def _save_entity_intel(self, intel:T, f:io.IOBase) -> int: ...
    @abc.abstractmethod
    def _load_entity_intel(self, f:io.IOBase, load_context:save_game.LoadContext, intel_entity_id:uuid.UUID, intel_entity_short_id:str, intel_entity_type:type, entity_id:uuid.UUID) -> T: ...

    def _save_intel(self, intel:T, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.to_len_pre_f(util.fullname(intel.intel_entity_type), f)
        bytes_written += s_util.uuid_to_f(intel.intel_entity_id, f)
        bytes_written += s_util.to_len_pre_f(intel.intel_entity_short_id, f)
        bytes_written += self._save_entity_intel(intel, f)
        return bytes_written

    def _load_intel(self, f:io.IOBase, load_context:save_game.LoadContext, entity_id:uuid.UUID) -> T:
        intel_entity_class_name = s_util.from_len_pre_f(f)
        intel_entity_type = pydoc.locate(intel_entity_class_name)
        assert(isinstance(intel_entity_type, type))
        intel_entity_id = s_util.uuid_from_f(f)
        intel_entity_short_id = s_util.from_len_pre_f(f)

        intel = self._load_entity_intel(f, load_context, intel_entity_id, intel_entity_short_id, intel_entity_type, entity_id)

        return intel

class EconAgentIntelSaver(EntityIntelSaver[intel.EconAgentIntel]):
    def _save_entity_intel(self, intel:intel.EconAgentIntel, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.uuid_to_f(intel.sector_entity_id, f)
        bytes_written += s_util.fancy_dict_to_f(intel.sell_offers, f, s_util.int_to_f, s_util.float_pair_to_f)
        bytes_written += s_util.fancy_dict_to_f(intel.buy_offers, f, s_util.int_to_f, s_util.float_pair_to_f)
        return bytes_written

    def _load_entity_intel(self, f:io.IOBase, load_context:save_game.LoadContext, intel_entity_id:uuid.UUID, intel_entity_short_id:str, intel_entity_type:type, entity_id:uuid.UUID) -> intel.EconAgentIntel:
        sector_entity_id = s_util.uuid_from_f(f)
        sell_offers = s_util.fancy_dict_from_f(f, s_util.int_from_f, lambda x: tuple(s_util.float_pair_from_f(x)))
        buy_offers = s_util.fancy_dict_from_f(f, s_util.int_from_f, lambda x: tuple(s_util.float_pair_from_f(x)))

        agent_intel = intel.EconAgentIntel(intel_entity_id, intel_entity_short_id, intel_entity_type, load_context.gamestate, entity_id=entity_id)
        agent_intel.sector_entity_id = sector_entity_id
        agent_intel.sell_offers = sell_offers
        agent_intel.buy_offers = sell_offers

        return agent_intel

class SectorEntityIntelSaver[SectorEntityIntel:intel.SectorEntityIntel](EntityIntelSaver[SectorEntityIntel]):
    def _save_sector_entity_intel(self, sector_entity_intel:SectorEntityIntel, f:io.IOBase) -> int:
        return 0
    def _load_sector_entity_intel(self, f:io.IOBase, load_context:save_game.LoadContext, sector_id:uuid.UUID, loc:npt.NDArray[np.float64], intel_entity_id:uuid.UUID, intel_entity_id_short:str, intel_entity_type:type, entity_id:uuid.UUID) -> SectorEntityIntel:
        return intel.SectorEntityIntel(sector_id, loc, intel_entity_id, intel_entity_id_short, intel_entity_type, load_context.gamestate, entity_id=entity_id) # type: ignore

    def _save_entity_intel(self, sector_entity_intel:SectorEntityIntel, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.uuid_to_f(sector_entity_intel.sector_id, f)
        bytes_written += s_util.float_pair_to_f(sector_entity_intel.loc, f)
        return bytes_written

    def _load_entity_intel(self, f:io.IOBase, load_context:save_game.LoadContext, intel_entity_id:uuid.UUID, intel_entity_short_id:str, intel_entity_type:type, entity_id:uuid.UUID) -> SectorEntityIntel:
        sector_id = s_util.uuid_from_f(f)
        loc = s_util.float_pair_from_f(f)
        sector_entity_intel = self._load_sector_entity_intel(f, load_context, sector_id, loc, intel_entity_id, intel_entity_short_id, intel_entity_type, entity_id)
        return sector_entity_intel

class AsteroidIntelSaver(SectorEntityIntelSaver[intel.AsteroidIntel]):
    def _save_sector_entity_intel(self, sector_entity_intel:intel.AsteroidIntel, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.int_to_f(sector_entity_intel.resource, f)
        bytes_written += s_util.float_to_f(sector_entity_intel.amount, f)
        return bytes_written

    def _load_sector_entity_intel(self, f:io.IOBase, load_context:save_game.LoadContext, sector_id:uuid.UUID, loc:npt.NDArray[np.float64], intel_entity_id:uuid.UUID, intel_entity_id_short:str, intel_entity_type:type, entity_id:uuid.UUID) -> intel.AsteroidIntel:
        resource = s_util.int_from_f(f)
        amount = s_util.float_from_f(f)
        sector_entity_intel = intel.AsteroidIntel(resource, amount, sector_id, loc, intel_entity_id, intel_entity_id_short, intel_entity_type, load_context.gamestate, entity_id=entity_id)
        return sector_entity_intel
