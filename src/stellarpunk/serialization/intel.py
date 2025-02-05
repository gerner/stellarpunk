import io
import uuid
import abc
import pydoc
from typing import Any, Type, Optional

import numpy as np
import numpy.typing as npt

from . import save_game, util as s_util, gamestate as s_gamestate

from stellarpunk import core, intel, util

class IntelManagerSaver(save_game.Saver[intel.IntelManager]):
    def fetch(self, klass:Type[intel.IntelManager], object_id:uuid.UUID, load_context:save_game.LoadContext) -> intel.IntelManager:
        intel_manager = load_context.gamestate.get_entity(object_id, core.Character).intel_manager
        assert(isinstance(intel_manager, intel.IntelManager))
        return intel_manager

    def save(self, intel_manager:intel.IntelManager, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.uuids_to_f(intel_manager._intel, f)
        # our character field is handled in CharacterSaver
        return bytes_written

    def load(self, f:io.IOBase, load_context:save_game.LoadContext) -> intel.IntelManager:
        intel_ids = s_util.uuids_from_f(f)
        intel_manager = intel.IntelManager(load_context.gamestate)
        # our character field is handled in CharacterSaver
        load_context.register_post_load(intel_manager, intel_ids)
        return intel_manager

    def post_load(self, intel_manager:intel.IntelManager, load_context:save_game.LoadContext, context:Any) -> None:
        intel_ids:list[uuid.UUID] = context
        for intel_id in intel_ids:
            intel = load_context.gamestate.get_entity(intel_id, core.AbstractIntel)
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
        load_context.store(task_id, task)
        load_context.register_post_load(task, intel_id)
        return task
    def post_load(self, obj:intel.ExpireIntelTask, load_context:save_game.LoadContext, context:Any) -> None:
        intel_id:uuid.UUID = context
        intel = load_context.gamestate.get_entity(intel_id, core.AbstractIntel)
        obj.intel = intel

    def fetch(self, klass:Type[intel.ExpireIntelTask], object_id:uuid.UUID, load_context:save_game.LoadContext) -> intel.ExpireIntelTask:
        return load_context.fetch(object_id, klass)

class IntelSaver[T: intel.Intel](s_gamestate.EntitySaver[T]):
    @abc.abstractmethod
    def _save_intel(self, intel:T, f:io.IOBase) -> int: ...
    @abc.abstractmethod
    def _load_intel(self, f:io.IOBase, load_context:save_game.LoadContext, entity_id:uuid.UUID, author_id:uuid.UUID) -> tuple[T, Any]: ...
    def _post_load_intel(self, obj:T, load_context:save_game.LoadContext, context:Any) -> None:
        pass

    def _save_entity(self, intel:T, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.uuid_to_f(intel.author_id, f)
        bytes_written += s_util.float_to_f(intel.fresh_until, f)
        bytes_written += s_util.float_to_f(intel.expires_at, f)
        bytes_written += s_util.optional_uuid_to_f(intel.expire_task.task_id if intel.expire_task else None, f)
        bytes_written += self._save_intel(intel, f)
        return bytes_written
    def _load_entity(self, f:io.IOBase, load_context:save_game.LoadContext, entity_id:uuid.UUID) -> T:
        author_id = s_util.uuid_from_f(f)
        fresh_until = s_util.float_from_f(f)
        expires_at = s_util.float_from_f(f)
        expire_task_id = s_util.optional_uuid_from_f(f)
        intel, extra_context = self._load_intel(f, load_context, entity_id, author_id)
        intel.fresh_until = fresh_until
        intel.expires_at = expires_at
        load_context.register_post_load(intel, (expire_task_id, extra_context))
        return intel

    def post_load(self, obj:T, load_context:save_game.LoadContext, context:Any) -> None:
        context_data:tuple[Optional[uuid.UUID], Any] = context
        expire_task_id, extra_context = context_data

        if expire_task_id:
            expire_task = self.save_game.fetch_object(intel.ExpireIntelTask, expire_task_id, load_context)
            obj.expire_task = expire_task

        self._post_load_intel(obj, load_context, extra_context)

class EntityIntelSaver[T: intel.EntityIntel](IntelSaver[T]):
    @abc.abstractmethod
    def _save_entity_intel(self, intel:T, f:io.IOBase) -> int: ...
    @abc.abstractmethod
    def _load_entity_intel(self, f:io.IOBase, load_context:save_game.LoadContext, **kwargs:Any) -> tuple[T, Any]: ...

    def _save_intel(self, intel:T, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.uuid_to_f(intel.intel_entity_id, f)
        bytes_written += s_util.to_len_pre_f(util.fullname(intel.intel_entity_type), f)
        bytes_written += s_util.to_len_pre_f(intel.intel_entity_name, f)
        bytes_written += s_util.to_len_pre_f(intel.intel_entity_description, f)
        bytes_written += self._save_entity_intel(intel, f)
        return bytes_written

    def _load_intel(self, f:io.IOBase, load_context:save_game.LoadContext, entity_id:uuid.UUID, author_id:uuid.UUID) -> tuple[T, Any]:
        intel_entity_id = s_util.uuid_from_f(f)
        intel_entity_class_name = s_util.from_len_pre_f(f)
        intel_entity_type = pydoc.locate(intel_entity_class_name)
        assert(isinstance(intel_entity_type, type))
        intel_entity_name = s_util.from_len_pre_f(f)
        intel_entity_description = s_util.from_len_pre_f(f)

        intel, extra_context = self._load_entity_intel(f, load_context, intel_entity_id=intel_entity_id, intel_entity_type=intel_entity_type, intel_entity_name=intel_entity_name, intel_entity_description=intel_entity_description, entity_id=entity_id, author_id=author_id)

        return intel, extra_context

class SectorIntelSaver(EntityIntelSaver[intel.SectorIntel]):
    def _save_entity_intel(self, intel:intel.SectorIntel, f:io.IOBase) -> int:
        return 0

    def _load_entity_intel(self, f:io.IOBase, load_context:save_game.LoadContext, **kwargs:Any) -> tuple[intel.SectorIntel, Any]:
        sector_intel = intel.SectorIntel(load_context.gamestate, **kwargs, _check_flag=True)
        return sector_intel, None

class SectorHexIntelSaver(IntelSaver[intel.SectorHexIntel]):
    def _save_intel(self, intel:intel.SectorHexIntel, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.uuid_to_f(intel.sector_id, f)
        bytes_written += s_util.float_pair_to_f(intel.hex_loc, f)
        bytes_written += s_util.bool_to_f(intel.is_static, f)
        bytes_written += s_util.int_to_f(intel.entity_count, f)
        bytes_written += s_util.fancy_dict_to_f(intel.type_counts, f, s_util.type_to_f, s_util.int_to_f)
        return bytes_written

    def _load_intel(self, f:io.IOBase, load_context:save_game.LoadContext, entity_id:uuid.UUID, author_id:uuid.UUID) -> tuple[intel.SectorHexIntel, Any]:
        sector_id = s_util.uuid_from_f(f)
        hex_loc = s_util.float_pair_from_f(f)
        is_static = s_util.bool_from_f(f)
        entity_count = s_util.int_from_f(f)
        type_counts = s_util.fancy_dict_from_f(f, lambda f: s_util.type_from_f(f, core.SectorEntity), s_util.int_from_f)

        hex_intel = intel.SectorHexIntel(sector_id, hex_loc, is_static, entity_count, type_counts, load_context.gamestate, _check_flag=True, entity_id=entity_id, author_id=author_id)

        return hex_intel, None


class EconAgentIntelSaver(EntityIntelSaver[intel.EconAgentIntel]):
    def _save_entity_intel(self, intel:intel.EconAgentIntel, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.uuid_to_f(intel.underlying_entity_id, f)
        bytes_written += s_util.type_to_f(intel.underlying_entity_type, f)
        bytes_written += s_util.fancy_dict_to_f(intel.sell_offers, f, s_util.int_to_f, s_util.float_pair_to_f)
        bytes_written += s_util.fancy_dict_to_f(intel.buy_offers, f, s_util.int_to_f, s_util.float_pair_to_f)
        return bytes_written

    def _load_entity_intel(self, f:io.IOBase, load_context:save_game.LoadContext, **kwargs:Any) -> tuple[intel.EconAgentIntel, Any]:
        underlying_entity_id = s_util.uuid_from_f(f)
        underlying_entity_type = s_util.type_from_f(f, core.Entity)
        sell_offers = s_util.fancy_dict_from_f(f, s_util.int_from_f, lambda x: tuple(s_util.float_pair_from_f(x)))
        buy_offers = s_util.fancy_dict_from_f(f, s_util.int_from_f, lambda x: tuple(s_util.float_pair_from_f(x)))

        agent_intel = intel.EconAgentIntel(load_context.gamestate, **kwargs, _check_flag=True)
        agent_intel.underlying_entity_id = underlying_entity_id
        agent_intel.underlying_entity_type = underlying_entity_type
        agent_intel.sell_offers = sell_offers
        agent_intel.buy_offers = buy_offers

        return agent_intel, None

class SectorEntityIntelSaver[SectorEntityIntel:intel.SectorEntityIntel](EntityIntelSaver[SectorEntityIntel]):
    def _save_sector_entity_intel(self, sector_entity_intel:SectorEntityIntel, f:io.IOBase) -> int:
        return 0
    def _load_sector_entity_intel(self, f:io.IOBase, load_context:save_game.LoadContext, sector_id:uuid.UUID, loc:npt.NDArray[np.float64], radius:float, is_static:bool, **kwargs:Any) -> tuple[SectorEntityIntel, Any]:
        return intel.SectorEntityIntel(sector_id, loc, radius, is_static, load_context.gamestate, **kwargs, _check_flag=True), None # type: ignore

    def _save_entity_intel(self, sector_entity_intel:SectorEntityIntel, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += self.save_game.debug_string_w("basic fields", f)
        bytes_written += s_util.uuid_to_f(sector_entity_intel.sector_id, f)
        bytes_written += s_util.float_pair_to_f(sector_entity_intel.loc, f)
        bytes_written += s_util.float_to_f(sector_entity_intel.radius, f)
        bytes_written += s_util.bool_to_f(sector_entity_intel.is_static, f)
        bytes_written += self.save_game.debug_string_w("seintel specific", f)
        bytes_written += self._save_sector_entity_intel(sector_entity_intel, f)
        return bytes_written

    def _load_entity_intel(self, f:io.IOBase, load_context:save_game.LoadContext, **kwargs:Any) -> tuple[SectorEntityIntel, Any]:
        load_context.debug_string_r("basic fields", f)
        sector_id = s_util.uuid_from_f(f)
        loc = s_util.float_pair_from_f(f)
        radius = s_util.float_from_f(f)
        is_static = s_util.bool_from_f(f)
        load_context.debug_string_r("seintel specific", f)
        sector_entity_intel, extra_context = self._load_sector_entity_intel(f, load_context, sector_id, loc, radius, is_static, **kwargs)
        return sector_entity_intel, extra_context

class AsteroidIntelSaver(SectorEntityIntelSaver[intel.AsteroidIntel]):
    def _save_sector_entity_intel(self, sector_entity_intel:intel.AsteroidIntel, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.int_to_f(int(sector_entity_intel.resource), f)
        bytes_written += s_util.float_to_f(sector_entity_intel.amount, f)
        return bytes_written

    def _load_sector_entity_intel(self, f:io.IOBase, load_context:save_game.LoadContext, sector_id:uuid.UUID, loc:npt.NDArray[np.float64], radius:float, is_static:bool, **kwargs:Any) -> tuple[intel.AsteroidIntel, Any]:
        resource = s_util.int_from_f(f)
        amount = s_util.float_from_f(f)
        sector_entity_intel = intel.AsteroidIntel(resource, amount, sector_id, loc, radius, is_static, load_context.gamestate, **kwargs, _check_flag=True)
        return sector_entity_intel, None

class StationIntelSaver(SectorEntityIntelSaver[intel.StationIntel]):
    def _save_sector_entity_intel(self, sector_entity_intel:intel.StationIntel, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.int_to_f(int(sector_entity_intel.resource), f)
        bytes_written += s_util.ints_to_f(list(int(x) for x in sector_entity_intel.inputs), f)
        return bytes_written

    def _load_sector_entity_intel(self, f:io.IOBase, load_context:save_game.LoadContext, sector_id:uuid.UUID, loc:npt.NDArray[np.float64], radius:float, is_static:bool, **kwargs:Any) -> tuple[intel.StationIntel, Any]:
        resource = s_util.int_from_f(f)
        inputs = set(s_util.ints_from_f(f))
        sector_entity_intel = intel.StationIntel(resource, inputs, sector_id, loc, radius, is_static, load_context.gamestate, **kwargs, _check_flag=True)
        return sector_entity_intel, None

class TravelGateIntelSaver(SectorEntityIntelSaver[intel.TravelGateIntel]):
    def _save_sector_entity_intel(self, sector_entity_intel:intel.TravelGateIntel, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.uuid_to_f(sector_entity_intel.destination_id, f)
        bytes_written += s_util.float_to_f(sector_entity_intel.direction, f)
        return bytes_written

    def _load_sector_entity_intel(self, f:io.IOBase, load_context:save_game.LoadContext, sector_id:uuid.UUID, loc:npt.NDArray[np.float64], radius:float, is_static:bool, **kwargs:Any) -> tuple[intel.TravelGateIntel, Any]:
        destination_id = s_util.uuid_from_f(f)
        direction = s_util.float_from_f(f)
        sector_entity_intel = intel.TravelGateIntel(sector_id, loc, radius, is_static, load_context.gamestate, **kwargs, _check_flag=True, destination_id=destination_id, direction=direction)
        return sector_entity_intel, None


# IntelCriteria savers

class EntityIntelMatchCriteriaSaver(save_game.Saver[intel.EntityIntelMatchCriteria]):
    def save(self, obj:intel.EntityIntelMatchCriteria, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.uuid_to_f(obj.entity_id, f)
        return bytes_written

    def load(self, f:io.IOBase, load_context:save_game.LoadContext) -> intel.EntityIntelMatchCriteria:
        entity_id = s_util.uuid_from_f(f)
        obj = intel.EntityIntelMatchCriteria(entity_id)
        return obj

class SectorHexMatchCriteriaSaver(save_game.Saver[intel.SectorHexMatchCriteria]):
    def save(self, obj:intel.SectorHexMatchCriteria, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.uuid_to_f(obj.sector_id, f)
        bytes_written += s_util.float_pair_to_f(obj.hex_loc, f)
        bytes_written += s_util.bool_to_f(obj.is_static, f)
        return bytes_written

    def load(self, f:io.IOBase, load_context:save_game.LoadContext) -> intel.SectorHexMatchCriteria:
        sector_id = s_util.uuid_from_f(f)
        hex_loc = s_util.float_pair_from_f(f)
        is_static = s_util.bool_from_f(f)
        obj = intel.SectorHexMatchCriteria(sector_id, hex_loc, is_static)
        return obj


class SectorEntityPartialCriteriaSaver(save_game.Saver[intel.SectorEntityPartialCriteria]):
    def save(self, obj:intel.SectorEntityPartialCriteria, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.to_len_pre_f(util.fullname(obj.cls), f)
        bytes_written += s_util.optional_obj_to_f(obj.is_static, f, s_util.bool_to_f)
        bytes_written += s_util.optional_uuid_to_f(obj.sector_id, f)
        return bytes_written

    def load(self, f:io.IOBase, load_context:save_game.LoadContext) -> intel.SectorEntityPartialCriteria:
        cls_name = s_util.from_len_pre_f(f)
        cls = pydoc.locate(cls_name)
        assert isinstance(cls, type)
        assert issubclass(cls, core.SectorEntity)
        is_static = s_util.optional_obj_from_f(f, s_util.bool_from_f)
        sector_id = s_util.optional_uuid_from_f(f)
        obj = intel.SectorEntityPartialCriteria(cls=cls, is_static=is_static, sector_id=sector_id)
        return obj

class AsteroidIntelPartialCriteriaSaver(save_game.Saver[intel.AsteroidIntelPartialCriteria]):
    def save(self, obj:intel.AsteroidIntelPartialCriteria, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.to_len_pre_f(util.fullname(obj.cls), f)
        bytes_written += s_util.optional_obj_to_f(obj.is_static, f, s_util.bool_to_f)
        bytes_written += s_util.optional_uuid_to_f(obj.sector_id, f)
        bytes_written += s_util.optional_obj_to_f(obj.resources, f, s_util.ints_to_f)
        return bytes_written

    def load(self, f:io.IOBase, load_context:save_game.LoadContext) -> intel.AsteroidIntelPartialCriteria:
        cls_name = s_util.from_len_pre_f(f)
        cls = pydoc.locate(cls_name)
        assert isinstance(cls, type)
        assert issubclass(cls, core.SectorEntity)
        is_static = s_util.optional_obj_from_f(f, s_util.bool_from_f)
        sector_id = s_util.optional_uuid_from_f(f)
        resources = s_util.optional_obj_from_f(f, s_util.ints_from_f)
        obj = intel.AsteroidIntelPartialCriteria(cls=cls, is_static=is_static, sector_id=sector_id, resources=frozenset(resources) if resources else None)
        return obj

class StationIntelPartialCriteriaSaver(save_game.Saver[intel.StationIntelPartialCriteria]):
    def save(self, obj:intel.StationIntelPartialCriteria, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.to_len_pre_f(util.fullname(obj.cls), f)
        bytes_written += s_util.optional_obj_to_f(obj.is_static, f, s_util.bool_to_f)
        bytes_written += s_util.optional_uuid_to_f(obj.sector_id, f)
        bytes_written += s_util.optional_obj_to_f(obj.resources, f, s_util.ints_to_f)
        bytes_written += s_util.optional_obj_to_f(obj.inputs, f, s_util.ints_to_f)
        return bytes_written

    def load(self, f:io.IOBase, load_context:save_game.LoadContext) -> intel.StationIntelPartialCriteria:
        cls_name = s_util.from_len_pre_f(f)
        cls = pydoc.locate(cls_name)
        assert isinstance(cls, type)
        assert issubclass(cls, core.SectorEntity)
        is_static = s_util.optional_obj_from_f(f, s_util.bool_from_f)
        sector_id = s_util.optional_uuid_from_f(f)
        resources = s_util.optional_obj_from_f(f, s_util.ints_from_f)
        inputs = s_util.optional_obj_from_f(f, s_util.ints_from_f)
        obj = intel.StationIntelPartialCriteria(cls=cls, is_static=is_static, sector_id=sector_id, resources=frozenset(resources) if resources else None, inputs=frozenset(inputs) if inputs else None)
        return obj

class SectorHexPartialCriteriaSaver(save_game.Saver[intel.SectorHexPartialCriteria]):
    def save(self, obj:intel.SectorHexPartialCriteria, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.optional_uuid_to_f(obj.sector_id, f)
        bytes_written += s_util.optional_obj_to_f(obj.hex_loc, f, s_util.float_pair_to_f)
        bytes_written += s_util.optional_obj_to_f(obj.hex_dist, f, s_util.float_to_f)
        bytes_written += s_util.optional_obj_to_f(obj.is_static, f, s_util.bool_to_f)
        return bytes_written

    def load(self, f:io.IOBase, load_context:save_game.LoadContext) -> intel.SectorHexPartialCriteria:
        sector_id = s_util.optional_uuid_from_f(f)
        hex_loc = s_util.optional_obj_from_f(f, s_util.float_pair_from_f)
        hex_dist = s_util.optional_obj_from_f(f, s_util.float_from_f)
        is_static = s_util.optional_obj_from_f(f, s_util.bool_from_f)
        obj = intel.SectorHexPartialCriteria(sector_id=sector_id, is_static=is_static, hex_loc=hex_loc, hex_dist=hex_dist)
        return obj

class EconAgentSectorEntityPartialCriteriaSaver(save_game.Saver[intel.EconAgentSectorEntityPartialCriteria]):
    def save(self, obj:intel.EconAgentSectorEntityPartialCriteria, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.optional_uuid_to_f(obj.sector_id, f)
        bytes_written += s_util.optional_uuid_to_f(obj.underlying_entity_id, f)
        bytes_written += s_util.to_len_pre_f(util.fullname(obj.underlying_entity_type), f)
        bytes_written += s_util.optional_obj_to_f(obj.buy_resources, f, s_util.ints_to_f)
        bytes_written += s_util.optional_obj_to_f(obj.sell_resources, f, s_util.ints_to_f)
        return bytes_written

    def load(self, f:io.IOBase, load_context:save_game.LoadContext) -> intel.EconAgentSectorEntityPartialCriteria:
        sector_id = s_util.optional_uuid_from_f(f)
        underlying_entity_id = s_util.optional_uuid_from_f(f)
        cls_name = s_util.from_len_pre_f(f)
        underlying_entity_type = pydoc.locate(cls_name)
        assert isinstance(underlying_entity_type, type)
        assert issubclass(underlying_entity_type, core.SectorEntity)
        buy_resources = s_util.optional_obj_from_f(f, s_util.ints_from_f)
        sell_resources = s_util.optional_obj_from_f(f, s_util.ints_from_f)
        obj = intel.EconAgentSectorEntityPartialCriteria(
                sector_id=sector_id,
                underlying_entity_id=underlying_entity_id,
                underlying_entity_type=underlying_entity_type,
                buy_resources=frozenset(buy_resources) if buy_resources else None,
                sell_resources=frozenset(sell_resources) if sell_resources else None
        )
        return obj

