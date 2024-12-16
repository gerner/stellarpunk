import time
import io
import os
import glob
import abc
import pydoc
import uuid
import datetime
import logging
import contextlib
from typing import Any, Optional, TypeVar

import numpy as np
import cymunk # type: ignore

from stellarpunk import util, sim, core, narrative
from stellarpunk.serialization import serialize_econ_sim, util as s_util

class LoadContext:
    """ State for one load cycle. """

    def __init__(self, sg:"GameSaver") -> None:
        self.save_game = sg
        self.debug = False
        self._post_loads:list[tuple[Any, Any]] = []

    def register_post_load(self, obj:Any, context:Any) -> None:
        self._post_loads.append((obj, context))

    def load_complete(self) -> None:
        for obj, context in self._post_loads:
            self.save_game.post_load_object(obj, self, context)

class Saver[T](abc.ABC):
    def __init__(self, save_game:"GameSaver"):
        self.save_game = save_game

    @abc.abstractmethod
    def save(self, obj:T, f:io.IOBase) -> int: ...
    @abc.abstractmethod
    def load(self, f:io.IOBase, load_context:LoadContext) -> T: ...

    def post_load(self, obj:T, load_context:LoadContext, context:Any) -> None:
        pass

class NoneSaver(Saver[None]):
    def save(self, obj:None, f:io.IOBase) -> int:
        return 0
    def load(self, f:io.IOBase, load_context:LoadContext) -> None:
        return None

class SaveGame:
    def __init__(self, filename:str):
        self.filename = filename

class GameSaver:
    """ Central point for saving a game.

    Organizes configuration and dispatches to various sorts of save logic. """

    def __init__(self) -> None:
        self.logger = logging.getLogger(util.fullname(self))

        self.debug = True
        self._save_path:str = "/tmp/stellarpunk_saves"
        self._save_file_glob = "save_*.stpnk"
        self._save_register:dict[type, Saver] = {}
        self._class_key_lookup:dict[type, int] = {}
        self._key_class_lookup:dict[int, type] = {}

    def _gen_save_filename(self) -> str:
        return f'save_{time.time()}.stpnk'

    def _save_registry(self, f:io.IOBase) -> int:
        bytes_written = s_util.size_to_f(len(self._class_key_lookup), f)
        for klass, key in self._class_key_lookup.items():
            bytes_written += s_util.to_len_pre_f(util.fullname(klass), f)
            bytes_written += s_util.int_to_f(key, f)
        return bytes_written

    def _load_registry(self, f:io.IOBase) -> None:
        # all the savers must be registered already, we're just loading the
        # class -> key correspondence the file originally used when saving
        # we want to verify that we have all the right savers registered and
        # that they have the same id keys
        # note that in general it's ok for us to have some extra savers
        count = s_util.size_from_f(f)
        for _ in range(count):
            fullname = s_util.from_len_pre_f(f)
            key = s_util.int_from_f(f)
            klass = pydoc.locate(fullname)
            assert(isinstance(klass, type))

            #TODO: should we just stomp on the registry?
            # if this doesn't match we might have a problem as it implies logic
            # has changed since we saved
            assert(self._class_key_lookup[klass] == key)

        #TODO: we probably don't want to assert this. we might add new types in
        # new versions of the code, but that doesn't necessarily invalidate
        # prior savegames that didn't know about these new types
        assert(count == len(self._class_key_lookup))

    def key_from_class(self, klass:type) -> int:
        return self._class_key_lookup[klass]

    def class_from_key(self, key:int) -> type:
        return self._key_class_lookup[key]

    def register_saver(self, klass:type, saver:Saver) -> int:
        """ Registers type specific save/load logic

        returns the key used to identify that type in save files """
        self._save_register[klass] = saver
        key = len(self._class_key_lookup)
        self._class_key_lookup[klass] = key
        self._key_class_lookup[key] = klass
        return key

    def ignore_saver(self, klass:type) -> int:
        if issubclass(klass, core.Entity):
            return self.register_saver(klass, NoneEntitySaver(self))
        else:
            return self.register_saver(klass, NoneSaver(self))

    def save_object(self, obj:Any, f:io.IOBase, klass:Optional[type]=None) -> int:
        if klass is None:
            klass = type(obj)

        bytes_written = 0
        if self.debug:
            bytes_written = s_util.to_len_pre_f(f'__so:{util.fullname(klass)}', f)
            bytes_written = s_util.to_len_pre_f(f'__so:{util.fullname(obj)}', f)
        return bytes_written+self._save_register[klass].save(obj, f)

    def load_object(self, klass:type, f:io.IOBase, load_context:LoadContext) -> Any:
        if load_context.debug:
            klassname = s_util.from_len_pre_f(f)
            fullname = s_util.from_len_pre_f(f)
            assert(klassname == f'__so:{util.fullname(klass)}')
        return self._save_register[klass].load(f, load_context)

    def post_load_object(self, obj:Any, load_context:LoadContext, context:Any) -> None:
        self._save_register[type(obj)].post_load(obj, load_context, context)

    def save(self, gamestate:core.Gamestate, save_file:Optional[io.IOBase]=None) -> str:
        self.logger.info("saving...")
        start_time = time.perf_counter()
        save_filename = self._gen_save_filename()
        save_filename = os.path.join(self._save_path, save_filename)
        bytes_written = 0
        with contextlib.ExitStack() as context_stack:
            if save_file is None:
                save_file = context_stack.enter_context(open(save_filename, "wb"))
            #TODO: put metadata about the save game at the top for quick retrieval
            if self.debug:
                save_file.write(b'1')
            else:
                save_file.write(b'0')

            # save class -> key registration
            self._save_registry(save_file)

            # save the simulator which will recursively save everything
            bytes_written += self.save_object(gamestate, save_file)

        self.logger.info(f'saved {bytes_written}bytes in {time.perf_counter()-start_time}s')

        return save_filename

    def list_save_games(self) -> list[SaveGame]:
        save_games = [SaveGame(x) for x in glob.glob(os.path.join(self._save_path, self._save_file_glob))]
        return save_games

    def load(self, save_filename:str, save_file:Optional[io.IOBase]=None) -> core.Gamestate:
        load_context = LoadContext(self)
        with contextlib.ExitStack() as context_stack:
            if save_file is None:
                save_file = context_stack.enter_context(open(save_filename, "rb"))
            #TODO: read metadata
            load_context.debug = save_file.read(1) == b'1'

            # load the class -> key registration
            self._load_registry(save_file)

            gamestate = self.load_object(core.Gamestate, save_file, load_context)

            #TODO: do we need to do any final setup?

            return gamestate

class GamestateSaver(Saver[core.Gamestate]):
    def save(self, gamestate:core.Gamestate, f:io.IOBase) -> int:
        bytes_written = 0

        # simple fields
        bytes_written += s_util.to_len_pre_f("simple fields", f)
        bytes_written += s_util.random_state_to_f(gamestate.random, f)
        bytes_written += s_util.to_len_pre_f(gamestate.base_date.isoformat(), f)
        bytes_written += s_util.float_to_f(gamestate.timestamp, f)
        bytes_written += s_util.float_to_f(gamestate.desired_dt, f)
        # no need to save dt, we should reload with desired dt
        #bytes_written += s_util.float_to_f(gamestate.dt, f)
        bytes_written += s_util.float_to_f(gamestate.min_tick_sleep, f)
        bytes_written += s_util.int_to_f(gamestate.ticks, f)
        assert(gamestate.force_pause_holder is None) # can't save while force paused
        bytes_written += s_util.uuid_to_f(gamestate.player.entity_id, f)
        #TODO: should we save counters?

        # production chain
        bytes_written += s_util.to_len_pre_f("production chain", f)
        pchain_bytes = serialize_econ_sim.save_production_chain(gamestate.production_chain)
        bytes_written += s_util.size_to_f(len(pchain_bytes), f)
        bytes_written += f.write(pchain_bytes)

        # entities
        bytes_written += s_util.to_len_pre_f("entities", f)
        bytes_written += s_util.size_to_f(len(gamestate.entities), f)
        for entity in gamestate.entities.values():
            # we save as a generic entity which will handle its own dispatch
            bytes_written += self.save_game.save_object(entity, f, klass=core.Entity)

        # sectors
        # save the sector ids in the right order
        # this gives us enough info to reconstruct sectors dict, sector_idx,
        # sector_spatial by pulling desired entity from the entity store
        bytes_written += s_util.to_len_pre_f("sector ids", f)
        bytes_written += s_util.size_to_f(len(gamestate.sector_ids), f)
        for sector_id in gamestate.sector_ids:
            s_util.uuid_to_f(sector_id, f)
        bytes_written += s_util.matrix_to_f(gamestate.sector_edges, f)

        # econ agents
        bytes_written += s_util.to_len_pre_f("econ agents", f)
        bytes_written += s_util.size_to_f(len(gamestate.econ_agents), f)
        for entity_id, agent in gamestate.econ_agents.items():
            bytes_written += s_util.uuid_to_f(entity_id, f)
            bytes_written += s_util.uuid_to_f(agent.entity_id, f)

        #TODO: task lists
        #TODO: starfields

        # entity destroy list
        bytes_written += s_util.to_len_pre_f("entity destroy list", f)
        bytes_written += s_util.size_to_f(len(gamestate.entity_destroy_list), f)
        for entity in gamestate.entity_destroy_list:
            bytes_written += s_util.uuid_to_f(entity.entity_id, f)

        bytes_written += s_util.to_len_pre_f("last colliders", f)
        bytes_written += s_util.size_to_f(len(gamestate.last_colliders), f)
        for colliders in gamestate.last_colliders:
            bytes_written += s_util.to_len_pre_f(colliders, f)

        bytes_written += s_util.to_len_pre_f("gamestate done", f)
        return bytes_written

    def load(self, f:io.IOBase, load_context:LoadContext) -> core.Gamestate:
        #TODO: make sure that core.Gamestate.gamestate is this one?
        gamestate = core.Gamestate()

        # simple fields
        debug_string = s_util.from_len_pre_f(f)
        assert(debug_string == "simple fields")
        gamestate.random = s_util.random_state_from_f(f)
        gamestate.base_date = datetime.datetime.fromisoformat(s_util.from_len_pre_f(f))
        gamestate.timestamp = s_util.float_from_f(f)
        gamestate.desired_dt = s_util.float_from_f(f)
        gamestate.dt = gamestate.desired_dt
        gamestate.min_tick_sleep = s_util.float_from_f(f)
        gamestate.ticks = s_util.int_from_f(f)
        player_id = s_util.uuid_from_f(f)

        debug_string = s_util.from_len_pre_f(f)
        assert(debug_string == "production chain")
        pchain_bytes_count = s_util.size_from_f(f)
        pchain_bytes = f.read(pchain_bytes_count)
        gamestate.production_chain = serialize_econ_sim.load_production_chain(pchain_bytes)

        # load entities
        debug_string = s_util.from_len_pre_f(f)
        assert(debug_string == "entities")
        count = s_util.size_from_f(f)
        for i in range(count):
            entity = self.save_game.load_object(core.Entity, f, load_context)
            # entities should have registered themselves with the gamestate,
            # getting a fresh, empty context and then populated it

        # needed to wait for entities to be loaded before we could get the
        # actual player entity
        player = gamestate.entities[player_id]
        assert(isinstance(player, core.Player))
        gamestate.player = player

        # sectors
        # load sector ids and sector edges from file
        debug_string = s_util.from_len_pre_f(f)
        sector_ids:list[uuid.UUID] = []
        count = s_util.size_from_f(f)
        for i in range(count):
            sector_id = s_util.uuid_from_f(f)
            sector_ids.append(sector_id)
        sector_edges = s_util.matrix_from_f(f)

        # now reconstruct all the sector state from entites already loaded
        sector_coords = np.zeros((len(sector_ids), 2), dtype=np.float64)
        for i, sector_id in enumerate(sector_ids):
            sector = gamestate.entities[sector_id]
            assert(isinstance(sector, core.Sector))
            sector_coords[i] = sector.loc
            gamestate.add_sector(sector, i)
        gamestate.update_edges(sector_edges, np.array(sector_ids), sector_coords)

        # econ agents
        debug_string = s_util.from_len_pre_f(f)
        count = s_util.size_from_f(f)
        for _ in range(count):
            entity_id = s_util.uuid_from_f(f)
            agent_id = s_util.uuid_from_f(f)
            agent = gamestate.entities[agent_id]
            assert(isinstance(agent, core.EconAgent))
            gamestate.representing_agent(entity_id, agent)

        #TODO: task lists

        #TODO: starfields

        # entity destroy list
        debug_string = s_util.from_len_pre_f(f)
        count = s_util.size_from_f(f)
        for _ in range(count):
            entity_id = s_util.uuid_from_f(f)
            gamestate.destroy_entity(gamestate.entities[entity_id])

        # last colliders
        debug_string = s_util.from_len_pre_f(f)
        count = s_util.size_from_f(f)
        last_colliders = set()
        for _ in range(count):
            last_colliders.add(s_util.from_len_pre_f(f))
        gamestate.last_colliders = last_colliders

        return gamestate

class EntityDispatchSaver(Saver[core.Entity]):
    """ Saves and loads Entities by dispatching to class specific logic.

    Gamestate doesn't know the type of entity when loading, so this saver reads
    a type code it writes during saving. all other saving/loading logic is
    dispatched to class specific logic. """

    def save(self, entity:core.Entity, f:io.IOBase) -> int:
        bytes_written = 0

        # save key so we know what type of entity to load!
        class_id = self.save_game.key_from_class(type(entity))
        bytes_written += s_util.int_to_f(class_id, f)

        # dispatch to specific entity type to save the rest
        bytes_written += self.save_game.save_object(entity, f)

        return bytes_written

    def load(self, f:io.IOBase, load_context:LoadContext) -> core.Entity:
        # read entity type
        class_id = s_util.int_from_f(f)
        klass = self.save_game.class_from_key(class_id)

        # dispatch to specific type
        entity = self.save_game.load_object(klass, f, load_context)

        return entity

#EntityType = TypeVar('EntityType', bound=core.Entity)
class EntitySaver[EntityType: core.Entity](Saver[EntityType], abc.ABC):

    @abc.abstractmethod
    def _save_entity(self, entity:EntityType, f:io.IOBase) -> int: ...
    @abc.abstractmethod
    def _load_entity(self, f:io.IOBase, load_context:LoadContext, entity_id:uuid.UUID) -> EntityType: ...

    def save(self, entity:EntityType, f:io.IOBase) -> int:
        i = 0

        # common fields we need for entity creation
        i += s_util.uuid_to_f(entity.entity_id, f)

        i += self._save_entity(entity, f)

        # save some common fields
        i += s_util.to_len_pre_f(entity.name, f, 2)
        i += s_util.to_len_pre_f(entity.description, f, 2)
        i += s_util.float_to_f(entity.created_at, f)

        context_dict = entity.context.to_dict()
        i += s_util.size_to_f(len(context_dict), f)
        for k,v in context_dict.items():
            i += s_util.int_to_f(k, f, 8)
            i += s_util.int_to_f(v, f, 8)

        return i

    def load(self, f:io.IOBase, load_context:LoadContext) -> EntityType:
        # fields we need for entity creation
        entity_id = s_util.uuid_from_f(f)

        # type specific loading logic
        entity = self._load_entity(f, load_context, entity_id)

        #TODO: is this how we want to get the generic fields into the entity?
        # alternatively we could read these and make them available to the
        # class specific loader somehow
        # read common fields
        entity.name = s_util.from_len_pre_f(f)
        entity.description = s_util.from_len_pre_f(f)
        entity.created_at = s_util.float_from_f(f)

        # creating the entity made an empty event context, we just have to
        # populate it
        count = s_util.size_from_f(f)
        for _ in range(count):
            k = s_util.int_from_f(f)
            v = s_util.int_from_f(f)
            entity.context.set_flag(k, v)

        return entity

class NoneEntitySaver(EntitySaver[core.Entity]):
    def _save_entity(self, entity:core.Entity, f:io.IOBase) -> int:
        return 0
    def _load_entity(self, f:io.IOBase, load_context:LoadContext, entity_id:uuid.UUID) -> core.Entity:
        return core.Entity(core.Gamestate.gamestate, entity_id=entity_id)

class PlayerSaver(EntitySaver[core.Player]):
    def _save_entity(self, entity:core.Player, f:io.IOBase) -> int:
        #TODO: which character are we
        #TODO: which econ agent
        #TODO: messages
        return 0

    def _load_entity(self, f:io.IOBase, load_context:LoadContext, entity_id:uuid.UUID) -> core.Player:
        #TODO: which character are we
        #TODO: which econ agent
        #TODO: messages
        #TODO: register for post_load
        return core.Player(core.Gamestate.gamestate, entity_id=entity_id)

    def post_load(self, player:core.Player, load_context:LoadContext, context:Any) -> None:
        context_tuple:tuple[uuid.UUID, uuid.UUID, list[uuid.UUID]] = context
        character_id, agent_id, messages = context_tuple
        #pull out fully loaded character
        character = core.Gamestate.gamestate.entities[character_id]
        assert(isinstance(character, core.Character))
        player.set_character(character)

        #pull out fully loaded econ agent
        agent = core.Gamestate.gamestate.entities[agent_id]
        assert(isinstance(agent, core.EconAgent))
        player.agent = agent

        #pull out fully loaded messages
        for message_id in messages:
            message = core.Gamestate.gamestate.entities[message_id]
            assert(isinstance(message, core.Message))
            player.messages[message_id] = message

class SectorSaver(EntitySaver[core.Sector]):
    def _save_entity(self, sector:core.Sector, f:io.IOBase) -> int:
        bytes_written = 0

        # basic fields: loc, radius, culture
        bytes_written += s_util.matrix_to_f(sector.loc, f)
        bytes_written += s_util.float_to_f(sector.radius, f)
        bytes_written += s_util.to_len_pre_f(sector.culture, f)

        # entities. we'll reconstruct planets, etc. from entities
        bytes_written += s_util.size_to_f(len(sector.entities), f)
        for entity_id in sector.entities.keys():
            bytes_written += s_util.uuid_to_f(entity_id, f)

        # effects
        bytes_written += s_util.size_to_f(len(sector._effects), f)
        for effect in sector._effects:
            bytes_written += self.save_game.save_object(effect, f)

        #TODO: collision observers

        # weather. we'll reconstruct weather index on load from weather regions
        bytes_written += s_util.size_to_f(len(sector._weathers), f)
        # we assume weathers are in compact index order (so first element is
        # index 0, second is 1, and so on)
        for weather in sector._weathers.values():
            bytes_written += self.save_game.save_object(weather, f)

        return bytes_written

    def _load_entity(self, f:io.IOBase, load_context:LoadContext, entity_id:uuid.UUID) -> core.Sector:
        #we can't fully load the sector until all entities have been loaded.

        # basic fields: loc, radius, culture
        loc = s_util.matrix_from_f(f)
        radius = s_util.float_from_f(f)
        culture = s_util.from_len_pre_f(f)

        sector = core.Sector(loc, radius, cymunk.Space(), core.Gamestate.gamestate, entity_id=entity_id, culture=culture)

        # entities. we'll reconstruct these in post load
        entities:list[uuid.UUID] = []
        count = s_util.size_from_f(f)
        for _ in range(count):
            entities.append(s_util.uuid_from_f(f))

        # effects
        count = s_util.size_from_f(f)
        for _ in range(count):
            # can't use add_effect since it calls begin_effect
            sector._effects.append(self.save_game.load_object(core.Effect, f, load_context))

        #TODO: collision observers

        # weather
        count = s_util.size_from_f(f)
        for _ in range(count):
            sector.add_region(self.save_game.load_object(core.SectorWeatherRegion, f, load_context))

        load_context.register_post_load(sector, entities)
        return sector

    def post_load(self, sector:core.Sector, load_context:LoadContext, context:Any) -> None:
        entities:list[uuid.UUID] = context
        for entity_id in entities:
            entity = core.Gamestate.gamestate.entities[entity_id]
            assert(isinstance(entity, core.SectorEntity))
            sector.add_entity(entity)

class SectorWeatherRegionSaver(Saver[core.SectorWeatherRegion]):
    def save(self, weather:core.SectorWeatherRegion, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.matrix_to_f(weather.loc, f)
        bytes_written += s_util.float_to_f(weather.radius, f)
        bytes_written += s_util.float_to_f(weather.sensor_factor, f)
        return bytes_written

    def load(self, f:io.IOBase, load_context:LoadContext) -> core.SectorWeatherRegion:
        loc = s_util.matrix_from_f(f)
        radius = s_util.float_from_f(f)
        sensor_factor = s_util.float_from_f(f)
        weather = core.SectorWeatherRegion(loc, radius, sensor_factor)
        return weather

class SectorEntitySaver[SectorEntity: core.SectorEntity](EntitySaver[SectorEntity], abc.ABC):
    @abc.abstractmethod
    def _save_sector_entity(self, sector_entity:SectorEntity, f:io.IOBase) -> int: ...
    @abc.abstractmethod
    def _load_sector_entity(self, f:io.IOBase, load_context:LoadContext, entity_id:uuid.UUID) -> SectorEntity: ...
    def _save_entity(self, sector_entity:SectorEntity, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += self._save_sector_entity(sector_entity, f)
        return bytes_written

    def _load_entity(self, f:io.IOBase, load_context:LoadContext, entity_id:uuid.UUID) -> SectorEntity:
        sector_entity = self._load_sector_entity(f, load_context, entity_id)
        return sector_entity
