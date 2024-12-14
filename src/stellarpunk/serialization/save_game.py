import time
import io
import os
import abc
import pydoc
import struct
import uuid
import datetime
from typing import Any

import numpy as np

from stellarpunk import util, sim, core, narrative
from stellarpunk.serialization import serialize_econ_sim, util as s_util

class Saver[T](abc.ABC):
    def __init__(self, save_game:"SaveGame"):
        self.save_game = save_game

    @abc.abstractmethod
    def save(self, obj:T, f:io.BufferedWriter) -> int: ...
    @abc.abstractmethod
    def load(self, f:io.BufferedReader) -> T: ...

class SaveGame:
    """ Central point for saving a game.

    Organizes configuration and dispatches to various sorts of save logic. """

    def __init__(self) -> None:
        self._save_path:str = "/tmp/"
        self._save_register:dict[type, Saver] = {}
        self._class_key_lookup:dict[type, int] = {}
        self._key_class_lookup:dict[int, type] = {}

    def _gen_save_filename(self) -> str:
        return f'save_{time.time()}'

    def _save_registry(self, f:io.BufferedWriter) -> int:
        bytes_written = s_util.size_to_f(len(self._class_key_lookup), f)
        for klass, key in self._class_key_lookup.items():
            bytes_written += s_util.to_len_pre_f(util.fullname(klass), f)
            bytes_written += f.write(key.to_bytes(4))
        return bytes_written

    def _load_registry(self, f:io.BufferedReader) -> None:
        # all the savers must be registered already, we're just loading the
        # class -> key correspondence the file originally used when saving
        count = s_util.size_from_f(f)
        for _ in range(count):
            fullname = s_util.from_len_pre_f(f)
            key = int.from_bytes(f.read(4))
            klass = pydoc.locate(fullname)
            assert(isinstance(klass, type))

            #TODO: should we just stomp on the registry?
            # if this doesn't match we might have a problem as it implies logic
            # has changed since we saved
            assert(self._class_key_lookup[klass] == key)

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

    def save_object(self, obj:Any, f:io.BufferedWriter) -> int:
        return self._save_register[type(obj)].save(obj, f)

    def load_object(self, klass:type, f:io.BufferedReader) -> Any:
        return self._save_register[klass].load(f)

    def save(self, gamestate:core.Gamestate) -> str:
        save_filename = self._gen_save_filename()
        save_filename = os.path.join(self._save_path, save_filename)
        bytes_written = 0
        with open(save_filename, "wb") as save_file:
            # save class -> key registration
            self._save_registry(save_file)

            # save the simulator which will recursively save everything
            bytes_written += self.save_object(gamestate, save_file)

        return save_filename

    def load(self, save_filename:str) -> core.Gamestate:
        with open(save_filename, "rb") as save_file:
            # load the class -> key registration
            self._load_registry(save_file)

            gamestate = self.load_object(core.Gamestate, save_file)

            return gamestate

"""
no reason to save Simulator
class SimSaver(Saver[sim.Simulator]):
    def save(self, sim:sim.Simulator, f:io.BufferedWriter) -> int:
        bytes_written = 0
        #TODO: save simulator fields we care about
        bytes_written += self.save_game.save_object(sim.gamestate, f)
        return bytes_written

    def load(self, f:io.BufferedReader) -> sim.Simulator:
        sim = sim.Simulator()
        sim.gamestate = self.save_game.load_object(core.Gamestate, f)
        return sim
"""

class GamestateSaver(Saver[core.Gamestate]):
    def save(self, gamestate:core.Gamestate, f:io.BufferedWriter) -> int:
        bytes_written = 0

        # simple fields
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
        pchain_bytes = serialize_econ_sim.save_production_chain(gamestate.production_chain)
        s_util.size_to_f(len(pchain_bytes), f)
        bytes_written += f.write(pchain_bytes)

        # entities, including context
        bytes_written += s_util.size_to_f(len(gamestate.entities), f)
        for entity in gamestate.entities.values():
            bytes_written += self.save_game.save_object(entity, f)
            bytes_written += self.save_game.save_object(entity.context, f)

        # sectors
        # save the sector ids in the right order
        # this gives us enough info to reconstruct sectors dict, sector_idx,
        # sector_spatial by pulling desired entity from the entity store
        bytes_written += s_util.size_to_f(len(gamestate.sector_ids), f)
        for sector_id in gamestate.sector_ids:
            s_util.uuid_to_f(sector_id, f)
        bytes_written += s_util.matrix_to_f(gamestate.sector_edges, f)

        # econ agents
        bytes_written += s_util.size_to_f(len(gamestate.econ_agents), f)
        for entity_id, agent in gamestate.econ_agents.items():
            bytes_written += s_util.uuid_to_f(entity_id, f)
            bytes_written += s_util.uuid_to_f(agent.entity_id, f)

        #TODO: task lists
        #TODO: starfields

        # entity destroy list
        bytes_written += s_util.size_to_f(len(gamestate.entity_destroy_list), f)
        for entity in gamestate.entity_destroy_list:
            bytes_written += s_util.uuid_to_f(entity.entity_id, f)

        bytes_written += s_util.size_to_f(len(gamestate.last_colliders), f)
        for colliders in gamestate.last_colliders:
            bytes_written += s_util.to_len_pre_f(colliders, f)

        return bytes_written

    def load(self, f:io.BufferedReader) -> core.Gamestate:
        gamestate = core.Gamestate()

        # simple fields
        gamestate.random = s_util.random_state_from_f(f)
        gamestate.base_date = datetime.datetime.fromisoformat(s_util.from_len_pre_f(f))
        gamestate.timestamp = s_util.float_from_f(f)
        gamestate.desired_dt = s_util.float_from_f(f)
        gamestate.dt = gamestate.desired_dt
        gamestate.min_tick_sleep = s_util.float_from_f(f)
        gamestate.ticks = s_util.int_from_f(f)
        player_id = s_util.uuid_from_f(f)

        pchain_bytes_count = s_util.size_from_f(f)
        pchain_bytes = f.read(pchain_bytes_count)
        gamestate.production_chain = serialize_econ_sim.load_production_chain(pchain_bytes)

        # load entities, including context
        count = s_util.size_from_f(f)
        for _ in range(count):
            entity = self.save_game.load_object(core.Entity, f)
            # entities should have registered themselves with the gamestate,
            # getting a fresh, empty context
            context = self.save_game.load_object(narrative.EventContext, f)
            for k,v in context._to_dict():
                entity.context.set_flag(k,v)

        # needed to wait for entities to be loaded before we could get the
        # actual player entity
        player = gamestate.entities[player_id]
        assert(isinstance(player, core.Player))
        gamestate.player = player

        # sectors
        # load sector ids and sector edges from file
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
        count = s_util.size_from_f(f)
        for _ in range(count):
            entity_id = s_util.uuid_from_f(f)
            gamestate.destroy_entity(gamestate.entities[entity_id])

        # last colliders
        count = s_util.size_from_f(f)
        last_colliders = set()
        for _ in range(count):
            last_colliders.add(s_util.from_len_pre_f(f))
        gamestate.last_colliders = last_colliders

        return gamestate

class EntitySaver(Saver[core.Entity]):
    def save(self, entity:core.Entity, f:io.BufferedWriter) -> int:
        i = 0

        # save key so we know what type of entity to load!
        i += f.write(self.save_game.key_from_class(type(entity)).to_bytes(4))

        # dispatch to specific entity type to save the rest
        i += self.save_game.save_object(entity, f)

        # save some common fields
        i += s_util.uuid_to_f(entity.entity_id, f)
        i += s_util.to_len_pre_f(entity.name, f, 2)
        i += s_util.to_len_pre_f(entity.description, f, 2)
        i += f.write(struct.pack(">f", entity.created_at))
        # note: gamestate saves our entity context

        return i

    def load(self, f:io.BufferedReader) -> core.Entity:
        # read entity type
        klass = self.save_game.class_from_key(int.from_bytes(f.read(4)))

        # dispatch to specific type
        entity = self.save_game.load_object(klass, f)

        #TODO: is this how we want to get the generic fields into the entity?
        # alternatively we could read these and make them available to the
        # class specific loader somehow
        # read common fields
        entity.entity_id = s_util.uuid_from_f(f)
        entity.name = s_util.from_len_pre_f(f)
        entity.description = s_util.from_len_pre_f(f)
        entity.created_at = struct.unpack(">f", f.read(4))
        # note: gamestate loads our context

        return entity
