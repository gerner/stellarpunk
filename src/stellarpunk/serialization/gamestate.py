import uuid
import io
import datetime
import abc
from typing import Any

import numpy as np

from stellarpunk import core

from . import serialize_econ_sim, save_game, util as s_util

class GamestateSaver(save_game.Saver[core.Gamestate]):
    def save(self, gamestate:core.Gamestate, f:io.IOBase) -> int:
        bytes_written = 0

        # simple fields
        bytes_written += s_util.debug_string_w("simple fields", f)
        bytes_written += s_util.random_state_to_f(gamestate.random, f)
        bytes_written += s_util.to_len_pre_f(gamestate.base_date.isoformat(), f)
        bytes_written += s_util.float_to_f(gamestate.timestamp, f)
        #bytes_written += s_util.float_to_f(gamestate.desired_dt, f)
        # no need to save dt, we should reload with desired dt
        #bytes_written += s_util.float_to_f(gamestate.dt, f)
        #bytes_written += s_util.float_to_f(gamestate.min_tick_sleep, f)
        bytes_written += s_util.int_to_f(gamestate.ticks, f)

        # in general we can't save while force paused since we force pause
        # because we're at risk of putting the gamestate in an inconsistent
        # state (e.g. middle of dialog)
        assert(not gamestate.is_force_paused())

        bytes_written += s_util.uuid_to_f(gamestate.player.entity_id, f)
        #TODO: should we save counters?

        # production chain
        bytes_written += s_util.debug_string_w("production chain", f)
        pchain_bytes = serialize_econ_sim.save_production_chain(gamestate.production_chain)
        bytes_written += s_util.size_to_f(len(pchain_bytes), f)
        bytes_written += f.write(pchain_bytes)

        # entities
        bytes_written += s_util.debug_string_w("entities", f)
        bytes_written += s_util.size_to_f(len(gamestate.entities), f)
        for entity in gamestate.entities.values():
            # we save as a generic entity which will handle its own dispatch
            bytes_written += self.save_game.save_object(entity, f, klass=core.Entity)

        # orders
        bytes_written += s_util.debug_string_w("orders", f)
        bytes_written += s_util.size_to_f(len(gamestate.orders), f)
        for order in gamestate.orders.values():
            # we save as a generic order which will handle its own dispatch
            bytes_written += self.save_game.save_object(order, f, klass=core.Order)


        # effects
        bytes_written += s_util.debug_string_w("effects", f)
        bytes_written += s_util.size_to_f(len(gamestate.effects), f)
        for effect in gamestate.effects.values():
            # we save as a generic effect which will handle its own dispatch
            bytes_written += self.save_game.save_object(effect, f, klass=core.Effect)

        # sectors
        # save the sector ids in the right order
        # this gives us enough info to reconstruct sectors dict, sector_idx,
        # sector_spatial by pulling desired entity from the entity store
        bytes_written += s_util.debug_string_w("sector ids", f)
        bytes_written += s_util.size_to_f(len(gamestate.sector_ids), f)
        for sector_id in gamestate.sector_ids:
            s_util.uuid_to_f(sector_id, f)
        bytes_written += s_util.matrix_to_f(gamestate.sector_edges, f)

        # econ agents
        bytes_written += s_util.debug_string_w("econ agents", f)
        bytes_written += s_util.size_to_f(core.EconAgent._next_id, f)
        bytes_written += s_util.size_to_f(len(gamestate.econ_agents), f)
        for entity_id, agent in gamestate.econ_agents.items():
            bytes_written += s_util.uuid_to_f(entity_id, f)
            bytes_written += s_util.uuid_to_f(agent.entity_id, f)

        #TODO: task lists
        # tricky bit here is that orders, effects, agenda are actually saved
        # elsewhere and we've just got references to them in the schedule. we
        # need to somehow fetch those references somehow. however, there's not
        # global repository of them and they don't have some identifier we can
        # use.

        # order schedule
        # effect schedule
        # agenda schedule
        # task schedule

        # starfields
        bytes_written += s_util.debug_string_w("starfields", f)
        bytes_written += s_util.size_to_f(len(gamestate.starfield), f)
        for starfield in gamestate.starfield:
            bytes_written += self.save_game.save_object(starfield, f)
        bytes_written += s_util.size_to_f(len(gamestate.sector_starfield), f)
        for starfield in gamestate.starfield:
            bytes_written += self.save_game.save_object(starfield, f)
        bytes_written += s_util.size_to_f(len(gamestate.portrait_starfield), f)
        for starfield in gamestate.starfield:
            bytes_written += self.save_game.save_object(starfield, f)

        # entity destroy list
        bytes_written += s_util.debug_string_w("entity destroy list", f)
        bytes_written += s_util.size_to_f(len(gamestate.entity_destroy_list), f)
        for entity in gamestate.entity_destroy_list:
            bytes_written += s_util.uuid_to_f(entity.entity_id, f)

        bytes_written += s_util.debug_string_w("last colliders", f)
        bytes_written += s_util.size_to_f(len(gamestate.last_colliders), f)
        for colliders in gamestate.last_colliders:
            bytes_written += s_util.to_len_pre_f(colliders, f)

        bytes_written += s_util.debug_string_w("gamestate done", f)
        return bytes_written

    def load(self, f:io.IOBase, load_context:save_game.LoadContext) -> core.Gamestate:
        #TODO: make sure that load_context.gamestate is this one?
        gamestate = core.Gamestate()
        load_context.gamestate = gamestate

        # simple fields
        s_util.debug_string_r("simple fields", f)
        gamestate.random = s_util.random_state_from_f(f)
        gamestate.base_date = datetime.datetime.fromisoformat(s_util.from_len_pre_f(f))
        gamestate.timestamp = s_util.float_from_f(f)
        #gamestate.desired_dt = s_util.float_from_f(f)
        #gamestate.dt = gamestate.desired_dt
        #gamestate.min_tick_sleep = s_util.float_from_f(f)
        gamestate.ticks = s_util.int_from_f(f)
        player_id = s_util.uuid_from_f(f)

        s_util.debug_string_r("production chain", f)
        pchain_bytes_count = s_util.size_from_f(f)
        pchain_bytes = f.read(pchain_bytes_count)
        gamestate.production_chain = serialize_econ_sim.load_production_chain(pchain_bytes)

        # load entities
        s_util.debug_string_r("entities", f)
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

        # load orders
        s_util.debug_string_r("orders", f)
        count = s_util.size_from_f(f)
        for i in range(count):
            order = self.save_game.load_object(core.Order, f, load_context)
            # orders should have registered themselves with the gamestate,

        # load effects
        s_util.debug_string_r("effects", f)
        count = s_util.size_from_f(f)
        for i in range(count):
            effect = self.save_game.load_object(core.Effect, f, load_context)
            # effects should have registered themselves with the gamestate,

        # sectors
        # load sector ids and sector edges from file
        s_util.debug_string_r("sector ids", f)
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
        s_util.debug_string_r("econ agents", f)
        core.EconAgent._next_id = s_util.size_from_f(f)
        count = s_util.size_from_f(f)
        for i in range(count):
            entity_id = s_util.uuid_from_f(f)
            agent_id = s_util.uuid_from_f(f)
            agent = gamestate.entities[agent_id]
            assert(isinstance(agent, core.EconAgent))
            gamestate.representing_agent(entity_id, agent)

        #TODO: task lists
        #TODO: event manager: event_queue and action schedule

        # starfields
        s_util.debug_string_r("starfields", f)
        count = s_util.size_from_f(f)
        for i in range(count):
            starfield = self.save_game.load_object(core.StarfieldLayer, f, load_context)
            gamestate.starfield.append(starfield)
        count = s_util.size_from_f(f)
        for i in range(count):
            starfield = self.save_game.load_object(core.StarfieldLayer, f, load_context)
            gamestate.sector_starfield.append(starfield)
        count = s_util.size_from_f(f)
        for i in range(count):
            starfield = self.save_game.load_object(core.StarfieldLayer, f, load_context)
            gamestate.portrait_starfield.append(starfield)

        # entity destroy list
        s_util.debug_string_r("entity destroy list", f)
        count = s_util.size_from_f(f)
        for i in range(count):
            entity_id = s_util.uuid_from_f(f)
            gamestate.destroy_entity(gamestate.entities[entity_id])

        # last colliders
        s_util.debug_string_r("last colliders", f)
        count = s_util.size_from_f(f)
        last_colliders = set()
        for i in range(count):
            last_colliders.add(s_util.from_len_pre_f(f))
        gamestate.last_colliders = last_colliders


        s_util.debug_string_r("gamestate done", f)

        return gamestate

class StarfieldLayerSaver(save_game.Saver[core.StarfieldLayer]):
    def save(self, starfield:core.StarfieldLayer, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.floats_to_f(starfield.bbox, f)
        bytes_written += s_util.float_to_f(starfield.zoom, f)

        # num_stars and density will get computed for us on load
        bytes_written += s_util.size_to_f(len(starfield._star_list), f)
        for loc, size, spectral_class in starfield._star_list:
            bytes_written += s_util.float_to_f(loc[0], f)
            bytes_written += s_util.float_to_f(loc[1], f)
            bytes_written += s_util.float_to_f(size, f)
            bytes_written += s_util.int_to_f(int(spectral_class), f)

        return bytes_written

    def load(self, f:io.IOBase, load_context:save_game.LoadContext) -> core.StarfieldLayer:
        bbox_list = s_util.floats_from_f(f)
        assert(len(bbox_list) == 4)
        bbox:tuple[float, float, float, float] = tuple(bbox_list) # type: ignore
        zoom = s_util.float_from_f(f)
        starfield = core.StarfieldLayer(bbox, zoom)

        # num_stars and density will get computed for us by add_star
        count = s_util.size_from_f(f)
        for i in range(count):
            loc = (s_util.float_from_f(f), s_util.float_from_f(f))
            size = s_util.float_from_f(f)
            spectral_class = s_util.int_from_f(f)
            starfield.add_star(loc, size, spectral_class)

        return starfield

class EntitySaver[EntityType: core.Entity](save_game.Saver[EntityType], abc.ABC):

    @abc.abstractmethod
    def _save_entity(self, entity:EntityType, f:io.IOBase) -> int: ...
    @abc.abstractmethod
    def _load_entity(self, f:io.IOBase, load_context:save_game.LoadContext, entity_id:uuid.UUID) -> EntityType: ...

    def save(self, entity:EntityType, f:io.IOBase) -> int:
        i = 0

        # common fields we need for entity creation
        s_util.debug_string_w("entity id", f)
        i += s_util.uuid_to_f(entity.entity_id, f)

        s_util.debug_string_w("dispatch", f)
        i += self._save_entity(entity, f)

        # save some common fields
        s_util.debug_string_w("common fields", f)
        i += s_util.to_len_pre_f(entity.name, f, 2)
        i += s_util.to_len_pre_f(entity.description, f, 2)
        i += s_util.float_to_f(entity.created_at, f)

        s_util.debug_string_w("context", f)
        context_dict = entity.context.to_dict()
        i += s_util.size_to_f(len(context_dict), f)
        for k,v in context_dict.items():
            i += s_util.int_to_f(k, f, 8)
            i += s_util.int_to_f(v, f, 8)

        return i

    def load(self, f:io.IOBase, load_context:save_game.LoadContext) -> EntityType:
        # fields we need for entity creation
        s_util.debug_string_r("entity id", f)
        entity_id = s_util.uuid_from_f(f)

        # type specific loading logic
        s_util.debug_string_r("dispatch", f)
        entity = self._load_entity(f, load_context, entity_id)

        #TODO: is this how we want to get the generic fields into the entity?
        # alternatively we could read these and make them available to the
        # class specific loader somehow
        # read common fields
        s_util.debug_string_r("common fields", f)
        entity.name = s_util.from_len_pre_f(f)
        entity.description = s_util.from_len_pre_f(f)
        entity.created_at = s_util.float_from_f(f)

        # creating the entity made an empty event context, we just have to
        # populate it
        s_util.debug_string_r("context", f)
        count = s_util.size_from_f(f)
        for _ in range(count):
            k = s_util.int_from_f(f, 8)
            v = s_util.int_from_f(f, 8)
            entity.context.set_flag(k, v)

        return entity

class NoneEntitySaver(EntitySaver[core.Entity]):
    def _save_entity(self, entity:core.Entity, f:io.IOBase) -> int:
        return 0
    def _load_entity(self, f:io.IOBase, load_context:save_game.LoadContext, entity_id:uuid.UUID) -> core.Entity:
        return core.Entity(load_context.gamestate, entity_id=entity_id)

