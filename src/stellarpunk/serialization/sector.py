import io
import uuid
from typing import Any

from stellarpunk import core

import cymunk # type: ignore

from . import save_game, util as s_util, gamestate as s_gamestate

class SectorSaver(s_gamestate.EntitySaver[core.Sector]):
    def _save_entity(self, sector:core.Sector, f:io.IOBase) -> int:
        bytes_written = 0

        #TODO: put more debug strings throughout

        # basic fields: loc, radius, culture
        bytes_written += s_util.matrix_to_f(sector.loc, f)
        bytes_written += s_util.float_to_f(sector.radius, f)
        bytes_written += s_util.to_len_pre_f(sector.culture, f)

        # entities. we'll reconstruct planets, etc. from entities
        bytes_written += s_util.uuids_to_f(sector.entities.keys(), f)

        # effects
        bytes_written += s_util.size_to_f(len(sector._effects), f)
        for effect in sector._effects:
            bytes_written += self.save_game.save_object(effect, f, klass=core.Effect)

        #TODO: collision observers

        # weather. we'll reconstruct weather index on load from weather regions
        bytes_written += s_util.size_to_f(len(sector._weathers), f)
        # we assume weathers are in compact index order (so first element is
        # index 0, second is 1, and so on)
        for weather in sector._weathers.values():
            bytes_written += self.save_game.save_object(weather, f)

        return bytes_written

    def _load_entity(self, f:io.IOBase, load_context:save_game.LoadContext, entity_id:uuid.UUID) -> core.Sector:
        #we can't fully load the sector until all entities have been loaded.

        # basic fields: loc, radius, culture
        loc = s_util.matrix_from_f(f)
        radius = s_util.float_from_f(f)
        culture = s_util.from_len_pre_f(f)

        # we can create a blank space here and let it be populated by
        # SectorEntity objects in post_load.
        # SectorEntity is responsible for saving/loading its own physics body
        # and shape
        sector = core.Sector(loc, radius, cymunk.Space(), load_context.gamestate, entity_id=entity_id, culture=culture)

        # entities. we'll reconstruct these in post load
        entities:list[uuid.UUID] = list(s_util.uuids_from_f(f))

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

    def post_load(self, sector:core.Sector, load_context:save_game.LoadContext, context:Any) -> None:
        entities:list[uuid.UUID] = context
        for entity_id in entities:
            entity = load_context.gamestate.entities[entity_id]
            assert(isinstance(entity, core.SectorEntity))
            sector.add_entity(entity)

class SectorWeatherRegionSaver(save_game.Saver[core.SectorWeatherRegion]):
    def save(self, weather:core.SectorWeatherRegion, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.matrix_to_f(weather.loc, f)
        bytes_written += s_util.float_to_f(weather.radius, f)
        bytes_written += s_util.float_to_f(weather.sensor_factor, f)
        return bytes_written

    def load(self, f:io.IOBase, load_context:save_game.LoadContext) -> core.SectorWeatherRegion:
        loc = s_util.matrix_from_f(f)
        radius = s_util.float_from_f(f)
        sensor_factor = s_util.float_from_f(f)
        weather = core.SectorWeatherRegion(loc, radius, sensor_factor)
        return weather

