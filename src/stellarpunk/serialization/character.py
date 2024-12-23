import uuid
import io
from typing import Any, Optional

from stellarpunk import core

from . import save_game, util as s_util, gamestate as s_gamestate

class PlayerSaver(s_gamestate.EntitySaver[core.Player]):
    def _save_entity(self, player:core.Player, f:io.IOBase) -> int:
        bytes_written = 0
        if player.character:
            bytes_written += s_util.int_to_f(1, f, blen=1)
            bytes_written += s_util.uuid_to_f(player.character.entity_id, f)
        else:
            bytes_written += s_util.int_to_f(0, f, blen=1)
        bytes_written += s_util.uuid_to_f(player.agent.entity_id, f)
        bytes_written += s_util.uuids_to_f(player.messages.keys(), f)
        return bytes_written

    def _load_entity(self, f:io.IOBase, load_context:save_game.LoadContext, entity_id:uuid.UUID) -> core.Player:
        has_character = s_util.int_from_f(f, blen=1)
        character_id:Optional[uuid.UUID] = None
        if has_character:
            character_id = s_util.uuid_from_f(f)
        agent_id = s_util.uuid_from_f(f)
        message_ids = s_util.uuids_from_f(f)
        player = core.Player(load_context.gamestate, entity_id=entity_id)
        load_context.register_post_load(player, (character_id, agent_id, message_ids))
        return player

    def post_load(self, player:core.Player, load_context:save_game.LoadContext, context:Any) -> None:
        context_tuple:tuple[Optional[uuid.UUID], uuid.UUID, list[uuid.UUID]] = context
        character_id, agent_id, messages = context_tuple
        #pull out fully loaded character
        if character_id:
            character = load_context.gamestate.entities[character_id]
            assert(isinstance(character, core.Character))
            player.character = character

        #pull out fully loaded econ agent
        agent = load_context.gamestate.entities[agent_id]
        assert(isinstance(agent, core.EconAgent))
        player.agent = agent

        #pull out fully loaded messages
        for message_id in messages:
            message = load_context.gamestate.entities[message_id]
            assert(isinstance(message, core.Message))
            player.messages[message_id] = message

class CharacterSaver(s_gamestate.EntitySaver[core.Character]):
    def _save_entity(self, character:core.Character, f:io.IOBase) -> int:
        bytes_written = 0
        if character.location is not None:
            bytes_written += s_util.int_to_f(1, f, blen=1)
            bytes_written += s_util.uuid_to_f(character.location.entity_id, f)
        else:
            bytes_written += s_util.int_to_f(0, f, blen=1)
        bytes_written += s_util.to_len_pre_f(character.portrait.sprite_id, f)
        bytes_written += s_util.float_to_f(character.balance, f)
        bytes_written += s_util.uuids_to_f(list(x.entity_id for x in character.assets), f)
        bytes_written += s_util.uuid_to_f(character.home_sector_id, f)

        bytes_written += s_util.size_to_f(len(character.agenda), f)
        for agendum in character.agenda:
            bytes_written += s_util.uuid_to_f(agendum.agenda_id, f)

        #TODO: observers
        return bytes_written

    def _load_entity(self, f:io.IOBase, load_context:save_game.LoadContext, entity_id:uuid.UUID) -> core.Character:
        has_location = s_util.int_from_f(f, blen=1)
        location_id:Optional[uuid.UUID] = None
        if has_location:
            location_id = s_util.uuid_from_f(f)
        sprite_id = s_util.from_len_pre_f(f)
        balance = s_util.float_from_f(f)
        asset_ids = s_util.uuids_from_f(f)
        home_sector_id = s_util.uuid_from_f(f)

        agenda_ids = []
        count = s_util.size_from_f(f)
        for i in range(count):
            agenda_ids.append(s_util.uuid_from_f(f))

        character = core.Character(
            load_context.generator.sprite_store[sprite_id],
            load_context.gamestate,
            entity_id=entity_id,
            home_sector_id=home_sector_id
        )
        character.balance = balance
        load_context.register_post_load(character, (location_id, asset_ids, agenda_ids))
        return character

    def post_load(self, character:core.Character, load_context:save_game.LoadContext, context:Any) -> None:
        context_data:tuple[Optional[uuid.UUID], list[uuid.UUID], list[uuid.UUID]] = context
        location_id, asset_ids, agenda_ids = context

        if location_id is not None:
            location = load_context.gamestate.entities[location_id]
            assert(isinstance(location, core.SectorEntity))
            character.location = location

        for asset_id in asset_ids:
            asset = load_context.gamestate.entities[asset_id]
            assert(isinstance(asset, core.Asset))
            character.assets.append(asset)

        #TODO: agenda
        #for agenda_id in agenda_ids:
        #    agendum = load_context.gamestate.agenda[agenda_id]
        #    character.agenda.append(agenda_id)

class MessageSaver(s_gamestate.EntitySaver[core.Message]):
    def _save_entity(self, message:core.Message, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.int_to_f(message.message_id, f)
        bytes_written += s_util.to_len_pre_f(message.subject, f)
        bytes_written += s_util.to_len_pre_f(message.message, f)
        bytes_written += s_util.float_to_f(message.timestamp, f)
        bytes_written += s_util.uuid_to_f(message.reply_to, f)
        if message.replied_at is not None:
            bytes_written += s_util.int_to_f(1, f, blen=1)
            bytes_written += s_util.float_to_f(message.replied_at, f)
        else:
            bytes_written += s_util.int_to_f(0, f, blen=1)

        return bytes_written

    def _load_entity(self, f:io.IOBase, load_context:save_game.LoadContext, entity_id:uuid.UUID) -> core.Message:
        message_id = s_util.int_from_f(f)
        subject = s_util.from_len_pre_f(f)
        message_body = s_util.from_len_pre_f(f)
        timestamp = s_util.float_from_f(f)
        reply_to = s_util.uuid_from_f(f)
        message = core.Message(message_id, subject, message_body, timestamp, reply_to, load_context.gamestate, entity_id=entity_id)
        has_replied_at = s_util.int_from_f(f, blen=1) == 1
        if has_replied_at:
            message.replied_at = s_util.float_from_f(f)

        return message
