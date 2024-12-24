import io
import uuid
import abc
from typing import Any

from stellarpunk import core, econ
from stellarpunk.core import sector_entity

from . import save_game, util as s_util, gamestate as s_gamestate

class EconAgentSaver[EconAgent: core.EconAgent](s_gamestate.EntitySaver[EconAgent], abc.ABC):
    @abc.abstractmethod
    def _save_econ_agent(self, econ_agent:EconAgent, f:io.IOBase) -> int: ...
    @abc.abstractmethod
    def _load_econ_agent(self, f:io.IOBase, load_context:save_game.LoadContext, entity_id:uuid.UUID) -> EconAgent: ...

    def _save_entity(self, econ_agent:EconAgent, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += self._save_econ_agent(econ_agent, f)
        bytes_written += s_util.int_to_f(econ_agent.agent_id, f)
        return bytes_written

    def _load_entity(self, f:io.IOBase, load_context:save_game.LoadContext, entity_id:uuid.UUID) -> EconAgent:
        econ_agent = self._load_econ_agent(f, load_context, entity_id)
        econ_agent.agent_id = s_util.int_from_f(f)
        return econ_agent

class PlayerAgentSaver(EconAgentSaver[econ.PlayerAgent]):
    def _save_econ_agent(self, econ_agent:econ.PlayerAgent, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.uuid_to_f(econ_agent.player.entity_id, f)
        return bytes_written

    def _load_econ_agent(self, f:io.IOBase, load_context:save_game.LoadContext, entity_id:uuid.UUID) -> econ.PlayerAgent:
        player_entity_id = s_util.uuid_from_f(f)
        player_agent = econ.PlayerAgent(load_context.gamestate, entity_id=entity_id)
        load_context.register_post_load(player_agent, player_entity_id)
        return player_agent

    def post_load(self, player_agent:econ.PlayerAgent, load_context:save_game.LoadContext, context:Any) -> None:
        player_id:uuid.UUID = context
        player = load_context.gamestate.entities[player_id]
        assert(isinstance(player, core.Player))
        player_agent.player = player

class StationAgentSaver(EconAgentSaver[econ.StationAgent]):
    def _save_econ_agent(self, econ_agent:econ.StationAgent, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.ints_to_f(econ_agent._buy_resources, f)
        bytes_written += s_util.ints_to_f(econ_agent._sell_resources, f)
        bytes_written += s_util.matrix_to_f(econ_agent._buy_price, f)
        bytes_written += s_util.matrix_to_f(econ_agent._sell_price, f)
        bytes_written += s_util.matrix_to_f(econ_agent._budget, f)
        bytes_written += s_util.uuid_to_f(econ_agent.station.entity_id, f)
        bytes_written += s_util.uuid_to_f(econ_agent.owner.entity_id, f)
        bytes_written += s_util.uuid_to_f(econ_agent.character.entity_id, f)

        return bytes_written

    def _load_econ_agent(self, f:io.IOBase, load_context:save_game.LoadContext, entity_id:uuid.UUID) -> econ.StationAgent:
        agent = econ.StationAgent(load_context.gamestate.production_chain, load_context.gamestate, entity_id=entity_id)
        agent._buy_resources = list(s_util.ints_from_f(f))
        agent._sell_resources = list(s_util.ints_from_f(f))
        agent._buy_price = s_util.matrix_from_f(f)
        agent._sell_price = s_util.matrix_from_f(f)
        agent._budget = s_util.matrix_from_f(f)

        station_id = s_util.uuid_from_f(f)
        owner_id = s_util.uuid_from_f(f)
        character_id = s_util.uuid_from_f(f)
        load_context.register_post_load(agent, (station_id, owner_id, character_id))

        return agent

    def post_load(self, agent:econ.StationAgent, load_context:save_game.LoadContext, context:Any) -> None:
        context_tuple:tuple[uuid.UUID, uuid.UUID, uuid.UUID] = context
        station_id, owner_id, character_id = context_tuple
        station = load_context.gamestate.entities[station_id]
        assert(isinstance(station, sector_entity.Station | sector_entity.Planet))
        agent.station = station
        owner = load_context.gamestate.entities[owner_id]
        assert(isinstance(owner, core.Character))
        agent.owner = owner
        character = load_context.gamestate.entities[character_id]
        assert(isinstance(character, core.Character))
        agent.character = character


class ShipTraderAgentSaver(EconAgentSaver[econ.ShipTraderAgent]):
    def _save_econ_agent(self, econ_agent:econ.ShipTraderAgent, f:io.IOBase) -> int:
        bytes_written = 0
        bytes_written += s_util.uuid_to_f(econ_agent.ship.entity_id, f)
        bytes_written += s_util.uuid_to_f(econ_agent.character.entity_id, f)
        return bytes_written

    def _load_econ_agent(self, f:io.IOBase, load_context:save_game.LoadContext, entity_id:uuid.UUID) -> econ.ShipTraderAgent:
        agent = econ.ShipTraderAgent(load_context.gamestate, entity_id=entity_id)
        ship_id = s_util.uuid_from_f(f)
        character_id = s_util.uuid_from_f(f)
        load_context.register_post_load(agent, (ship_id, character_id))
        return agent

    def post_load(self, agent:econ.ShipTraderAgent, load_context:save_game.LoadContext, context:Any) -> None:
        context_tuple:tuple[uuid.UUID, uuid.UUID] = context
        ship_id, character_id = context_tuple
        ship = load_context.gamestate.entities[ship_id]
        assert(isinstance(ship, core.Ship))
        agent.ship = ship
        character = load_context.gamestate.entities[character_id]
        assert(isinstance(character, core.Character))
        agent.character = character

