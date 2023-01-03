""" Manages Events """

from typing import List, Any
import logging

from stellarpunk import core, config, util

class DemoEvent(core.Event):
    def __init__(self) -> None:
        super().__init__("demo_event")

    def is_relevant(self, gamestate:core.Gamestate, player:core.Player) -> bool:
        return gamestate.timestamp > 5. and self.event_id not in player.flags

    def act(self, gamestate:core.Gamestate, player:core.Player) -> None:
        player.send_message(core.Message("what's up?", gamestate.timestamp))
        player.set_flag(self.event_id, gamestate.timestamp)

class EventManager:
    def __init__(self) -> None:
        self.logger = logging.getLogger(util.fullname(self))
        self.gamestate:core.Gamestate = None # type: ignore[assignment]
        self.events:List[core.Event] = []

    def initialize(self, gamestate:core.Gamestate) -> None:
        self.gamestate = gamestate
        self.events.append(DemoEvent())

    def tick(self) -> None:
        # check for relevant events and process them
        for event in self.events:
            if event.is_relevant(self.gamestate, self.gamestate.player):
                self.logger.debug(f'starting event {event.event_id}')
                event.act(self.gamestate, self.gamestate.player)
