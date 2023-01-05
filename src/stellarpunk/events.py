""" Manages Events """

from typing import List, Any, Sequence
import logging

from stellarpunk import core, config, util, dialog

class DemoEvent(core.Event):
    def __init__(self) -> None:
        super().__init__("demo_event")

    def is_relevant(self, gamestate:core.Gamestate, player:core.Player) -> bool:
        return (
            (gamestate.timestamp > 5. and self.event_id not in player.flags) or
            (
                self.event_id in player.flags and
                f'{self.event_id}_ack' not in player.flags and
                (gamestate.timestamp - player.flags[self.event_id] > 5.)
            )
        )

    def act(self, gamestate:core.Gamestate, player:core.Player) -> None:
        if self.event_id not in player.flags:
            player.send_message(core.Message(
                "what's up? "*20,
                gamestate.timestamp,
                reply_to=player.character,
                reply_dialog=dialog.load_dialog("dialog_demo"),
            ))
            player.set_flag(self.event_id, gamestate.timestamp)
        elif f'{self.event_id}_ack' not in player.flags:
            player.send_message(core.Message(
                "stop ignoring me! "*20,
                gamestate.timestamp,
                reply_to=player.character,
                reply_dialog=dialog.load_dialog("dialog_demo"),
            ))
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

class DialogManager:
    def __init__(self, dialog:dialog.DialogGraph, gamestate:core.Gamestate, player:core.Player) -> None:
        self.dialog = dialog
        self.current_id = dialog.root_id
        self.gamestate = gamestate
        self.player = player

    @property
    def node(self) -> dialog.DialogNode:
        return self.dialog.nodes[self.current_id]

    @property
    def choices(self) -> Sequence[dialog.DialogChoice]:
        return self.dialog.nodes[self.current_id].choices

    def do_node(self) -> None:
        for node_event_id in self.dialog.nodes[self.current_id].event_id:
            self.player.set_flag(node_event_id, self.gamestate.timestamp)

    def choose(self, choice:dialog.DialogChoice) -> None:
        if choice not in self.choices:
            raise ValueError(f'given choice is not current node {self.current_id} choices')

        for event_id in choice.event_id:
            self.player.set_flag(event_id, self.gamestate.timestamp)
        self.current_id = choice.node_id
