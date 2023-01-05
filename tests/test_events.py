""" Test cases for events, EventManager, DialogManager, etc. """

import uuid

import pytest

from stellarpunk import core, events, dialog

class EventMock(core.Event):
    message_entity_id = uuid.UUID('6b34e2b3-f6c0-4ee4-aaf1-d4e4cec9d1a6')

    def __init__(self) -> None:
        super().__init__("test_event")

    def is_relevant(self, gamestate:core.Gamestate, player:core.Player) -> bool:
        return self.event_id not in player.flags

    def act(self, gamestate:core.Gamestate, player:core.Player) -> None:
        player.send_message(core.Message(
            "what's up? "*20,
            gamestate.timestamp,
            reply_to=player.character,
            reply_dialog=dialog.load_dialog("dialog_demo"),
            entity_id=EventMock.message_entity_id,
        ))
        player.set_flag(self.event_id, gamestate.timestamp)

def test_event_tick(gamestate):
    event_manager = events.EventManager()
    event_manager.initialize(gamestate)
    test_event = EventMock()
    event_manager.events = [test_event]

    player = gamestate.player

    assert test_event.event_id not in player.flags
    assert len(player.messages) == 0

    event_manager.tick()

    assert test_event.event_id in player.flags
    assert len(player.messages) == 1
    assert player.messages[EventMock.message_entity_id].reply_dialog.dialog_id == "dialog_demo"

def test_dialog_manager(gamestate):
    dialog_graph = dialog.load_dialog("dialog_demo")
    dialog_manager = events.DialogManager(dialog_graph, gamestate, gamestate.player)

    assert "demo_event_ack" not in gamestate.player.flags

    dialog_manager.do_node()

    assert "demo_event_ack" in gamestate.player.flags
    assert gamestate.player.flags["demo_event_ack"] == gamestate.timestamp

    dialog_manager.node.node_id == dialog_graph.root_id

    assert len(dialog_manager.choices) == 3

    choice = dialog_manager.choices[1]

    assert "demo_event_longchoice" not in gamestate.player.flags
    dialog_manager.choose(choice)
    assert "demo_event_longchoice" in gamestate.player.flags
    assert dialog_manager.node.node_id == choice.node_id

    with pytest.raises(ValueError):
        dialog_manager.choose(choice)
