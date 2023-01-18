""" Test cases for events, EventManager, DialogManager, etc. """

import uuid

import pytest

from stellarpunk import core, events, dialog, predicates

class EventMock(events.Event):
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

def test_event_tick(gamestate, player):
    event_manager = events.EventManager()
    event_manager.initialize(gamestate)
    test_event = EventMock()
    event_manager.events = [test_event]

    assert test_event.event_id not in player.flags
    assert len(player.messages) == 0

    event_manager.tick()

    assert test_event.event_id in player.flags
    assert len(player.messages) == 1
    assert player.messages[EventMock.message_entity_id].reply_dialog.dialog_id == "dialog_demo"

def test_dialog_manager(gamestate, player):
    dialog_graph = dialog.load_dialog("dialog_demo")
    dialog_manager = events.DialogManager(dialog_graph, gamestate, gamestate.player)

    assert "demo_event_ack" not in player.flags

    dialog_manager.do_node()

    assert "demo_event_ack" in player.flags
    assert player.flags["demo_event_ack"] == gamestate.timestamp

    dialog_manager.node.node_id == dialog_graph.root_id

    assert len(dialog_manager.choices) == 3

    choice = dialog_manager.choices[1]

    assert "demo_event_longchoice" not in player.flags
    dialog_manager.choose(choice)
    assert "demo_event_longchoice" in player.flags
    assert dialog_manager.node.node_id == choice.node_id

    with pytest.raises(ValueError):
        dialog_manager.choose(choice)

def test_load_criteria_good():
    criteria1 = events.load_criteria("foo")
    assert isinstance(criteria1, events.FlagCriteria)
    assert criteria1.flag == "foo"

    criteria1 = events.load_criteria("(foo)")
    assert isinstance(criteria1, events.FlagCriteria)
    assert criteria1.flag == "foo"

    criteria2 = events.load_criteria("!foo")
    assert isinstance(criteria2, predicates.Negation)
    assert isinstance(criteria2.inner, events.FlagCriteria)

    criteria3 = events.load_criteria("foo | bar")
    assert isinstance(criteria3, predicates.Disjunction)
    assert isinstance(criteria3.a, events.FlagCriteria)
    assert criteria3.a.flag == "foo"
    assert isinstance(criteria3.b, events.FlagCriteria)
    assert criteria3.b.flag == "bar"

    criteria4 = events.load_criteria("foo & bar")
    assert isinstance(criteria4, predicates.Conjunction)
    assert isinstance(criteria4.a, events.FlagCriteria)
    assert criteria4.a.flag == "foo"
    assert isinstance(criteria4.b, events.FlagCriteria)
    assert criteria4.b.flag == "bar"

    criteria5= events.load_criteria("(foo & bar) | !blerf")
    assert isinstance(criteria5, predicates.Disjunction)
    assert isinstance(criteria5.a, predicates.Conjunction)
    assert isinstance(criteria5.a.a, events.FlagCriteria)
    assert criteria5.a.a.flag == "foo"
    assert isinstance(criteria5.a.b, events.FlagCriteria)
    assert criteria5.a.b.flag == "bar"
    assert isinstance(criteria5.b, predicates.Negation)
    assert isinstance(criteria5.b.inner, events.FlagCriteria)
    assert criteria5.b.inner.flag == "blerf"

    # equiv to foo & (bar | !blerf)
    criteria6 = events.load_criteria("foo & bar | !blerf")
    assert isinstance(criteria6, predicates.Conjunction)
    assert isinstance(criteria6.a, events.FlagCriteria)
    assert criteria6.a.flag == "foo"
    assert isinstance(criteria6.b, predicates.Disjunction)
    assert isinstance(criteria6.b.a, events.FlagCriteria)
    assert criteria6.b.a.flag == "bar"
    assert isinstance(criteria6.b.b, predicates.Negation)
    assert isinstance(criteria6.b.b.inner, events.FlagCriteria)
    assert criteria6.b.b.inner.flag == "blerf"

    criteria7 = events.load_criteria("foo & (bar | !blerf)")
    assert isinstance(criteria7, predicates.Conjunction)
    assert isinstance(criteria7.a, events.FlagCriteria)
    assert criteria7.a.flag == "foo"
    assert isinstance(criteria7.b, predicates.Disjunction)
    assert isinstance(criteria7.b.a, events.FlagCriteria)
    assert criteria7.b.a.flag == "bar"
    assert isinstance(criteria7.b.b, predicates.Negation)
    assert isinstance(criteria7.b.b.inner, events.FlagCriteria)
    assert criteria7.b.b.inner.flag == "blerf"

    # make sure that negation doesn't grab junctions
    criteria8 = events.load_criteria("!foo & (blerf | bar)")
    assert isinstance(criteria8, predicates.Conjunction)
    assert isinstance(criteria8.a, predicates.Negation)
    assert isinstance(criteria8.b, predicates.Disjunction)

def test_load_criteria_cmp():
    criteria1 = events.load_criteria("CMP(NOW > 2)")
    assert isinstance(criteria1, events.CompareCriteria)
    assert criteria1.left == "NOW"
    assert criteria1.op == ">"
    assert not criteria1.relative
    assert criteria1.right == 2.

def test_load_criteria_bad():
    with pytest.raises(ValueError):
        events.load_criteria("foo & bar blerf")

    with pytest.raises(ValueError):
        events.load_criteria("(foo & bar")

    with pytest.raises(ValueError):
        events.load_criteria("!")

    with pytest.raises(ValueError):
        events.load_criteria("foo)")
