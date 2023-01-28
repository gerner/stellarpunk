""" Test cases for director and narrative package in general """

import enum

from stellarpunk.narrative import rule_parser, director

class ET(enum.IntEnum):
    start_game = enum.auto()

class CK(enum.IntEnum):
    is_player = enum.auto()
    stuff = enum.auto()
    foo = enum.auto()
    bar = enum.auto()
    baz = enum.auto()

class A(enum.IntEnum):
    broadcast = enum.auto()
    other_stuff = enum.auto()
    message = enum.auto()

def test_parse_criteria():
    criteria = rule_parser.parse_criteria("1 <= is_player <= 1", {x.name: x.value for x in CK})
    assert isinstance(criteria, director.FlagCriteria)
    assert criteria.fact == CK.is_player
    assert criteria.low == 1
    assert criteria.high == 1

    criteria = rule_parser.parse_criteria("4 <= $foo.bar <= 15", {x.name: x.value for x in CK})
    assert isinstance(criteria, director.EntityCriteria)
    assert criteria.entity_fact == CK.foo
    assert criteria.sub_fact == CK.bar
    assert criteria.low == 4
    assert criteria.high == 15

def test_parse_eval():
    test_config = """
        [start_game_help]
        type = "start_game"
        priority = 0
        criteria = ["1 <= is_player <= 1"]
        [[start_game_help.actions]]
        _action = "broadcast"
        message = "what up gang?"
        delay = 500
        [[start_game_help.actions]]
        _action = "message"
        message = "{recipient} how ya doin?"
        recipient = "PLAYER"
        sender = "CHARACTER"

        [other_rule]
        type = "start_game"
        priority = 1
        criteria = [ "0 <= is_player <= 0", "5 <= stuff <= 10", "4 <= $foo.bar <= 15", "21 <= $bar.baz <= 21"]
        [[other_rule.actions]]
        _action = "message"
        message = "what up fool?"
    """

    d = rule_parser.loads(test_config, {x.name: x.value for x in ET}, {x.name: x.value for x in CK}, {x.name: x.value for x in A})

    event_context = director.context({CK.foo: 100, CK.bar: 200})
    entity_context = {
        100: director.context({CK.bar: 7}),
        200: director.context({CK.baz: 21}),
    }

    event = director.Event(
        ET.start_game,
        event_context,
        entity_context,
        {"awesome": "sauce"}
    )

    char_a = director.context({CK.is_player: 1})
    char_b = director.context({})
    char_c = director.context({CK.stuff: 3})
    char_d = director.context({CK.stuff: 7})
    char_e = director.context({CK.stuff: 12})

    actions = d.evaluate(
        event,
        [
            director.CharacterCandidate(char_a, "alice"),
            director.CharacterCandidate(char_b, "bob"),
            director.CharacterCandidate(char_c, "charlie"),
            director.CharacterCandidate(char_d, "doug"),
            director.CharacterCandidate(char_e, "emma"),
        ]
    )

    assert len(actions) == 3
    assert actions[0].action_id == A.broadcast
    assert actions[0].character_candidate.data == "alice"
    assert actions[1].action_id == A.message
    assert actions[1].character_candidate.data == "alice"
    assert actions[2].action_id == A.message
    assert actions[2].character_candidate.data == "doug"
