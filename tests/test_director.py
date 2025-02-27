""" Test cases for director and narrative package in general """

import enum
from typing import Dict

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
    builder = director.CriteriaBuilder()
    rule_parser.parse_criteria("1 <= is_player <= 1", {x.name: x.value for x in CK}, builder)
    assert isinstance(builder.last_fact, director.FlagRef)
    assert builder.last_fact.fact == CK.is_player
    assert isinstance(builder.last_low, director.IntRef)
    assert builder.last_low.value == 1
    assert isinstance(builder.last_high, director.IntRef)
    assert builder.last_high.value == 1

    rule_parser.parse_criteria("is_player <= 1", {x.name: x.value for x in CK}, builder)
    assert isinstance(builder.last_low, director.FlagRef)
    assert builder.last_low.fact == CK.is_player
    assert isinstance(builder.last_fact, director.IntRef)
    assert builder.last_fact.value == 1
    assert isinstance(builder.last_high, director.IntRef)
    assert builder.last_high.value == rule_parser.POS_INF

    rule_parser.parse_criteria("4 <= $foo.bar <= 15", {x.name: x.value for x in CK}, builder)
    assert isinstance(builder.last_fact, director.EntityRef)
    assert builder.last_fact.entity_fact == CK.foo
    assert builder.last_fact.sub_fact == CK.bar
    assert isinstance(builder.last_low, director.IntRef)
    assert builder.last_low.value == 4
    assert isinstance(builder.last_high, director.IntRef)
    assert builder.last_high.value == 15

    rule_parser.parse_criteria("$bar.baz <= $foo.bar <= stuff", {x.name: x.value for x in CK}, builder)
    assert isinstance(builder.last_fact, director.EntityRef)
    assert builder.last_fact.entity_fact == CK.foo
    assert builder.last_fact.sub_fact == CK.bar
    assert isinstance(builder.last_low, director.EntityRef)
    assert builder.last_low.entity_fact == CK.bar
    assert builder.last_low.sub_fact == CK.baz
    assert isinstance(builder.last_high, director.FlagRef)
    assert builder.last_high.fact == CK.stuff

    rule_parser.parse_criteria("!$bar.baz", {x.name: x.value for x in CK}, builder)
    assert isinstance(builder.last_fact, director.EntityRef)
    assert builder.last_fact.entity_fact == CK.bar
    assert builder.last_fact.sub_fact == CK.baz
    assert isinstance(builder.last_low, director.IntRef)
    assert builder.last_low.value == 0
    assert isinstance(builder.last_high, director.IntRef)
    assert builder.last_high.value == 0

    rule_parser.parse_criteria("baz", {x.name: x.value for x in CK}, builder)
    assert isinstance(builder.last_fact, director.FlagRef)
    assert builder.last_fact.fact == CK.baz
    assert isinstance(builder.last_low, director.IntRef)
    assert builder.last_low.value == 1
    assert isinstance(builder.last_high, director.IntRef)
    assert builder.last_high.value == rule_parser.POS_INF

    rule_parser.parse_criteria("baz = foo", {x.name: x.value for x in CK}, builder)
    assert isinstance(builder.last_low, director.FlagRef)
    assert builder.last_low.fact == CK.baz
    assert isinstance(builder.last_fact, director.FlagRef)
    assert builder.last_fact.fact == CK.foo
    assert isinstance(builder.last_high, director.IntRef)
    assert builder.last_high.value == rule_parser.POS_INF

def test_parse_eval():
    # notice that prority order (small comes first) is inverted w.r.t. the
    # order the rules are defined. this helps to test that "high" (i.e. small)
    # priority rules will win over "low" (i.e. large) priority rules, even if
    # defined out of order
    test_config = """
        [other_rule]
        type = "start_game"
        priority = 1
        criteria = [ "5 <= stuff <= 10", "4 <= $foo.bar <= 15", "21 <= $bar.baz <= 21"]
        [[other_rule.actions]]
        _action = "message"
        message = "what up fool?"
        id_for_testing = 3

        [start_game_help]
        type = "start_game"
        priority = 0
        criteria = ["1 <= is_player <= 1"]
        [[start_game_help.actions]]
        _action = "broadcast"
        message = "what up gang?"
        delay = 500
        id_for_testing = 0
        [[start_game_help.actions]]
        _action = "message"
        message = "{recipient} how ya doin?"
        recipient = "PLAYER"
        sender = "CHARACTER"
        id_for_testing = 1
    """

    directors = rule_parser.loads(test_config, {x.name: x.value for x in ET}, {x.name: x.value for x in CK}, {x.name: x.value for x in A})
    assert(len(directors) == 1)
    d = directors[0]
    #d.check_refcounts()

    event_context:Dict[int,int] = {CK.foo: 100, CK.bar: 200}
    entity_store = director.EntityStore()
    e1 = entity_store.register_entity(100)
    e1.set_flag(CK.bar, 7)
    e2 = entity_store.register_entity(200)
    e2.set_flag(CK.baz, 21)

    event = director.Event(
        int(ET.start_game),
        event_context,
        entity_store,
        {"awesome": "sauce"}
    )

    char_a = entity_store.register_entity(300)
    char_a.set_flag(CK.is_player, 1)
    char_a.set_flag(CK.stuff, 7)
    char_b = entity_store.register_entity(400)
    char_c = entity_store.register_entity(500)
    char_c.set_flag(CK.stuff, 3)
    char_d = entity_store.register_entity(600)
    char_d.set_flag(CK.stuff, 7)
    char_e = entity_store.register_entity(700)
    char_e.set_flag(CK.stuff, 12)

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

    # alice should get both actions for the first rule,
    # alice should get none for the second, even though they match the rule
    # doug should get the action for the second, only matching, rule
    # other folk should not show up
    assert len(actions) == 3
    assert actions[0].action_id == A.broadcast
    assert actions[0].character_candidate.data == "alice"
    assert actions[0].args["id_for_testing"] == 0
    assert actions[1].action_id == A.message
    assert actions[1].character_candidate.data == "alice"
    assert actions[1].args["id_for_testing"] == 1
    assert actions[2].action_id == A.message
    assert actions[2].character_candidate.data == "doug"
    assert actions[2].args["id_for_testing"] == 3
