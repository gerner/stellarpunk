""" Narrative Event Rule Parsing """

import sys
import re
import collections
import logging
from collections.abc import Mapping, MutableMapping
from typing import Mapping, Union, Any, Callable, MutableMapping, Optional

import toml # type: ignore

from . import director

INT_RE = re.compile("[0-9]+")
FLAG_RE = re.compile("[a-zA-Z_][a-zA-Z0-9_:]*")

POS_INF = (1<<64)-1

def parse_ref(data: str, context_keys: Mapping[str, int]) -> tuple[Union[director.IntRef, director.FlagRef, director.EntityRef], int]:
    m = INT_RE.match(data)
    if m:
        return director.IntRef(int(m.group(0))), m.end()

    if data[0] == "$":
        data = data[1:]
        m = FLAG_RE.match(data)
        if not m:
            raise ValueError("bad ename in entity ref")
        entity_name = m.group(0)
        data = data[m.end():]
        if not data[0] == ".":
            raise ValueError("bad dotted notation in entity ref")
        data = data[1:]
        m = FLAG_RE.match(data)
        if not m:
            raise ValueError("bad fname in entity ref")
        flag_name = m.group(0)
        data = data[m.end():].lstrip()

        return director.EntityRef(context_keys[entity_name], context_keys[flag_name]), 1 + len(entity_name) + 1 + len(flag_name)
    else:
        m = FLAG_RE.match(data)
        if not m:
            raise ValueError("bad flag ref")
        flag_name = m.group(0)
        return director.FlagRef(context_keys[flag_name]), m.end()


def parse_criteria(cri: str, context_keys: Mapping[str, int], builder: director.CriteriaBuilder) -> None:

    if not isinstance(cri, str):
        raise ValueError("criteria must be a string, got {cri}")

    # CRITERIA := [REF "<="] REF ["<=" REF] | INVERTED_REF | EQUALITY_REF
    # INVERTED_REF := "!" REF
    # EQUALITY_REF := REF "=" REF
    # REF := int | FLAG_REF | ENTITY_REF
    # INT_REF := [0-9]+
    # FLAG_REF := [a-zA-Z_][a-zA-Z0-9_]*
    # ENTITY_REF := $ FLAG_REF "." FLAG_REF

    data = cri.lstrip()

    if data[0] == "!":
        data = data[1:]
        # inverted ref case
        ref, pos = parse_ref(data, context_keys)
        data = data[pos:].lstrip()
        if data != "":
            raise ValueError(f'had left-over string in inverted criteria "{data}"')

        builder.add_low(director.IntRef(0))
        builder.add_fact(ref)
        builder.add_high(director.IntRef(0))
        builder.build()
        return

    ref, pos = parse_ref(data, context_keys)
    data = data[pos:].lstrip()

    if data == "":
        # single ref case
        builder.add_low(director.IntRef(1))
        builder.add_fact(ref)
        builder.add_high(director.IntRef(POS_INF))
        builder.build()
        return

    elif data[0] == "=":
        # equality ref case
        data = data[1:].lstrip()
        rhs_ref, pos = parse_ref(data, context_keys)
        data = data[pos:].lstrip()
        if data != "":
            raise ValueError(f'had left-over string in equality criteria "{data}"')

        builder.add_low(ref)
        builder.add_fact(rhs_ref)
        builder.add_high(director.IntRef(POS_INF))
        return

    if not data.startswith("<="):
        raise ValueError(f'epected lower bound as "<=" in "{data}"')
    data = data[2:].lstrip()

    lref = ref
    ref, pos = parse_ref(data, context_keys)
    data = data[pos:].lstrip()

    if data == "":
        # single bound case
        builder.add_low(lref)
        builder.add_fact(ref)
        builder.add_high(director.IntRef(POS_INF))
        builder.build()
        return

    if not data.startswith("<="):
        raise ValueError(f'epected upper bound as "<=" in "{data}"')
    data = data[2:].lstrip()

    href, pos = parse_ref(data, context_keys)
    data = data[pos:].lstrip()
    if data != "":
        raise ValueError(f'had left-over string in criteria "{data}"')

    builder.add_low(lref)
    builder.add_fact(ref)
    builder.add_high(href)
    builder.build()
    return


def parse_action(
    rule_id: str,
    act: Mapping[str, Any],
    action_ids: Mapping[str, int],
    action_validators: Mapping[int, Callable[[Mapping], bool]],
    context_keys: Mapping[str, int],
) -> director.ActionTemplate:
    if not isinstance(act, dict):
        raise ValueError(f'actions for {rule_id} must be a list of tables')
    if "_action" not in act:
        raise ValueError(f'actions for {rule_id} must all have _action field')
    action_name = act["_action"]
    if action_name not in action_ids:
        raise ValueError(f'rule {rule_id} had unknown action {action_name}')
    action_id = action_ids[action_name]

    # translate "ref:" type arguments to their integer context key id
    for k, v in act.items():
        if isinstance(v, str) and v.startswith("_ref:"):
            ref_key = v[len("_ref:"):]
            if ref_key not in context_keys:
                raise ValueError(f'an action arg for {rule_id} had context key ref "{v}" that was not found in context keys')
            act[k] = context_keys[ref_key]

    if action_id in action_validators and not action_validators[action_id](act):
        raise ValueError(f'rule {rule_id} had invalid action args for action {action_name}')

    action_template = director.ActionTemplate(action_id, act)
    return action_template


def loads(
    data: str,
    event_types: Mapping[str, int],
    context_keys: Mapping[str, int],
    action_ids: Mapping[str, int],
    action_validators: Mapping[int, Callable[[Mapping], bool]] = {},
) -> list[director.Director]:
    """
    Loads rules from a toml string into a narrative director.

    Parameters
    ----------
    data : str
        toml encoded rule data
    event_types : dict of str to int
        mapping from string event names to event int identifiers
    context_keys : dict of str to int
        mapping from string context key names to int identifiers
    acction_ids : dict of str to int
        mapping from string action names to int identifiers

    Returns
    -------
    out : narrative.director
        a director instance loaded with rules as parsed from the input
    """

    # load data as toml
    rule_data = toml.loads(data)
    return loadd(rule_data, event_types, context_keys, action_ids, action_validators)


def loadd(
    rule_data: Mapping[str, Any],
    event_types: Mapping[str, int],
    context_keys: Mapping[str, int],
    action_ids: Mapping[str, int],
    action_validators: Mapping[int, Callable[[Mapping], bool]] = {},
) -> list[director.Director]:
    """
    Loads rules from a rule data dict into a narrative director.

    Parameters
    ----------
    rule_data : dict
        dictionary of rules. Keys are rule ids. Values are the rule data to
        decode.
    event_types : dict of str to int
        mapping from string event names to event int identifiers
    context_keys : dict of str to int
        mapping from string context key names to int identifiers
    acction_ids : dict of str to int
        mapping from string action names to int identifiers

    Returns
    -------
    out : narrative.director
        a director instance loaded with rules as parsed from the input
    """
    base_rules = collections.defaultdict(list)
    rules_by_group:MutableMapping[str, MutableMapping[int, list[director.Rule]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for rule_id, rule in rule_data.items():
        # get event type this rule subscribes to
        if "type" not in rule:
            raise ValueError(f'no event type in rule {rule_id}')
        event_type = rule["type"]
        if event_type not in event_types:
            raise ValueError(f'rule {rule_id} had unknown event type {event_type}')
        event_type_id = event_types[event_type]

        if "priority" not in rule or not isinstance(rule["priority"], int):
            raise ValueError(f'missing or bad priority in rule {rule_id}')
        priority = rule["priority"]

        group:Optional[str] = None
        if "group" in rule:
            if isinstance(rule["group"], str):
                group = rule["group"]
            else:
                raise ValueError(f'group must be a string in {rule_id}, got {rule["group"]}')

        # find and parse all the various criteria
        criteria_data:list[str]
        if "criteria" not in rule:
            criteria_data = []
        elif not isinstance(rule["criteria"], list):
            raise ValueError(f'criteria for {rule_id} must be a list')
        else:
            criteria_data = rule["criteria"]

        criteria_builder = director.CriteriaBuilder()

        for cri in criteria_data:
            #try:
            parse_criteria(cri, context_keys, criteria_builder)
            #except ValueError as e:
            #    raise ValueError(f'bad criteria for {rule_id}') from e

        # find and parse the actions
        action_data:list[dict[str, Any]]
        if "actions" not in rule:
            action_data = []
        elif not isinstance(rule["actions"], list):
            raise ValueError(f'actions for {rule_id} must be a list')
        else:
            action_data = rule["actions"]

        actions:list[director.ActionTemplate] = []

        for act in action_data:
            actions.append(parse_action(rule_id, act, action_ids, action_validators, context_keys))

        # create a rule record
        if not group:
            base_rules[event_type_id].append(director.Rule(event_type_id, priority, criteria_builder, actions))
        else:
            rules_by_group[group][event_type_id].append(director.Rule(event_type_id, priority, criteria_builder, actions))

    # make sure the rules are in priority order
    sorted_rule_groups:list[MutableMapping[int, list[director.Rule]]] = []
    sorted_rules:MutableMapping[int, list[director.Rule]] = {}
    for event_type_id, rule_values in base_rules.items():
        sorted_rules[event_type_id] = sorted(rule_values, key=lambda x: (x.get_priority()))
    sorted_rule_groups.append(sorted_rules)

    for rule_group in rules_by_group.values():
        sorted_rule_group:MutableMapping[int, list[director.Rule]] = {}
        for event_type_id, rule_values in rule_group.items():
            sorted_rule_group[event_type_id] = sorted(rule_values, key=lambda x: (x.get_priority()))
        sorted_rule_groups.append(sorted_rule_group)

    # create and return event directors, one for each group
    return list(director.Director(rule_group) for rule_group in sorted_rule_groups)
