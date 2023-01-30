""" Narrative Event Rule Parsing """

import sys
import re
import collections
from typing import Dict, Mapping, List, Union, Any, Callable

import toml # type: ignore

from . import director

INT_RE = re.compile("[0-9]+")
FLAG_RE = re.compile("[a-zA-Z_][a-zA-Z0-9_]*")

def parse_criteria(cri: str, context_keys: Mapping[str, int]) -> Union[director.FlagCriteria, director.EntityCriteria]:

    if not isinstance(cri, str):
        raise ValueError("criteria must be a string, got {cri}")

    # CRITERIA := RANGE_CRITERIA
    # RANGE_CRITERIA := int "<=" REF "<=" int
    # REF := FLAG_REF | ENTITY_REF
    # FLAG_REF := [a-zA-Z_][a-zA-Z0-9_]*
    # ENTITY_REF := $ FLAG_REF "." FLAG_REF

    data = cri

    # lower bound value
    m = INT_RE.match(data)
    if not m:
        raise ValueError("bad low value")
    low = int(m.group(0))
    data = data[m.end():].lstrip()

    # lower bound indicator
    if data[0:2] != "<=":
        raise ValueError(f'epected lower bound as "<=" in "{data}"')
    data = data[2:].strip()

    # flag or entity flag
    entity_name = None
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
    else:
        m = FLAG_RE.match(data)
        if not m:
            raise ValueError("bad flag ref")
        flag_name = m.group(0)
        data = data[m.end():].lstrip()

    # upper bound indicator
    if data[0:2] != "<=":
        raise ValueError(f'epected upper bound as "<=" in "{data}"')
    data = data[2:].strip()

    # upper bound value
    m = INT_RE.match(data)
    if not m:
        raise ValueError("bad high value")
    high = int(m.group(0))
    data = data[m.end():].strip()

    # nothing else left
    if data != "":
        raise ValueError(f'had left-over string in criteria {data}')

    if entity_name:
        entity_id = context_keys[entity_name]
        flag_id = context_keys[flag_name]
        return director.EntityCriteria(entity_id, flag_id, low, high)
    else:
        flag_id = context_keys[flag_name]
        return director.FlagCriteria(flag_id, low, high)


def parse_action(
    rule_id: str,
    act: Mapping[str, Any],
    action_ids: Mapping[str, int],
    action_validators: Mapping[int, Callable[[Mapping], bool]],
) -> director.ActionTemplate:
    if not isinstance(act, Dict):
        raise ValueError(f'actions for {rule_id} must be a list of tables')
    if "_action" not in act:
        raise ValueError(f'actions for {rule_id} must all have _action field')
    action_name = act["_action"]
    if action_name not in action_ids:
        raise ValueError(f'rule {rule_id} had unknown action {action_name}')
    action_id = action_ids[action_name]
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
) -> director.Director:
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
) -> director.Director:
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
    rules = collections.defaultdict(list)
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

        # find and parse all the various criteria
        criteria_data: List[str]
        if "criteria" not in rule:
            criteria_data = []
        elif not isinstance(rule["criteria"], List):
            raise ValueError(f'criteria for {rule_id} must be a list')
        else:
            criteria_data = rule["criteria"]

        criteria: List[director.FlagCriteria] = []
        entity_criteria: List[director.EntityCriteria] = []

        for cri in criteria_data:
            #try:
            parsed_criteria = parse_criteria(cri, context_keys)
            #except ValueError as e:
            #    raise ValueError(f'bad criteria for {rule_id}') from e

            if isinstance(parsed_criteria, director.FlagCriteria):
                criteria.append(parsed_criteria)
            elif isinstance(parsed_criteria, director.EntityCriteria):
                entity_criteria.append(parsed_criteria)
            else:
                raise Exception(f'for {rule_id} "{cri}" parsed to bad criteria')

        # find and parse the actions
        action_data: List[Dict[str, Any]]
        if "actions" not in rule:
            action_data = []
        elif not isinstance(rule["actions"], List):
            raise ValueError(f'actions for {rule_id} must be a list')
        else:
            action_data = rule["actions"]

        actions: List[director.ActionTemplate] = []

        for act in action_data:
            actions.append(parse_action(rule_id, act, action_ids, action_validators))

        # create a rule record
        rules[event_type_id].append(director.Rule(event_type_id, priority, criteria, entity_criteria, actions))

    # create and return an event director
    return director.Director(rules)
