""" Manages Events """

import abc
import logging
import re
from typing import List, Any, Sequence, Dict, Tuple, Optional

from stellarpunk import core, util, dialog, predicates, config

class Event(abc.ABC):
    def __init__(self, event_id:str) -> None:
        self.event_id = event_id

    def is_relevant(self, gamestate:core.Gamestate, player:core.Player) -> bool: ...
    def act(self, gamestate:core.Gamestate, player:core.Player) -> None: ...

class DemoEvent(Event):
    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__("demo_event", *args, **kwargs)

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

EventCriteria = predicates.Criteria[Tuple[core.Gamestate, core.Player]]
class FlagCriteria(EventCriteria):
    def __init__(self, flag:str) -> None:
        self.flag = flag

    def evaluate(self, universe:Tuple[core.Gamestate, core.Player]) -> bool:
        gamestate, player = universe
        return self.flag in player.flags

class CompareCriteria(EventCriteria):
    def __init__(self, left:str, op:str, relative:bool, right:float) -> None:
        self.left = left
        self.op = op
        self.relative = relative
        self.right = right

    def evaluate(self, universe:Tuple[core.Gamestate, core.Player]) -> bool:
        gamestate, player = universe
        if self.left == "NOW":
            leftv = gamestate.timestamp
        elif self.left not in player.flags:
            return False
        else:
            leftv = player.flags[self.left]

        if self.relative:
            rightv = self.right + gamestate.timestamp
        else:
            rightv = self.right

        if self.op == ">":
            return leftv > rightv
        elif self.op == "<":
            return leftv < rightv
        else:
            raise Exception(f'unknown operator {self.op}')

FLAG_RE = re.compile(r'[ \t]*([a-zA-Z0-9_.]+)[ \t]*')
JUNCTION_RE = re.compile(r'[ \t]*([|&])[ \t]*')
OP_RE = re.compile(r'[ \t]*(CMP)[(]([^)]*)[)][ \t]*')
CMP_ARGS_RE = re.compile(r'[ \t]*([a-zA-Z0-9_.]+)[ \t]*(<|>)[ \t]*(r?)([0-9]+[.]?[0-9]*)[ \t]*')
WS_RE = re.compile(r'[ \t]*')

def parse_junction(criteria_text:str, start:int, m:re.Match, left:EventCriteria) -> Tuple[int, EventCriteria]:
    # JUNCTION := CRITERIA "|" CRITERIA | CRITERIA "&" CRITERIA
    new_start, right = parse_criteria(criteria_text, start+m.end())

    if m.group(1) == "|":
        return new_start, predicates.Disjunction(left, right)
    elif m.group(1) == "&":
        return new_start, predicates.Conjunction(left, right)
    else:
        raise Exception(f'JUNCTION_RE incorrectly matched "{m.group(0)}"')

def parse_op(m:re.Match) -> EventCriteria:
    if m.group(1) == "CMP":
        m_args = CMP_ARGS_RE.match(m.group(2))
        if m_args is None:
            raise ValueError(f'unparseable CMP arguments: "{m.group(2)}" parsing operator {m.group(0)}')
        left = m_args.group(1)
        op = m_args.group(2)
        relative = m_args.group(3) == "r"
        right = float(m_args.group(4))
        return CompareCriteria(left, op, relative, right)
    else:
        raise ValueError(f'unsupported operation "{m.group(1)}" parsing operator {m.group(0)}')

def parse_criteria(criteria_text:str, start:int=0, grab_junction:bool=True) -> Tuple[int, EventCriteria]:

    # CRITERIA :=  "!" CRITERIA | OP | FLAG | ( CRITERIA ) | CRITERIA JUNCTION
    # OP := OP_NAME "(" OP_ARGS ")"
    # OP_NAME := "CMP"
    # OP_ARGS := FLAG OP_OP OP_NUM
    # OP_OP := "<" | ">"
    # OP_NUM := [r]FLOAT # leading r makes the float relative to "now" at eval time
    # FLAG := a string matching FLAG_RE
    # JUNCTION := "|" CRITERIA | "&" CRITERIA

    # (( foo | bar ) & baz) | (blerf & bling)

    # fast foward past leading whitespace
    m = WS_RE.match(criteria_text[start:])
    assert m is not None
    start = start+m.end()

    if len(criteria_text[start:]) == 0:
        raise ValueError(f'no input to process')

    if criteria_text[start] == "!":
        new_start, criteria = parse_criteria(criteria_text, start+1, grab_junction=False)
        criteria = predicates.Negation(criteria)
        start = new_start
    elif m := OP_RE.match(criteria_text[start:]):
        criteria = parse_op(m)
        start = start+m.end()
    elif m := FLAG_RE.match(criteria_text[start:]):
        criteria = FlagCriteria(m.group(1))
        start = start+m.end()
    elif criteria_text[start] == "(":
        new_start, criteria = parse_criteria(criteria_text, start+1)
        if len(criteria_text) <= new_start or criteria_text[new_start] != ")":
            raise ValueError(f'expected ")" at position {new_start},{new_start-start} while parsing "{criteria_text[start:new_start+8]}..."')
        m = WS_RE.match(criteria_text[new_start+1:])
        assert m is not None
        start = new_start+1+m.end()

    if grab_junction:
        m = JUNCTION_RE.match(criteria_text[start:])
        if m is not None:
            return parse_junction(criteria_text, start, m, criteria)
        else:
            return start, criteria
    else:
        return start, criteria

def load_criteria(criteria_text:str) -> EventCriteria:
    new_start, criteria = parse_criteria(criteria_text)
    if new_start != len(criteria_text):
        raise ValueError(f'after processing, had left over text: "{criteria_text[new_start:]}"')
    return criteria

def load_events(event_data:Dict[str, Any]) -> Dict[str, Event]:
    return {}

class ScriptedEvent(Event):
    """ Event that's driven from a scripted configuration. """

    @staticmethod
    def load_from_config(event_id:str, data:Dict[str, Any]) -> "ScriptedEvent":
        criteria = load_criteria(data["criteria"])

        return ScriptedEvent(event_id, criteria, notification=data.get("notification"))

    def __init__(self,
            event_id:str,
            criteria:EventCriteria,
            *args:Any,
            notification:Optional[str]=None,
            **kwargs:Any) -> None:
        super().__init__(event_id, *args, **kwargs)
        self.criteria = criteria
        self.notification = notification

    def is_relevant(self, gamestate:core.Gamestate, player:core.Player) -> bool:
        return self.criteria.evaluate((gamestate, player))

    def act(self, gamestate:core.Gamestate, player:core.Player) -> None:
        player.set_flag(self.event_id, gamestate.timestamp)
        if self.notification is not None:
            player.send_notification(self.notification)

class EventManager:
    def __init__(self) -> None:
        self.logger = logging.getLogger(util.fullname(self))
        self.gamestate:core.Gamestate = None # type: ignore[assignment]
        self.events:List[Event] = []

    def initialize(self, gamestate:core.Gamestate) -> None:
        self.gamestate = gamestate
        for event_id, event_data in config.Events.items():
            self.events.append(ScriptedEvent.load_from_config(event_id, event_data))

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
