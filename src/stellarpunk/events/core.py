""" Bridges between Stellarpunk and narrative director library.

Responsible for:
 * packaging Stellarpunk inferface for events as narrative events
 * queuing events for processing
 * passing into director for evaluation
 * recognizing action identifiers that come back from rule eval
 * running actions
"""

import enum
import logging
import collections
import numbers
from typing import Any, Optional, Union, Callable
from collections.abc import Iterable, Sequence, Mapping, MutableMapping

from stellarpunk import core, util, dialog, narrative, task_schedule

class Action:
    def __init__(self) -> None:
        self.gamestate: core.Gamestate = None # type: ignore[assignment]

    def _required_keys(self, key_types: Sequence[tuple[str, type]], action_args: Mapping[str, Any]) -> bool:
        return all(
            k in action_args and isinstance(action_args[k], t) for k,t in key_types
        )

    def _optional_keys(self, key_types: Sequence[tuple[str, type]], action_args: Mapping[str, Any]) -> bool:
        return all(
            k not in action_args or isinstance(action_args[k], t) for k,t in key_types
        )

    def _validate(self, action_args: Mapping[str, Any]) -> bool:
        return True

    def validate(self, action_args: Mapping[str, Any]) -> bool:
        """ Validate a set of action_args associated with a configured instance
        of the action.

        This is useful to help catch event configuration errors (e.g. missing
        or malformed required params). """

        if "_delay" in action_args and not isinstance(action_args["_delay"], numbers.Real):
            return False

        return self._validate(action_args)

    def initialize(self, gamestate: core.Gamestate) -> None:
        self.gamestate = gamestate

    def act(
        self,
        character: core.Character,
        event_type: int,
        event_context: Mapping[int, int],
        event_args: MutableMapping[str, Any],
        action_args: Mapping[str, Any]
    ) -> None:
        pass


_event_offset = 0
RegisteredEventSpaces:dict[enum.EnumMeta, int] = {}
_context_key_offset = 0
RegisteredContextSpaces:dict[enum.EnumMeta, int] = {}
RegisteredActions:dict[Action, str] = {}


def e(event_id: enum.IntEnum) -> int:
    return event_id + RegisteredEventSpaces[event_id.__class__]


def ck(context_key: enum.IntEnum) -> int:
    return context_key + RegisteredContextSpaces[context_key.__class__]


def register_events(events: enum.EnumMeta) -> None:
    if len(events) > 0:
        if not all(isinstance(x, int) for x in events): # type: ignore[var-annotated]
            raise ValueError("members of events must all be int-like")
        if min(events) not in [0, 1]:
            raise ValueError("events must start at 0 or 1")
        if max(events) > len(events):
            raise ValueError("events must be continuous")
    global _event_offset
    RegisteredEventSpaces[events] = _event_offset
    _event_offset += max(events)+1


def register_context_keys(context_keys: enum.EnumMeta) -> None:
    if len(context_keys) > 0:
        if not all(isinstance(x, int) for x in context_keys): # type: ignore[var-annotated]
            raise ValueError("members of context keys must all be int-like")
        if min(context_keys) not in [0, 1]:
            raise ValueError("context keys must start at 0 or 1")
        if max(context_keys) > len(context_keys):
            raise ValueError("context keys must be continuous")
    global _context_key_offset
    RegisteredContextSpaces[context_keys] = _context_key_offset
    _context_key_offset += max(context_keys)+1


def register_action(action: Action, name:str) -> None:
    #if name is None:
    #    name = util.camel_to_snake(action.__class__.__name__)
    #    if name.endswith("_action"):
    #        name = name[:-len("_action")]
    RegisteredActions[action] = name


"""
events.register_events(MyEvents)
events.register_context_keys(MyContextKeys)

events.register_action(A()) for A in [Action1, Action2, Action3]

trigger_event([dude], events.e(MyEvents.coolio), narrative.context({events.ck(MyContextKeys.foo): 27}), bob, billy, stuff="yes", other_stuff=42)
"""

class EventManager(core.AbstractEventManager):
    def __init__(
        self,
    ) -> None:
        self.logger = logging.getLogger(util.fullname(self))
        self.gamestate:core.Gamestate = None # type: ignore[assignment]
        self.director:narrative.Director = None # type: ignore[assignment]
        self.event_queue:collections.deque[tuple[narrative.Event, Iterable[narrative.CharacterCandidate]]] = collections.deque()
        self.actions:dict[int, Action] = {}
        self.event_types:dict[str, int] = {}
        self.event_type_lookup:dict[int, str] = {}
        self.context_keys:dict[str, int] = {}
        self.context_key_lookup:dict[int, str] = {}
        self.action_ids:dict[str, int] = {}
        self.action_id_lookup:dict[int, str] = {}

        self.action_schedule: task_schedule.TaskSchedule[tuple[narrative.Event, narrative.Action]] = task_schedule.TaskSchedule()

    def initialize(self, gamestate: core.Gamestate, events: Mapping[str, Any]) -> None:
        self.gamestate = gamestate
        old_event_manager = self.gamestate.event_manager
        self.gamestate.event_manager = self

        # assign integer ids for events, contexts, actions
        action_validators:dict[int, Callable[[Mapping], bool]] = {}

        for event_enum in RegisteredEventSpaces:
            for event_key in event_enum: # type: ignore[var-annotated]
                self.event_types[util.camel_to_snake(event_key.name)] = event_key + RegisteredEventSpaces[event_enum]

        for context_enum in RegisteredContextSpaces:
            for context_key in context_enum: # type: ignore[var-annotated]
                self.context_keys[util.camel_to_snake(context_key.name)] = context_key.value + RegisteredContextSpaces[context_enum]

        action_count = 0
        for action, action_name in RegisteredActions.items():
            action.initialize(self.gamestate)
            self.action_ids[action_name] = action_count
            self.actions[action_count] = action
            action_validators[action_count] = action.validate
            action_count += 1

        self.event_type_lookup = dict((v,k) for k,v in self.event_types.items())
        self.context_key_lookup = dict((v,k) for k,v in self.context_keys.items())
        self.action_id_lookup = dict((v,k) for k,v in self.action_ids.items())

        self.logger.info(f'known events {self.event_types.keys()}')
        self.logger.info(f'known context keys {self.context_keys.keys()}')
        self.logger.info(f'known actions {self.action_ids.keys()}')
        self.director = narrative.loadd(events, self.event_types, self.context_keys, self.action_ids, action_validators)

        self.logger.info(f'transferring events from {old_event_manager} to {self}')
        old_event_manager.transfer_events(self)

        self.logger.info(f'event manager initialized')

    def trigger_event(
        self,
        characters: Iterable[core.Character],
        event_type: int,
        context: Mapping[int, int],
        event_args: MutableMapping[str, Any],
    ) -> None:
        self.logger.debug(f'enqueuing event {self.event_type_lookup[event_type]} ({event_type})')
        self.event_queue.append((
            narrative.Event(
                event_type,
                context,
                self.gamestate.entity_context_store,
                event_args,
            ),
            [narrative.CharacterCandidate(c.context, c) for c in characters]
        ))

    def trigger_event_immediate(
        self,
        characters: Iterable[core.Character],
        event_type: int,
        context: Mapping[int, int],
        event_args: MutableMapping[str, Any],
    ) -> None:
        actions_processed = self._do_event(
            narrative.Event(
                event_type,
                context,
                self.gamestate.entity_context_store,
                event_args,
            ),
            [narrative.CharacterCandidate(c.context, c) for c in characters]
        )
        self.gamestate.counters[core.Counters.EVENTS_PROCESSED_OOB] += 1
        self.gamestate.counters[core.Counters.EVENT_ACTIONS_PROCESSED_OOB] += actions_processed

    def tick(self) -> None:
        # check for relevant events and process them
        events_processed = 0
        actions_processed = 0
        while len(self.event_queue) > 0:
            event, candidates = self.event_queue.popleft()
            actions_processed += self._do_event(event, candidates)
            events_processed += 1

        for event, action in self.action_schedule.pop_current_tasks(self.gamestate.timestamp):
            self._do_action(event, action)
            actions_processed += 1

        self.gamestate.counters[core.Counters.EVENTS_PROCESSED] += events_processed
        self.gamestate.counters[core.Counters.EVENT_ACTIONS_PROCESSED] += actions_processed

    def _do_event(self, event: narrative.Event, candidates:Iterable[narrative.CharacterCandidate]) -> int:
        actions_processed = 0
        self.logger.debug(f'evaluating event {self.event_type_lookup[event.event_type]} ({event.event_type}) for {list(x.data.short_id() for x in candidates)}')
        actions = self.director.evaluate(event, candidates)
        for action in actions:
            self.logger.debug(f'triggered action {self.action_id_lookup[action.action_id]} ({action.action_id}) for {action.character_candidate.data.short_id()}')

            if "_delay" in action.args:
                delay = action.args["_delay"]
                self.logger.debug(f'delaying action {self.action_id_lookup[action.action_id]} {action.action_id} by {delay}')
                self.action_schedule.push_task(
                    self.gamestate.timestamp+delay,
                    (event, action)
                )
            else:
                self._do_action(event, action)
                actions_processed += 1

        return actions_processed

    def _do_action(self, event: narrative.Event, action: narrative.Action) -> None:
        s_action = self.actions[action.action_id]

        self.logger.debug(f'processing action {s_action}')
        s_action.act(
            action.character_candidate.data,
            event.event_type,
            event.event_context,
            event.args,
            action.args
        )


class DialogManager:
    def __init__(self, dialog:dialog.DialogGraph, gamestate:core.Gamestate, event_manager:EventManager, character: core.Character, speaker: core.Character) -> None:
        self.dialog = dialog
        self.current_id = dialog.root_id
        self.gamestate = gamestate
        self.event_manager = event_manager
        self.character = character
        self.speaker = speaker

    @property
    def node(self) -> dialog.DialogNode:
        return self.dialog.nodes[self.current_id]

    @property
    def choices(self) -> Sequence[dialog.DialogChoice]:
        return self.dialog.nodes[self.current_id].choices

    def do_node(self) -> None:
        for flag in self.dialog.nodes[self.current_id].flags:
            self.character.context.set_flag(self.event_manager.context_keys[flag], 1)
        for flag in self.dialog.nodes[self.current_id].speaker_flags:
            self.speaker.context.set_flag(self.event_manager.context_keys[flag], 1)

    def choose(self, choice:dialog.DialogChoice) -> None:
        if choice not in self.choices:
            raise ValueError(f'given choice is not current node {self.current_id} choices')

        for flag in choice.flags:
            self.character.context.set_flag(self.event_manager.context_keys[flag], 1)
        for flag in choice.speaker_flags:
            self.speaker.context.set_flag(self.event_manager.context_keys[flag], 1)
        self.current_id = choice.node_id
