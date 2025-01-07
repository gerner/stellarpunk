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
import uuid
from typing import Any, Optional, Union, Callable
from collections.abc import Collection, Iterable, Sequence, Mapping, MutableMapping

import numpy as np

from stellarpunk import core, util, dialog, narrative, task_schedule

class Action:
    def __init__(self) -> None:
        self.gamestate: core.Gamestate = None # type: ignore[assignment]

    def ck(self, event_context:Mapping[int, int], context_key:enum.IntEnum) -> int:
        ck_id = self.gamestate.event_manager.ck(context_key)
        try:
            return event_context[ck_id]
        except KeyError as ke:
            raise KeyError(f'{self.gamestate.event_manager.ck_rev(ck_id)}({ck_id}) not found in event context with keys {list(self.gamestate.event_manager.ck_rev(c) for c in event_context.keys())}') from ke

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
        event_args: MutableMapping[str, Union[int,float,str,bool]],
        action_args: Mapping[str, Union[int,float,str,bool]]
    ) -> None:
        pass

"""
events.register_events(MyEvents)
events.register_context_keys(MyContextKeys)

events.register_action(A()) for A in [Action1, Action2, Action3]

trigger_event([dude], events.e(MyEvents.coolio), narrative.context({events.ck(MyContextKeys.foo): 27}), bob, billy, stuff="yes", other_stuff=42)
"""

class EventState:
    def __init__(self) -> None:
        self.event_queue:collections.deque[tuple[narrative.Event, Iterable["narrative.CharacterCandidate[core.Character]"]]] = collections.deque()
        self.action_schedule: task_schedule.TaskSchedule[tuple[narrative.Event, "narrative.Action[core.Character]"]] = task_schedule.TaskSchedule()
        self.last_event_trigger:dict[uuid.UUID, MutableMapping[int, float]] = collections.defaultdict(lambda: collections.defaultdict(lambda: -np.inf))


class EventManager(core.AbstractEventManager):
    def __init__(
        self,
        event_throttle_secs:float=30.0
    ) -> None:
        self.logger = logging.getLogger(util.fullname(self))
        self.event_throttle_secs = event_throttle_secs
        self.gamestate:core.Gamestate = None # type: ignore[assignment]
        self.directors:list[narrative.Director] = None # type: ignore[assignment]

        # this is mapping to/from code specific logic and EventManager logic
        self._event_offset = 0
        self.RegisteredEventSpaces:dict[enum.EnumMeta, tuple[int, str]] = {}
        self._context_key_offset = 0
        self.RegisteredContextSpaces:dict[enum.EnumMeta, tuple[int, str]] = {}
        self.RegisteredActions:dict[Action, tuple[str, str]] = {}

        # this is mapping to/from EventContext id space and EventManager logic
        self.actions:dict[int, Action] = {}
        self.event_types:dict[str, int] = {}
        self.event_type_lookup:dict[int, str] = {}
        self.context_keys:dict[str, int] = {}
        self.context_key_lookup:dict[int, str] = {}
        self.action_ids:dict[str, int] = {}
        self.action_id_lookup:dict[int, str] = {}

        # this is actual dynamic state
        self.event_state = EventState()

    # logic helping code interact with the event system
    def e(self, event_id: enum.IntEnum) -> int:
        """ map code specific event id to the global EventContext event id """
        return event_id + self.RegisteredEventSpaces[event_id.__class__][0]

    def ck(self, context_key: enum.IntEnum) -> int:
        """ map code specific context key to the global EventContext key """
        return context_key + self.RegisteredContextSpaces[context_key.__class__][0]

    def f(self, flag:str) -> int:
        """ map global flag string name to EventContext key """
        return self.context_keys[flag]

    def e_rev(self, event_id:int) -> str:
        return self.event_type_lookup[event_id]

    def ck_rev(self, context_key:int) -> str:
        return self.context_key_lookup[context_key]

    def register_events(self, events: enum.EnumMeta, namespace:str="") -> None:
        """ Registers a set of events code might trigger later

        Code keeps its own notion of events and this registration maps into a
        global event space. The names of the enum items

        events: is an enum specific code will use to identify the events later

        """
        if len(events) > 0:
            if not all(isinstance(x, int) for x in events): # type: ignore[var-annotated]
                raise ValueError("members of events must all be int-like")
            if min(events) not in [0, 1]:
                raise ValueError("events must start at 0 or 1")
            if max(events) > len(events):
                raise ValueError("events must be continuous")
        self.RegisteredEventSpaces[events] = (self._event_offset, namespace)
        self._event_offset += max(events)+1

    def register_context_keys(self, context_keys: enum.EnumMeta, namespace:str="") -> None:
        if len(context_keys) > 0:
            if not all(isinstance(x, int) for x in context_keys): # type: ignore[var-annotated]
                raise ValueError("members of context keys must all be int-like")
            if min(context_keys) not in [0, 1]:
                raise ValueError("context keys must start at 0 or 1")
            if max(context_keys) > len(context_keys):
                raise ValueError("context keys must be continuous")
        self.RegisteredContextSpaces[context_keys] = (self._context_key_offset, namespace)
        self._context_key_offset += max(context_keys)+1

    def register_action(self, action: Action, name:str, namespace:str="") -> None:
        #if name is None:
        #    name = util.camel_to_snake(action.__class__.__name__)
        #    if name.endswith("_action"):
        #        name = name[:-len("_action")]
        self.RegisteredActions[action] = (name, namespace)

    def pre_initialize(self, events: Mapping[str, Any]) -> None:
        """ pre-gamestate creation/loading initialization. """
        # assign integer ids for events, contexts, actions
        action_validators:dict[int, Callable[[Mapping], bool]] = {}

        for event_enum, (offset, namespace) in self.RegisteredEventSpaces.items():
            for event_key in event_enum: # type: ignore[var-annotated]
                if namespace:
                    event_name = f'{namespace}:{util.camel_to_snake(event_key.name)}'
                else:
                    event_name = util.camel_to_snake(event_key.name)
                self.event_types[event_name] = event_key + offset

        for context_enum, (offset, namespace) in self.RegisteredContextSpaces.items():
            for context_key in context_enum: # type: ignore[var-annotated]
                if namespace:
                    context_name = f'{namespace}:{util.camel_to_snake(context_key.name)}'
                else:
                    context_name = util.camel_to_snake(context_key.name)
                self.context_keys[context_name] = context_key.value + offset

        action_count = 0
        for action, (action_name, namespace) in self.RegisteredActions.items():
            if namespace:
                action_name = f'{namespace}:{action_name}'
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
        self.directors = narrative.loadd(events, self.event_types, self.context_keys, self.action_ids, action_validators)

        self.logger.info(f'event manager initialized')

    def initialize_gamestate(self, event_state:EventState, gamestate:core.Gamestate) -> None:
        """ post gamestate creation/loading initialization. """
        self.event_state = event_state
        self.gamestate = gamestate
        self.gamestate.event_manager = self

        for action in self.actions.values():
            action.initialize(self.gamestate)

    def trigger_event(
        self,
        characters: Collection[core.Character],
        event_type: int,
        context: Mapping[int, int],
        event_args: dict[str, Union[int,float,str,bool]],
    ) -> None:
        candidates = [narrative.CharacterCandidate(c.context, c) for c in characters if self.event_state.last_event_trigger[c.entity_id][event_type] + self.event_throttle_secs < self.gamestate.timestamp]
        self.gamestate.counters[core.Counters.EVENT_CANDIDATES_THROTTLED] += float(len(characters) - len(candidates))
        if len(candidates) == 0:
            self.gamestate.counters[core.Counters.EVENTS_TOTAL_THROTTLED] += 1
            self.logger.debug(f'skipping event {self.event_type_lookup[event_type]} ({event_type}) with no non-throttled candidates')
            return
        self.logger.debug(f'enqueuing event {self.event_type_lookup[event_type]} ({event_type})')
        self.event_state.event_queue.append((
            narrative.Event(
                event_type,
                context,
                self.gamestate.entity_context_store,
                event_args,
            ),
            candidates
        ))

    def trigger_event_immediate(
        self,
        characters: Iterable[core.Character],
        event_type: int,
        context: Mapping[int, int],
        event_args: dict[str, Union[int,float,str,bool]],
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
        while len(self.event_state.event_queue) > 0:
            event, candidates = self.event_state.event_queue.popleft()
            actions_processed += self._do_event(event, candidates)
            events_processed += 1

        for event, action in self.event_state.action_schedule.pop_current_tasks(self.gamestate.timestamp):
            self._do_action(event, action)
            actions_processed += 1

        self.gamestate.counters[core.Counters.EVENTS_PROCESSED] += events_processed
        self.gamestate.counters[core.Counters.EVENT_ACTIONS_PROCESSED] += actions_processed

    def _do_event(self, event: narrative.Event, candidates:Iterable["narrative.CharacterCandidate[core.Character]"]) -> int:
        actions_processed = 0
        self.logger.debug(f'evaluating event {self.event_type_lookup[event.event_type]} ({event.event_type}) for {list(x.data.short_id() for x in candidates)}')
        actions:list[narrative.Action] = []
        for director in self.directors:
            actions.extend(director.evaluate(event, candidates))

        for action in actions:
            self.event_state.last_event_trigger[action.character_candidate.data.entity_id][event.event_type] = self.gamestate.timestamp
            self.logger.debug(f'triggered action {self.action_id_lookup[action.action_id]} ({action.action_id}) for {action.character_candidate.data.short_id()}')

            if "_delay" in action.args:
                delay = action.args["_delay"]
                assert(isinstance(delay, int) or isinstance(delay, float))
                self.logger.debug(f'delaying action {self.action_id_lookup[action.action_id]} {action.action_id} by {delay}')
                self.event_state.action_schedule.push_task(
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
    def __init__(self, dialog:dialog.DialogGraph, gamestate:core.Gamestate, character: core.Character, speaker: core.Character) -> None:
        self.dialog = dialog
        self.current_id = dialog.root_id
        self.gamestate = gamestate
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
            self.character.context.set_flag(self.gamestate.event_manager.f(flag), 1)
        for flag in self.dialog.nodes[self.current_id].speaker_flags:
            self.speaker.context.set_flag(self.gamestate.event_manager.f(flag), 1)

    def choose(self, choice:dialog.DialogChoice) -> None:
        if choice not in self.choices:
            raise ValueError(f'given choice is not current node {self.current_id} choices')

        for flag in choice.flags:
            self.character.context.set_flag(self.gamestate.event_manager.f(flag), 1)
        for flag in choice.speaker_flags:
            self.speaker.context.set_flag(self.gamestate.event_manager.f(flag), 1)
        self.current_id = choice.node_id
