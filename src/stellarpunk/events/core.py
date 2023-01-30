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
from typing import Iterable, Any, Sequence, Mapping, Dict, Tuple, Optional, Union, Deque, Callable

from stellarpunk import core, util, dialog, predicates, task_schedule
from stellarpunk import narrative


class AbstractEventManager:
    def trigger_event(
        self,
        characters: Iterable[core.Character],
        event_type: int,
        context: narrative.EventContext,
        *entities: core.Entity,
        **kwargs: Any,
    ) -> None:
        pass

    def tick(self) -> None:
        pass


class Action:
    def __init__(self) -> None:
        self.gamestate: core.Gamestate = None # type: ignore[assignment]

    def validate(self, action_args: Mapping[str, Any]) -> bool:
        """ Validate a set of action_args associated with a configured instance
        of the action.

        This is useful to help catch event configuration errors (e.g. missing
        or malformed required params). """
        return True

    def initialize(self, gamestate: core.Gamestate) -> None:
        self.gamestate = gamestate

    def act(
        self,
        character: core.Character,
        event_type: int,
        event_context: narrative.EventContext,
        entities: Mapping[int, core.Entity],
        event_args: Mapping[str, Any],
        action_args: Mapping[str, Any]
    ) -> None:
        pass


RegisteredEventSpaces: Dict[enum.EnumMeta, int] = {}
RegisteredContextSpaces: Dict[enum.EnumMeta, int] = {}
RegisteredActions: Dict[Action, str] = {}


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
    RegisteredEventSpaces[events] = -1


def register_context_keys(context_keys: enum.EnumMeta) -> None:
    if len(context_keys) > 0:
        if not all(isinstance(x, int) for x in context_keys): # type: ignore[var-annotated]
            raise ValueError("members of context keys must all be int-like")
        if min(context_keys) not in [0, 1]:
            raise ValueError("context keys must start at 0 or 1")
        if max(context_keys) > len(context_keys):
            raise ValueError("context keys must be continuous")
    RegisteredContextSpaces[context_keys] = -1


def register_action(action: Action, name: Optional[str] = None) -> None:
    if name is None:
        name = util.camel_to_snake(action.__class__.__name__)
        if name.endswith("_action"):
            name = name[:-len("_action")]
    RegisteredActions[action] = name


"""
events.register_events(MyEvents)
events.register_context_keys(MyContextKeys)

events.register_action(A()) for A in [Action1, Action2, Action3]

trigger_event([dude], events.e(MyEvents.coolio), narrative.context({events.ck(MyContextKeys.foo): 27}), bob, billy, stuff="yes", other_stuff=42)
"""

class EventManager(AbstractEventManager):
    def __init__(
        self,
    ) -> None:
        self.logger = logging.getLogger(util.fullname(self))
        self.gamestate: core.Gamestate = None # type: ignore[assignment]
        self.director: narrative.Director = None # type: ignore[assignment]
        self.event_queue: Deque[Tuple[narrative.Event, Iterable[narrative.CharacterCandidate]]] = collections.deque()
        self.actions: Dict[int, Action] = {}
        self.event_types: Dict[str, int] = {}
        self.context_keys: Dict[str, int] = {}
        self.action_ids: Dict[str, int] = {}

    def initialize(self, gamestate: core.Gamestate, events: Mapping[str, Any]) -> None:
        self.gamestate = gamestate

        # assign integer ids for events, contexts, actions
        action_validators: Dict[int, Callable[[Mapping], bool]] = {}

        event_offset = 0
        for event_enum in RegisteredEventSpaces:
            RegisteredEventSpaces[event_enum] = event_offset
            for event_key in event_enum: # type: ignore[var-annotated]
                self.event_types[util.camel_to_snake(event_key.name)] = event_key + event_offset
            event_offset += max(event_enum)+1

        context_key_offset = 0
        for context_enum in RegisteredContextSpaces:
            RegisteredContextSpaces[context_enum] = context_key_offset
            for context_key in context_enum: # type: ignore[var-annotated]
                self.context_keys[util.camel_to_snake(context_key.name)] = context_key.value + context_key_offset
            context_key_offset += max(context_enum)+1

        action_count = 0
        for action, action_name in RegisteredActions.items():
            action.initialize(self.gamestate)
            self.action_ids[action_name] = action_count
            self.actions[action_count] = action
            action_validators[action_count] = action.validate
            action_count += 1

        self.director = narrative.loadd(events, self.event_types, self.context_keys, self.action_ids, action_validators)

    def trigger_event(
        self,
        characters: Iterable[core.Character],
        event_type: int,
        context: narrative.EventContext,
        *entities: core.Entity,
        **kwargs: Any,
    ) -> None:
        entity_context: Dict[int, narrative.EventContext] = {}
        entity_dict: Dict[int, core.Entity] = {}
        for entity in entities:
            entity_context[entity.short_id_int()] = entity.context
            entity_dict[entity.short_id_int()] = entity
        self.event_queue.append((
            narrative.Event(
                event_type,
                context,
                entity_context,
                (entity_dict, kwargs)
            ),
            [narrative.CharacterCandidate(c.context, c) for c in characters]
        ))

    def tick(self) -> None:
        # check for relevant events and process them
        events_processed = 0
        actions_processed = 0
        while len(self.event_queue) > 0:
            event, candidates = self.event_queue.popleft()

            for action in self.director.evaluate(event, candidates):
                self.logger.debug(f'processing action {action.action_id}')
                delay = action.args.get("_delay", 0)
                self._do_action(event, action)
                actions_processed += 1

            events_processed += 1

        self.gamestate.counters[core.Counters.EVENTS_PROCESSED] += events_processed
        self.gamestate.counters[core.Counters.EVENT_ACTIONS_PROCESSED] += actions_processed

    def _do_action(self, event: narrative.Event, action: narrative.Action) -> None:
        s_action = self.actions[action.action_id]
        s_action.act(
            action.character_candidate.data,
            event.event_type,
            event.event_context,
            event.args[0],
            event.args[1],
            action.args
        )


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
        #for node_event_id in self.dialog.nodes[self.current_id].event_id:
        #    self.player.set_flag(node_event_id, self.gamestate.timestamp)
        pass

    def choose(self, choice:dialog.DialogChoice) -> None:
        if choice not in self.choices:
            raise ValueError(f'given choice is not current node {self.current_id} choices')

        #for event_id in choice.event_id:
        #    self.player.set_flag(event_id, self.gamestate.timestamp)
        self.current_id = choice.node_id
