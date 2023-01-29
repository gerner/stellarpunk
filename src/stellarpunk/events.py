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
from typing import Iterable, Any, Sequence, Mapping, Dict, Tuple, Optional, Union, Deque

from stellarpunk import core, util, dialog, predicates, config, task_schedule
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
        pass

    def act(
        self,
        character: core.Character,
        event_type: int,
        event_context: narrative.EventContext,
        entity_context: Mapping[int, narrative.EventContext],
        event_args: Mapping[str, Any],
        action_args: Mapping[str, Any]
    ) -> None:
        pass


"""
class BroadcastAction:
    def __init__(
        self,
        sector: core.Sector,
        loc: Tuple[float, float],
        message_id: int,
        message: str,
        *args: Any,
        radius: float = 5e3,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.sector = sector
        self.loc = loc
        self.radius = radius
        self.message_id = message_id
        self.message = message

    def act(self) -> None:
        nearby_characters = list(
            x.captain for x in self.sector.spatial_point(self.loc, self.radius) if x.captain is not None and x.captain != self.character
        )

        # TODO: this triggers many events, one for each character in range.
        # maybe we actually want to trigger a single event, but associate it
        # somehow with all of these potential characters.
        for receiver in nearby_characters:
            destination = self.event.get_entity(core.ContextKey.DESTINATION)
            self.gamestate.trigger_event(
                receiver,
                core.EventType.BROADCAST,
                narrative.context({
                    core.ContextKey.MESSAGE_SENDER: self.character.short_id_int(),
                    core.ContextKey.MESSAGE_ID: self.message_id,
                    core.ContextKey.DESTINATION: self.event.context.get_flag(core.ContextKey.DESTINATION),
                    core.ContextKey.SHIP: self.event.context.get_flag(core.ContextKey.SHIP),
                }),
                self.character,
                self.event.get_entity(core.ContextKey.DESTINATION),
                self.event.get_entity(core.ContextKey.SHIP),
                message=self.message,
            )
"""


RegisteredEventSpaces: Dict[enum.EnumMeta, int] = {}
RegisteredContextSpaces: Dict[enum.EnumMeta, int] = {}
RegisteredActions: Dict[Action, str] = {}


def e(event_id: enum.IntEnum) -> int:
    return event_id + RegisteredEventSpaces[event_id.__class__]


def ck(context_key: enum.IntEnum) -> int:
    return context_key + RegisteredContextSpaces[context_key.__class__]


def register_events(events: enum.EnumMeta) -> None:
    RegisteredEventSpaces[events] = -1


def register_context_keys(context_keys: enum.EnumMeta) -> None:
    RegisteredContextSpaces[context_keys] = -1


def register_action(action: Action, name: Optional[str] = None) -> None:
    if name is None:
        name = util.camel_to_snake(action.__class__.__name__)
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
        self.actions:Mapping[int, Action] = {}

    def initialize(self, gamestate:core.Gamestate) -> None:
        self.gamestate = gamestate


        # assign integer ids for events, contexts, actions
        event_types: Dict[str, int] = {}
        context_keys: Dict[str, int] = {}
        action_ids: Dict[str, int] = {}
        actions: Dict[int, Action] = {}

        event_offset = 0
        for event_enum in RegisteredEventSpaces:
            RegisteredEventSpaces[event_enum] = event_offset
            for event_key in event_enum: # type: ignore[var-annotated]
                event_types[event_key.name] = event_key + event_offset
            event_offset += max(event_enum)+1

        context_key_offset = 0
        for context_enum in RegisteredContextSpaces:
            RegisteredContextSpaces[context_enum] = context_key_offset
            for context_key in context_enum: # type: ignore[var-annotated]
                context_keys[context_key.name] = context_key.value + context_key_offset
            context_key_offset += max(context_enum)+1

        action_count = 0
        for action, action_name in RegisteredActions.items():
            action_ids[action_name] = action_count
            actions[action_count] = action
            action_count += 1

        self.director = narrative.loadd(config.Events, event_types, context_keys, action_ids)
        self.actions = actions

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
            event.entity_context,
            event.args,
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
