from typing import Mapping, List, Iterable, Any, Union


class EventContext:
    def __init__(self) -> None: ...
    def set_flag(self, flag: int, value: int) -> None: ...
    def get_flag(self, flag: int) -> int: ...


class EntityStore:
    def register_entity(self, entity_id: int) -> EventContext: ...
    def unregister_entity(self, entity_id: int) -> None: ...


class IntRef:
    def __init__(self, value: int) -> None: ...
    @property
    def value(self) -> int: ...


class FlagRef:
    def __init__(self, flag: int) -> None: ...
    @property
    def fact(self) -> int: ...


class EntityRef:
    def __init__(self, entity_fact: int, sub_fact: int) -> None: ...
    @property
    def entity_fact(self) -> int: ...
    @property
    def sub_fact(self) -> int: ...


class CriteriaBuilder:
    last_low: Union[IntRef, FlagRef, EntityRef]
    last_fact: Union[IntRef, FlagRef, EntityRef]
    last_high: Union[IntRef, FlagRef, EntityRef]

    def add_low(self, low: Union[IntRef, FlagRef, EntityRef]) -> None: ...
    def add_fact(self, fact: Union[IntRef, FlagRef, EntityRef]) -> None: ...
    def add_high(self, high: Union[IntRef, FlagRef, EntityRef]) -> None: ...
    def build(self) -> None: ...

class ActionTemplate:
    def __init__(self, action_id: int, args: Any) -> None: ...


class Rule:
    def __init__(
        self,
        event_type: int,
        priority: int,
        criteria: CriteriaBuilder,
        actions: List[ActionTemplate] = [],
    ) -> None: ...


class Event:
    event_type: int
    event_context: Mapping[int, int]
    entity_context: EntityStore
    args: Any

    def __init__(
        self,
        event_type: int,
        event_context: Mapping[int, int],
        entity_context: EntityStore,
        args: Any
    ) -> None: ...


class CharacterCandidate:
    character_context: EventContext = ...
    data: Any = ...

    def __init__(self, character_context: EventContext, data: Any) -> None: ...


class Action:
    action_id: int = ...
    character_candidate: CharacterCandidate = ...
    args: Any = ...

    def __init__(
        self,
        action_id: int,
        character_candidate: CharacterCandidate,
        args: Any,
    ) -> None: ...


class Director:
    def __init__(self, rules: Mapping[int, Iterable[Rule]]) -> None: ...
    def evaluate(self, event: Event, character_candidates: Iterable[CharacterCandidate]) -> List[Action]: ...
