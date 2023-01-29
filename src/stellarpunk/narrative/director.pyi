from typing import Mapping, List, Iterable, Any


class EventContext:
    def __init__(self) -> None: ...
    def set_flag(self, flag: int, value: int) -> None: ...
    def get_flag(self, flag: int) -> int: ...


def context(c: Mapping[int, int]) -> EventContext: ...


class FlagCriteria:
    def __init__(self, flag: int, low: int, high: int) -> None: ...

    @property
    def fact(self) -> int: ...
    @property
    def low(self) -> int: ...
    @property
    def high(self) -> int: ...


class EntityCriteria:
    def __init__(self, ef: int, sf: int, l: int, h: int) -> None: ...

    @property
    def entity_fact(self) -> int: ...
    @property
    def sub_fact(self) -> int: ...
    @property
    def low(self) -> int: ...
    @property
    def high(self) -> int: ...


class ActionTemplate:
    def __init__(self, action_id: int, args: Any) -> None: ...


class Rule:
    def __init__(
        self,
        event_type: int,
        priority: int,
        criteria: List[FlagCriteria] = [],
        entity_criteria: List[EntityCriteria] = [],
        actions: List[ActionTemplate] = [],
    ) -> None: ...


class Event:
    event_type: int
    event_context: EventContext
    entity_context: Mapping[int, EventContext]
    args: Any

    def __init__(
        self,
        event_type: int,
        event_context: EventContext,
        entity_context: Mapping[int, EventContext],
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
