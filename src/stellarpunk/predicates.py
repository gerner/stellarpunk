""" A boolean logic predicate library """

import abc
from typing import TypeVar, Generic

T = TypeVar('T')

class Criteria(Generic[T], abc.ABC):
    @abc.abstractmethod
    def evaluate(self, universe:T) -> bool: ...

class Literal(Criteria[T]):
    def __init__(self, value:bool) -> None:
        self.value = value
    def evaluate(self, universe:T) -> bool:
        return self.value

class Negation(Criteria[T]):
    def __init__(self, inner:Criteria[T]) -> None:
        self.inner = inner

    def evaluate(self, universe:T) -> bool:
        return not self.inner.evaluate(universe)

class Disjunction(Criteria[T]):
    def __init__(self, a:Criteria[T], b:Criteria[T]) -> None:
        self.a = a
        self.b = b

    def evaluate(self, universe:T) -> bool:
        return self.a.evaluate(universe) or self.b.evaluate(universe)

class Conjunction(Criteria[T]):
    def __init__(self, a:Criteria[T], b:Criteria[T]) -> None:
        self.a = a
        self.b = b

    def evaluate(self, universe:T) -> bool:
        return self.a.evaluate(universe) and self.b.evaluate(universe)


