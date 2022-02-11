""" Effects that take place as interactions between ships/inside a sector """

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from stellarpunk import core

class MiningEffect(core.Effect):
    def __init__(self, resource:int, amount:float, source:core.Asteroid, destination:core.SectorEntity, *args: Any, **kwargs: Any) -> None:
        super(*args, **kwargs)
        self.resource = resource
        self.amount = amount
        self.sofar = 0.
        self.resource_per_second = 1e2

        self.source = source
        self.destination = destination

    def bbox(self) -> Tuple[float, float, float, float]:
        locs = np.asarray((self.source.loc, self.destination.loc))
        min_x, min_y = np.min(locs, axis=0)
        max_x, max_y = np.max(locs, axis=0)
        return (min_x, max_y, max_x, min_y)

    def is_complete(self) -> bool:
        #TODO: max cargo space?
        return self.sofar == self.amount or self.source.amount <= 0.

    def act(self, dt:float) -> None:
        assert self.source.resource == self.resource
        #TODO: distance between source and dest?
        #TODO: max cargo space?
        amount = np.clip(self.amount-self.sofar, 0, self.source.amount) / (self.resource_per_second * dt)
        self.source.amount -= amount
        self.sofar += amount
        self.destination.cargo[self.resource] += amount

class TransferCargoEffect(core.Effect):
    def __init__(self, resource:int, amount:float, source:core.SectorEntity, destination:core.SectorEntity, *args: Any, **kwargs: Any) -> None:
        super(*args, **kwargs)
        self.resource = resource
        self.amount = amount
        self.sofar = 0.
        self.resource_per_second = 1e2

        self.source = source
        self.destination = destination

    def bbox(self) -> Tuple[float, float, float, float]:
        locs = np.asarray((self.source, self.destination))
        min_x, min_y = np.min(locs, axis=0)
        max_x, max_y = np.max(locs, axis=0)
        return (min_x, max_y, max_x, min_y)

    def is_complete(self) -> bool:
        #TODO: distance between source and dest?
        #TODO: max cargo space?
        return self.sofar == self.amount or self.source.cargo[self.resource] <= 0.

    def act(self, dt:float) -> None:
        #TODO: max cargo space?
        amount = np.clip(self.amount-self.sofar, 0, self.source.cargo[self.resource]) / (self.resource_per_second * dt)
        self.source.cargo[self.resource] -= amount
        self.sofar += amount
        self.destination.cargo[self.resource] += amount
