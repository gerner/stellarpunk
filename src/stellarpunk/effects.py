""" Effects that take place as interactions between ships/inside a sector """

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from stellarpunk import core

class MiningEffect(core.Effect):
    def __init__(self, resource:int, amount:float, source:core.Asteroid, destination:core.SectorEntity, *args: Any, mining_rate:float=1e2, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.resource = resource
        self.amount = amount
        self.sofar = 0.
        self.mining_rate = mining_rate

        self.source = source
        self.destination = destination

    def bbox(self) -> Tuple[float, float, float, float]:
        locs = np.asarray((self.source.loc, self.destination.loc))
        min_x, min_y = np.min(locs, axis=0)
        max_x, max_y = np.max(locs, axis=0)
        return (min_x, max_y, max_x, min_y)

    def is_complete(self) -> bool:
        #TODO: max cargo space?
        return self.sofar == self.amount or self.source.amount <= 0. or self.destination.cargo_full()

    def act(self, dt:float) -> None:
        assert self.source.resource == self.resource
        #TODO: distance between source and dest?
        #TODO: max cargo space?
        amount = min(self.destination.cargo_capacity - np.sum(self.destination.cargo), np.clip(self.amount-self.sofar, 0, self.source.amount)) / (self.mining_rate * dt)
        self.source.amount -= amount
        self.sofar += amount
        self.destination.cargo[self.resource] += amount

class TransferCargoEffect(core.Effect):
    def __init__(self, resource:int, amount:float, source:core.SectorEntity, destination:core.SectorEntity, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.resource = resource
        self.amount = amount
        self.sofar = 0.
        self.transfer_rate = 1e2

        self.source = source
        self.destination = destination

    def bbox(self) -> Tuple[float, float, float, float]:
        locs = np.asarray((self.source.loc, self.destination.loc))
        min_x, min_y = np.min(locs, axis=0)
        max_x, max_y = np.max(locs, axis=0)
        return (min_x, max_y, max_x, min_y)

    def is_complete(self) -> bool:
        #TODO: distance between source and dest?
        #TODO: max cargo space?
        return self.sofar == self.amount or self.source.cargo[self.resource] <= 0. or self.destination.cargo_full()

    def act(self, dt:float) -> None:
        #TODO: max cargo space?
        amount = (
            min(
                self.destination.cargo_capacity - np.sum(self.destination.cargo),
                np.clip(self.amount-self.sofar, 0, self.source.cargo[self.resource])
            )
        ) / (self.transfer_rate * dt)
        self.source.cargo[self.resource] -= amount
        self.sofar += amount
        self.destination.cargo[self.resource] += amount
