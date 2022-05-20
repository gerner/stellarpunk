""" Effects that take place as interactions between ships/inside a sector """

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from stellarpunk import core

class MiningEffect(core.Effect):
    def __init__(self, resource:int, amount:float, source:core.Asteroid, destination:core.SectorEntity, *args: Any, mining_rate:float=1e2, max_distance=2.5e3, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.resource = resource
        self.amount = amount
        self.sofar = 0.
        self.mining_rate = mining_rate
        self.max_distance=max_distance

        self.source = source
        self.destination = destination

    def bbox(self) -> Tuple[float, float, float, float]:
        locs = np.asarray((self.source.loc, self.destination.loc))
        min_x, min_y = np.min(locs, axis=0)
        max_x, max_y = np.max(locs, axis=0)
        return (min_x, max_y, max_x, min_y)

    def is_complete(self) -> bool:
        #TODO: distance between source and target
        return self.destination.distance_to(self.source) > self.max_distance or self.sofar == self.amount or self.source.amount <= 0. or self.destination.cargo_full()

    def act(self, dt:float) -> None:
        assert self.source.resource == self.resource
        #TODO: distance between source and dest?
        amount = min(
                self.destination.cargo_capacity - np.sum(self.destination.cargo),
                np.clip(self.amount-self.sofar, 0, self.source.amount)
        )
        amount = min((self.mining_rate * dt), amount)
        self.source.amount -= amount
        if np.isclose(self.source.amount, 0.):
            self.source.amount = 0.
        assert self.source.amount >= 0.
        self.sofar += amount
        self.destination.cargo[self.resource] += amount

class TransferCargoEffect(core.Effect):
    def __init__(self, resource:int, amount:float, source:core.SectorEntity, destination:core.SectorEntity, *args: Any, transfer_rate:float=1e2, max_distance=2.5e3, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.resource = resource
        self.amount = amount
        self.sofar = 0.
        self.transfer_rate = transfer_rate
        self.max_distance=max_distance

        self.source = source
        self.destination = destination

    def bbox(self) -> Tuple[float, float, float, float]:
        locs = np.asarray((self.source.loc, self.destination.loc))
        min_x, min_y = np.min(locs, axis=0)
        max_x, max_y = np.max(locs, axis=0)
        return (min_x, max_y, max_x, min_y)

    def is_complete(self) -> bool:
        #TODO: distance between source and dest?
        return self.source.distance_to(self.destination) > self.max_distance or self.sofar == self.amount or self.source.cargo[self.resource] <= 0. or self.destination.cargo_full()

    def act(self, dt:float) -> None:
        #TODO: distance between source and dest?
        amount = min(
                self.destination.cargo_capacity - np.sum(self.destination.cargo),
                np.clip(self.amount-self.sofar, 0, self.source.cargo[self.resource])
        )
        amount = min((self.transfer_rate * dt), amount)
        self.source.cargo[self.resource] -= amount
        if np.isclose(self.source.cargo[self.resource], 0.):
            self.source.cargo[self.resource] = 0.
        assert self.source.cargo[self.resource] >= 0.
        self.sofar += amount
        self.destination.cargo[self.resource] += amount
