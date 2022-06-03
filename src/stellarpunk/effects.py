""" Effects that take place as interactions between ships/inside a sector """

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from stellarpunk import core, util

AMOUNT_EPS = 0.5
TRANSFER_PERIOD = 0.5

class MiningEffect(core.Effect):
    def __init__(self, resource:int, amount:float, source:core.Asteroid, destination:core.SectorEntity, *args: Any, mining_rate:float=1e2, max_distance=2.5e3, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.resource = resource
        self.amount = amount
        self.sofar = 0.
        self.mining_rate = mining_rate
        self.max_distance=max_distance

        self._completed_transfer = False

        self.source = source
        self.destination = destination

        # next timestamp we should act on
        self.next_effect_time = 0.

        # amount held in escrow between actions
        self.escrow = 0.

    def bbox(self) -> Tuple[float, float, float, float]:
        locs = np.asarray((self.source.loc, self.destination.loc))
        min_x, min_y = np.min(locs, axis=0)
        max_x, max_y = np.max(locs, axis=0)
        return (min_x, max_y, max_x, min_y)

    def is_complete(self) -> bool:
        return self._completed_transfer

    def act(self, dt:float) -> None:
        if self.gamestate.timestamp < self.next_effect_time:
            return

        if self.destination.distance_to(self.source) > self.max_distance or self.sofar == self.amount:
            self._completed_transfer = True
            return

        assert self.source.resource == self.resource

        if self.destination.cargo_capacity - np.sum(self.destination.cargo) < self.escrow:
            self.logger.info(f'dropping {self.escrow - np.sum(self.destination.cargo)} units of resource {self.resource} because no more cargo space')
            self.escrow = self.destination.cargo_capacity - np.sum(self.destination.cargo)

        self.destination.cargo[self.resource] += self.escrow
        self.sofar += self.escrow
        self.escrow = 0.

        if self.destination.cargo_full() or self.source.amount <= 0.:
            self._completed_transfer = True
            return

        amount = min(
                self.destination.cargo_capacity - np.sum(self.destination.cargo),
                util.clip(self.amount-self.sofar, 0, self.source.amount)
        )
        amount = min((self.mining_rate * TRANSFER_PERIOD), amount)
        self.escrow = amount
        self.source.amount -= amount
        if self.source.amount < AMOUNT_EPS:
            self.source.amount = 0.

        self.next_effect_time = self.gamestate.timestamp + TRANSFER_PERIOD

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
                util.clip(self.amount-self.sofar, 0, self.source.cargo[self.resource])
        )
        amount = min((self.transfer_rate * dt), amount)
        self.source.cargo[self.resource] -= amount
        if np.isclose(self.source.cargo[self.resource], 0.):
            self.source.cargo[self.resource] = 0.
        assert self.source.cargo[self.resource] >= 0.
        self.sofar += amount
        self.destination.cargo[self.resource] += amount
