""" Effects that take place as interactions between ships/inside a sector """

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from stellarpunk import core, util

AMOUNT_EPS = 0.5
TRANSFER_PERIOD = 0.5

class TransferCargoEffect(core.Effect):
    def __init__(
            self,
            resource:int, amount:float, source:core.SectorEntity,
            destination:core.SectorEntity,
            *args: Any,
            transfer_rate:float=1e2, max_distance:float=2.5e3,
            **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.resource = resource
        self.amount = amount
        self.sofar = 0.
        self.transfer_rate = transfer_rate
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

        if self.destination.distance_to(self.source) > self.max_distance:
            self.cancel_effect()
            self._completed_transfer = True
            return

        if self.sofar == self.amount:
            self._completed_transfer = True
            return

        if self.destination.cargo_capacity - np.sum(self.destination.cargo) < self.escrow:
            self.logger.info(f'dropping {self.escrow - np.sum(self.destination.cargo)} units of resource {self.resource} because no more cargo space')
            self.escrow = self.destination.cargo_capacity - np.sum(self.destination.cargo)

        self.destination.cargo[self.resource] += self.escrow
        self.sofar += self.escrow
        self.escrow = 0.

        if self.destination.cargo_full() or self.source.cargo[self.resource] <= 0.:
            self._completed_transfer = True
            return

        amount = min(
                self.destination.cargo_capacity - np.sum(self.destination.cargo),
                util.clip(self.amount-self.sofar, 0, self.source.cargo[self.resource])
        )
        amount = min((self.transfer_rate * TRANSFER_PERIOD), amount)
        self.escrow = amount
        self.source.cargo[self.resource] -= amount
        if self.source.cargo[self.resource] < AMOUNT_EPS:
            self.source.cargo[self.resource] = 0.

        self.next_effect_time = self.gamestate.timestamp + TRANSFER_PERIOD

    def _cancel(self) -> None:
        """ cancel the mining effect and return escrow amount back to source.
        """
        if self.escrow > 0:
            #TODO: worry about max capacity at the source?
            self.source.cargo[self.resource] += self.escrow

class MiningEffect(TransferCargoEffect):
    """ Subclass of TransferCargoEffect to get different visuals. """
    pass
