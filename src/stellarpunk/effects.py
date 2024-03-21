""" Effects that take place as interactions between ships/inside a sector """

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import numpy.typing as npt

from stellarpunk import core, econ, util, events

AMOUNT_EPS = 0.5
TRANSFER_PERIOD = 1.0

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

    def estimate_eta(self) -> float:
        return self.amount / self.transfer_rate - (self.gamestate.timestamp - self.started_at)

    def _begin(self) -> None:
        amount = self._amount()
        if amount == 0.:
            self.gamestate.schedule_effect_immediate(self, jitter=1.0)
        else:
            self.gamestate.schedule_effect(self.gamestate.timestamp + (amount / self.transfer_rate), self, jitter=1.0)

    def bbox(self) -> Tuple[float, float, float, float]:
        locs = np.asarray((self.source.loc, self.destination.loc))
        min_x, min_y = np.min(locs, axis=0)
        max_x, max_y = np.max(locs, axis=0)
        return (min_x, max_y, max_x, min_y)

    def is_complete(self) -> bool:
        return self._completed_transfer

    def act(self, dt:float) -> None:
        if self._completed_transfer:
            return

        if self.destination.distance_to(self.source) > self.max_distance:
            #TODO: the transfer isn't really complete, how to communicate that?
            self._completed_transfer = True
            return

        amount = self._amount()
        if not self._continue_transfer(amount):
            self.logger.debug(f'aborting transfer')
            #TODO: the transfer isn't really complete, how to communicate that?
            self._completed_transfer = True
            self.complete_effect()#cancel_effect()
            return

        self.sofar += amount
        self._deliver(amount)

        self._completed_transfer = True
        self.complete_effect()

    def _amount(self) -> float:
        amount = min(
                self.destination.cargo_capacity - np.sum(self.destination.cargo),
                util.clip(self.amount-self.sofar, 0, self.source.cargo[self.resource])
        )
        return amount

    def _continue_transfer(self, amount:float) -> bool:
        """ Called during the transfer to see if transfer should continue.

        amount: the next amount to transfer atomically
        return true iff transfer should continue
        """

        if amount <= 0:
            return False
        if self.destination.cargo_capacity - float(np.sum(self.destination.cargo)) < amount:
            return False
        return True

    def _deliver(self, amount:float) -> None:
        """ Called after one unit of transfer is completed. """
        self.source.cargo[self.resource] -= amount
        if self.source.cargo[self.resource] < AMOUNT_EPS:
            self.source.cargo[self.resource] = 0.
        self.destination.cargo[self.resource] += amount

class TradeTransferEffect(TransferCargoEffect):
    #TODO: do we want to log trading somewhere?
    def __init__(self, buyer:core.EconAgent, seller:core.EconAgent, current_price:econ.PriceFn, *args:Any, floor_price:float=0., ceiling_price:float=np.inf, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.buyer = buyer
        self.seller = seller
        self.floor_price = floor_price
        self.ceiling_price = ceiling_price
        self.current_price = current_price

    def _amount(self) -> float:
        amount = super()._amount()
        price = self.current_price(self.buyer, self.seller, self.resource)

        if amount * price > self.buyer.budget(self.resource):
            amount = np.floor(self.buyer.budget(self.resource) / price)
        if amount * price > self.buyer.balance():
            amount = np.floor(self.buyer.balance() / price)

        # always pick a non-zero amount so we can bail if things are
        # unaffordable later
        if amount < 1.:
            amount = 1.
        return amount

    def _continue_transfer(self, amount:float) -> bool:
        if not super()._continue_transfer(amount):
            return False
        if not self._continue_trade(amount):
            return False
        return True

    def _deliver(self, amount:float) -> None:
        #TODO: make sure the buyer still wants to buy more at the given price
        price = self.current_price(self.buyer, self.seller, self.resource)
        self.gamestate.transact(self.resource, self.buyer, self.seller, price, amount)

    def _continue_trade(self, amount:float) -> bool:
        #TODO: what if the agents are invalid now (e.g. change agent for the station)
        #TODO: what if one or both parties has been destroyed?
        price = self.current_price(self.buyer, self.seller, self.resource)
        return econ.trade_valid(
                self.buyer, self.seller, self.resource, price, amount,
                self.floor_price, self.ceiling_price)


class MiningEffect(TransferCargoEffect):
    """ Subclass of TransferCargoEffect to get different visuals. """
    #TODO: do we want to log mining somewhere?
    def _deliver(self, amount: float) -> None:
        super()._deliver(amount)
        if self.destination.captain is not None:
            self.gamestate.trigger_event(
                [self.destination.captain],
                events.Events.MINED,
                {
                    events.ContextKeys.TARGET: self.source.short_id_int(),
                    events.ContextKeys.RESOURCE: self.resource,
                    events.ContextKeys.AMOUNT: int(amount),
                    events.ContextKeys.AMOUNT_ON_HAND: int(self.destination.cargo[self.resource]),
                },
            )


class WarpOutEffect(core.Effect):
    def __init__(self, loc:npt.NDArray[np.float64], *args:Any, radius:float=1e4, ttl:float=2., **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.loc = loc
        self.radius = radius
        self.ttl = ttl
        self.expiration_time = np.inf

    def _begin(self) -> None:
        self.expiration_time = self.gamestate.timestamp + self.ttl
        self.gamestate.schedule_effect(self.expiration_time + self.gamestate.desired_dt, self)

    def bbox(self) -> Tuple[float, float, float, float]:
        ll = self.loc - self.radius
        ur = self.loc + self.radius
        return (ll[0], ll[1], ur[0], ur[1])

    def is_complete(self) -> bool:
        return self.gamestate.timestamp >= self.expiration_time

class WarpInEffect(core.Effect):
    def __init__(self, loc:npt.NDArray[np.float64], *args:Any, radius:float=1e4, ttl:float=2., **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.loc = loc
        self.radius = radius
        self.ttl = ttl
        self.expiration_time = np.inf

    def _begin(self) -> None:
        self.expiration_time = self.gamestate.timestamp + self.ttl
        self.gamestate.schedule_effect(self.expiration_time + self.gamestate.desired_dt, self)

    def bbox(self) -> Tuple[float, float, float, float]:
        ll = self.loc - self.radius
        ur = self.loc + self.radius
        return (ll[0], ll[1], ur[0], ur[1])

    def is_complete(self) -> bool:
        return self.gamestate.timestamp >= self.expiration_time

