""" Effects that take place as interactions between ships/inside a sector """

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import numpy.typing as npt

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
            #TODO: this is not really complete, how to communicate that?
            self._completed_transfer = True
            return

        if self.sofar == self.amount:
            self._completed_transfer = True
            return

        amount = min(
                self.destination.cargo_capacity - np.sum(self.destination.cargo),
                util.clip(self.amount-self.sofar, 0, self.source.cargo[self.resource])
        )
        amount = min((self.transfer_rate * TRANSFER_PERIOD), amount)

        if not self._continue_transfer(amount):
            #TODO: this is not really complete, how to communicate that?
            self._completed_transfer = True
            return

        self.source.cargo[self.resource] -= amount
        if self.source.cargo[self.resource] < AMOUNT_EPS:
            self.source.cargo[self.resource] = 0.
        self.destination.cargo[self.resource] += amount
        self.sofar += amount
        self._deliver(amount)
        self.next_effect_time = self.gamestate.timestamp + TRANSFER_PERIOD

    def _continue_transfer(self, amount:float) -> bool:
        """ Called during the transfer to see if transfer should continue.

        amount: the next amount to transfer atomically
        return true iff transfer should continue
        """

        return amount > 0 and self.destination.cargo_capacity - float(np.sum(self.destination.cargo)) >= amount

    def _deliver(self, amount:float) -> None:
        """ Called after one unit of transfer is completed. """
        pass

class TradeTransferEffect(TransferCargoEffect):
    #TODO: do we want to log trading somewhere?
    def __init__(self, buyer:core.Character, seller:core.Character, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.buyer = buyer
        self.seller = seller
        self.value = 0.

    def _continue_transfer(self, amount:float) -> bool:
        return super()._continue_transfer(amount) and self._continue_trade(amount)

    def _deliver(self, amount:float) -> None:
        #TODO: make sure the buyer still wants to buy more at the given price
        value = self._current_price() * amount
        self.buyer.balance -= value
        self.seller.balance += value

    def _continue_trade(self, amount:float) -> bool:
        price = self._current_price()
        value = amount * price
        return self.buyer.balance >= value

    def _current_price(self) -> float:
        return 0.

class SellToStationEffect(TradeTransferEffect):
    def __init__(self, floor_price:float, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.floor_price = floor_price

    def _continue_trade(self, amount:float) -> bool:
        price = self._current_price()
        value = amount * price
        #return price >= self.floor_price and self.destination.budget[self.resource] >= value and self.buyer.balance >= value
        return True


    def _current_price(self) -> float:
        #return self.destination.price[self.resource]
        return True

    def _deliver(self, amount:float) -> None:
        price = self._current_price()
        value = amount * price
        #self.destination.budget[self.resource] -= value

class BuyFromStationEffect(TradeTransferEffect):
    def __init__(self, ceiling_price:float, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.ceiling_price = ceiling_price

    def _continue_trade(self, amount:float) -> bool:
        price = self._current_price()
        value = amount * price
        return price <= self.ceiling_price and self.buyer.balance >= value

    def _current_price(self) -> float:
        #return self.source.price[self.resource]
        return True

class MiningEffect(TransferCargoEffect):
    """ Subclass of TransferCargoEffect to get different visuals. """
    #TODO: do we want to log mining somewhere?
    pass

class WarpOutEffect(core.Effect):
    def __init__(self, loc:npt.NDArray[np.float64], *args:Any, radius:float=1e4, ttl:float=2., **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.loc = loc
        self.radius = radius
        self.ttl = ttl
        self.expiration_time = np.inf

    def _begin(self) -> None:
        self.expiration_time = self.gamestate.timestamp + self.ttl

    def bbox(self) -> Tuple[float, float, float, float]:
        ll = self.loc - self.radius
        ur = self.loc + self.radius
        return (ll[0], ll[1], ur[0], ur[1])

    def is_complete(self) -> bool:
        return self.gamestate.timestamp > self.expiration_time

class WarpInEffect(core.Effect):
    def __init__(self, loc:npt.NDArray[np.float64], *args:Any, radius:float=1e4, ttl:float=2., **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.loc = loc
        self.radius = radius
        self.ttl = ttl
        self.expiration_time = np.inf

    def _begin(self) -> None:
        self.expiration_time = self.gamestate.timestamp + self.ttl

    def bbox(self) -> Tuple[float, float, float, float]:
        ll = self.loc - self.radius
        ur = self.loc + self.radius
        return (ll[0], ll[1], ur[0], ur[1])

    def is_complete(self) -> bool:
        return self.gamestate.timestamp > self.expiration_time

