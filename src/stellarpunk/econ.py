""" Stuff facilitating economic modelling. """

import abc
from typing import Callable

import numpy as np

from stellarpunk import core, util

PriceFn = Callable[[core.EconAgent, core.EconAgent, int], float]

def buyer_price(buyer:core.EconAgent, seller:core.EconAgent, resource:int) -> float:
    return buyer.buy_price(resource)

def seller_price(buyer:core.EconAgent, seller:core.EconAgent, resource:int) -> float:
    return seller.sell_price(resource)

class StationAgent(core.EconAgent):
    """ Agent for a Station.

    Exposes buy/sell prices and budget for trades.
    """

    def __init__(self, station:core.Station, production_chain:core.ProductionChain) -> None:
        self._buy_price = np.zeros((production_chain.num_products,))
        self._sell_price = np.full((production_chain.num_products,), np.inf)
        self._budget = np.zeros((production_chain.num_products,))
        self.station = station

    def buy_price(self, resource:int) -> float:
        return self._buy_price[resource]

    def sell_price(self, resource:int) -> float:
        return self._sell_price[resource]

    def balance(self) -> float:
        assert self.station.owner is not None
        return self.station.owner.balance

    def budget(self, resource:int) -> float:
        return self._budget[resource]

    def inventory(self, resource:int) -> float:
        return self.station.cargo[resource]

    def buy(self, resource:int, price:float, amount:float) -> None:
        assert self.station.owner is not None
        value = price * amount

        assert self._budget[resource] >= value
        assert self.balance() >= value
        assert self.station.cargo.sum() <= self.station.cargo_capacity

        self._budget[resource] -= value
        self.station.cargo[resource] += amount
        self.station.owner.balance -= value
        if util.isclose(self.station.owner.balance, 0.):
            self.station.owner.balance = 0.

    def sell(self, resource:int, price:float, amount:float) -> None:
        assert self.station.owner is not None
        assert self.inventory(resource) >= amount

        self.station.cargo[resource] -= amount
        if util.isclose(self.station.cargo[resource], 0.):
            self.station.cargo[resource] = 0.
        self.station.owner.balance += price * amount

class ShipTraderAgent(core.EconAgent):
    """ Agent for a trader on a Ship.

    Buy/sell prices and budget are irrelevant for this agent. We assume this
    agent is "active" and decisions are handled elsewhere. """

    def __init__(self, ship:core.Ship) -> None:
        self.ship = ship

    def buy_price(self, resource:int) -> float:
        return np.inf

    def sell_price(self, resource:int) -> float:
        return 0.

    def balance(self) -> float:
        assert self.ship.owner is not None
        return self.ship.owner.balance

    def budget(self, resource:int) -> float:
        return np.inf

    def inventory(self, resource:int) -> float:
        return self.ship.cargo[resource]

    def buy(self, resource:int, price:float, amount:float) -> None:
        assert self.ship.owner is not None
        value = price * amount

        assert self.balance() >= value
        assert self.ship.cargo.sum() <= self.ship.cargo_capacity

        self.ship.cargo[resource] += amount
        self.ship.owner.balance -= value
        if util.isclose(self.ship.owner.balance, 0.):
            self.ship.owner.balance = 0.

    def sell(self, resource:int, price:float, amount:float) -> None:
        assert self.ship.owner is not None
        assert self.inventory(resource) >= amount

        self.ship.cargo[resource] -= amount
        if util.isclose(self.ship.cargo[resource], 0.):
            self.ship.cargo[resource] = 0.
        self.ship.owner.balance += price * amount

