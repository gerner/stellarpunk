""" Stuff facilitating economic modelling. """

import abc
from typing import Callable, Collection, Tuple

import numpy as np
import numpy.typing as npt

from stellarpunk import core, util

def assign_agents_to_products(gamestate:core.Gamestate, num_agents:int, assign_start:int=0, assign_end:int=-1) -> npt.NDArray[np.float64]:
    """ Creates an assignment matrix from agents to goods.

    gamestate: core.Gamestate
        populated gamestate (e.g. rng and production chain)
    num_agents: int
        how many agents to assign
    assign_start: int, default 0
        product id to start assigning
    assign_end: int, optional
        product id to stop assigning (default is num_products)

    returns: np.ndarray
        assignment matrix (num_agents, num_products)
    """
    num_products = gamestate.production_chain.num_products

    if assign_end < 0:
        assign_end = num_products

    if num_agents < assign_end-assign_start:
        raise ValueError(f'cannot assign fewer agents {num_agents} than desired products {assign_end}-{assign_start}={assign_end-assign_start}')

    # set up agents, at least one in every good, the rest can be random
    agent_goods = np.zeros((num_agents, num_products))
    agent_goods[0:assign_end-assign_start,assign_start:assign_end] = np.eye(num_products)[assign_start:assign_end,assign_start:assign_end]

    # randomly assign goods to the rest of the agents where higher ranked
    # goods are exponentially less likely
    #ranks = gamestate.production_chain.ranks
    #p = np.hstack([[3**(len(ranks)-i)]*v for (i,v) in enumerate(ranks)])
    p = 1.0 / gamestate.production_chain.prices[assign_start:assign_end]
    p = p/p.sum()
    single_products = gamestate.random.choice(a=np.arange(assign_start, assign_end), size=num_agents-(assign_end-assign_start), replace=True, p=p)
    agent_goods[np.arange(assign_end-assign_start, num_agents), single_products] = 1

    # make sure only one product per agent (for now, we might want to lift
    # this simplifying restriction in the future...)
    assert np.all(agent_goods.sum(axis=1) == 1)
    assert np.all(agent_goods[:,assign_start:assign_end].sum(axis=0) >= 1)

    return agent_goods

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
        if station.resource is None:
            raise ValueError("cannot make a StationAgent if station has no resource set")
        self._buy_resources = tuple(np.where(production_chain.adj_matrix[:,station.resource] > 0)[0])
        self._sell_resources:Tuple[int] = (station.resource,)
        self._buy_price = np.zeros((production_chain.num_products,))
        self._sell_price = np.full((production_chain.num_products,), np.inf)
        self._budget = np.zeros((production_chain.num_products,))
        self.station = station

    def buy_resources(self) -> Collection:
        return self._buy_resources

    def sell_resources(self) -> Collection:
        return self._sell_resources

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

EMPTY_TUPLE:Tuple = tuple()

class ShipTraderAgent(core.EconAgent):
    """ Agent for a trader on a Ship.

    Buy/sell prices and budget are irrelevant for this agent. We assume this
    agent is "active" and decisions are handled elsewhere. """

    def __init__(self, ship:core.Ship) -> None:
        self.ship = ship

    def buy_resources(self) -> Collection:
        return EMPTY_TUPLE

    def sell_resources(self) -> Collection:
        return EMPTY_TUPLE

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

