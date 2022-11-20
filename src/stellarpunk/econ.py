""" Stuff facilitating economic modelling. """

import abc
from typing import Callable, Collection, Tuple

import numpy as np
import numpy.typing as npt

from stellarpunk import core, util

#TODO: unify this with the one in effects
AMOUNT_EPS = 0.5
PRICE_EPS = 1e-05

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

def trade_valid(buyer:core.EconAgent, seller:core.EconAgent, resource:int, price:float, amount:float, floor_price:float=0., ceiling_price:float=np.inf) -> bool:
    if price < floor_price:
        return False
    if price > ceiling_price:
        return False
    if amount > seller.inventory(resource):
        return False
    value = amount * price
    if buyer.balance()+PRICE_EPS < value:
        return False
    if buyer.budget(resource)+PRICE_EPS < value:
        return False
    return True

PriceFn = Callable[[core.EconAgent, core.EconAgent, int], float]

def buyer_price(buyer:core.EconAgent, seller:core.EconAgent, resource:int) -> float:
    return buyer.buy_price(resource)

def seller_price(buyer:core.EconAgent, seller:core.EconAgent, resource:int) -> float:
    return seller.sell_price(resource)

class YesAgent(core.EconAgent):
    """ EconAgent that always buys, sells, has budget, etc. """
    def __init__(self, production_chain:core.ProductionChain) -> None:
        self._resources = tuple(range(production_chain.num_products))

    def buy_resources(self) -> Collection:
        return self._resources

    def sell_resources(self) -> Collection:
        return self._resources

    def buy_price(self, resource:int) -> float:
        return np.inf

    def sell_price(self, resource:int) -> float:
        return np.inf

    def balance(self) -> float:
        return np.inf

    def budget(self, resource:int) -> float:
        return np.inf

    def inventory(self, resource:int) -> float:
        return np.inf

    def buy(self, resource:int, price:float, amount:float) -> None:
        raise ValueError("do not trade with the YesAgent")

    def sell(self, resource:int, price:float, amount:float) -> None:
        raise ValueError("do not trade with the YesAgent")

class StationAgent(core.EconAgent):
    """ Agent for a Station.

    Exposes buy/sell prices and budget for trades.
    """

    @classmethod
    def create_station_agent(cls, station:core.Station, production_chain:core.ProductionChain) -> "StationAgent":
        if station.resource is None:
            raise ValueError(f'cannot create station agent for station that has no resource')

        if station.owner is None:
            raise ValueError(f'cannot create station agent for station that has no owner')

        station_agent = StationAgent(station, station.owner, production_chain)

        resource = station.resource
        inputs = production_chain.inputs_of(resource)

        station_agent._buy_resources = tuple(inputs) # type: ignore
        station_agent._sell_resources = (resource,)

        # start with buy price half the markup on our resource, reserving the
        # other half for the trader to transport it
        inputs_markup = 1 + (0.5 * (production_chain.markup[resource]-1))
        assert inputs_markup > 1
        station_agent._buy_price[inputs] = production_chain.prices[inputs] * inputs_markup

        station_agent._sell_price[resource] = production_chain.prices[resource]

        station_agent._budget[inputs] = np.inf

        return station_agent

    @classmethod
    def create_planet_agent(cls, planet:core.Planet,  production_chain:core.ProductionChain) -> "StationAgent":
        if planet.owner is None:
            raise ValueError(f'cannot create station agent for planet that has no owner')

        station_agent = StationAgent(planet, planet.owner, production_chain)
        consumer_product_id = production_chain.final_product_ids()[core.RESOURCE_REL_CONSUMER]
        station_agent._buy_resources = (consumer_product_id,)

        # start with buy price the markup for the consumer good again
        station_agent._buy_price[consumer_product_id] = production_chain.prices[consumer_product_id] * production_chain.markup[consumer_product_id]
        station_agent._budget[consumer_product_id] = np.inf

        return station_agent

    def __init__(self, station:core.SectorEntity, owner:core.Character, production_chain:core.ProductionChain) -> None:
        self._buy_resources:Tuple[int] = tuple() # type: ignore
        self._sell_resources:Tuple[int] = tuple() # type: ignore

        self._buy_price = np.zeros((production_chain.num_products,))
        self._sell_price = np.full((production_chain.num_products,), np.inf)
        self._budget = np.zeros((production_chain.num_products,))
        self.station = station
        self.owner = owner

    def buy_resources(self) -> Collection:
        return self._buy_resources

    def sell_resources(self) -> Collection:
        return self._sell_resources

    def buy_price(self, resource:int) -> float:
        if self.station.cargo.sum()+AMOUNT_EPS > self.station.cargo_capacity:
            return 0.
        else:
            return self._buy_price[resource]

    def sell_price(self, resource:int) -> float:
        return self._sell_price[resource]

    def balance(self) -> float:
        assert self.owner is not None
        return self.owner.balance

    def budget(self, resource:int) -> float:
        return self._budget[resource]

    def inventory(self, resource:int) -> float:
        return self.station.cargo[resource]

    def buy(self, resource:int, price:float, amount:float) -> None:
        assert self.owner is not None
        value = price * amount

        assert self._budget[resource] >= value
        assert self.balance()+PRICE_EPS >= value
        assert self.station.cargo.sum() <= self.station.cargo_capacity

        self._budget[resource] -= value
        self.station.cargo[resource] += amount
        self.owner.balance -= value
        if util.isclose(self.owner.balance, 0.):
            self.owner.balance = 0.

    def sell(self, resource:int, price:float, amount:float) -> None:
        assert self.owner is not None
        assert self.inventory(resource) >= amount

        self.station.cargo[resource] -= amount
        if util.isclose(self.station.cargo[resource], 0.):
            self.station.cargo[resource] = 0.
        self.owner.balance += price * amount

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

        assert self.balance()+PRICE_EPS >= value
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

