""" Agenda items for characters, reflecting activites they are involved in. """

from typing import Optional, List, Any, Iterable, Tuple, DefaultDict
import enum
import collections
import itertools

import numpy as np
import numpy.typing as npt

from stellarpunk import core, econ, util
import stellarpunk.orders.core as ocore
from stellarpunk.orders import movement

def choose_station_to_buy_from(
        gamestate:core.Gamestate,
        ship:core.Ship,
        allowed_resources:List[int],
        buy_from_stations:Optional[List[core.SectorEntity]],
        sell_to_stations:Optional[List[core.SectorEntity]]
        ) -> Optional[Tuple[int, core.Station, core.EconAgent]]:

    if ship.sector is None:
        raise ValueError(f'{ship} in no sector')

    # compute diffs of allowed/known stations
    # pick a trade that maximizes profit discounting by time spent travelling
    # and transferring

    # (sell - buy price) * amount
    # transfer time = buy transfer + sell transfer
    # travel time  = time to buy + time to sell

    #TODO: how to get prices and stations without magically having global knowledge?

    # figure out possible buys by resource
    buys:DefaultDict[int, List[Tuple[float, float, core.SectorEntity]]] = collections.defaultdict(list)
    buy_hits:Iterable[core.SectorEntity]
    if buy_from_stations is None:
        buy_hits = ship.sector.spatial_point(ship.loc, mask=core.ObjectFlag.STATION)
    else:
        buy_hits = buy_from_stations
    for hit in buy_hits:
        if not isinstance(hit, core.Station):
            continue
        agent = gamestate.econ_agents[hit.entity_id]
        for resource in agent.sell_resources():
            if resource not in allowed_resources:
                continue
            if not (agent.sell_price(resource) < np.inf):
                continue
            if agent.inventory(resource) == 0.:
                continue

            buys[resource].append((
                agent.sell_price(resource),
                agent.inventory(resource),
                hit,
            ))

    # figure out possible sales by resource
    sales:DefaultDict[int, List[Tuple[float, float, core.SectorEntity]]] = collections.defaultdict(list)
    sale_hits:Iterable[core.SectorEntity]
    if sell_to_stations is None:
        sale_hits = ship.sector.spatial_point(ship.loc, mask=core.ObjectFlag.STATION)
    else:
        sale_hits = sell_to_stations
    for hit in sale_hits:
        if not isinstance(hit, core.Station):
            continue
        agent = gamestate.econ_agents[hit.entity_id]
        for resource in agent.buy_resources():
            if agent.buy_price(resource) <= 0:
                continue
            if agent.budget(resource) <= 0.:
                continue

            sales[resource].append((
                agent.buy_price(resource),
                np.floor(agent.budget(resource)/agent.buy_price(resource)),
                hit,
            ))

    best_profit_per_time = 0.
    best_trade:Optional[Tuple[int, core.Station, core.EconAgent]] = None
    for resource in buys.keys():
        for ((buy_price, buy_amount, buy_station), (sale_price, sale_amount, sale_station)) in itertools.product(buys[resource], sales[resource]):
            amount = min(buy_amount, sale_amount)
            profit = (sale_price - buy_price)*amount
            transfer_time = ocore.TradeCargoFromStation.transfer_rate() * amount + ocore.TradeCargoToStation.transfer_rate() * amount
            travel_time = movement.GoToLocation.compute_eta(ship, buy_station.loc) + movement.GoToLocation.compute_eta(ship, sale_station.loc, starting_loc=buy_station.loc)

            profit_per_time = profit / (transfer_time + travel_time)
            if profit_per_time > best_profit_per_time:
                best_profit_per_time = profit_per_time
                best_trade = (resource, buy_station, gamestate.econ_agents[buy_station.entity_id]) # type: ignore

    return best_trade

def choose_station_to_sell_to(
        gamestate:core.Gamestate,
        ship:core.Ship,
        ship_agent:core.EconAgent,
        allowed_resources:List[int],
        allowed_stations:Optional[List[core.SectorEntity]]
        ) -> Optional[Tuple[int, core.Station, core.EconAgent]]:
    """ Choose a station to sell goods from ship to """

    if ship.sector is None:
        raise ValueError(f'{ship} in no sector')

    # pick the station where we'll get the best profit for our cargo
    # biggest profit-per-tick for our cargo

    #TODO: how do we access price information? shouldn't this be
    # somehow limited to a view specific to us?
    #TODO: what if no stations seem to trade the resources we have?
    #TODO: what if no stations seem to trade any allowed resources?

    # figure out possible sales by resource
    sales:DefaultDict[int, List[Tuple[float, float, core.SectorEntity]]] = collections.defaultdict(list)
    sale_hits:Iterable[core.SectorEntity]
    if allowed_stations is None:
        sale_hits = ship.sector.spatial_point(ship.loc, mask=core.ObjectFlag.STATION)
    else:
        sale_hits = allowed_stations
    for hit in sale_hits:
        if not isinstance(hit, core.Station):
            continue
        agent = gamestate.econ_agents[hit.entity_id]
        for resource in agent.buy_resources():
            price = agent.buy_price(resource)
            if not econ.trade_valid(agent, ship_agent, resource, price, 1.):
                continue

            sales[resource].append((
                agent.buy_price(resource),
                min(
                    np.floor(agent.budget(resource)/agent.buy_price(resource)),
                    ship.cargo[resource],
                ),
                hit,
            ))

    best_profit_per_time = 0.
    best_trade:Optional[Tuple[int, core.Station, core.EconAgent]] = None
    for resource in sales.keys():
        for sale_price, amount, sale_station in sales[resource]:
            profit = sale_price * amount
            transfer_time = ocore.TradeCargoToStation.transfer_rate() * amount
            travel_time = movement.GoToLocation.compute_eta(ship, sale_station.loc)

            profit_per_time = profit / (transfer_time + travel_time)
            if profit_per_time > best_profit_per_time:
                best_profit_per_time = profit_per_time
                best_trade = (resource, sale_station, gamestate.econ_agents[sale_station.entity_id]) # type: ignore

    return best_trade

# how long to wait, idle if we can't do work
MINING_SLEEP_TIME = 60.
TRADING_SLEEP_TIME = 60.

class MiningAgendum(core.Agendum, core.OrderObserver):
    """ Managing a ship for mining.

    Operates as a state machine as we mine asteroids and sell the resources to
    relevant stations. """

    class State(enum.Enum):
        IDLE = enum.auto()
        MINING = enum.auto()
        TRADING = enum.auto()
        COMPLETE = enum.auto()

    def __init__(self, ship:core.Ship, *args:Any, allowed_resources:Optional[List[int]]=None, allowed_stations:Optional[List[core.SectorEntity]]=None, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        self.ship = ship
        self.agent = econ.ShipTraderAgent(ship)

        # resources we're allowed to mine
        if allowed_resources is None:
            allowed_resources = list(range(self.gamestate.production_chain.ranks[0]))
        self.allowed_resources = allowed_resources

        self.allowed_stations:Optional[List[core.SectorEntity]] = allowed_stations

        # state machine to keep track of what we're doing
        self.state:MiningAgendum.State = MiningAgendum.State.IDLE

        # keep track of a couple sorts of actions
        self.mining_order:Optional[ocore.MineOrder] = None
        self.transfer_order:Optional[ocore.TradeCargoToStation] = None

        self.round_trips = 0
        self.max_trips = -1

    def order_begin(self, order:core.Order) -> None:
        pass

    def order_complete(self, order:core.Order) -> None:
        if self.state == MiningAgendum.State.MINING:
            assert order == self.mining_order
            self.mining_order = None
            # go back into idle state to start things off again
            self.state = MiningAgendum.State.IDLE
            self.gamestate.schedule_agendum_immediate(self)
        elif self.state == MiningAgendum.State.TRADING:
            assert order == self.transfer_order
            self.transfer_order = None
            # go back into idle state to start things off again
            self.state = MiningAgendum.State.IDLE
            self.gamestate.schedule_agendum_immediate(self)
            self.round_trips += 1
        else:
            raise ValueError("got order_complete in wrong state {self.state}")

    def order_cancel(self, order:core.Order) -> None:
        if self.state == MiningAgendum.State.MINING:
            assert order == self.mining_order
            self.mining_order = None
        elif self.state == MiningAgendum.State.TRADING:
            assert order == self.transfer_order
            self.transfer_order = None
        else:
            raise ValueError("got order_cancel in wrong state {self.state}")

        self.state = MiningAgendum.State.IDLE
        self.gamestate.schedule_agendum_immediate(self)

    def _choose_asteroid(self) -> Optional[core.Asteroid]:
        if self.ship.sector is None:
            raise ValueError(f'{self.ship} in no sector')

        nearest = None
        nearest_dist = np.inf
        distances = []
        candidates = []
        for hit in self.ship.sector.spatial_point(self.ship.loc, mask=core.ObjectFlag.ASTEROID):
            if not isinstance(hit, core.Asteroid):
                continue
            if hit.resource not in self.allowed_resources:
                continue
            if hit.cargo[hit.resource] <= 0:
                continue

            dist = util.distance(self.ship.loc, hit.loc)
            distances.append(dist)
            candidates.append(hit)

        if len(candidates) == 0:
            return None

        #TODO: choose asteroids in a more sensible way
        p = 1.0 / np.array(distances)
        p = p / p.sum()
        idx = self.gamestate.random.choice(len(candidates), 1, p=p)[0]
        target = candidates[idx]

        #TODO: worry about other people harvesting asteroids
        return target

    def start(self) -> None:
        assert self.state == MiningAgendum.State.IDLE
        self.gamestate.schedule_agendum_immediate(self)

    def is_complete(self) -> bool:
        return self.max_trips >= 0 and self.round_trips >= self.max_trips

    def act(self) -> None:
        assert self.state == MiningAgendum.State.IDLE

        if self.is_complete():
            self.state = MiningAgendum.State.COMPLETE
            return

        if np.any(self.ship.cargo[self.allowed_resources] > 0.):
            # if we've got resources to sell, find a station to sell to

            station_ret = choose_station_to_sell_to(
                    self.gamestate, self.ship, self.agent,
                    self.allowed_resources, self.allowed_stations,
            )
            if station_ret is None:
                self.logger.debug(f'cannot find a station buying my mined resources. Sleeping...')
                self.gamestate.schedule_agendum(self.gamestate.timestamp + MINING_SLEEP_TIME, self)
                return

            resource, station, station_agent = station_ret
            assert station_agent.buy_price(resource) > 0
            assert station_agent.budget(resource) > 0
            #TODO: sensibly have a floor for selling the good
            # basically we pick a station and hope for the best
            floor_price = 0.

            self.state = MiningAgendum.State.TRADING
            self.transfer_order = ocore.TradeCargoToStation(
                    station_agent, self.agent, floor_price,
                    station, resource, self.ship.cargo[resource],
                    self.ship, self.gamestate)
            self.transfer_order.observe(self)
            self.ship.prepend_order(self.transfer_order)
        else:
            # if we don't have any resources in cargo, go mine some

            target = self._choose_asteroid()
            if target is None:
                #TODO: notify someone?
                self.logger.debug(f'could not find asteroid of type {self.allowed_resources} in {self.ship.sector}, sleeping...')
                self.gamestate.schedule_agendum(self.gamestate.timestamp + MINING_SLEEP_TIME, self)
                return

            #TODO: choose amount to harvest
            # push mining order
            self.state = MiningAgendum.State.MINING
            self.mining_order = ocore.MineOrder(target, 1e3, self.ship, self.gamestate)
            self.mining_order.observe(self)
            self.ship.prepend_order(self.mining_order)

class TradingAgendum(core.Agendum, core.OrderObserver):

    class State(enum.Enum):
        IDLE = enum.auto()
        BUYING = enum.auto()
        SELLING = enum.auto()
        COMPLETE = enum.auto()

    def __init__(self,
            ship:core.Ship, *args:Any,
            allowed_goods:Optional[List[int]]=None,
            buy_from_stations:Optional[List[core.SectorEntity]]=None,
            sell_to_stations:Optional[List[core.SectorEntity]]=None,
            **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.ship = ship
        self.agent = econ.ShipTraderAgent(ship)
        self.state = TradingAgendum.State.IDLE

        # goods we're allowed to trade
        if allowed_goods is None:
            allowed_goods = list(range(self.gamestate.production_chain.ranks[0], self.gamestate.production_chain.ranks.cumsum()[-2]))
        self.allowed_goods = allowed_goods

        self.buy_from_stations:Optional[List[core.SectorEntity]] = buy_from_stations
        self.sell_to_stations:Optional[List[core.SectorEntity]] = sell_to_stations

        self.buy_order:Optional[ocore.TradeCargoFromStation] = None
        self.sell_order:Optional[ocore.TradeCargoToStation] = None

        self.max_trips = -1
        self.trade_trips = 0

    def order_begin(self, order:core.Order) -> None:
        pass

    def order_complete(self, order:core.Order) -> None:
        if self.state == TradingAgendum.State.BUYING:
            assert order == self.buy_order
            self.buy_order = None
            # go back into idle state to start things off again
            self.state = TradingAgendum.State.IDLE
            self.gamestate.schedule_agendum_immediate(self)
        elif self.state == TradingAgendum.State.SELLING:
            assert order == self.sell_order
            self.sell_order = None
            # go back into idle state to start things off again
            self.state = TradingAgendum.State.IDLE
            self.gamestate.schedule_agendum_immediate(self)
            self.trade_trips += 1
        else:
            raise ValueError("got order_complete in wrong state {self.state}")

    def order_cancel(self, order:core.Order) -> None:
        if self.state == TradingAgendum.State.BUYING:
            assert order == self.buy_order
            self.buy_order = None
        elif self.state == TradingAgendum.State.SELLING:
            assert order == self.sell_order
            self.sell_order = None
        else:
            raise ValueError("got order_cancel in wrong state {self.state}")

        self.state = TradingAgendum.State.IDLE
        self.gamestate.schedule_agendum_immediate(self)

    def start(self) -> None:
        pass

    def is_complete(self) -> bool:
        return self.max_trips >= 0 and self.trade_trips >= self.max_trips

    def act(self) -> None:
        assert self.state == TradingAgendum.State.IDLE

        if self.is_complete():
            self.state = TradingAgendum.State.COMPLETE
            return

        if np.any(self.ship.cargo[self.allowed_goods] > 0.):
            # if we've got resources to sell, find a station to sell to

            station_ret = choose_station_to_sell_to(
                    self.gamestate, self.ship, self.agent,
                    self.allowed_goods, self.sell_to_stations,
            )
            if station_ret is None:
                self.logger.debug(f'cannot find a station buying my trade goods. Sleeping...')
                self.gamestate.schedule_agendum(self.gamestate.timestamp + TRADING_SLEEP_TIME, self)
                return

            resource, station, station_agent = station_ret
            assert station_agent.buy_price(resource) > 0
            assert station_agent.budget(resource) > 0
            #TODO: sensibly have a floor for selling the good
            # basically we pick a station and hope for the best
            floor_price = 0.

            self.state = TradingAgendum.State.SELLING
            self.sell_order = ocore.TradeCargoToStation(
                    station_agent, self.agent, floor_price,
                    station, resource, self.ship.cargo[resource],
                    self.ship, self.gamestate)
            self.sell_order.observe(self)
            self.ship.prepend_order(self.sell_order)
        else:
            station_ret = choose_station_to_buy_from(
                    self.gamestate, self.ship,
                    self.allowed_goods,
                    self.buy_from_stations, self.sell_to_stations)
            if station_ret is None:
                self.logger.debug(f'cannot find a valid trade for my trade goods. Sleeping...')
                self.gamestate.schedule_agendum(self.gamestate.timestamp + TRADING_SLEEP_TIME, self)
                return
            resource, station, station_agent = station_ret
            assert station_agent.sell_price(resource) < np.inf
            assert station_agent.inventory(resource) > 0.

            #TODO: sensibly have a ceiling for buying the good
            # basically we pick a station and hope for the best
            ceiling_price = np.inf
            amount = min(station.cargo[resource], self.ship.cargo_capacity - self.ship.cargo.sum())

            self.state = TradingAgendum.State.BUYING
            self.buy_order = ocore.TradeCargoFromStation(
                    self.agent, station_agent, ceiling_price,
                    station, resource, amount,
                    self.ship, self.gamestate)
            self.buy_order.observe(self)
            self.ship.prepend_order(self.buy_order)

class StationManager(core.Agendum):
    def __init__(self, station:core.Station, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        self.station = station
        self.agent = econ.StationAgent(station, self.gamestate.production_chain)

        #TODO: how do we keep this up to date if there's a change?
        self.gamestate.representing_agent(station.entity_id, self.agent)

    def start(self) -> None:
        input_idx = np.where(self.gamestate.production_chain.adj_matrix[:,self.station.resource])
        self.agent._buy_price[input_idx] = self.gamestate.production_chain.prices[input_idx]
        self.agent._budget[input_idx] = np.inf
        self.agent._sell_price[self.station.resource] = self.gamestate.production_chain.prices[self.station.resource]

    def act(self) -> None:
        # price and budget setting stuff goes here and should run periodically
        pass

