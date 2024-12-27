""" Agenda items for characters, reflecting activites they are involved in. """

import enum
import collections
import itertools
import abc
import uuid
from typing import Optional, List, Any, Iterable, Tuple, DefaultDict, Mapping, Type

import numpy as np
import numpy.typing as npt

from stellarpunk import core, econ, util
import stellarpunk.orders.core as ocore
from stellarpunk.core import combat, sector_entity
from stellarpunk.orders import movement

class Agendum(core.AbstractAgendum, abc.ABC):

    def __init__(self, gamestate:core.Gamestate, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.gamestate = gamestate
        self.started_at = -1.0
        self.stopped_at = -1.0

    def register(self) -> None:
        self.gamestate.register_agendum(self)

    def unregister(self) -> None:
        self.gamestate.unregister_agendum(self)

    def start(self) -> None:
        assert(self.started_at < 0.0)
        self.stopped_at = -1.0
        self.started_at = self.gamestate.timestamp
        self._start()

    def pause(self) -> None:
        self._pause()
        self.gamestate.unschedule_agendum(self)

    def stop(self) -> None:
        assert(self.started_at >= 0.0)
        assert(self.stopped_at < 0.0)
        self._stop()
        self.stopped_at = self.gamestate.timestamp
        self.gamestate.unschedule_agendum(self)


def possible_buys(
        gamestate:core.Gamestate,
        ship:core.SectorEntity,
        ship_agent:core.EconAgent,
        allowed_resources:List[int],
        buy_from_stations:Optional[List[core.SectorEntity]],
        ) -> Mapping[int, List[Tuple[float, float, core.SectorEntity]]]:
    assert ship.sector is not None
    # figure out possible buys by resource
    buys:DefaultDict[int, List[Tuple[float, float, core.SectorEntity]]] = collections.defaultdict(list)
    buy_hits:Iterable[core.SectorEntity]
    if buy_from_stations is None:
        buy_hits = ship.sector.spatial_point(ship.loc)
    else:
        buy_hits = buy_from_stations
    for hit in buy_hits:
        if not isinstance(hit, sector_entity.Station):
            continue
        agent = gamestate.econ_agents[hit.entity_id]
        for resource in agent.sell_resources():
            if resource not in allowed_resources:
                continue
            price = agent.sell_price(resource)
            if not (price < np.inf):
                continue
            if not econ.trade_valid(ship_agent, agent, resource, price, 1.):
                continue

            buys[resource].append((
                agent.sell_price(resource),
                agent.inventory(resource),
                hit,
            ))
    return buys

def possible_sales(
        gamestate:core.Gamestate,
        ship:core.SectorEntity,
        ship_agent:core.EconAgent,
        allowed_resources:List[int],
        allowed_stations:Optional[List[core.SectorEntity]],
        ) -> Mapping[int, List[Tuple[float, float, core.SectorEntity]]]:
    assert ship.sector is not None
    # figure out possible sales by resource
    sales:DefaultDict[int, List[Tuple[float, float, core.SectorEntity]]] = collections.defaultdict(list)
    sale_hits:Iterable[core.SectorEntity]
    if allowed_stations is None:
        sale_hits = ship.sector.spatial_point(ship.loc)
    else:
        sale_hits = allowed_stations
    for hit in sale_hits:
        if not isinstance(hit, sector_entity.Station):
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
                    ship_agent.inventory(resource),
                ),
                hit,
            ))
    return sales

def choose_station_to_buy_from(
        gamestate:core.Gamestate,
        ship:core.Ship,
        ship_agent:core.EconAgent,
        allowed_resources:List[int],
        buy_from_stations:Optional[List[core.SectorEntity]],
        sell_to_stations:Optional[List[core.SectorEntity]]
        ) -> Optional[Tuple[int, sector_entity.Station, core.EconAgent, float, float]]:

    if ship.sector is None:
        raise ValueError(f'{ship} in no sector')

    # compute diffs of allowed/known stations
    # pick a trade that maximizes profit discounting by time spent travelling
    # and transferring

    # (sell - buy price) * amount
    # transfer time = buy transfer + sell transfer
    # travel time  = time to buy + time to sell

    #TODO: how to get prices and stations without magically having global knowledge?

    buys = possible_buys(gamestate, ship, ship_agent, allowed_resources, buy_from_stations)

    # find sales, assuming we can acquire whatever resource we need
    sales = possible_sales(gamestate, ship, econ.YesAgent(gamestate.production_chain), allowed_resources, sell_to_stations)

    #best_profit_per_time = 0.
    #best_trade:Optional[Tuple[int, core.Station, core.EconAgent]] = None
    profits_per_time = []
    trades = []
    for resource in buys.keys():
        for ((buy_price, buy_amount, buy_station), (sale_price, sale_amount, sale_station)) in itertools.product(buys[resource], sales[resource]):
            amount = min(buy_amount, sale_amount)
            profit = (sale_price - buy_price)*amount
            transfer_time = ocore.TradeCargoFromStation.transfer_rate() * amount + ocore.TradeCargoToStation.transfer_rate() * amount
            travel_time = movement.GoToLocation.compute_eta(ship, buy_station.loc) + movement.GoToLocation.compute_eta(ship, sale_station.loc, starting_loc=buy_station.loc)

            profit_per_time = profit / (transfer_time + travel_time)
            profits_per_time.append(profit_per_time)
            trades.append((resource, buy_station, gamestate.econ_agents[buy_station.entity_id], profit, transfer_time + travel_time)) # type: ignore
            #if profit_per_time > best_profit_per_time:
            #    best_profit_per_time = profit_per_time
            #    best_trade = (resource, buy_station, gamestate.econ_agents[buy_station.entity_id]) # type: ignore

    #assert best_trade is None or best_profit_per_time > 0
    #return best_trade
    if len(trades) == 0:
        return None
    else:
        x = np.array(profits_per_time)
        p = x/x.sum()
        t = gamestate.random.choice(trades, p=p) # type: ignore
        return t

def choose_station_to_sell_to(
        gamestate:core.Gamestate,
        ship:core.Ship,
        ship_agent:core.EconAgent,
        allowed_resources:List[int],
        sell_to_stations:Optional[List[core.SectorEntity]]
        ) -> Optional[Tuple[int, sector_entity.Station, core.EconAgent]]:
    """ Choose a station to sell goods from ship to """

    if ship.sector is None:
        raise ValueError(f'{ship} in no sector')

    # pick the station where we'll get the best profit for our cargo
    # biggest profit-per-tick for our cargo

    #TODO: how do we access price information? shouldn't this be
    # somehow limited to a view specific to us?
    #TODO: what if no stations seem to trade the resources we have?
    #TODO: what if no stations seem to trade any allowed resources?

    sales = possible_sales(gamestate, ship, ship_agent, allowed_resources, sell_to_stations)

    best_profit_per_time = 0.
    best_trade:Optional[Tuple[int, sector_entity.Station, core.EconAgent]] = None
    for resource in sales.keys():
        for sale_price, amount, sale_station in sales[resource]:
            profit = sale_price * amount
            transfer_time = ocore.TradeCargoToStation.transfer_rate() * amount
            travel_time = movement.GoToLocation.compute_eta(ship, sale_station.loc)

            profit_per_time = profit / (transfer_time + travel_time)
            if profit_per_time > best_profit_per_time:
                best_profit_per_time = profit_per_time
                best_trade = (resource, sale_station, gamestate.econ_agents[sale_station.entity_id]) # type: ignore

    assert best_trade is None or best_profit_per_time > 0.
    return best_trade

# how long to wait, idle if we can't do work
MINING_SLEEP_TIME = 60.
TRADING_SLEEP_TIME = 60.

class EntityOperatorAgendum(Agendum, core.SectorEntityObserver):
    @classmethod
    def create_eoa[T:EntityOperatorAgendum](cls:Type[T], craft: core.CrewedSectorEntity, *args: Any, **kwargs: Any) -> T:
        a = cls.create_agendum(*args, **kwargs)
        a.craft=craft
        return a

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.craft:core.CrewedSectorEntity = None # type: ignore

    def _start(self) -> None:
        if self.character.location is None:
            raise ValueError(f'{self.character.short_id()} tried to operate {self.craft.short_id()} but they are nowhere')
        if self.character.location != self.craft:
            raise ValueError(f'{self.character.short_id()} tried to operate {self.craft.short_id()} but they are on {self.character.location.short_id()}')
        if self.craft.captain != self.character:
            raise ValueError(f'{self.character.short_id()} tried to operate {self.craft.short_id()} but they are not the captain')
        self.craft.observe(self)

    def _stop(self) -> None:
        self.craft.unobserve(self)

    @property
    def observer_id(self) -> uuid.UUID:
        return self.agenda_id

    def entity_destroyed(self, entity:core.SectorEntity) -> None:
        if entity == self.craft:
            self.stop()

class CaptainAgendum(EntityOperatorAgendum, core.OrderObserver):
    def __init__(self, *args: Any, enable_threat_response:bool=True, start_transponder:bool=False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.enable_threat_response = enable_threat_response
        self.threat_response:Optional[combat.FleeOrder] = None
        self._start_transponder = start_transponder

    def _start(self) -> None:
        # become captain before underlying start so we'll be captain by that point
        self.craft.captain = self.character
        super()._start()
        for a in self.character.agenda:
            if a != self and isinstance(a, CaptainAgendum):
                raise ValueError(f'{self.character.short_id()} already had a captain agendum: {a}')

        self.craft.sensor_settings.set_transponder(self._start_transponder)

    def _stop(self) -> None:
        super()._stop()
        self.craft.captain = None
        #TODO: kill other EOAs?

    def order_completed(self, order:core.Order) -> None:
        if order == self.threat_response:
            self.threat_response = None
        for a in self.character.agenda:
            if isinstance(a, EntityOperatorAgendum):
                a.unpause()

    def order_cancelled(self, order:core.Order) -> None:
        if order == self.threat_response:
            self.threat_response = None
        for a in self.character.agenda:
            if isinstance(a, EntityOperatorAgendum):
                a.unpause()

    def entity_targeted(self, craft:core.SectorEntity, threat:core.SectorEntity) -> None:
        assert craft == self.craft
        assert self.craft.sector
        if not self.enable_threat_response:
            return

        # ignore if we're already handling threats
        if self.threat_response:
            return

        # determine in threat is hostile
        hostile = combat.is_hostile(self.craft, threat)
        # decide how to proceed:
        if not hostile:
            return
        # if first threat, pause other ship-operating activities (agenda), start fleeing
        self.logger.debug(f'{self.craft.short_id} initiating defensive maneuvers against threat {threat}')
        #TODO: is it weird for one agendum to manipulate another?
        for a in self.character.agenda:
            if isinstance(a, EntityOperatorAgendum):
                a.pause()

        # engage in defense
        if self.threat_response:
            threat_image = self.craft.sector.sensor_manager.target(threat, craft)
            self.threat_response.add_threat(threat_image)
            return
        assert(isinstance(self.craft, core.Ship))
        self.threat_response = combat.FleeOrder.create_flee_order(self.craft, self.gamestate)
        self.threat_response.observe(self)
        threat_image = self.craft.sector.sensor_manager.target(threat, self.craft)
        self.threat_response.add_threat(threat_image)
        self.craft.prepend_order(self.threat_response)

class MiningAgendum(EntityOperatorAgendum, core.OrderObserver):
    """ Managing a ship for mining.

    Operates as a state machine as we mine asteroids and sell the resources to
    relevant stations. """

    class State(enum.IntEnum):
        IDLE = enum.auto()
        MINING = enum.auto()
        TRADING = enum.auto()
        COMPLETE = enum.auto()

    @classmethod
    def create_mining_agendum[T:MiningAgendum](cls:Type[T], *args:Any, allowed_stations:Optional[list[core.SectorEntity]]=None, **kwargs:Any) -> T:
        a = cls.create_eoa(*args, **kwargs)
        a.initialize_mining_agendum(allowed_stations)
        return a

    def __init__(
        self,
        *args: Any,
        allowed_resources: Optional[list[int]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.agent:econ.ShipTraderAgent = None # type: ignore
        # resources we're allowed to mine
        if allowed_resources is None:
            allowed_resources = list(range(self.gamestate.production_chain.ranks[0]))
        self.allowed_resources = allowed_resources

        self.allowed_stations:Optional[list[core.SectorEntity]] = None

        # state machine to keep track of what we're doing
        self.state:MiningAgendum.State = MiningAgendum.State.IDLE

        # keep track of a couple sorts of actions
        self.mining_order:Optional[ocore.MineOrder] = None
        self.transfer_order:Optional[ocore.TradeCargoToStation] = None

        self.round_trips = 0
        self.max_trips = -1

    def initialize_mining_agendum(self, allowed_stations:Optional[list[core.SectorEntity]] = None) -> None:
        assert(isinstance(self.craft, core.Ship))
        self.agent = econ.ShipTraderAgent.create_ship_trader_agent(self.craft, self.character, self.gamestate)
        self.allowed_stations = allowed_stations

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

    def _choose_asteroid(self) -> Optional[sector_entity.Asteroid]:
        if self.craft.sector is None:
            raise ValueError(f'{self.craft} in no sector')

        nearest = None
        nearest_dist = np.inf
        distances = []
        candidates = []
        for hit in self.craft.sector.spatial_point(self.craft.loc):
            if not isinstance(hit, sector_entity.Asteroid):
                continue
            if hit.resource not in self.allowed_resources:
                continue
            if hit.cargo[hit.resource] <= 0:
                continue

            dist = util.distance(self.craft.loc, hit.loc)
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

    def _start(self) -> None:
        super()._start()
        assert self.state == MiningAgendum.State.IDLE
        self.gamestate.schedule_agendum_immediate(self, jitter=5.)

    def _unpause(self) -> None:
        super()._unpause()
        assert self.state == MiningAgendum.State.IDLE
        self.gamestate.schedule_agendum_immediate(self, jitter=5.)

    def _pause(self) -> None:
        super()._pause()
        if self.state == MiningAgendum.State.MINING:
            assert self.mining_order is not None
            self.mining_order.cancel_order()
        elif self.state == MiningAgendum.State.TRADING:
            assert self.transfer_order is not None
            self.transfer_order.cancel_order()

    def _stop(self) -> None:
        super()._stop()
        if self.state == MiningAgendum.State.MINING:
            assert self.mining_order is not None
            self.mining_order.cancel_order()
        elif self.state == MiningAgendum.State.TRADING:
            assert self.transfer_order is not None
            self.transfer_order.cancel_order()

    def is_complete(self) -> bool:
        return self.max_trips >= 0 and self.round_trips >= self.max_trips

    def act(self) -> None:
        assert(isinstance(self.craft, core.Ship))
        assert self.state == MiningAgendum.State.IDLE

        if self.is_complete():
            self.state = MiningAgendum.State.COMPLETE
            return

        if np.any(self.craft.cargo[self.allowed_resources] > 0.):
            # if we've got resources to sell, find a station to sell to

            sell_station_ret = choose_station_to_sell_to(
                    self.gamestate, self.craft, self.agent,
                    self.allowed_resources, self.allowed_stations,
            )
            if sell_station_ret is None:
                self.logger.debug(f'cannot find a station buying my mined resources ({np.where(self.craft.cargo[self.allowed_resources] > 0.)}). Sleeping...')
                sleep_jitter = self.gamestate.random.uniform(high=MINING_SLEEP_TIME)
                self.gamestate.schedule_agendum(self.gamestate.timestamp + MINING_SLEEP_TIME/2 + sleep_jitter, self)
                return

            resource, station, station_agent = sell_station_ret
            assert station_agent.buy_price(resource) > 0
            assert station_agent.budget(resource) > 0
            #TODO: sensibly have a floor for selling the good
            # basically we pick a station and hope for the best
            floor_price = 0.

            self.state = MiningAgendum.State.TRADING
            self.transfer_order = ocore.TradeCargoToStation.create_trade_cargo_to_station(
                    station_agent, self.agent, floor_price,
                    station, resource, self.craft.cargo[resource],
                    self.craft, self.gamestate)
            self.transfer_order.observe(self)
            self.craft.prepend_order(self.transfer_order)
        else:
            # if we don't have any resources in cargo, go mine some

            target = self._choose_asteroid()
            if target is None:
                #TODO: notify someone?
                self.logger.debug(f'could not find asteroid of type {self.allowed_resources} in {self.craft.sector}, sleeping...')
                self.gamestate.schedule_agendum(self.gamestate.timestamp + MINING_SLEEP_TIME, self)
                return

            #TODO: choose amount to harvest
            # push mining order
            self.state = MiningAgendum.State.MINING
            self.mining_order = ocore.MineOrder.create_mine_order(target, 1e3, self.craft, self.gamestate)
            self.mining_order.observe(self)
            self.craft.prepend_order(self.mining_order)

class TradingAgendum(EntityOperatorAgendum, core.OrderObserver):

    class State(enum.IntEnum):
        IDLE = enum.auto()
        BUYING = enum.auto()
        SELLING = enum.auto()
        COMPLETE = enum.auto()
        SLEEP_NO_BUYS = enum.auto()
        SLEEP_NO_SALES = enum.auto()

    @classmethod
    def create_trading_agendum[T:TradingAgendum](
            cls:Type[T],
            *args:Any,
            buy_from_stations: Optional[List[core.SectorEntity]] = None,
            sell_to_stations: Optional[List[core.SectorEntity]] = None,
            **kwargs:Any
    ) -> T:
        a = cls.create_eoa(*args, **kwargs)
        a.initialize_trading_agendum(buy_from_stations, sell_to_stations)
        return a

    def __init__(self,
        *args: Any,
        allowed_goods: Optional[List[int]] = None,
        **kwargs:Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.agent:econ.ShipTraderAgent = None # type: ignore
        self.state = TradingAgendum.State.IDLE

        # goods we're allowed to trade
        if allowed_goods is None:
            allowed_goods = list(range(self.gamestate.production_chain.ranks[0], self.gamestate.production_chain.ranks.cumsum()[-2]))
        self.allowed_goods = allowed_goods

        self.buy_from_stations:Optional[List[core.SectorEntity]] = None
        self.sell_to_stations:Optional[List[core.SectorEntity]] = None

        self.buy_order:Optional[ocore.TradeCargoFromStation] = None
        self.sell_order:Optional[ocore.TradeCargoToStation] = None

        self.trade_trips = 0
        self.max_trips = -1

    def initialize_trading_agendum(
            self,
            buy_from_stations: Optional[List[core.SectorEntity]] = None,
            sell_to_stations: Optional[List[core.SectorEntity]] = None,
    ) -> None:
        assert(isinstance(self.craft, core.Ship))
        self.agent = econ.ShipTraderAgent.create_ship_trader_agent(self.craft, self.character, self.gamestate)
        self.buy_from_stations = buy_from_stations
        self.sell_to_stations = sell_to_stations

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

    def _start(self) -> None:
        super()._start()
        assert self.state == TradingAgendum.State.IDLE
        self.gamestate.schedule_agendum_immediate(self, jitter=5.)

    def _unpause(self) -> None:
        super()._unpause()
        assert self.state == TradingAgendum.State.IDLE
        self.gamestate.schedule_agendum_immediate(self, jitter=5.)

    def _pause(self) -> None:
        super()._pause()
        if self.state == TradingAgendum.State.BUYING:
            assert self.buy_order is not None
            self.buy_order.cancel_order()
        elif self.state == TradingAgendum.State.SELLING:
            assert self.sell_order is not None
            self.sell_order.cancel_order()

    def _stop(self) -> None:
        super()._stop()
        if self.state == TradingAgendum.State.BUYING:
            assert self.buy_order is not None
            self.buy_order.cancel_order()
        elif self.state == TradingAgendum.State.SELLING:
            assert self.sell_order is not None
            self.sell_order.cancel_order()

    def is_complete(self) -> bool:
        return self.max_trips >= 0 and self.trade_trips >= self.max_trips

    def _buy_goods(self) -> None:
        assert(isinstance(self.craft, core.Ship))
        buy_station_ret = choose_station_to_buy_from(
                self.gamestate, self.craft, self.agent,
                self.allowed_goods,
                self.buy_from_stations, self.sell_to_stations)
        if buy_station_ret is None:
            self.state = TradingAgendum.State.SLEEP_NO_SALES
            self.logger.debug(f'cannot find a valid trade for my trade goods. Sleeping...')
            sleep_jitter = self.gamestate.random.uniform(high=TRADING_SLEEP_TIME)
            self.gamestate.schedule_agendum(self.gamestate.timestamp + TRADING_SLEEP_TIME/2 + sleep_jitter, self)
            return
        resource, station, station_agent, est_profit, est_time = buy_station_ret
        assert station_agent.sell_price(resource) < np.inf
        assert station_agent.inventory(resource) > 0.

        self.logger.debug(f'buying {resource=} from {station=} {est_profit=} {est_time=}')

        #TODO: sensibly have a ceiling for buying the good
        # basically we pick a station and hope for the best
        ceiling_price = np.inf
        amount = min(station.cargo[resource], self.craft.cargo_capacity - self.craft.cargo.sum())

        self.state = TradingAgendum.State.BUYING
        self.buy_order = ocore.TradeCargoFromStation.create_trade_cargo_from_station(
                self.agent, station_agent, ceiling_price,
                station, resource, amount,
                self.craft, self.gamestate)
        self.buy_order.observe(self)
        self.craft.prepend_order(self.buy_order)

    def _sell_goods(self) -> bool:
        # if we've got resources to sell, find a station to sell to
        assert(isinstance(self.craft, core.Ship))

        sell_station_ret = choose_station_to_sell_to(
                self.gamestate, self.craft, self.agent,
                self.allowed_goods, self.sell_to_stations,
        )
        if sell_station_ret is None:
            #TODO: revisit sleeping and tracking that as a state
            #self.logger.debug(f'cannot find a station buying my trade goods ({np.where(self.ship.cargo[self.allowed_goods] > 0.)}). Sleeping...')
            #self.state = TradingAgendum.State.SLEEP_NO_BUYS
            #sleep_jitter = self.gamestate.random.uniform(high=TRADING_SLEEP_TIME)
            #self.gamestate.schedule_agendum(self.gamestate.timestamp + TRADING_SLEEP_TIME/2 + sleep_jitter, self)
            #return
            return False

        resource, station, station_agent = sell_station_ret
        assert station_agent.buy_price(resource) > 0
        assert station_agent.budget(resource) > 0

        self.logger.debug(f'selling {resource=} to {station=}')

        #TODO: sensibly have a floor for selling the good
        # basically we pick a station and hope for the best
        floor_price = 0.

        self.state = TradingAgendum.State.SELLING
        self.sell_order = ocore.TradeCargoToStation.create_trade_cargo_to_station(
                station_agent, self.agent, floor_price,
                station, resource, self.craft.cargo[resource],
                self.craft, self.gamestate)
        self.sell_order.observe(self)
        self.craft.prepend_order(self.sell_order)

        return True

    def act(self) -> None:
        assert self.state in [TradingAgendum.State.IDLE, TradingAgendum.State.SLEEP_NO_BUYS, TradingAgendum.State.SLEEP_NO_SALES]

        if self.is_complete():
            self.state = TradingAgendum.State.COMPLETE
            return

        if np.any(self.craft.cargo[self.allowed_goods] > 0.):
            if not self._sell_goods():
                self._buy_goods()
        else:
            self._buy_goods()

class StationManager(EntityOperatorAgendum):
    """ Manage production and trading for a station.

    Responsible for actually driving the production at the station as well as
    trading, price setting, although it might delegate those reponsiblities.
    """

    @classmethod
    def create_station_manager[T:"StationManager"](cls:Type[T], *args:Any, **kwargs:Any) -> T:
        a = cls.create_eoa(*args, **kwargs)
        a.initialize_station_manager()
        return a

    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.agent:econ.StationAgent = None # type: ignore
        self.produced_batches = 0

    def initialize_station_manager(self) -> None:
        assert(isinstance(self.craft, sector_entity.Station))
        self.agent = econ.StationAgent.create_station_agent(
            self.character,
            self.craft,
            self.gamestate.production_chain,
            self.gamestate,
        )

    def _start(self) -> None:
        # become captain before underlying start so we'll be captain by that point
        self.craft.captain = self.character
        super()._start()
        self.gamestate.representing_agent(self.craft.entity_id, self.agent)
        self.gamestate.schedule_agendum_immediate(self)

        #TODO; should managing the transponder live here?
        #TODO: don't always want the transponder on
        self.craft.sensor_settings.set_transponder(True)

    def _stop(self) -> None:
        super()._stop()
        self.gamestate.withdraw_agent(self.craft.entity_id)
        self.craft.captain = None

    def _produce_at_station(self) -> float:
        """ Run production at this agendum's station.

        returns when we should next check for production.
        """

        assert(isinstance(self.craft, sector_entity.Station))

        # waiting for production to finish case
        if self.craft.next_batch_time > 0:
            # batch is ready case
            if self.craft.next_batch_time <= self.gamestate.timestamp:
                # add the batch to cargo
                amount = self.gamestate.production_chain.batch_sizes[self.craft.resource]
                self.craft.cargo[self.craft.resource] += amount
                #TODO: record the production somehow
                #self.gamestate.production_chain.goods_produced[station.resource] += amount
                self.craft.next_batch_time = 0.
                self.craft.next_production_time = 0.
                self.produced_batches += 1
                return self.gamestate.timestamp + 1.0
            # batch is not ready case
            else:
                return self.craft.next_batch_time
        # waiting for enough cargo to produce case
        elif self.craft.next_production_time <= self.gamestate.timestamp:
            # check if we have enough resource to start a batch
            resources_needed = self.gamestate.production_chain.adj_matrix[:,self.craft.resource] * self.gamestate.production_chain.batch_sizes[self.craft.resource]

            # we have enough cargo to produce case
            if np.all(self.craft.cargo >= resources_needed):
                self.craft.cargo -= resources_needed
                # TODO: float vs floating type issues with numpy (arg!)
                self.craft.next_batch_time = self.gamestate.timestamp + self.gamestate.production_chain.production_times[self.craft.resource] # type: ignore
                return self.craft.next_batch_time
            # we do not have enough cargo to produce
            else:
                # wait a cooling off period to avoid needlesss expensive checks
                self.craft.next_production_time = self.gamestate.timestamp + self.gamestate.production_chain.production_coolingoff_time
                return self.craft.next_production_time
        else:
            return self.craft.next_production_time

    def act(self) -> None:
        # we must always be the representing agent
        assert self.gamestate.econ_agents[self.craft.entity_id] == self.agent

        # do production
        next_production_ts = self._produce_at_station()

        #TODO: price and budget setting stuff goes here and should run periodically
        self.gamestate.schedule_agendum(next_production_ts, self, jitter=1.0)

class PlanetManager(EntityOperatorAgendum):
    """ Manage consumption and trading for planet/hab. """

    @classmethod
    def create_planet_manager[T:"PlanetManager"](cls:Type[T], *args:Any, **kwargs:Any) -> T:
        a = cls.create_eoa(*args, **kwargs)
        a.initialize_planet_manager()
        return a

    def __init__(self, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.agent:econ.StationAgent = None # type: ignore

    def initialize_planet_manager(self) -> None:
        assert(isinstance(self.craft, sector_entity.Planet))
        self.agent = econ.StationAgent.create_planet_agent(
            self.character,
            self.craft,
            self.gamestate.production_chain,
            self.gamestate
        )

    def _start(self) -> None:
        # become captain before underlying start so we'll be captain by that point
        self.craft.captain = self.character
        super()._start()
        self.gamestate.representing_agent(self.craft.entity_id, self.agent)
        self.gamestate.schedule_agendum_immediate(self)

    def _stop(self) -> None:
        super()._stop()
        self.gamestate.withdraw_agent(self.craft.entity_id)
        self.craft.captain = None

    def act(self) -> None:
        assert self.gamestate.econ_agents[self.craft.entity_id] == self.agent
        #TODO: price and budget setting stuff goes here and should run periodically
        pass
