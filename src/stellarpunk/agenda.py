""" Agenda items for characters, reflecting activites they are involved in. """

from typing import Optional, List, Any, Iterable, Tuple, DefaultDict, Mapping
import enum
import collections
import itertools

import numpy as np
import numpy.typing as npt

from stellarpunk import core, econ, util
import stellarpunk.orders.core as ocore
from stellarpunk.orders import movement

def possible_buys(
        gamestate:core.Gamestate,
        ship:core.Ship,
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
        if not isinstance(hit, core.Station):
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
        ship:core.Ship,
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
        ) -> Optional[Tuple[int, core.Station, core.EconAgent, float, float]]:

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

    sales = possible_sales(gamestate, ship, ship_agent, allowed_resources, sell_to_stations)

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

    assert best_trade is None or best_profit_per_time > 0.
    return best_trade

# how long to wait, idle if we can't do work
MINING_SLEEP_TIME = 60.
TRADING_SLEEP_TIME = 60.

class EntityOperatorAgendum(core.Agendum, core.SectorEntityObserver):
    def __init__(self, craft: core.SectorEntity, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.craft = craft

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

    def entity_destroyed(self, entity:core.SectorEntity) -> None:
        if entity == self.craft:
            self.stop()

class CaptainAgendum(EntityOperatorAgendum):
    def __init__(self, *args: Any, threat_response:bool=True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.threat_response = threat_response

    def _start(self) -> None:
        # become captain before underlying start so we'll be captain by that point
        self.craft.captain = self.character
        super()._start()
        for a in self.character.agenda:
            if a != self and isinstance(a, CaptainAgendum):
                raise ValueError(f'{self.character.short_id()} already had a captain agendum: {a}')

    def _stop(self) -> None:
        super()._stop()
        self.craft.captain = None
        #TODO: kill other EOAs?

    def entity_targeted(self, craft:core.SectorEntity, threat:core.SectorEntity) -> None:
        if not self.threat_response:
            return

        #TODO: determine if threat is hostile:
        hostile = False
        # is it a weapon (e.g. missile)
        # is it known to be hostile?
        # is it running with its transponder off?
        if not threat.sensor_settings.transponder:
            hostile = True

        #TODO: decide how to proceed:
        # if not hostile, ignore
        # if first threat, pause other activities (agenda), start fleeing

class MiningAgendum(EntityOperatorAgendum, core.OrderObserver):
    """ Managing a ship for mining.

    Operates as a state machine as we mine asteroids and sell the resources to
    relevant stations. """

    class State(enum.Enum):
        IDLE = enum.auto()
        MINING = enum.auto()
        TRADING = enum.auto()
        COMPLETE = enum.auto()

    def __init__(
        self,
        ship:core.Ship,
        *args: Any,
        allowed_resources: Optional[List[int]] = None,
        allowed_stations: Optional[List[core.SectorEntity]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(ship, *args, **kwargs)

        self.ship = ship
        self.agent = econ.ShipTraderAgent(ship, self.character, self.gamestate)

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
        for hit in self.ship.sector.spatial_point(self.ship.loc):
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

    def _start(self) -> None:
        super()._start()
        assert self.state == MiningAgendum.State.IDLE
        self.gamestate.schedule_agendum_immediate(self, jitter=5.)

    def _stop(self) -> None:
        super()._stop()
        if self.state == MiningAgendum.State.MINING:
            assert self.mining_order is not None
            self.mining_order.cancel_order()
        elif self.state == MiningAgendum.State.TRADING:
            assert self.transfer_order is not None
            self.transfer_order.cancel_order()
        self.gamestate.unschedule_agendum(self)

    def is_complete(self) -> bool:
        return self.max_trips >= 0 and self.round_trips >= self.max_trips

    def act(self) -> None:
        assert self.state == MiningAgendum.State.IDLE

        if self.is_complete():
            self.state = MiningAgendum.State.COMPLETE
            return

        if np.any(self.ship.cargo[self.allowed_resources] > 0.):
            # if we've got resources to sell, find a station to sell to

            sell_station_ret = choose_station_to_sell_to(
                    self.gamestate, self.ship, self.agent,
                    self.allowed_resources, self.allowed_stations,
            )
            if sell_station_ret is None:
                self.logger.debug(f'cannot find a station buying my mined resources ({np.where(self.ship.cargo[self.allowed_resources] > 0.)}). Sleeping...')
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

class TradingAgendum(EntityOperatorAgendum, core.OrderObserver):

    class State(enum.Enum):
        IDLE = enum.auto()
        BUYING = enum.auto()
        SELLING = enum.auto()
        COMPLETE = enum.auto()
        SLEEP_NO_BUYS = enum.auto()
        SLEEP_NO_SALES = enum.auto()

    def __init__(self,
        ship: core.Ship,
        *args: Any,
        allowed_goods: Optional[List[int]] = None,
        buy_from_stations: Optional[List[core.SectorEntity]] = None,
        sell_to_stations: Optional[List[core.SectorEntity]] = None,
        **kwargs:Any
    ) -> None:
        super().__init__(ship, *args, **kwargs)
        self.ship = ship
        self.agent = econ.ShipTraderAgent(ship, self.character, self.gamestate)
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

    def _start(self) -> None:
        super()._start()
        assert self.state == TradingAgendum.State.IDLE
        self.gamestate.schedule_agendum_immediate(self, jitter=5.)

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
        buy_station_ret = choose_station_to_buy_from(
                self.gamestate, self.ship, self.agent,
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
        amount = min(station.cargo[resource], self.ship.cargo_capacity - self.ship.cargo.sum())

        self.state = TradingAgendum.State.BUYING
        self.buy_order = ocore.TradeCargoFromStation(
                self.agent, station_agent, ceiling_price,
                station, resource, amount,
                self.ship, self.gamestate)
        self.buy_order.observe(self)
        self.ship.prepend_order(self.buy_order)

    def _sell_goods(self) -> bool:
        # if we've got resources to sell, find a station to sell to

        sell_station_ret = choose_station_to_sell_to(
                self.gamestate, self.ship, self.agent,
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
        self.sell_order = ocore.TradeCargoToStation(
                station_agent, self.agent, floor_price,
                station, resource, self.ship.cargo[resource],
                self.ship, self.gamestate)
        self.sell_order.observe(self)
        self.ship.prepend_order(self.sell_order)

        return True

    def act(self) -> None:
        assert self.state in [TradingAgendum.State.IDLE, TradingAgendum.State.SLEEP_NO_BUYS, TradingAgendum.State.SLEEP_NO_SALES]

        if self.is_complete():
            self.state = TradingAgendum.State.COMPLETE
            return

        if np.any(self.ship.cargo[self.allowed_goods] > 0.):
            if not self._sell_goods():
                self._buy_goods()
        else:
            self._buy_goods()

class StationManager(EntityOperatorAgendum):
    """ Manage production and trading for a station.

    Responsible for actually driving the production at the station as well as
    trading, price setting, although it might delegate those reponsiblities.
    """

    def __init__(self, station:core.Station, *args:Any, **kwargs:Any) -> None:
        super().__init__(station, *args, **kwargs)

        self.station = station
        self.station.observe(self)
        self.agent = econ.StationAgent.create_station_agent(
            self.character,
            station,
            self.gamestate.production_chain,
            self.gamestate,
        )
        self.produced_batches = 0

        #TODO: how do we keep this up to date if there's a change?
        self.gamestate.representing_agent(station.entity_id, self.agent)

    def _start(self) -> None:
        # become captain before underlying start so we'll be captain by that point
        self.craft.captain = self.character
        super()._start()
        self.gamestate.schedule_agendum_immediate(self)

    def _stop(self) -> None:
        super()._stop()
        self.gamestate.withdraw_agent(self.station.entity_id)
        self.craft.captain = None

    def _produce_at_station(self) -> float:
        """ Run production at this agendum's station.

        returns when we should next check for production.
        """

        # waiting for production to finish case
        if self.station.next_batch_time > 0:
            # batch is ready case
            if self.station.next_batch_time <= self.gamestate.timestamp:
                # add the batch to cargo
                amount = self.gamestate.production_chain.batch_sizes[self.station.resource]
                self.station.cargo[self.station.resource] += amount
                #TODO: record the production somehow
                #self.gamestate.production_chain.goods_produced[station.resource] += amount
                self.station.next_batch_time = 0.
                self.station.next_production_time = 0.
                self.produced_batches += 1
                return self.gamestate.timestamp + 1.0
            # batch is not ready case
            else:
                return self.station.next_batch_time
        # waiting for enough cargo to produce case
        elif self.station.next_production_time <= self.gamestate.timestamp:
            # check if we have enough resource to start a batch
            resources_needed = self.gamestate.production_chain.adj_matrix[:,self.station.resource] * self.gamestate.production_chain.batch_sizes[self.station.resource]

            # we have enough cargo to produce case
            if np.all(self.station.cargo >= resources_needed):
                self.station.cargo -= resources_needed
                # TODO: float vs floating type issues with numpy (arg!)
                self.station.next_batch_time = self.gamestate.timestamp + self.gamestate.production_chain.production_times[self.station.resource] # type: ignore
                return self.station.next_batch_time
            # we do not have enough cargo to produce
            else:
                # wait a cooling off period to avoid needlesss expensive checks
                self.station.next_production_time = self.gamestate.timestamp + self.gamestate.production_chain.production_coolingoff_time
                return self.station.next_production_time
        else:
            return self.station.next_production_time

    def act(self) -> None:
        # we must always be the representing agent
        assert self.gamestate.econ_agents[self.station.entity_id] == self.agent

        # do production
        next_production_ts = self._produce_at_station()

        #TODO: price and budget setting stuff goes here and should run periodically

        self.gamestate.schedule_agendum(next_production_ts, self, jitter=1.0)

class PlanetManager(EntityOperatorAgendum):
    """ Manage consumption and trading for planet/hab. """

    def __init__(self, planet:core.Planet, *args:Any, **kwargs:Any) -> None:
        super().__init__(planet, *args, **kwargs)

        self.planet = planet
        self.planet.observe(self)
        self.agent = econ.StationAgent.create_planet_agent(
            self.character,
            planet,
            self.gamestate.production_chain,
            self.gamestate
        )

        #TODO: how do we keep this up to date if there's a change?
        self.gamestate.representing_agent(planet.entity_id, self.agent)

    def _start(self) -> None:
        # become captain before underlying start so we'll be captain by that point
        self.craft.captain = self.character
        super()._start()
        self.gamestate.schedule_agendum_immediate(self)

    def _stop(self) -> None:
        super()._stop()
        self.gamestate.withdraw_agent(self.planet.entity_id)
        self.craft.captain = None

    def act(self) -> None:
        assert self.gamestate.econ_agents[self.planet.entity_id] == self.agent
        # price and budget setting stuff goes here and should run periodically
        pass
