""" Economic agenda items like mining, production, trade. """

import enum
import collections
import itertools
import uuid
from collections.abc import Iterable, Mapping
from typing import Optional, Any, Type

import numpy as np

from stellarpunk import core, econ, util, intel
import stellarpunk.orders.core as ocore
from stellarpunk.core import sector_entity
from stellarpunk.orders import movement

from .core import EntityOperatorAgendum

def possible_buys(
        character:core.Character,
        gamestate:core.Gamestate,
        ship:core.SectorEntity,
        ship_agent:core.EconAgent,
        allowed_resources:list[int],
        buy_from_stations:Optional[list[uuid.UUID]],
        ) -> Mapping[int, list[tuple[float, float, intel.StationIntel, intel.EconAgentIntel]]]:
    assert ship.sector is not None
    # figure out possible buys by resource
    buys:collections.defaultdict[int, list[tuple[float, float, intel.StationIntel, intel.EconAgentIntel]]] = collections.defaultdict(list)
    buy_hits:list[tuple[intel.EconAgentIntel, intel.StationIntel]] = []

    resources = frozenset(ship_agent.buy_resources()).intersection(allowed_resources)
    if len(resources) == 0:
        return buys

    for econ_agent_intel in character.intel_manager.intel(intel.EconAgentSectorEntityPartialCriteria(sell_resources=resources), intel.EconAgentIntel):
        if buy_from_stations is not None and econ_agent_intel.intel_entity_id not in buy_from_stations:
            continue
        # get the corresponding station intel so we know where to go
        #TODO: what about planets?
        station_intel = character.intel_manager.get_intel(intel.EntityIntelMatchCriteria(econ_agent_intel.underlying_entity_id), intel.StationIntel)
        if not station_intel:
            continue
        buy_hits.append((econ_agent_intel, station_intel))

    for econ_agent_intel, station_intel in buy_hits:
        for resource, (price, amount) in econ_agent_intel.sell_offers.items():
            if amount < 1.:
                continue
            assert price < np.inf

            buys[resource].append((price, amount, station_intel, econ_agent_intel))

    return buys

def possible_sales(
        character:core.Character,
        gamestate:core.Gamestate,
        ship:core.SectorEntity,
        ship_agent:core.EconAgent,
        allowed_resources:list[int],
        allowed_stations:Optional[list[uuid.UUID]],
        ) -> Mapping[int, list[tuple[float, float, intel.StationIntel, intel.EconAgentIntel]]]:
    assert ship.sector is not None
    # figure out possible sales by resource
    sales:collections.defaultdict[int, list[tuple[float, float, intel.StationIntel, intel.EconAgentIntel]]] = collections.defaultdict(list)
    sale_hits:list[tuple[intel.EconAgentIntel, intel.StationIntel]] = []

    resources = frozenset(ship_agent.sell_resources()).intersection(allowed_resources)
    if len(resources) == 0:
        return sales

    for econ_agent_intel in character.intel_manager.intel(intel.EconAgentSectorEntityPartialCriteria(buy_resources=resources), intel.EconAgentIntel):
        if allowed_stations is not None and econ_agent_intel.intel_entity_id not in allowed_stations:
            continue
        # get the corresponding station intel so we know where to go
        #TODO: what about planets?
        station_intel = character.intel_manager.get_intel(intel.EntityIntelMatchCriteria(econ_agent_intel.underlying_entity_id), intel.StationIntel)
        if not station_intel:
            continue
        sale_hits.append((econ_agent_intel, station_intel))

    for econ_agent_intel, station_intel in sale_hits:
        for resource, (price, amount) in econ_agent_intel.buy_offers.items():
            if ship_agent.inventory(resource) < 1.:
                continue
            if amount < 1.:
                continue

            sales[resource].append((
                price,
                min(amount, ship_agent.inventory(resource)),
                station_intel,
                econ_agent_intel
            ))
    return sales

def choose_station_to_buy_from(
        character:core.Character,
        gamestate:core.Gamestate,
        ship:core.Ship,
        ship_agent:core.EconAgent,
        allowed_resources:list[int],
        buy_from_stations:Optional[list[uuid.UUID]],
        sell_to_stations:Optional[list[uuid.UUID]]
        ) -> Optional[tuple[int, intel.StationIntel, intel.EconAgentIntel, float, float]]:

    if ship.sector is None:
        raise ValueError(f'{ship} in no sector')

    # compute diffs of allowed/known stations
    # pick a trade that maximizes profit discounting by time spent travelling
    # and transferring

    # (sell - buy price) * amount
    # transfer time = buy transfer + sell transfer
    # travel time  = time to buy + time to sell

    buys = possible_buys(character, gamestate, ship, ship_agent, allowed_resources, buy_from_stations)

    # find sales, assuming we can acquire whatever resource we need
    sales = possible_sales(character, gamestate, ship, econ.YesAgent(gamestate.production_chain), allowed_resources, sell_to_stations)

    #best_profit_per_time = 0.
    #best_trade:Optional[tuple[int, core.Station, core.EconAgent]] = None
    profits_per_time = []
    trades = []
    for resource in buys.keys():
        for ((buy_price, buy_amount, buy_station, buy_agent), (sale_price, sale_amount, sale_station, sale_agent)) in itertools.product(buys[resource], sales[resource]):
            amount = min(buy_amount, sale_amount)
            profit = (sale_price - buy_price)*amount
            if profit < 0.:
                continue
            transfer_time = ocore.TradeCargoFromStation.transfer_rate() * amount + ocore.TradeCargoToStation.transfer_rate() * amount
            travel_time = movement.GoToLocation.compute_eta(ship, buy_station.loc) + movement.GoToLocation.compute_eta(ship, sale_station.loc, starting_loc=buy_station.loc)

            profit_per_time = profit / (transfer_time + travel_time)
            profits_per_time.append(profit_per_time)
            trades.append((resource, buy_station, buy_agent, profit, transfer_time + travel_time)) # type: ignore
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
        character:core.Character,
        gamestate:core.Gamestate,
        ship:core.Ship,
        ship_agent:core.EconAgent,
        allowed_resources:list[int],
        sell_to_stations:Optional[list[uuid.UUID]]
        ) -> Optional[tuple[int, intel.StationIntel, intel.EconAgentIntel]]:
    """ Choose a station to sell goods from ship to """

    if ship.sector is None:
        raise ValueError(f'{ship} in no sector')

    # pick the station where we'll get the best profit for our cargo
    # biggest profit-per-tick for our cargo

    #TODO: how do we access price information? shouldn't this be
    # somehow limited to a view specific to us?
    #TODO: what if no stations seem to trade the resources we have?
    #TODO: what if no stations seem to trade any allowed resources?

    sales = possible_sales(character, gamestate, ship, ship_agent, allowed_resources, sell_to_stations)

    best_profit_per_time = 0.
    best_trade:Optional[tuple[int, intel.StationIntel, intel.EconAgentIntel]] = None
    for resource in sales.keys():
        for sale_price, amount, sale_station, sale_agent in sales[resource]:
            profit = sale_price * amount
            transfer_time = ocore.TradeCargoToStation.transfer_rate() * amount
            travel_time = movement.GoToLocation.compute_eta(ship, sale_station.loc)

            profit_per_time = profit / (transfer_time + travel_time)
            if profit_per_time > best_profit_per_time:
                best_profit_per_time = profit_per_time
                best_trade = (resource, sale_station, sale_agent)

    assert best_trade is None or best_profit_per_time > 0.
    return best_trade

# how long to wait while trying to acquire primary agenda position
MINING_SLEEP_TIME = 10.
MINING_SLEEP_TIME_NO_INTEL = 30.
TRADING_SLEEP_TIME = 10.
TRADING_SLEEP_TIME_NO_INTEL = 30.

class MiningAgendum(core.OrderObserver, core.IntelManagerObserver, EntityOperatorAgendum):
    """ Managing a ship for mining.

    Operates as a state machine as we mine asteroids and sell the resources to
    relevant stations. """

    class State(enum.IntEnum):
        # we're in between mine/sell cycles
        IDLE = enum.auto()
        # we're actively pursuing an asteroid
        MINING = enum.auto()
        # we're actively pursuing a sale to station
        TRADING = enum.auto()
        # we're done with all our work
        COMPLETE = enum.auto()
        # we're sleeping, waiting for a mine/sell opportunity
        SLEEP = enum.auto()
        # we're trying to become the primary agendum
        WAIT_PRIMARY = enum.auto()
        # we can't get enough intel to do mining
        NO_INTEL_WAIT_PRIMARY = enum.auto()
        NO_INTEL = enum.auto()

    @classmethod
    def create_mining_agendum[T:MiningAgendum](cls:Type[T], *args:Any, **kwargs:Any) -> T:
        a = cls.create_eoa(*args, **kwargs)
        a.initialize_mining_agendum()
        return a

    def __init__(
        self,
        *args: Any,
        allowed_resources: Optional[list[int]] = None,
        allowed_stations:Optional[list[uuid.UUID]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.agent:econ.ShipTraderAgent = None # type: ignore
        # resources we're allowed to mine
        if allowed_resources is None:
            allowed_resources = list(range(self.gamestate.production_chain.ranks[0]))
        self.allowed_resources = allowed_resources

        self.allowed_stations:Optional[list[uuid.UUID]] = None

        # state machine to keep track of what we're doing
        self.state:MiningAgendum.State = MiningAgendum.State.IDLE

        # keep track of a couple sorts of actions
        self.mining_order:Optional[ocore.MineOrder] = None
        self.transfer_order:Optional[ocore.TradeCargoToStation] = None

        self.round_trips = 0
        self.max_trips = -1

        self._pending_intel_interest:Optional[core.IntelMatchCriteria] = None

    def __str__(self) -> str:
        return f'{util.fullname(self)} {self.state.name}'

    def initialize_mining_agendum(self) -> None:
        assert(isinstance(self.craft, core.Ship))
        self.agent = econ.ShipTraderAgent.create_ship_trader_agent(self.craft, self.character, self.gamestate)

    # core.OrderObserver

    @property
    def observer_id(self) -> uuid.UUID:
        return self.agenda_id

    def order_begin(self, order:core.Order) -> None:
        pass

    def order_complete(self, order:core.Order) -> None:
        if self.state == MiningAgendum.State.MINING:
            assert order == self.mining_order
            self.mining_order = None
        elif self.state == MiningAgendum.State.TRADING:
            assert order == self.transfer_order
            self.transfer_order = None
            self.round_trips += 1
        else:
            raise ValueError("got order_complete in wrong state {self.state}")

        self.state = MiningAgendum.State.IDLE
        self.gamestate.schedule_agendum_immediate(self)

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

    # core.IntelManagerObserver

    def intel_added(self, intel_manager:core.AbstractIntelManager, intel:core.AbstractIntel) -> None:
        # either we're not sleeping or we're waiting on some specific kind of intel
        assert self.state != MiningAgendum.State.SLEEP or self._pending_intel_interest
        # check if this is intel we've been waiting for
        if self._pending_intel_interest and self._pending_intel_interest.matches(intel):
            assert not self.is_primary()
            assert self.state == MiningAgendum.State.SLEEP
            self._pending_intel_interest = None
            # we can wake up, taking primary and try going back to work
            # note, we can't take primary right now, but we'll be able to do
            # that when we act
            self.state = MiningAgendum.State.WAIT_PRIMARY
            self.gamestate.schedule_agendum_immediate(self, jitter=1.0)

    def intel_undesired(self, intel_manager:core.AbstractIntelManager, intel_criteria:core.IntelMatchCriteria) -> None:
        if self._pending_intel_interest and self._pending_intel_interest == intel_criteria:
            assert not self.is_primary()
            assert self.state == MiningAgendum.State.SLEEP
            # we should try taking back primary, but we will not be able to go
            # back to work.
            # clear pending_intel_interest avoid an inconsistent state
            self._pending_intel_interest = None
            self.state = MiningAgendum.State.NO_INTEL_WAIT_PRIMARY
            self.gamestate.schedule_agendum_immediate(self, jitter=1.0)


    def _choose_asteroid(self) -> Optional[intel.AsteroidIntel]:
        if self.craft.sector is None:
            raise ValueError(f'{self.craft} in no sector')

        nearest = None
        nearest_dist = np.inf
        distances = []
        candidates = []
        for a in self.character.intel_manager.intel(intel.AsteroidIntelPartialCriteria(resources=frozenset(self.allowed_resources)), intel.AsteroidIntel):
            #TODO: handle out of sector mining
            if a.sector_id != self.craft.sector.entity_id:
                continue
            if a.amount > 0.:
                candidates.append(a)
                distances.append(util.distance(self.craft.loc, a.loc))

        if len(candidates) == 0:
            return None

        #TODO: choose asteroids in a more sensible way
        p = 1.0 / np.array(distances)
        p = p / p.sum()
        idx = self.gamestate.random.choice(len(candidates), 1, p=p)[0]
        return candidates[idx]

    def _preempt_primary(self) -> bool:
        #TODO: decide if we want to relinquish being primary
        return False

    def _start(self) -> None:
        self.state = MiningAgendum.State.WAIT_PRIMARY
        self.gamestate.schedule_agendum_immediate(self, jitter=5.)
        self.character.intel_manager.observe(self)

    def _unpause(self) -> None:
        assert self.state == MiningAgendum.State.IDLE
        assert self.is_primary()
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
        self.character.intel_manager.unobserve(self)

    def _do_selling(self) -> None:
        assert(isinstance(self.craft, core.Ship))
        # if we've got resources to sell, find a station to sell to

        sell_station_ret = choose_station_to_sell_to(
                self.character,
                self.gamestate, self.craft, self.agent,
                self.allowed_resources, self.allowed_stations,
        )
        if sell_station_ret is None:
            #TODO: we could also mine something else instead
            self._pending_intel_interest = intel.EconAgentSectorEntityPartialCriteria(buy_resources=frozenset(np.flatnonzero(self.craft.cargo)).intersection(self.allowed_resources))
            self.character.intel_manager.register_intel_interest(self._pending_intel_interest)
            self.logger.debug(f'cannot find a station buying my mined resources ({np.where(self.craft.cargo[self.allowed_resources] > 0.)}). Sleeping...')
            # we cannot trade until we find a station that buys our cargo
            self.state = MiningAgendum.State.SLEEP
            self.relinquish_primary()
            return

        resource, station, station_agent = sell_station_ret
        assert station_agent.buy_offers[resource][0] > 0
        assert station_agent.buy_offers[resource][1] > 0
        #TODO: sensibly have a floor for selling the good
        # basically we pick a station and hope for the best
        floor_price = 0.

        self.state = MiningAgendum.State.TRADING

        assert self.craft.sector
        assert station.sector_id == self.craft.sector.entity_id
        #TODO: multiple sectors
        #TODO: we should probably not reach into the actual entities here
        actual_agent = self.gamestate.get_entity(station_agent.intel_entity_id, core.EconAgent)

        self.transfer_order = ocore.TradeCargoToStation.create_trade_cargo_to_station(
                actual_agent, self.agent, floor_price,
                station, resource, self.craft.cargo[resource],
                self.craft, self.gamestate)
        self.transfer_order.observe(self)
        self.craft.prepend_order(self.transfer_order)

    def _do_mining(self) -> None:
        assert(isinstance(self.craft, core.Ship))
        # if we don't have any resources in cargo, go mine some

        target = self._choose_asteroid()
        if target is None:
            self._pending_intel_interest = intel.AsteroidIntelPartialCriteria(resources=frozenset(self.allowed_resources))
            self.character.intel_manager.register_intel_interest(self._pending_intel_interest)
            self.logger.debug(f'could not find asteroid of type {self.allowed_resources} in {self.craft.sector}, sleeping...')
            # we cannot mine until we learn of some asteroids (via intel_added)
            self.state = MiningAgendum.State.SLEEP
            self.relinquish_primary()
            return

        #TODO: choose amount to harvest
        # push mining order
        self.state = MiningAgendum.State.MINING
        self.mining_order = ocore.MineOrder.create_mine_order(target, 1e3, self.craft, self.gamestate)
        self.mining_order.observe(self)
        self.craft.prepend_order(self.mining_order)

    def is_complete(self) -> bool:
        return self.max_trips >= 0 and self.round_trips >= self.max_trips

    def act(self) -> None:
        if self.paused:
            return

        if not self.is_primary():
            assert self.state in (MiningAgendum.State.WAIT_PRIMARY, MiningAgendum.State.NO_INTEL_WAIT_PRIMARY)
            # we must have been sleeping, waiting for intel and now we might
            # have what we need
            # we had relinquished primary so intel collection could work
            # they might be done, or might not, let's see
            if self.character.find_primary_agendum():
                # sleep some more hoping whoever is primary will relinquish
                self.gamestate.schedule_agendum(self.gamestate.timestamp + MINING_SLEEP_TIME, self)
                return
            # take back control
            self.make_primary()
            assert isinstance(self.craft, core.Ship)
            self.craft.clear_orders()

            if self.state == MiningAgendum.State.WAIT_PRIMARY:
                # we successfully discovered what we needed, go back to work
                self.state = MiningAgendum.State.IDLE
            else:
                # we failed to discover what we needed, all we can do is sit
                # around and try again
                self.state = MiningAgendum.State.NO_INTEL
                self._pending_intel_interest = None
                self.craft.prepend_order(movement.WaitOrder.create_wait_order(self.craft, self.gamestate))
                self.gamestate.schedule_agendum(self.gamestate.timestamp + MINING_SLEEP_TIME_NO_INTEL, self)
                return

        #TODO: periodically wake up and check if there's nearby asteroids or
        # stations that we don't have good econ intel for
        # we can be opportunistic and grab that intel while we're operating
        # do a sensor scan trying to identify asteroids and stations
        # if there are stations "nearby" that buy the resources we mine, dock
        # at them

        assert self.state in (MiningAgendum.State.IDLE, MiningAgendum.State.NO_INTEL)

        if self.is_complete():
            self.state = MiningAgendum.State.COMPLETE
            #TODO: should we give up primary? should we stop ourselves?
            # this is mostly a state for test purposes
            return

        if np.any(self.craft.cargo[self.allowed_resources] > 0.):
            self._do_selling()
        else:
            self._do_mining()

class TradingAgendum(core.OrderObserver, core.IntelManagerObserver, EntityOperatorAgendum):

    class State(enum.IntEnum):
        # we're between buy/sell cycles
        IDLE = enum.auto()
        # we're actively buying a good we think we can sell
        BUYING = enum.auto()
        # we're actively selling a good
        SELLING = enum.auto()
        # we're done with all our work
        COMPLETE = enum.auto()
        # we're waiting on a buy/sell opportunity
        SLEEP = enum.auto()
        # we're trying to become the primary agendum
        WAIT_PRIMARY = enum.auto()
        # we can't get enough intel to do trading
        NO_INTEL_WAIT_PRIMARY = enum.auto()
        NO_INTEL = enum.auto()

    @classmethod
    def create_trading_agendum[T:TradingAgendum](
            cls:Type[T],
            *args:Any,
            **kwargs:Any
    ) -> T:
        a = cls.create_eoa(*args, **kwargs)
        a.initialize_trading_agendum()
        return a

    def __init__(self,
        *args: Any,
        allowed_goods:Optional[list[int]]=None,
        buy_from_stations:Optional[list[uuid.UUID]]=None,
        sell_to_stations:Optional[list[uuid.UUID]]=None,
        **kwargs:Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.agent:econ.ShipTraderAgent = None # type: ignore
        self.state = TradingAgendum.State.IDLE

        # goods we're allowed to trade
        if allowed_goods is None:
            allowed_goods = list(range(self.gamestate.production_chain.ranks[0], self.gamestate.production_chain.ranks.cumsum()[-2]))
        self.allowed_goods = allowed_goods

        self.buy_from_stations = buy_from_stations
        self.sell_to_stations = sell_to_stations

        self.buy_order:Optional[ocore.TradeCargoFromStation] = None
        self.sell_order:Optional[ocore.TradeCargoToStation] = None

        # known economic state, and our effort to expand that
        self._pending_intel_interests:set[core.IntelMatchCriteria] = set()
        self._known_sold_resources:frozenset[int] = frozenset()
        self._known_bought_resources:frozenset[int] = frozenset()

        self.trade_trips = 0
        self.max_trips = -1

    def __str__(self) -> str:
        return f'{util.fullname(self)} {self.state.name}'

    def initialize_trading_agendum(self) -> None:
        assert(isinstance(self.craft, core.Ship))
        self.agent = econ.ShipTraderAgent.create_ship_trader_agent(self.craft, self.character, self.gamestate)

    # core.OrderObserver

    def order_begin(self, order:core.Order) -> None:
        pass

    def order_complete(self, order:core.Order) -> None:
        if self.state == TradingAgendum.State.BUYING:
            assert order == self.buy_order
            self.buy_order = None
        elif self.state == TradingAgendum.State.SELLING:
            assert order == self.sell_order
            self.sell_order = None
            self.trade_trips += 1
        else:
            raise ValueError("got order_complete in wrong state {self.state}")

        # go back into idle state to start things off again
        self.state = TradingAgendum.State.IDLE
        self.gamestate.schedule_agendum_immediate(self)

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

    # core.IntelManagerObserver

    def intel_added(self, intel_manager:core.AbstractIntelManager, intel_item:core.AbstractIntel) -> None:
        # either we're not sleeping or we have some intel interest we're waiting on
        assert self.state != TradingAgendum.State.SLEEP or self._pending_intel_interests
        # check if this is intel we've been waiting for
        matches:list[core.IntelMatchCriteria] = []
        for interest in self._pending_intel_interests:
            if interest.matches(intel_item):
                matches.append(interest)

        for match in matches:
            self._pending_intel_interests.remove(match)

        # we might have already woken up because of some other match
        if self.state == TradingAgendum.State.SLEEP and matches:
            assert not self.is_primary()
            # we don't need to wait on any more intel, try with this
            # if it isn't enough, we'll ask for more in the next act cycle
            self._pending_intel_interests.clear()
            # we can wake up, taking primary and try going back to work
            # note, we can't take primary right now, but we'll be able to do
            # that when we act
            self.state = TradingAgendum.State.WAIT_PRIMARY
            self.gamestate.schedule_agendum_immediate(self, jitter=1.0)

    def intel_undesired(self, intel_manager:core.AbstractIntelManager, intel_criteria:core.IntelMatchCriteria) -> None:
        if intel_criteria in self._pending_intel_interests:
            # we were waiting on this, but it's unsatisfiable, give up on it.
            self._pending_intel_interests.remove(intel_criteria)

            if self.state == TradingAgendum.State.SLEEP and len(self._pending_intel_interests) == 0:
                assert not self.is_primary()
                # because we're still sleeping we must never have gotten a
                # piece of intel that woke us up.
                # this last piece of intel that could have woken us up is
                # unsatisfiable
                # we should try taking back primary, but we will not be able to
                # go back to work.
                self.state = TradingAgendum.State.NO_INTEL_WAIT_PRIMARY
                self.gamestate.schedule_agendum_immediate(self, jitter=1.0)


    # Agendum

    def _preempt_primary(self) -> bool:
        #TODO: decide if we want to relinquish being primary
        return False

    def _start(self) -> None:
        self.state = TradingAgendum.State.WAIT_PRIMARY
        self.gamestate.schedule_agendum_immediate(self, jitter=5.)
        self.character.intel_manager.observe(self)

    def _unpause(self) -> None:
        assert self.state == TradingAgendum.State.IDLE
        assert self.is_primary()
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
        self.character.intel_manager.unobserve(self)

    def is_complete(self) -> bool:
        return self.max_trips >= 0 and self.trade_trips >= self.max_trips

    def _register_interest(self, interest:core.IntelMatchCriteria) -> None:
        if interest not in self._pending_intel_interests:
            self._pending_intel_interests.add(interest)
            self.character.intel_manager.register_intel_interest(interest)

    def _register_interests(self) -> None:
        # we need a buy/sell pair. cases:
        # * a buyer for a good we have
        # * a buyer for a known good that's sold
        # * a seller for a known good that's bought
        # * any buyer or seller for allowed goods

        # we operate in two phases:
        # fresh: look for anything that might be useful
        # delta: we've learned something new, try to exploit that

        buys = possible_buys(
                self.character,
                self.gamestate, self.craft, self.agent,
                self.allowed_goods, self.buy_from_stations
        )
        sales = possible_sales(
                self.character,
                self.gamestate, self.craft, self.agent,
                self.allowed_goods, self.sell_to_stations,
        )
        known_sold_resources = frozenset(buys.keys()).intersection(self.allowed_goods)
        known_bought_resources = frozenset(sales.keys()).intersection(self.allowed_goods)

        # if we have new buy/sell knowledge since the last time we looked,
        # just try to exploit that (e.g. find buyer for newly known sold good)
        # failing that, look for everything
        delta_sold = known_sold_resources - self._known_sold_resources
        delta_bought = known_bought_resources - self._known_bought_resources
        self._known_sold_resources = known_sold_resources
        self._known_bought_resources = known_bought_resources

        if delta_sold or delta_bought:
            self._register_delta_interests(delta_sold, delta_bought)
        else:
            self._register_fresh_interests()

    def _register_delta_interests(self, delta_sold:frozenset[int], delta_bought:frozenset[int]) -> None:
        #TODO: multi-sector?
        # seek buyers for delta_sold resources
        if delta_sold:
            interest = intel.EconAgentSectorEntityPartialCriteria(buy_resources=delta_sold)
            self._register_interest(interest)
        if delta_bought:
            interest = intel.EconAgentSectorEntityPartialCriteria(sell_resources=delta_bought)
            self._register_interest(interest)


        # seek sellers for delta_bought resources

    def _register_fresh_interests(self) -> None:
        #TODO: multi-sector?
        # buyer for our goods (if we have any)
        if np.sum(np.take(self.craft.cargo, self.allowed_goods)) > 0.:
            interest:core.IntelMatchCriteria = intel.EconAgentSectorEntityPartialCriteria(buy_resources=frozenset(np.flatnonzero(self.craft.cargo)).intersection(self.allowed_goods))
            self._register_interest(interest)
        # buyer for known sold good
        if len(self._known_sold_resources) > 0:
            interest = intel.EconAgentSectorEntityPartialCriteria(buy_resources=self._known_sold_resources)
            self._register_interest(interest)

        # seller for known bought goods
        if len(self._known_bought_resources) > 0:
            interest = intel.EconAgentSectorEntityPartialCriteria(sell_resources=self._known_bought_resources)
            self._register_interest(interest)

        # any buyers or sellers
        interest = intel.EconAgentSectorEntityPartialCriteria(buy_resources=frozenset(self.allowed_goods))
        self._register_interest(interest)
        interest = intel.EconAgentSectorEntityPartialCriteria(sell_resources=frozenset(self.allowed_goods))
        self._register_interest(interest)

    def _buy_goods(self) -> None:
        assert(isinstance(self.craft, core.Ship))
        buy_station_ret = choose_station_to_buy_from(
                self.character,
                self.gamestate, self.craft, self.agent,
                self.allowed_goods,
                self.buy_from_stations, self.sell_to_stations)

        if buy_station_ret is None:
            self._register_interests()
            self.logger.debug(f'cannot find a valid trade for my trade goods. Sleeping...')
            # we cannot mine until we learn of some allowed buy/sell pair
            self.state = TradingAgendum.State.SLEEP
            self.relinquish_primary()
            return
        resource, station, station_agent, est_profit, est_time = buy_station_ret
        assert station_agent.sell_offers[resource][0] < np.inf
        assert station_agent.sell_offers[resource][1] > 0.

        self.logger.debug(f'buying {resource=} from {station=} {est_profit=} {est_time=}')

        #TODO: sensibly have a ceiling for buying the good
        # basically we pick a station and hope for the best
        ceiling_price = np.inf
        amount = min(station_agent.sell_offers[resource][1], self.craft.cargo_capacity - self.craft.cargo.sum())

        self.state = TradingAgendum.State.BUYING
        assert self.craft.sector
        #TODO: multiple sectors
        #TODO: we should probably not reach into the actual entities here
        actual_agent = self.gamestate.get_entity(station_agent.intel_entity_id, core.EconAgent)
        self.buy_order = ocore.TradeCargoFromStation.create_trade_cargo_from_station(
                self.agent, actual_agent, ceiling_price,
                station, resource, amount,
                self.craft, self.gamestate)
        self.buy_order.observe(self)
        self.craft.prepend_order(self.buy_order)

    def _sell_goods(self) -> bool:
        # if we've got resources to sell, find a station to sell to
        assert(isinstance(self.craft, core.Ship))

        sell_station_ret = choose_station_to_sell_to(
                self.character,
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
        assert station_agent.buy_offers[resource][0] > 0
        assert station_agent.buy_offers[resource][1] > 0

        self.logger.debug(f'selling {resource=} to {station=}')

        #TODO: sensibly have a floor for selling the good
        # basically we pick a station and hope for the best
        floor_price = 0.

        self.state = TradingAgendum.State.SELLING
        assert self.craft.sector
        assert station.sector_id == self.craft.sector.entity_id
        #TODO: we should probably not reach into the actual entities here
        actual_agent = self.gamestate.get_entity(station_agent.intel_entity_id, core.EconAgent)
        self.sell_order = ocore.TradeCargoToStation.create_trade_cargo_to_station(
                actual_agent, self.agent, floor_price,
                station, resource, self.craft.cargo[resource],
                self.craft, self.gamestate)
        self.sell_order.observe(self)
        self.craft.prepend_order(self.sell_order)

        return True

    def act(self) -> None:
        if self.paused:
            return

        if not self.is_primary():
            assert self.state in (TradingAgendum.State.WAIT_PRIMARY, TradingAgendum.State.NO_INTEL_WAIT_PRIMARY)
            current_primary = self.character.find_primary_agendum()
            # try and grab back primary
            if current_primary and not current_primary.preempt_primary():
                # sleep some more hoping whoever is primary will relinquish
                self.gamestate.schedule_agendum(self.gamestate.timestamp + TRADING_SLEEP_TIME, self, jitter=1.0)
                return
            # take back control
            self.make_primary()
            assert isinstance(self.craft, core.Ship)
            self.craft.clear_orders()

            if self.state == TradingAgendum.State.WAIT_PRIMARY:
                # we discovered what we were looking for, try trading again
                self.state = TradingAgendum.State.IDLE
            else:
                # we failed to discover what we needed, all we can do is sit
                # around and try again
                self.state = TradingAgendum.State.NO_INTEL
                self._pending_intel_interests.clear()
                self.craft.prepend_order(movement.WaitOrder.create_wait_order(self.craft, self.gamestate))
                self.gamestate.schedule_agendum(self.gamestate.timestamp + TRADING_SLEEP_TIME_NO_INTEL, self)
                return

        assert self.state in (TradingAgendum.State.IDLE, TradingAgendum.State.NO_INTEL)

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

    def _preempt_primary(self) -> bool:
        # we can always be preempted. Well resume production on unpause
        return True

    def _unpause(self) -> None:
        self.gamestate.schedule_agendum_immediate(self)

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

    def _preempt_primary(self) -> bool:
        # we can always be preempted. Well resume production on unpause
        return True

    def _unpause(self) -> None:
        self.gamestate.schedule_agendum_immediate(self)

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
