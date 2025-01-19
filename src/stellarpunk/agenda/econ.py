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
        buy_from_stations:Optional[list[core.SectorEntity]],
        ) -> Mapping[int, list[tuple[float, float, core.SectorEntity]]]:
    assert ship.sector is not None
    # figure out possible buys by resource
    buys:collections.defaultdict[int, list[tuple[float, float, core.SectorEntity]]] = collections.defaultdict(list)
    buy_hits:list[tuple[intel.EconAgentIntel, intel.StationIntel]] = []

    for econ_agent_intel in character.intel_manager.intel(intel.EconAgentSectorEntityPartialCriteria(sell_resources=frozenset(ship_agent.buy_resources()).intersection(allowed_resources)), intel.EconAgentIntel):
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

            #TODO: we should not reaching into gamestate for the actual station
            station = gamestate.get_entity(station_intel.intel_entity_id, sector_entity.Station)
            buys[resource].append((price, amount, station))

    return buys

def possible_sales(
        character:core.Character,
        gamestate:core.Gamestate,
        ship:core.SectorEntity,
        ship_agent:core.EconAgent,
        allowed_resources:list[int],
        allowed_stations:Optional[list[uuid.UUID]],
        ) -> Mapping[int, list[tuple[float, float, core.SectorEntity]]]:
    assert ship.sector is not None
    # figure out possible sales by resource
    sales:collections.defaultdict[int, list[tuple[float, float, core.SectorEntity]]] = collections.defaultdict(list)
    sale_hits:list[tuple[intel.EconAgentIntel, intel.StationIntel]] = []

    for econ_agent_intel in character.intel_manager.intel(intel.EconAgentSectorEntityPartialCriteria(buy_resources=frozenset(ship_agent.sell_resources()).intersection(allowed_resources)), intel.EconAgentIntel):
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
            assert ship_agent.inventory(resource) > 0.
            if amount < 1.:
                continue

            #TODO: we should not reaching into gamestate for the actual station
            station = gamestate.get_entity(station_intel.intel_entity_id, sector_entity.Station)
            sales[resource].append((
                price,
                min(amount, ship_agent.inventory(resource)),
                station,
            ))
    return sales

def choose_station_to_buy_from(
        character:core.Character,
        gamestate:core.Gamestate,
        ship:core.Ship,
        ship_agent:core.EconAgent,
        allowed_resources:list[int],
        buy_from_stations:Optional[list[core.SectorEntity]],
        sell_to_stations:Optional[list[uuid.UUID]]
        ) -> Optional[tuple[int, sector_entity.Station, core.EconAgent, float, float]]:

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
        for ((buy_price, buy_amount, buy_station), (sale_price, sale_amount, sale_station)) in itertools.product(buys[resource], sales[resource]):
            amount = min(buy_amount, sale_amount)
            profit = (sale_price - buy_price)*amount
            if profit < 0.:
                continue
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
        character:core.Character,
        gamestate:core.Gamestate,
        ship:core.Ship,
        ship_agent:core.EconAgent,
        allowed_resources:list[int],
        sell_to_stations:Optional[list[uuid.UUID]]
        ) -> Optional[tuple[int, sector_entity.Station, core.EconAgent]]:
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
    best_trade:Optional[tuple[int, sector_entity.Station, core.EconAgent]] = None
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

        # ephemeral state, doesn't need to exist outside an act call
        # all of these represent intel opportunities
        self.nearby_hexes:set[tuple[int, int]] = set()
        self.far_hexes:set[tuple[int, int]] = set()
        self.nearby_stations:set[uuid.UUID] = set()

        self._pending_intel_interest:Optional[core.IntelMatchCriteria] = None

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


    def _choose_asteroid(self) -> Optional[sector_entity.Asteroid]:
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
        #TODO: we should not directly retrieve the asteroid, and use the intel
        # or maybe a sensor image of the intel
        target = self.gamestate.get_entity(candidates[idx].intel_entity_id, sector_entity.Asteroid)
        return target

    def _preempt_primary(self) -> bool:
        #TODO: decide if we want to relinquish being primary
        return False

    def _start(self) -> None:
        current_primary = self.character.find_primary_agendum()
        if current_primary:
            raise ValueError(f'cannot start {self} because there is already a primary agendum {current_primary}')
        assert self.state == MiningAgendum.State.IDLE
        self.make_primary()
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

    def _has_enough_intel(self) -> bool:
        """ decide if, given our current position, we've got enough intel to
        successfully mine/trade. """

        #TODO: we're still integrating all this intel logic
        return True

        assert(self.craft.sector)
        # consider if we've got enough intel about asteroids to mine
        # consider if we've got enough fresh buy offers to sell mined goods at
        # we may need to travel (in sector? out of sector?) to get the intel

        # if we don't know about any mine/sell pair at all, we don't have
        # enough intel

        # if we know about any "nearby" asteroids and any "nearby" buyer for
        # those resources, we're happy
        known_all_resources:set[int] = set()
        known_all_buy_resources:set[int] = set()
        known_local_resources:set[int] = set()
        known_local_buy_resources:set[int] = set()
        known_buyers:set[uuid.UUID] = set()
        for a in self.character.intel_manager.intel(intel.AsteroidIntel):
            if a.resource in self.allowed_resources and a.amount > 0:
                known_all_resources.add(a.resource)
                if a.sector_id == self.craft.sector.entity_id:
                    #TODO: condition these on being very close by
                    known_local_resources.add(a.resource)
        for e in self.character.intel_manager.intel(intel.EconAgentIntel):
            if len(e.buy_offers.keys() & set(self.allowed_resources)) > 0:
                known_all_buy_resources.update(e.buy_offers.keys())
                #TODO: what about planets?
                if not issubclass(e.underlying_entity_type, sector_entity.Station):
                    continue
                known_buyers.add(e.underlying_entity_id)
                sentity_intel = self.character.intel_manager.get_intel(intel.EntityIntelMatchCriteria(e.underlying_entity_id), intel.StationIntel)
                if not sentity_intel:
                    continue
                if sentity_intel.sector_id == self.craft.sector.entity_id:
                    #TODO: condition these on being close-ish by
                    known_local_buy_resources.update(e.buy_offers.keys())

        no_mine_sell_pairs = (len(known_all_resources & known_all_buy_resources) == 0)

        if len(known_local_resources & known_local_buy_resources) > 0:
            #TODO: how to decide if these mining/trade opportunities are "good
            # enough" for us to do it?
            return True

        # at this point we don't know about about any nearby mine/sell pairs
        # and we know if we have far away options
        # let's continue to explore nearby intel options

        # if we don't have intel for "nearby" sector hexes, explore near, hope
        # to find asteroids and stations to sell resources at
        self.nearby_hexes = {util.int_coords(x) for x in util.hexes_within_pixel_dist(self.craft.loc, 5e5, self.craft.sector.hex_size)}
        self.far_hexes = {util.int_coords(x) for x in util.hexes_within_pixel_dist(np.array((0.0, 0.0)), self.craft.sector.radius*2, self.craft.sector.hex_size)}
        self.far_hexes -= self.nearby_hexes
        for sector_hex_intel in self.character.intel_manager.intel(intel.SectorHexIntel):
            if sector_hex_intel.sector_id != self.character or not sector_hex_intel.is_static:
                continue
            h = util.int_coords(sector_hex_intel.hex_loc)
            if h in self.nearby_hexes:
                self.nearby_hexes.remove(h)
            if h in self.far_hexes:
                self.far_hexes.remove(h)

        # if we have intel for "nearby" stations that buy our goods, but don't
        # have econ info, explore near, hope to find buy offers
        self.nearby_stations = set()
        for station_intel in self.character.intel_manager.intel(intel.StationIntel):
            if station_intel.sector_id == self.craft.sector.entity_id:
                allowed_resources = self.allowed_resources
                if any(x in allowed_resources for x in station_intel.inputs):
                    if station_intel.intel_entity_id not in known_buyers:
                        self.nearby_stations.add(station_intel.intel_entity_id)

        # these represent our intel opportunities to exploit
        # in this case, they're all doable right now
        if len(self.nearby_hexes) > 0:
            return False
        if len(self.nearby_stations) > 0:
            return False
        if len(self.far_hexes) > 0:
            return False

        # we have no nearby intel opportunities, if we have trade opps far
        # away, do that and check in again later. otherwise we'll have to
        # travel far away to get intel
        return no_mine_sell_pairs

    def _do_intel(self) -> None:
        #TODO: seek intel about asteroids and stations buying resources
        # this should be reusable logic other stuff can use to accumulate
        # relevant intel

        # if we have intel opportunities in this sector, go explore those
        if len(self.nearby_hexes) > 0:
            #TODO: pick one to go to do a sensor scan
            pass
        if len(self.nearby_stations) > 0:
            #TODO: pick one to go to and dock
            pass
        if len(self.far_hexes) > 0:
            #TODO: pick one to go to do a sensor scan
            pass

        #TODO: otherwise we'll need to travel out of sector
        pass

    def is_complete(self) -> bool:
        return self.max_trips >= 0 and self.round_trips >= self.max_trips

    def act(self) -> None:
        if self.paused:
            return

        if not self.is_primary():
            assert self.state == MiningAgendum.State.WAIT_PRIMARY
            # we must have been sleeping, waiting for intel and now we might
            # have what we need
            # we had relinquished primary so intel collection could work
            # they might be done, or might not, let's see
            assert self._pending_intel_interest is None
            if self.character.find_primary_agendum():
                # sleep some more hoping whoever is primary will relinquish
                self.gamestate.schedule_agendum(self.gamestate.timestamp + MINING_SLEEP_TIME, self)
                return
            # take back control
            self.make_primary()
            self.state = MiningAgendum.State.IDLE

        #TODO: periodically wake up and check if there's nearby asteroids or
        # stations that we don't have good econ intel for
        # we can be opportunistic and grab that intel while we're operating
        # do a sensor scan trying to identify asteroids and stations
        # if there are stations "nearby" that buy the resources we mine, dock
        # at them

        assert self.state == MiningAgendum.State.IDLE

        # do a scan of our current location. this will get us intel about the
        # sector and entities within it that we can see
        assert(self.craft.sector)
        self.craft.sector.sensor_manager.scan(self.craft)

        if self.is_complete():
            self.state = MiningAgendum.State.COMPLETE
            return

        if np.any(self.craft.cargo[self.allowed_resources] > 0.):
            self._do_selling()
        elif self._has_enough_intel():
            self._do_mining()
        else:
            self._do_intel()

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

    @classmethod
    def create_trading_agendum[T:TradingAgendum](
            cls:Type[T],
            *args:Any,
            buy_from_stations: Optional[list[core.SectorEntity]] = None,
            **kwargs:Any
    ) -> T:
        a = cls.create_eoa(*args, **kwargs)
        a.initialize_trading_agendum(buy_from_stations)
        return a

    def __init__(self,
        *args: Any,
        allowed_goods:Optional[list[int]]=None,
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

        self.buy_from_stations:Optional[list[core.SectorEntity]] = None
        self.sell_to_stations = sell_to_stations

        self.buy_order:Optional[ocore.TradeCargoFromStation] = None
        self.sell_order:Optional[ocore.TradeCargoToStation] = None

        self._pending_intel_interests:set[core.IntelMatchCriteria] = set()

        self.trade_trips = 0
        self.max_trips = -1

    def initialize_trading_agendum(
            self,
            buy_from_stations: Optional[list[core.SectorEntity]] = None,
            sell_to_stations: Optional[list[uuid.UUID]] = None,
    ) -> None:
        assert(isinstance(self.craft, core.Ship))
        self.agent = econ.ShipTraderAgent.create_ship_trader_agent(self.craft, self.character, self.gamestate)
        self.buy_from_stations = buy_from_stations
        self.sell_to_stations = sell_to_stations

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

    def intel_added(self, intel_manager:core.AbstractIntelManager, intel:core.AbstractIntel) -> None:
        # either we're not sleeping or we have some intel interest we're waiting on
        assert self.state != TradingAgendum.State.SLEEP or self._pending_intel_interests
        # check if this is intel we've been waiting for
        matches:list[core.IntelMatchCriteria] = []
        for interest in self._pending_intel_interests:
            if interest.matches(intel):
                matches.append(interest)

        for match in matches:
            self._pending_intel_interests.remove(match)

        # we might have already woken up because of some other match
        if self.state == TradingAgendum.State.SLEEP and matches:
            assert not self.is_primary()
            # we can wake up, taking primary and try going back to work
            # note, we can't take primary right now, but we'll be able to do
            # that when we act
            self.state = TradingAgendum.State.WAIT_PRIMARY
            self.gamestate.schedule_agendum_immediate(self, jitter=1.0)

    # Agendum

    def _preempt_primary(self) -> bool:
        #TODO: decide if we want to relinquish being primary
        return False

    def _start(self) -> None:
        current_primary = self.character.find_primary_agendum()
        if current_primary:
            raise ValueError(f'cannot start {self} because there is already a primary agendum {current_primary}')
        assert self.state == TradingAgendum.State.IDLE
        self.make_primary()
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

    def _register_interests(self) -> None:
        #TODO: prioritize these somehow
        # we need a buy/sell pair. cases:
        # * a buyer for a good we have
        # * a buyer for a known good that's sold
        # * a seller for a known good that's bought
        # * any buyer or seller for allowed goods

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
        known_sold_resources = buys.keys()
        known_bought_resources = sales.keys()

        # buyer for our goods (if we have any)
        if np.sum(np.take(self.craft.cargo, self.allowed_goods)) > 0.:
            self._pending_intel_interests.add(intel.EconAgentSectorEntityPartialCriteria(buy_resources=frozenset(np.flatnonzero(self.craft.cargo)).intersection(self.allowed_goods)))
        # buyer for known sold good
        self._pending_intel_interests.add(intel.EconAgentSectorEntityPartialCriteria(buy_resources=frozenset(known_sold_resources).intersection(self.allowed_goods)))
        # seller for known bought goods
        self._pending_intel_interests.add(intel.EconAgentSectorEntityPartialCriteria(sell_resources=frozenset(known_bought_resources).intersection(self.allowed_goods)))

        # any buyers or sellers
        self._pending_intel_interests.add(intel.EconAgentSectorEntityPartialCriteria(buy_resources=frozenset(self.allowed_goods)))
        self._pending_intel_interests.add(intel.EconAgentSectorEntityPartialCriteria(sell_resources=frozenset(self.allowed_goods)))

        for interest in self._pending_intel_interests:
            self.character.intel_manager.register_intel_interest(interest)


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
        if self.paused:
            return

        if not self.is_primary():
            assert self.state == TradingAgendum.State.WAIT_PRIMARY
            if self.character.find_primary_agendum():
                # sleep some more hoping whoever is primary will relinquish
                self.gamestate.schedule_agendum(self.gamestate.timestamp + TRADING_SLEEP_TIME, self, jitter=1.0)
                return
            # take back control
            self.make_primary()
            self.state = TradingAgendum.State.IDLE

        assert self.state == TradingAgendum.State.IDLE

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
