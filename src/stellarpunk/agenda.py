""" Agenda items for characters, reflecting activites they are involved in. """

from typing import Optional, List, Any, Iterable, Tuple
import enum

import numpy as np

from stellarpunk import core, econ
import stellarpunk.orders.core as ocore

# how long to wait, idle if we can't do work
MINING_SLEEP_TIME = 60.

class MiningAgendum(core.OrderObserver, core.Agendum):
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
            self.gamestate.schedule_agendum(self.gamestate.timestamp, self)
        elif self.state == MiningAgendum.State.TRADING:
            assert order == self.transfer_order
            self.transfer_order = None
            # go back into idle state to start things off again
            self.state = MiningAgendum.State.IDLE
            self.gamestate.schedule_agendum(self.gamestate.timestamp, self)
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
        self.gamestate.schedule_agendum(self.gamestate.timestamp, self)

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

            dist = np.linalg.norm(self.ship.loc - hit.loc)
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

    def _choose_station(self) -> Optional[Tuple[int, core.Station, core.EconAgent]]:
        if self.ship.sector is None:
            raise ValueError(f'{self.ship} in no sector')

        # pick the station where we'll get the best profit for our cargo
        # biggest profit-per-tick for our cargo

        #TODO: how do we access price information? shouldn't this be
        # somehow limited to a view specific to us?
        #TODO: what if no stations seem to trade the resources we have?
        #TODO: what if no stations seem to trade any allowed resources?

        resource = int(self.ship.cargo[self.allowed_resources].argmax())
        station = None
        hits:Iterable[core.SectorEntity]
        if self.allowed_stations is None:
            hits = self.ship.sector.spatial_point(self.ship.loc, mask=core.ObjectFlag.STATION)
        else:
            hits = self.allowed_stations

        buyer_resources = np.where(self.gamestate.production_chain.adj_matrix[resource] > 0)[0]
        station_agent = None
        for hit in hits:
            if not isinstance(hit, core.Station):
                continue
            if hit.resource not in buyer_resources:
                continue
            if hit.entity_id not in self.gamestate.econ_agents:
                continue
            station_agent = self.gamestate.econ_agents[hit.entity_id]
            if station_agent.balance() < station_agent.buy_price(resource):
                continue
            if station_agent.budget(resource) < station_agent.buy_price(resource):
                continue
            station = hit
            break

        if station is None:
            #TODO: we should try other resources in our cargo
            return None

        assert station_agent is not None
        return (resource, station, station_agent)

    def start(self) -> None:
        assert self.state == MiningAgendum.State.IDLE
        self.gamestate.schedule_agendum(self.gamestate.timestamp, self)

    def is_complete(self) -> bool:
        return self.max_trips >= 0 and self.round_trips >= self.max_trips

    def act(self) -> None:
        assert self.state == MiningAgendum.State.IDLE

        if self.is_complete():
            self.state = MiningAgendum.State.COMPLETE
            return

        if np.any(self.ship.cargo[self.allowed_resources] > 0.):
            # if we've got resources to sell, find a station to sell to

            station_ret = self._choose_station()
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
            self.ship.orders.appendleft(self.transfer_order)
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
            self.ship.orders.appendleft(self.mining_order)

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

