""" Agenda items for characters, reflecting activites they are involved in. """

from typing import Optional, List, Any, Iterable, Tuple
import enum

import numpy as np

from stellarpunk import core
import stellarpunk.orders.core as ocore

class ManagementAgendum(core.Agendum):
    def __init__(self, asset:core.Asset, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.asset = asset

class MiningAgendum(core.OrderObserver, ManagementAgendum):
    """ Managing a ship for mining.

    Operates as a state machine as we mine asteroids and sell the resources to
    relevant stations. """

    class State(enum.Enum):
        IDLE = enum.auto()
        MINING = enum.auto()
        TRADING = enum.auto()

    def __init__(self, *args:Any, allowed_resources:Optional[List[int]]=None, allowed_stations:Optional[List[core.SectorEntity]]=None, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        # resources we're allowed to mine
        if allowed_resources is None:
            allowed_resources = list(range(self.gamestate.production_chain.ranks[0]))
        self.allowed_resources = allowed_resources

        self.allowed_stations:Optional[List[core.SectorEntity]] = allowed_stations

        # state machine to keep track of what we're doing
        self.state:MiningAgendum.State = MiningAgendum.State.IDLE

        # keep track of a couple sorts of actions
        self.mining_order:Optional[ocore.MineOrder] = None
        self.transfer_order:Optional[ocore.TransferCargo] = None

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
        assert isinstance(self.asset, core.Ship)
        ship = self.asset
        if ship.sector is None:
            raise ValueError(f'{ship} in no sector')

        nearest = None
        nearest_dist = np.inf
        distances = []
        candidates = []
        for hit in ship.sector.spatial_point(ship.loc):
            if not isinstance(hit, core.Asteroid):
                continue
            if hit.resource not in self.allowed_resources:
                continue
            if hit.cargo[hit.resource] <= 0:
                continue

            dist = np.linalg.norm(ship.loc - hit.loc)
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

    def _choose_station(self) -> Tuple[int, core.Station]:
        assert isinstance(self.asset, core.Ship)
        ship = self.asset
        if ship.sector is None:
            raise ValueError(f'{ship} in no sector')

        # pick the station where we'll get the best profit for our cargo
        # biggest profit-per-tick for our cargo

        #TODO: how do we access price information? shouldn't this be
        # somehow limited to a view specific to us?
        #TODO: what if no stations seem to trade the resources we have?
        #TODO: what if no stations seem to trade any allowed resources?

        resource = int(ship.cargo[self.allowed_resources].argmax())
        station = None
        hits:Iterable[core.SectorEntity]
        if self.allowed_stations is None:
            hits = ship.sector.spatial_point(ship.loc)
        else:
            hits = self.allowed_stations

        buyer_resources = np.where(self.gamestate.production_chain.adj_matrix[resource] > 0)[0]
        for hit in hits:
            if not isinstance(hit, core.Station):
                continue
            if hit.resource not in buyer_resources:
                continue
            station = hit
            break
        if station is None:
            #TODO: we should try other resources in our cargo
            raise Exception(f'no station buys {resource} buyers: {buyer_resources}')
        assert self.gamestate.production_chain.adj_matrix[resource, station.resource] > 0
        return (resource, station)

    def act(self) -> None:
        assert self.state == MiningAgendum.State.IDLE

        assert isinstance(self.asset, core.Ship)
        ship = self.asset

        if np.any(ship.cargo[self.allowed_resources] > 0.):
            # if we've got resources to sell, find a station to sell to

            resource, station = self._choose_station()
            self.state = MiningAgendum.State.TRADING
            self.transfer_order = ocore.TransferCargo(station, resource, ship.cargo[resource], ship, self.gamestate)
            self.transfer_order.observe(self)
            ship.orders.appendleft(self.transfer_order)
        else:
            # if we don't have any resources in cargo, go mine some

            target = self._choose_asteroid()
            if target is None:
                #TODO: what do we do in this case?
                self.logger.info(f'could not find asteroid of type {self.allowed_resources} in {ship.sector}, stopping harvest')
                return

            #TODO: choose amount to harvest
            # push mining order
            self.state = MiningAgendum.State.MINING
            self.mining_order = ocore.MineOrder(target, 1e3, ship, self.gamestate)
            self.mining_order.observe(self)
            ship.orders.appendleft(self.mining_order)

