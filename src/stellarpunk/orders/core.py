""" Orders that can be given to ships. """

from __future__ import annotations

from typing import Optional, Any

import numpy as np

from stellarpunk import util, core, effects

from .movement import GoToLocation
from .steering import ZERO_VECTOR

class MineOrder(core.Order):
    def __init__(self, target: core.Asteroid, amount: float, *args: Any, max_dist:float=2e3, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.target = target
        self.max_dist = max_dist
        self.amount = amount
        self.mining_effect:Optional[effects.MiningEffect] = None
        self.mining_rate = 2e1

    def _begin(self) -> None:
        self.init_eta = (
                GoToLocation.compute_eta(self.ship, self.target.loc)
                + self.amount / self.mining_rate
        )

    def _cancel(self) -> None:
        if self.mining_effect:
            self.mining_effect.cancel_effect()

    def is_complete(self) -> bool:
        return self.mining_effect is not None and self.mining_effect.is_complete()

    def act(self, dt: float) -> None:
        # grab resources from the asteroid and add to our cargo
        distance = np.linalg.norm(self.ship.loc - self.target.loc) - self.target.radius
        if distance > self.max_dist:
            self.ship.orders.appendleft(GoToLocation.goto_entity(self.target, self.ship, self.gamestate, surface_distance=self.max_dist))
            return

        if not self.mining_effect:
            assert self.ship.sector is not None
            self.mining_effect = effects.MiningEffect(
                    self.target.resource, self.amount, self.target, self.ship, self.ship.sector, self.gamestate, mining_rate=self.mining_rate)
            self.ship.sector.effects.append(self.mining_effect)
        # else wait for the mining effect

class TransferCargo(core.Order):
    def __init__(self, target: core.SectorEntity, resource: int, amount: float, *args: Any, max_dist:float=2e3, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.target = target
        self.resource = resource
        self.amount = amount
        self.transferred = 0.
        self.max_dist = max_dist
        self.transfer_rate = 1e2

        self.transfer_effect:Optional[effects.TransferCargoEffect] = None

    def _begin(self) -> None:
        self.init_eta = (
                GoToLocation.compute_eta(self.ship, self.target.loc)
                + self.amount / self.transfer_rate
        )

    def _cancel(self) -> None:
        if self.transfer_effect:
            self.transfer_effect.cancel_effect()

    def is_complete(self) -> bool:
        return self.transfer_effect is not None and self.transfer_effect.is_complete()

    def act(self, dt:float) -> None:
        # if we're too far away, go to the target
        distance = np.linalg.norm(self.ship.loc - self.target.loc) - self.target.radius
        if distance > self.max_dist:
            self.ship.orders.appendleft(GoToLocation.goto_entity(self.target, self.ship, self.gamestate, surface_distance=self.max_dist))
            return

        #TODO: multiple goods? transfer from us to them?
        if not self.transfer_effect:
            assert self.ship.sector is not None
            self.transfer_effect = effects.TransferCargoEffect(
                    self.resource, self.amount, self.ship, self.target,
                    self.ship.sector, self.gamestate,
                    transfer_rate=self.transfer_rate)
            self.ship.sector.effects.append(self.transfer_effect)
        # else wait for the transfer effect

class HarvestOrder(core.Order):
    def __init__(self, base:core.SectorEntity, resource:int, *args:Any, max_trips:int=0, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        self.base = base
        self.resource = resource
        self.trips = 0
        self.max_trips = max_trips
        self.keep_harvesting = True

        self.mining_order:Optional[MineOrder] = None
        self.transfer_order:Optional[TransferCargo] = None

    def _begin(self) -> None:
        if self.max_trips > 0:
            assert self.ship.sector is not None
            # travel to an asteroid, mine, travel to base, transfer, repeat
            mining_loc = util.polar_to_cartesian(self.ship.sector.radius*2, 0)
            mining_rate = 2e1
            transfer_rate = 6e1
            self.init_eta = self.max_trips * (
                2*GoToLocation.compute_eta(self.ship, mining_loc)
                + self.ship.cargo_capacity / mining_rate
                + self.ship.cargo_capacity / transfer_rate
            )

    def is_complete(self) -> bool:
        #TODO: harvest forever?
        return not self.keep_harvesting

    def act(self, dt:float) -> None:
        if self.ship.sector is None:
            #TODO: shouldn't we be able to target harvesting at a particular
            # sector and it'll go there?
            raise Exception("ship must be in a sector to harvest")

        if self.transfer_order is not None and self.transfer_order.is_complete():
            self.trips += 1
            self.transfer_order = None
            if self.max_trips > 0 and self.trips >= self.max_trips:
                self.keep_harvesting = False
                return

        # if our cargo is full, send it back to home base
        if self.ship.cargo_full():
            self.logger.debug("cargo full, heading to {self.base} to dump cargo")
            self.transfer_order = TransferCargo(self.base, self.resource, self.ship.cargo[self.resource], self.ship, self.gamestate)
            self.ship.orders.appendleft(self.transfer_order)
            return

        # choose an asteroid to harvest
        self.logger.debug(f'searching for next asteroid {self.ship.cargo_capacity - np.sum(self.ship.cargo)} space left')
        #TODO: how should we find the nearest asteroid? point_query_nearest with ShipFilter?
        nearest = None
        nearest_dist = np.inf
        for hit in self.ship.sector.spatial_point(self.ship.loc):
            if not isinstance(hit, core.Asteroid):
                continue
            if hit.resource != self.resource:
                continue
            if hit.amount <= 0:
                continue

            dist = np.linalg.norm(self.ship.loc - hit.loc)
            if dist < nearest_dist:
                nearest = hit
                nearest_dist = dist

        if nearest is None:
            self.logger.info(f'could not find asteroid of type {self.resource} in {self.ship.sector}, stopping harvest')
            self.keep_harvesting = False
            return

        #TODO: worry about other people harvesting asteroids
        #TODO: choose amount to harvest
        # push mining order
        self.mining_order = MineOrder(nearest, 1e3, self.ship, self.gamestate)
        self.ship.orders.appendleft(self.mining_order)

class DisembarkToEntity(core.Order):
    @staticmethod
    def disembark_to(embark_to:core.SectorEntity, ship:core.Ship, gamestate:core.Gamestate, disembark_dist:float=5e3, disembark_margin:float=5e2) -> DisembarkToEntity:
        if ship.sector is None:
            raise Exception("ship must be in a sector to disembark to")
        hits = ship.sector.spatial_point(ship.loc, max_dist=disembark_dist)
        nearest_dist = np.inf
        nearest = None
        for entity in hits:
            if entity == ship:
                continue
            if np.allclose(entity.velocity, ZERO_VECTOR):
                dist = np.linalg.norm(entity.loc - ship.loc)-entity.radius
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest = entity

        return DisembarkToEntity(nearest, embark_to, ship, gamestate, disembark_dist=disembark_dist, disembark_margin=disembark_margin)

    def __init__(self, disembark_from: Optional[core.SectorEntity], embark_to: core.SectorEntity, *args: Any, disembark_dist:float=5e3, disembark_margin:float=5e2, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.disembark_dist = disembark_dist
        self.disembark_margin = disembark_margin

        self.disembark_from = disembark_from
        self.embark_to = embark_to

        self.disembark_order:Optional[GoToLocation] = None
        self.embark_order:Optional[GoToLocation] = None

    def is_complete(self) -> bool:
        return self.embark_order is not None and self.embark_order.is_complete()

    def _begin(self) -> None:
        # should be upper bound of distance to the disembarkation point
        disembark_loc = self.ship.loc + util.polar_to_cartesian(self.disembark_dist, -self.ship.angle)
        self.init_eta = (
                GoToLocation.compute_eta(self.ship, disembark_loc)
                + GoToLocation.compute_eta(self.ship, self.embark_to.loc)
        )

    def act(self, dt:float) -> None:
        self.embark_order = GoToLocation.goto_entity(self.embark_to, self.ship, self.gamestate)
        self.add_child(self.embark_order)
        if self.disembark_from and np.linalg.norm(self.disembark_from.loc - self.ship.loc)-self.disembark_from.radius < self.disembark_dist:
            # choose a location which is outside disembark_dist
            _, angle = util.cartesian_to_polar(*(self.ship.loc - self.disembark_from.loc))
            target_angle = angle + self.gamestate.random.uniform(-np.pi/2, np.pi/2)
            target_disembark_distance = self.disembark_from.radius+self.disembark_dist+self.disembark_margin
            target_loc = self.disembark_from.loc + util.polar_to_cartesian(target_disembark_distance, target_angle)

            self.disembark_order = GoToLocation(
                    target_loc, self.ship, self.gamestate,
                    arrival_distance=self.disembark_margin,
                    min_distance=0.
            )
            self.add_child(self.disembark_order)
