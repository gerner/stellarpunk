""" Steering behaviors that can be used in orders. """ 

from __future__ import annotations

import collections
import math
from typing import Optional, Deque, Any, Tuple
import logging

import numpy as np
import numpy.typing as npt
import numba as nb # type: ignore
from numba import jit # type: ignore
import cymunk # type: ignore

from stellarpunk import util, core
from stellarpunk.orders import collision

ANGLE_EPS = 8e-2 # about 3 degrees #2e-3 # about .06 degrees
PARALLEL_EPS = 0.5e-1
VELOCITY_EPS = 5e-1
COARSE_VELOCITY_MATCH = 2e0

# think of this as the gimballing angle (?)
# pi/16 is ~11 degrees
COARSE_ANGLE_MATCH = np.pi/16

COLLISION_MARGIN_HISTERESIS_FACTOR = 0.55#0.65

# a convenient zero vector to avoid needless array creations
ZERO_VECTOR = np.array((0.,0.))
ZERO_VECTOR.flags.writeable = False
CYZERO_VECTOR = cymunk.Vec2d(0.,0.)

class AbstractSteeringOrder(core.Order):
    def __init__(self, *args: Any,
            safety_factor:float=2.,
            collision_margin:float=2e2,
            neighborhood_radius: float = 8.5e3,
            **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.safety_factor = safety_factor

        assert self.ship.sector is not None
        self.neighbor_analyzer = collision.Navigator(
                self.ship.sector.space, self.ship.phys,
                self.ship.radius, self.ship.max_thrust, self.ship.max_torque,
                self.ship.max_speed(),
                collision_margin,
                neighborhood_radius,
        )

    def _cancel(self) -> None:
        self.logger.debug(f'cancel at {self.gamestate.timestamp}')
        self.ship.apply_force(ZERO_VECTOR, False)
        self.ship.apply_torque(0., False)

    def _complete(self) -> None:
        self.logger.debug(f'complete at {self.gamestate.timestamp}')
        self.ship.apply_force(ZERO_VECTOR, False)
        self.ship.apply_torque(0., False)

    @property
    def collision_cbdr(self) -> bool: return self.neighbor_analyzer.get_collision_cbdr()
    @property
    def cannot_avoid_collision(self) -> bool: return self.neighbor_analyzer.get_cannot_avoid_collision()
    @property
    def cannot_avoid_collision_hold(self) -> bool: return self.neighbor_analyzer.get_cannot_avoid_collision_hold()
    @property
    def threat_count(self) -> int: return self.neighbor_analyzer.get_threat_count()
    @property
    def nearest_neighbor_dist(self) -> float: return self.neighbor_analyzer.get_nearest_neighbor_dist()
    @property
    def num_neighbors(self) -> int: return self.neighbor_analyzer.get_num_neighbors()
    @property
    def collision_margin(self) -> float: return self.neighbor_analyzer.get_collision_margin()
    @property
    def computed_neighborhood_radius(self) -> float: return self.neighbor_analyzer.get_neighborhood_radius()

    def to_history(self) -> dict:
        history = super().to_history()
        history.update(self.neighbor_analyzer.get_telemetry())
        if "ct" in history:
            history["ct"] = str(history["ct"].data.entity_id)
            history["ct_dv"] = list(self.collision_dv)
        return history

    def _collision_neighbor(
            self,
            sector: core.Sector,
            max_distance: float
        ) -> tuple[
            Optional[core.SectorEntity],
            float,
            float,
            float,
            cymunk.Vec2d,
            cymunk.Vec2d,
            float,
            int,
            bool]:

        (
                threat,
                approach_time,
                _,
                _,
                minimum_separation,
                threat_count,
                coalesced_threats,
                non_coalesced_threats,
                threat_radius,
                threat_loc,
                threat_velocity,
                nearest_neighbor,
                nearest_neighbor_dist,
                neighborhood_density,
                num_neighbors,
                prior_threat_count,
        ) = self.neighbor_analyzer.analyze_neighbors(
            self.gamestate.timestamp,
            max_distance,
        )

        if num_neighbors > 0:
            self.gamestate.counters[core.Counters.COLLISION_NEIGHBOR_HAS_NEIGHBORS] += 1
            self.gamestate.counters[core.Counters.COLLISION_NEIGHBOR_NUM_NEIGHBORS] += num_neighbors

            self.gamestate.counters[core.Counters.COLLISION_THREATS_C] += coalesced_threats
            self.gamestate.counters[core.Counters.COLLISION_THREATS_NC] += non_coalesced_threats

        else:
            self.gamestate.counters[core.Counters.COLLISION_NEIGHBOR_NO_NEIGHBORS] += 1
            approach_time = np.inf
            minimum_separation = np.inf
            collision_loc = ZERO_VECTOR
            threat_count = 0
            coalesced_threats = 0
            non_coalesced_threats = 0
            nearest_neighbor = None
            nearest_neighbor_dist = np.inf
            threat_loc = CYZERO_VECTOR
            threat_velocity = CYZERO_VECTOR
            threat_radius = 0.
            neighborhood_density = 0.
            num_neighbors = 0

        if threat_count == 0:
            self.gamestate.counters[core.Counters.COLLISION_NEIGHBOR_NONE] += 1
            neighbor = None
        else:
            neighbor = threat.data

        return neighbor, approach_time, minimum_separation, threat_radius, threat_loc, threat_velocity, neighborhood_density, num_neighbors, prior_threat_count > 0

    def _avoid_collisions_dv(self, sector: core.Sector, max_distance: float=np.inf, desired_direction:Optional[cymunk.Vec2d]=None) -> tuple[np.ndarray, float]:
        """ Given current velocity, try to avoid collisions with neighbors

        sector
        neighborhood_loc: centerpoint for looking for threats
        margin: how far apart (between envelopes) to target
        max_distance: max distance to care about collisions
        v: our velocity
        desired_direction: the desired velocity we're shooting for when not avoiding a collision

        returns tuple of desired delta v to avoid collision, approach time, minimum separation and the target distance
        """


        if desired_direction is None:
            desired_direction = self.ship.phys.velocity

        # find neighbor with soonest closest appraoch
        neighbor, approach_time, minimum_separation, threat_radius, threat_loc, threat_velocity, neighborhood_density, num_neighbors, any_prior_threats = self._collision_neighbor(sector, max_distance)

        if neighbor is None:
            self.collision_dv = ZERO_VECTOR
            return ZERO_VECTOR, np.inf

        (delta_velocity, _, _) = self.neighbor_analyzer.collision_dv(self.gamestate.timestamp, desired_direction)

        delta_velocity = np.array((delta_velocity[0], delta_velocity[1]))
        assert not util.either_nan_or_inf(delta_velocity)
        self.collision_dv = delta_velocity

        return delta_velocity, approach_time
