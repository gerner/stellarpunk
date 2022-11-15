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

ANGLE_EPS = 5e-2 # about 3 degrees #2e-3 # about .06 degrees
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
        self.collision_margin = collision_margin
        self._neighborhood_radius = neighborhood_radius

        assert self.ship.sector is not None
        self.neighbor_analyzer = collision.Navigator(
                self.ship.sector.space, self.ship.phys,
                self.ship.radius, self.ship.max_thrust, self.ship.max_torque,
                self.ship.max_speed(),
                self.collision_margin,
                neighborhood_radius,
        )

        self.nearest_neighbor:Optional[core.SectorEntity] = None
        self.nearest_neighbor_dist = np.inf

        self.collision_threat:Optional[core.SectorEntity] = None
        self.collision_minimum_separation:Optional[float] = None
        self.collision_dv = ZERO_VECTOR
        self.collision_threat_time = 0.
        self.collision_approach_time = np.inf
        self.collision_cbdr = False
        self.collision_coalesced_threats = 0.
        self.collision_threat_loc = CYZERO_VECTOR
        self.collision_threat_radius = 0.

        self.cannot_avoid_collision = False

        # how many neighbors per m^2
        self.neighborhood_density = 0.
        self.num_neighbors = 0

    def to_history(self) -> dict:
        history = super().to_history()
        if self.collision_threat:
            history["ct"] = str(self.collision_threat.entity_id)
            history["ct_ms"] = self.collision_minimum_separation
            history["ct_loc"] = (self.collision_threat.loc[0], self.collision_threat.loc[1])
            history["ct_v"] = (self.collision_threat.velocity[0], self.collision_threat.velocity[1])
            history["ct_ts"] = self.collision_threat_time
            history["ct_at"] = self.collision_approach_time
            history["ct_dv"] = (self.collision_dv[0], self.collision_dv[1])
            history["ct_ct"] = self.collision_coalesced_threats
            history["ct_cloc"] = (self.collision_threat_loc[0], self.collision_threat_loc[1])
            history["ct_cradius"] = self.collision_threat_radius
            history["ct_cn"] = self.neighbor_analyzer.coalesced_neighbor_locations()
            history["ct_mh"] = self.neighbor_analyzer.get_collision_margin_histeresis()
            history["cac"] = bool(self.cannot_avoid_collision)
            history["cach"] = bool(self.neighbor_analyzer.get_cannot_avoid_collision_hold())
            history["cbdr"] = self.collision_cbdr

            # just get the oldest and newest cbdr history cause it's expensive
            # to iterating over it
            history["cbdr_hist"] = self.neighbor_analyzer.cbdr_history_summary()

        history["nd"] = self.neighborhood_density
        history["nnd"] = self.nearest_neighbor_dist

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

        self.cannot_avoid_collision = False
        self.nearest_neighbor_dist = np.inf

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

        if neighborhood_density > 0:
            self.gamestate.counters[core.Counters.COLLISION_NEIGHBOR_HAS_NEIGHBORS] += 1
            self.gamestate.counters[core.Counters.COLLISION_NEIGHBOR_NUM_NEIGHBORS] += num_neighbors

            self.gamestate.counters[core.Counters.COLLISION_THREATS_C] += coalesced_threats
            self.gamestate.counters[core.Counters.COLLISION_THREATS_NC] += non_coalesced_threats

            self.collision_threat_loc = threat_loc
            self.collision_threat_radius = threat_radius

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

            self.collision_threat_loc = CYZERO_VECTOR
            self.collision_threat_radius = 0

        if threat_count == 0:
            self.gamestate.counters[core.Counters.COLLISION_NEIGHBOR_NONE] += 1
            neighbor = None
        else:
            neighbor = threat.data

        # cache the collision threat
        if neighbor != self.collision_threat:
            self.collision_threat = neighbor
            self.collision_approach_time = approach_time
            self.collision_threat_time = self.gamestate.timestamp

        self.nearest_neighbor = nearest_neighbor
        self.nearest_neighbor_dist = nearest_neighbor_dist

        self.collision_minimum_separation = minimum_separation
        self.collision_coalesced_threats = coalesced_threats

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

        self.neighborhood_density = neighborhood_density
        self.num_neighbors = num_neighbors

        if neighbor is None:
            self.cannot_avoid_collision = False
            self.collision_threat = None
            self.collision_dv = ZERO_VECTOR
            self.collision_cbdr = False
            return ZERO_VECTOR, np.inf

        (delta_velocity, self.collision_cbdr, self.cannot_avoid_collision) = self.neighbor_analyzer.collision_dv(self.gamestate.timestamp, desired_direction)

        delta_velocity = np.array((delta_velocity[0], delta_velocity[1]))
        assert not util.either_nan_or_inf(delta_velocity)
        self.collision_dv = delta_velocity

        return delta_velocity, approach_time
