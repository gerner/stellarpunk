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

COLLISION_MARGIN_HISTERESIS_FACTOR = 0.65

# a convenient zero vector to avoid needless array creations
ZERO_VECTOR = np.array((0.,0.))
ZERO_VECTOR.flags.writeable = False
CYZERO_VECTOR = cymunk.Vec2d(0.,0.)

class AbstractSteeringOrder(core.Order):
    def __init__(self, *args: Any, safety_factor:float=2., **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.safety_factor = safety_factor

        assert self.ship.sector is not None
        self.neighbor_analyzer = collision.NeighborAnalyzer(self.ship.sector.space, self.ship.phys)

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
        self.collision_margin_histeresis = 0.

        self.cannot_avoid_collision = False
        # a flag we set to hold the cannot avoid collision state until some
        # point (see logic elsewhere)
        self.cannot_avoid_collision_hold = False

        # how many neighbors per m^2
        self.neighborhood_density = 0.

        self._next_accelerate_compute_ts = 0.
        self._accelerate_force:npt.NDArray[np.float64] = ZERO_VECTOR
        self._accelerate_torque:float = 0.
        self._accelerate_difference_angle:float = 0.
        self._accelerate_difference_mag:float = 0.

        # cache this so we can use it over and over
        self.worst_case_rot_time = math.sqrt(2. * math.pi / self.ship.max_angular_acceleration())
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
            history["ct_mh"] = self.collision_margin_histeresis
            history["cac"] = bool(self.cannot_avoid_collision)
            history["cach"] = bool(self.cannot_avoid_collision_hold)
            history["cbdr"] = self.collision_cbdr

            # just get the oldest and newest cbdr history cause it's expensive
            # to iterating over it
            history["cbdr_hist"] = self.neighbor_analyzer.cbdr_history_summary()

        history["nd"] = self.neighborhood_density
        history["nnd"] = self.nearest_neighbor_dist

        history["_nact"] = self._next_accelerate_compute_ts
        history["_ada"] = self._accelerate_difference_angle
        return history

    def _collision_neighbor(
            self,
            sector: core.Sector,
            neighborhood_dist: float,
            margin: float,
            max_distance: float
        ) -> tuple[
            Optional[core.SectorEntity],
            float,
            float,
            float,
            cymunk.Vec2d,
            cymunk.Vec2d,
            float,
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
            max_distance, self.ship.radius, margin, neighborhood_dist,
            self.ship.max_acceleration(),
        )

        if neighborhood_density > 0:
            self.gamestate.counters[core.Counters.COLLISION_NEIGHBOR_HAS_NEIGHBORS] += 1
            self.gamestate.counters[core.Counters.COLLISION_NEIGHBOR_NUM_NEIGHBORS] += num_neighbors

            self.gamestate.counters[core.Counters.COLLISION_THREATS_C] += coalesced_threats
            self.gamestate.counters[core.Counters.COLLISION_THREATS_NC] += non_coalesced_threats

            if not self.cannot_avoid_collision_hold and self.collision_threat_radius > 0:
                threat_loc, threat_radius = collision.migrate_threat_location(
                        self.ship.phys.position, self.ship.radius,
                        self.collision_threat_loc, self.collision_threat_radius,
                        threat_loc, threat_radius
                )

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

        return neighbor, approach_time, minimum_separation, threat_radius, threat_loc, threat_velocity, neighborhood_density, prior_threat_count > 0

    def _avoid_collisions_dv(self, sector: core.Sector, neighborhood_dist: float, margin: float, max_distance: float=np.inf, margin_histeresis:Optional[float]=None, desired_direction:Optional[np.ndarray]=None) -> tuple[np.ndarray, float, float, float]:
        """ Given current velocity, try to avoid collisions with neighbors

        sector
        neighborhood_loc: centerpoint for looking for threats
        neighborhood_dist: how far away to look for threats
        margin: how far apart (between envelopes) to target
        max_distance: max distance to care about collisions
        v: our velocity
        margin_histeresis: additional distance, on top of margin to target when avoiding collisions
        desired_direction: the desired velocity we're shooting for when not avoiding a collision

        returns tuple of desired delta v to avoid collision, approach time, minimum separation and the target distance
        """


        if margin_histeresis is None:
            # add additional margin of size this factor
            # the margin basically scales by 1 + this amount
            margin_histeresis = margin * 1

        # if we're taking emergency action to avoid a collision set a very
        # small margin
        if self.cannot_avoid_collision_hold:
            margin = self.ship.radius * self.safety_factor
            self.collision_margin_histeresis = 0.

        if desired_direction is None:
            v = self.ship.velocity
            desired_direction = v

        # if we already have a threat increase margin to get extra far from it
        neighbor_margin = margin + self.collision_margin_histeresis

        # find neighbor with soonest closest appraoch
        neighbor, approach_time, minimum_separation, threat_radius, threat_loc, threat_velocity, neighborhood_density, any_prior_threats = self._collision_neighbor(sector, neighborhood_dist, neighbor_margin, max_distance)

        if any_prior_threats:
            # if there's any overlap, keep the margin extra big
            #self.collision_margin_histeresis = margin_histeresis

            # expand margin up to margin histeresis
            # this is the iterative form of the inverse of expoential decay
            # which we use below to decay the margin histeresis when there's no
            # overlap
            if self.collision_margin_histeresis < margin_histeresis:
                y = self.collision_margin_histeresis
                b = margin_histeresis
                m = COLLISION_MARGIN_HISTERESIS_FACTOR
                y_next = (b - y)*(1-m)+y
                # numerical precision means we might exceed the upper bound
                if y_next > margin_histeresis:
                    y_next = margin_histeresis
                self.collision_margin_histeresis = y_next
            elif self.collision_margin_histeresis > margin_histeresis:
                self.collision_margin_histeresis *= COLLISION_MARGIN_HISTERESIS_FACTOR
                if self.collision_margin_histeresis < margin_histeresis:
                    self.collision_margin_histeresis = margin_histeresis

            assert self.collision_margin_histeresis >= 0

        else:
            # if there's no overlap, start collapsing collision margin
            self.collision_margin_histeresis *= COLLISION_MARGIN_HISTERESIS_FACTOR
            # if there's no overlap, clear the cannot avoid collision flag
            self.cannot_avoid_collision_hold = False

            assert self.collision_margin_histeresis >= 0
            #assert self.collision_margin_histeresis <= margin_histeresis

        self.neighborhood_density = neighborhood_density

        if neighbor is None:
            self.cannot_avoid_collision = False
            self.ship.collision_threat = None
            self.collision_dv = ZERO_VECTOR
            self.collision_cbdr = False
            return ZERO_VECTOR, np.inf, np.inf, 0

        base_bias = 2.
        if self.ship.entity_id < neighbor.entity_id and not util.both_almost_zero(neighbor.velocity):
            cbdr_bias = -base_bias
        else:
            cbdr_bias = base_bias


        self.collision_cbdr = self.neighbor_analyzer.detect_cbdr(self.gamestate.timestamp)

        desired_margin = neighbor_margin + threat_radius + self.ship.radius
        current_threat_loc = threat_loc-threat_velocity*approach_time
        current_threat_vec = current_threat_loc - self.ship.loc
        distance_to_threat = util.magnitude(current_threat_vec[0], current_threat_vec[1])

        if distance_to_threat <= desired_margin + VELOCITY_EPS:
            delta_velocity = (current_threat_loc - self.ship.loc) / distance_to_threat * self.ship.max_speed() * -1
            if minimum_separation < threat_radius:
                if distance_to_threat < threat_radius:
                    self.cannot_avoid_collision = True
                elif approach_time > 0.:
                    required_acceleration = 2*(threat_radius-minimum_separation)/(approach_time ** 2)
                    required_thrust = self.ship.mass * required_acceleration
                    self.cannot_avoid_collision = required_thrust > self.ship.max_thrust
        else:
            if minimum_separation < threat_radius:
                if approach_time > 0.:
                    # check if we can avoid collision
                    # s = u*t + 1/2 a * t^2
                    # s = threat_radius - minimum_separation
                    # ignore current velocity, we want to get displacement on
                    #   top of current situation
                    # t = approach_time
                    # a = 2 * s / t^2
                    required_acceleration = 2*(threat_radius-minimum_separation)/(approach_time ** 2)
                    required_thrust = self.ship.mass * required_acceleration
                    self.cannot_avoid_collision = required_thrust > self.ship.max_thrust
                else:
                    self.cannot_avoid_collision = True
            else:
                if approach_time > 0:
                    # check that the desired margin is feasible given our delta-v budget
                    # if not, set the margin to whatever our delta-v budget permits
                    required_acceleration = 2*(desired_margin-minimum_separation)/(approach_time ** 2)
                    required_thrust = self.ship.mass * required_acceleration
                    if required_thrust > self.ship.max_thrust:
                        # best case is we spend all our time accelerating in one direction
                        # worst case is we need to rotate 180 degrees
                        #worst_case_rot_time = math.sqrt(2. * math.pi / self.ship.max_angular_acceleration())
                        #desired_margin = (self.ship.max_acceleration() * (approach_time - worst_case_rot_time)** 2 + 2 * minimum_separation)/2
                        desired_margin = (self.ship.max_acceleration() * approach_time ** 2 + 2 * minimum_separation)/2

                self.cannot_avoid_collision = False

            desired_delta_velocity = desired_direction - self.ship.velocity
            ddv_mag = util.magnitude(desired_delta_velocity[0], desired_delta_velocity[1])
            max_dv_available = self.ship.max_thrust / self.ship.mass * approach_time
            if ddv_mag > max_dv_available:
                desired_delta_velocity = desired_delta_velocity / ddv_mag * max_dv_available

            delta_v_budget = self.ship.max_thrust / self.ship.mass * (approach_time - self.worst_case_rot_time)

            delta_velocity = collision.collision_dv(
                    current_threat_loc, threat_velocity,
                    self.ship.phys.position, self.ship.phys.velocity,
                    desired_margin, cymunk.Vec2d(*desired_delta_velocity),
                    self.collision_cbdr, cbdr_bias,
                    delta_v_budget,
            )

        # if we cannot currently avoid a collision, flip the flag, but don't
        # clear it just because we currently are ok, that happens elsewhere.
        self.cannot_avoid_collision_hold = self.cannot_avoid_collision_hold or self.cannot_avoid_collision

        delta_velocity = np.array((delta_velocity[0], delta_velocity[1]))
        assert not util.either_nan_or_inf(delta_velocity)
        self.collision_dv = delta_velocity

        return delta_velocity, approach_time, minimum_separation, self.ship.radius + neighbor.radius + margin - minimum_separation
