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

# the scale (per tick) we use to scale down threat radii if the new threat is
# still covered by the previous threat radius
THREAT_RADIUS_SCALE_FACTOR = 0.995
THREAT_LOCATION_ALPHA = 0.001
COLLISION_MARGIN_HISTERESIS_FACTOR = 0.65

# a convenient zero vector to avoid needless array creations
ZERO_VECTOR = np.array((0.,0.))
ZERO_VECTOR.flags.writeable = False

@jit(cache=True, nopython=True, fastmath=True)
def _collision_dv(entity_pos:npt.NDArray[np.float64], entity_vel:npt.NDArray[np.float64], pos:npt.NDArray[np.float64], vel:npt.NDArray[np.float64], margin:float, v_d:npt.NDArray[np.float64], cbdr:bool, cbdr_bias:float, delta_v_budget:float) -> npt.NDArray[np.float64]:
    """ Computes a divert vector (as in accelerate_to(v + dv)) to avoid a
    collision by at least distance m. This divert will be of minimum size
    relative to the desired velocity.

    entity_pos: location of the threat
    entity_pos: velocity of the threat
    pos: our position
    v: our velocity
    v_d: the desired delta velocity
    """

    # rel pos
    r = entity_pos - pos
    # rel vel
    v = entity_vel - vel
    # margin, including radii
    m = margin

    # desired diversion from v
    a = -v_d

    # check if the desired divert is already viable
    x = a[0]
    y = a[1]

    if util.isclose(v[0], 0.) and util.isclose(v[1], 0.):
        do_nothing_margin_sq = r[0]**2+r[1]**2
    else:
        do_nothing_margin_sq = r[0]**2+r[1]**2 - (r[0]*x+r[1]*y+(2*r[0]*v[0]+2*r[1]*v[1]))**2/((2*v[0]+x)**2+(2*v[1]+y)**2)
    if do_nothing_margin_sq > 0 and do_nothing_margin_sq >= m**2:
        #if util.magnitude(a[0], a[1]) <= delta_v_budget:
        return v_d

    if util.magnitude(r[0], r[1]) <= margin:
        raise ValueError("already inside margin")

    # given divert (x,y):
    # (r[0]**2+r[1]**2)-(2*(r[0]*v[0]+r[1]*v[1])+(r[0]*x+r[1]*y))**2/((2*v[0]+x)**2 + (2*v[1]+y)**2) > m**2
    # this forms a pair of intersecting lines with viable diverts between them

    # given divert (x,y):
    # cost_from desired = (a[0]-x)**2 +(a[1]-y)**2
    # see https://www.desmos.com/calculator/qvk8fpbw3k

    # to understand the margin, we end up with two circles whose intersection
    # points are points on the tangent lines that form the boundary of our
    # viable diverts
    # (x+2*v[0])**2 + (y+2*v[1])**2 = r[0]**2+r[1]**2-m**2
    # (x+2*v[0]-r[0])**2 + (y+2*v[1]-r[1])**2 = m**2
    # a couple of simlifying substitutions:
    # we can translate the whole system to the origin:
    # let s_x,s_y = (x + 2*v[0], y + 2*v[1])
    # let p = r[0]**2 + r[1]**2 - m**2
    # having done this we can subtract the one from the other, solve for y,
    # plug back into one of the equations
    # solve the resulting quadratic eq for x (two roots)
    # plug back into one of the equations to get y (two sets of two roots)
    # for the y roots, only one will satisfy the other equation, pick that one
    # also, we'll divide by r[1] below. So if that's zero we have an alternate
    # form where there's a single value for x

    p = r[0]**2 + r[1]**2 - m**2

    if util.isclose_flex(r[1]**2, 0, atol=1e-5):
        # this case would cause a divide by zero when computing the
        # coefficients of the quadratic equation below
        s_1x = s_2x = p/r[0]
    else:
        # note that r[0] and r[1] cannot both be zero (assuming m>0)
        q_a = r[0]**2/r[1]**2+1
        q_b = -2*p*r[0]/r[1]**2
        q_c = p**2/r[1]**2 - p
        # quadratic formula
        # note that we get real roots as long as the problem is feasible (i.e.
        # we're not already inside the margin
        s_1x = (-q_b-np.sqrt(q_b**2-4*q_a*q_c)) / (2*q_a)
        s_2x = (-q_b+np.sqrt(q_b**2-4*q_a*q_c)) / (2*q_a)

    # y roots are y_i and -y_i, but only one each for i=0,1 will be on the curve

    # if p and s_2x**2 are close, this can appear to go negative
    if p - s_1x**2 < 0.:
        s_1y = 0.
    else:
        s_1y = np.sqrt(p-s_1x**2)
    if not util.isclose((s_1x - r[0])**2 + (s_1y - r[1])**2, m**2):
        s_1y = -s_1y

    # if p and s_2x**2 are close, this can appear to go negative
    if p-s_2x**2 < 0.:
        s_2y = 0.
    else:
        s_2y = np.sqrt(p-s_2x**2)
    if not util.isclose((s_2x - r[0])**2 + (s_2y - r[1])**2, m**2):
        s_2y = -s_2y

    # subbing back in our x_hat,y_hat above,
    # these determine the slope of the boundry lines of our viable region
    # (1) y+2*v[1] = s_iy/s_ix * (x+2*v[0]) for i = 0,1 (careful if x_i = 0)
    # with perpendiculars going through the desired_divert point
    # (2) y-a[1] = -s_ix/s_iy * (x-a[0]) (careful if y_i == 0)
    # so find the intersection of each of these pairs of equations

    if util.isclose(s_1x, 0):
        # tangent line is vertical
        # implies perpendicular is horizontal
        y1 = a[1]
        x1 = -2*v[0]
    elif util.isclose(s_1y, 0):
        # tangent line is horizontal
        # implies perpendicular is vertical
        x1 = a[0]
        y1 = -2*v[1]
    else:
        # solve (1) for y in terms of x and plug into (2), solve for x
        x1 = (s_1x/s_1y*a[0]+a[1] - s_1y/s_1x*2*v[0] + 2*v[1]) / (s_1y/s_1x + s_1x/s_1y)
        # plug back into (1)
        y1 = s_1y/s_1x * (x1+2*v[0]) - 2*v[1]

    if util.isclose(s_2x, 0):
        y2 = a[1]
        x2 = -2*v[0]
    elif util.isclose(s_2y, 0):
        x2 = a[0]
        y2 = -2*v[1]
    else:
        x2 = (s_2x/s_2y*a[0]+a[1] - s_2y/s_2x*2*v[0] + 2*v[1]) / (s_2y/s_2x + s_2x/s_2y)
        y2 = s_2y/s_2x * (x2+2*v[0]) - 2*v[1]

    cost1 = (a[0]-x1)**2 +(a[1]-y1)**2
    cost2 = (a[0]-x2)**2 +(a[1]-y2)**2

    assert not (math.isnan(cost1) or math.isinf(cost1))
    assert not (math.isnan(cost2) or math.isinf(cost2))

    s_x = 0.
    s_y = 0.
    cost = 0.
    if not cost2 < cost1:
        x = x1
        y = y1
        s_x = s_1x
        s_y = s_1y
        cost = cost1
    elif not cost1 < cost2:
        x = x2
        y = y2
        s_x = s_2x
        s_y = s_2y
        cost = cost2
    else:
        # not exactly sure why either would be nan, but hopefully one is not
        assert not math.isnan(cost1) or not math.isnan(cost2)

    if cbdr:
        # prefer diversion in the same direction in case of cbdr
        # the sign of cross1 and cross2 indicate the direction of the divert
        # (clockwise or counter-clockwise)
        cross1 = v[0]*y1-v[1]*x1
        cross2 = v[0]*y2-v[1]*x2
        if cross1 > cross2:
            x = x1
            y = y1
            s_x = s_1x
            s_y = s_1y
            cost = cost1
        else:
            x = x2
            y = y2
            s_x = s_2x
            s_y = s_2y
            cost = cost2

    #TODO: this assumes the other guy will move which seems very risky
    # we should come back to this in the future and do something more
    # proactive, but still cooperative
    #if cbdr and cbdr_bias < 0:
    #    return np.array((0.,0.))

    #if cbdr:
    #    # swap our choices if our bias is negative
    #    if cbdr_bias < 0:
    #        if not cost2 < cost1:
    #            x = x2
    #            y = y2
    #            s_x = s_2x
    #            s_y = s_2y
    #            cost = cost2
    #        elif not cost1 < cost2:
    #            x = x1
    #            y = y1
    #            s_x = s_1x
    #            s_y = s_1y
    #            cost = cost1

    # useful assert when testing
    # this asserts that the resulting x,y point matches the the contraint on
    # the margin
    assert util.isclose_flex(
            (r[0]**2+r[1]**2)-(2*(r[0]*v[0]+r[1]*v[1])+(r[0]*x+r[1]*y))**2/((2*v[0]+x)**2 + (2*v[1]+y)**2),
            m**2,
            rtol=1e-3)
    return np.array((-x, -y))

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
        self.collision_threat_loc = ZERO_VECTOR
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
            np.ndarray,
            np.ndarray,
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
            threat_loc = np.array(threat_loc)
            threat_velocity = np.array(threat_velocity)

            self.gamestate.counters[core.Counters.COLLISION_THREATS_C] += coalesced_threats
            self.gamestate.counters[core.Counters.COLLISION_THREATS_NC] += non_coalesced_threats

            if not self.cannot_avoid_collision_hold:
                # we want to avoid nearby, dicontinuous changes to threat loc and
                # radius. this can happen when two threats are near each other.
                # if the new and old threat circles overlap "significantly", we are
                # careful about discontinuous changes
                new_old_dist = util.distance(self.collision_threat_loc, threat_loc)
                if new_old_dist < 2*threat_radius or new_old_dist < 2*self.collision_threat_radius:
                    old_loc = self.collision_threat_loc
                    old_radius = self.collision_threat_radius
                    if new_old_dist + threat_radius > old_radius + VELOCITY_EPS:
                        # the new circle does not completely eclipse the old
                        # find the smallest enclosing circle for both
                        old_loc, old_radius = util.enclosing_circle(threat_loc, threat_radius, self.collision_threat_loc, self.collision_threat_radius)
                        new_old_dist = util.distance(old_loc, threat_loc)

                    new_old_vec = threat_loc - old_loc

                    # the new threat is smaller than the old one and completely
                    # contained in the old one. let's scale and translate the old
                    # one toward the new one so that it still contains it, but
                    # asymptotically approaches it. this will avoid
                    # discontinuities.
                    new_radius = util.clip(
                        old_radius * THREAT_RADIUS_SCALE_FACTOR,
                        threat_radius,
                        old_radius
                    )
                    if util.isclose(new_old_dist, 0.):
                        new_loc = old_loc
                    else:
                        new_loc = (
                            new_old_vec/new_old_dist
                            * (old_radius-new_radius)
                            + old_loc
                        )
                    # useful assert during testing
                    #assert np.linalg.norm(threat_loc - new_loc) + threat_radius <= new_radius + VELOCITY_EPS

                    # if we're not already inside the margin for this new location
                    # take it, otherwise stick with the one computed for the actual
                    # threats we have, discontinuities be damned
                    if util.distance(self.ship.loc, new_loc) > new_radius + self.ship.radius:
                        threat_loc = new_loc
                        threat_radius = new_radius
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
            threat_loc = ZERO_VECTOR
            threat_velocity = ZERO_VECTOR
            threat_radius = 0.
            neighborhood_density = 0.

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
        self.collision_threat_loc = threat_loc
        self.collision_threat_radius = threat_radius

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


        if self.collision_coalesced_threats == 1 and not util.both_almost_zero(threat_velocity) and self.neighbor_analyzer.detect_cbdr(self.gamestate.timestamp):
            self.collision_cbdr = True
        else:
            self.collision_cbdr = False

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

            delta_velocity = _collision_dv(
                    current_threat_loc, threat_velocity,
                    self.ship.loc, self.ship.velocity,
                    desired_margin, desired_delta_velocity,
                    self.collision_cbdr, cbdr_bias,
                    delta_v_budget,
            )

        # if we cannot currently avoid a collision, flip the flag, but don't
        # clear it just because we currently are ok, that happens elsewhere.
        self.cannot_avoid_collision_hold = self.cannot_avoid_collision_hold or self.cannot_avoid_collision

        assert not util.either_nan_or_inf(delta_velocity)
        self.collision_dv = delta_velocity

        return delta_velocity, approach_time, minimum_separation, self.ship.radius + neighbor.radius + margin - minimum_separation
