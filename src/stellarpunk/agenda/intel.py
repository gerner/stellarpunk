import enum
import uuid
import abc
from typing import Any, Optional, Type

import numpy as np
import numpy.typing as npt

from stellarpunk import core, util, intel
from stellarpunk.core import sector_entity
from stellarpunk.orders import movement, core as ocore

from .core import Agendum

class IntelCollectionAgendum(core.IntelManagerObserver, Agendum):
    """ Behavior for any character to obtain desired intel.

    This agendum watches for registered intel interests and operates either
    passively, without getting in the way of other agenda, or actively, taking
    full control of the character, to collect that intel.

    It has specialized behavior for each kind of intel to collect."""

    class State(enum.IntEnum):
        IDLE = enum.auto()
        ACTIVE = enum.auto()
        PASSIVE = enum.auto()

    def __init__(self, collection_director:"IntelCollectionDirector", *args:Any, idle_period:float=120.0, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self._director = collection_director
        self._idle_period = idle_period
        self._state = IntelCollectionAgendum.State.PASSIVE
        self._interests:set[core.IntelMatchCriteria] = set()
        self._source_interests_by_dependency:dict[core.IntelMatchCriteria, core.IntelMatchCriteria] = {}
        self._source_interests_by_source:dict[core.IntelMatchCriteria, core.IntelMatchCriteria] = {}

        self._preempted_primary:Optional[Agendum] = None

    # core.IntelManagerObserver

    def intel_desired(self, intel_manager:core.AbstractIntelManager, intel_criteria:core.IntelMatchCriteria, source:Optional[core.IntelMatchCriteria]) -> None:
        if source is not None:
            # remove the source interest if we have it. we don't want to try to get
            # that before we get the dependent intel. but we'll keep track to make
            # sure we eventually get it or try fresh if we satisfy this dependency
            # without satisfying the source
            if source in self._interests:
                self._interests.remove(source)
            self._source_interests_by_dependency[intel_criteria] = source
            self._source_interests_by_source[source] = intel_criteria

        # make note that we want to find such intel
        assert(intel_manager == self.character.intel_manager)
        self._interests.add(intel_criteria)

        # if we're already passively or actively collecting intel, no sense
        # interrupting that, so wait for that to finish and we'll get a act
        # call when that should be complete. at that point we'll consider new
        # intel needs.
        if self._state == IntelCollectionAgendum.State.IDLE:
            # note: we might ask to be scheduled many times here if someone
            # registers several interests, but the schedule will dedupe
            self.gamestate.schedule_agendum_immediate(self, jitter=1.0)

    def _check_dependency_removal(self, dependency:core.IntelMatchCriteria, intel:core.Intel) -> None:
        # if this intel was a dependency for some other source interest
        # we need to pull that source back into our regular set of
        # interests so we can try and collect it again
        if dependency in self._source_interests_by_dependency:
            source = self._source_interests_by_dependency[dependency]
            if not source.matches(intel):
                # in this case we can keep any further dependnecy chain intact
                self._interests.add(source)
            else:
                # if this itself is a dependency, we need to recursively handle
                # the dependency chain
                self._check_dependency_removal(source, intel)
            del self._source_interests_by_source[source]
            del self._source_interests_by_dependency[dependency]


    def intel_added(self, intel_manager:core.AbstractIntelManager, intel:core.Intel) -> None:
        # first see if this satisfies some source criteria
        remove_sources:set[core.IntelMatchCriteria] = set()
        for source, dependency in list(self._source_interests_by_source.items()):
            if source.matches(intel):
                # we can stop tracking this
                del self._source_interests_by_source[source]
                del self._source_interests_by_dependency[dependency]
                self._check_dependency_removal(source, intel)

        # see if that intel satisfies any of our needs and drop that need.
        # if it doesn't actually satisfy the root need, they can re-register
        for criteria in self._interests.copy():
            if criteria.matches(intel):
                self._interests.remove(criteria)
                self._check_dependency_removal(criteria, intel)


    # Agendum

    def _unpause(self) -> None:
        self._go_idle()

    def _start(self) -> None:
        self.character.intel_manager.observe(self)
        self._go_idle()

    def _stop(self) -> None:
        self.character.intel_manager.unobserve(self)

    def _do_passive_collection(self) -> None:
        # opportunistically try to collect desired intel
        cheapest_cost = np.inf
        cheapest_criteria:Optional[core.IntelMatchCriteria] = None
        for criteria in self._interests:
            ret = self._director.estimate_cost(self.character, criteria)
            if not ret:
                continue
            is_active, cost = ret
            if is_active:
                continue
            if cost < cheapest_cost:
                cheapest_cost = cost
                cheapest_criteria = criteria

        # if there's no intel we can passively collect, bail
        if not cheapest_criteria:
            self._go_idle()
            return

        # preempt current primary so we're not fighting over behavior
        # ship orders will be preempted by prepending orders
        # we'll restore the preempted primary the next time we act
        if self._preempted_primary is None:
            #TODO: we need to be very careful here and not preempt if they are
            #in the middle of some critical operation (e.g. docked at a station
            #and not on their ship, in the middle of warping out of sector
            # which take some time, etc.)
            # we need some way to "lock" the character, agenda and/or ship
            # in that case we can just bail and check in again later

            current_primary = self.find_primary()
            # there must be a current primary, otherwise we'd be in ACTIVE mode
            assert(current_primary is not None)
            self._preempted_primary = current_primary # type: ignore
            current_primary.preempt_primary()
            current_primary.pause()
            self.make_primary()

        next_ts = self._director.collect_intel(self.character, cheapest_criteria)
        if next_ts > 0:
            self.gamestate.schedule_agendum(next_ts, self, jitter=1.0)
        else:
            self.gamestate.schedule_agendum_immediate(self)

    def _do_active_collection(self) -> None:
        # make big plans for intel collection, travelling, etc.
        cheapest_cost = np.inf
        cheapest_criteria:Optional[core.IntelMatchCriteria] = None
        for criteria in self._interests:
            ret = self._director.estimate_cost(self.character, criteria)
            if not ret:
                continue
            is_active, cost = ret
            if cost < cheapest_cost:
                cheapest_cost = cost
                cheapest_criteria = criteria

        if cheapest_criteria is None:
            # this means we have intel interests we cannot collect and no one
            # else is directing primary character behavior. Seems bad.
            # perhaps at a later point we will be able to
            self.logger.info(f'{self.character} has intel interests that we cannot actively satisfy')
            self._go_idle()
            return

        next_ts = self._director.collect_intel(self.character, cheapest_criteria)
        if next_ts > 0:
            self.gamestate.schedule_agendum(next_ts, self, jitter=1.0)
        else:
            self.gamestate.schedule_agendum_immediate(self)

    def _restore_preempted(self) -> None:
        assert(self._preempted_primary)
        assert(self.is_primary())
        assert(self._state == IntelCollectionAgendum.State.PASSIVE)
        self.preempt_primary()
        self._preempted_primary.make_primary()
        self._preempted_primary.unpause()
        self._preempted_primary = None

    def _go_idle(self) -> None:
        if self._preempted_primary:
            self._restore_preempted()
        self._state = IntelCollectionAgendum.State.IDLE

        # we'll wake ourselves up if someone registers an interest, no need to
        # force a wakeup that will do nothing
        if len(self._interests) > 0:
            self.gamestate.schedule_agendum(self.gamestate.timestamp + self._idle_period, self, jitter=1.0)


    def act(self) -> None:
        # no sense working if we have no intel to collect
        if len(self._interests) == 0:
            self._go_idle()
            return

        if self._preempted_primary is None:
            # we're not in the middle of a passive collection cycle
            # figure out what state we should be in based on other agenda
            assert(self._state != IntelCollectionAgendum.State.PASSIVE)
            if self._is_primary:
                assert(self._state == IntelCollectionAgendum.State.ACTIVE)
            else:
                assert(self._state == IntelCollectionAgendum.State.IDLE)
                primary_agendum = self.find_primary()
                if primary_agendum is None:
                    self.make_primary()
                    self._state = IntelCollectionAgendum.State.ACTIVE
                else:
                    # if we're primary our _is_primary flag should be true
                    assert(primary_agendum != self)
                    self._state = IntelCollectionAgendum.State.PASSIVE
        else:
            # we must be in the middle of a passive collection cycle.
            #TODO: should we limit how much passive collection we do? as
            # written we'll just keep collecting all passive intel until there
            # is none.
            assert(self._state == IntelCollectionAgendum.State.PASSIVE)

        if self._state == IntelCollectionAgendum.State.PASSIVE:
            self._do_passive_collection()
        elif self._state == IntelCollectionAgendum.State.ACTIVE:
            self._do_active_collection()
        else:
            raise ValueError(f'intel agendum in unexpected state: {self._state}')

class IntelCollectionDirector:
    def __init__(self) -> None:
        self._gatherers:list[tuple[Type[core.IntelMatchCriteria], IntelGatherer]] = []

    def _find_gatherer(self, klass:Type[core.IntelMatchCriteria]) -> Optional["IntelGatherer"]:
        for criteria_klass, gatherer in self._gatherers:
            if issubclass(klass, criteria_klass):
                return gatherer
        return None

    def register_gatherer(self, klass:Type[core.IntelMatchCriteria], gatherer:"IntelGatherer") -> None:
        # note, gatherers should be registered from most specific to least
        # specific so we can find the most specific gatherer first
        self._gatherers.append((klass, gatherer))

    def estimate_cost(self, character:core.Character, intel_criteria:core.IntelMatchCriteria) -> Optional[tuple[bool, float]]:
        gatherer = self._find_gatherer(type(intel_criteria))
        if gatherer is None:
            return None
        return gatherer.estimate_cost(character, intel_criteria)

    def collect_intel(self, character:core.Character, intel_criteria:core.IntelMatchCriteria) -> float:
        gatherer = self._find_gatherer(type(intel_criteria))
        assert(gatherer is not None)
        return gatherer.collect_intel(character, intel_criteria)

class IntelGatherer[T: core.IntelMatchCriteria](abc.ABC):
    def __init__(self) -> None:
        self.gamestate:core.Gamestate = None # type: ignore

    def initialize_gamestate(self, gamestate:core.Gamestate) -> None:
        self.gamestate = gamestate

    @abc.abstractmethod
    def estimate_cost(self, character:core.Character, intel_criteria:T) -> Optional[tuple[bool, float]]:
        """ Returns an estimate to collect associated intel, if any.

        returns if we can handle this criteria (Optional), if it's active or
        passive (bool) and an estimate of the cost to retrieve it in seconds
        """
        ...

    @abc.abstractmethod
    def collect_intel(self, character:core.Character, intel_criteria:T) -> float:
        """ Begins or continues collecting associated intel

        returns the timestamp we should check in again. """
        ...

class SectorHexIntelGatherer(IntelGatherer[intel.SectorHexPartialCriteria]):
    def _candidate_in_sector(self, character:core.Character, intel_criteria:intel.SectorHexPartialCriteria, sector_id:uuid.UUID) -> Optional[npt.NDArray[np.float64]]:
        if character.location is not None and character.location.sector is not None and character.location.sector.entity_id == sector_id:
            sector = character.location.sector
            loc = character.location.loc
        else:
            sector = self.gamestate.get_entity(sector_id, core.Sector)
            loc = np.array((0.0, 0.0))

        hex_loc = sector.get_hex_coords(loc)
        # look for hex options in current sector

        # start looking close-ish to where we are, within a sector radius
        target_loc = hex_loc
        target_dist = sector.radius / (np.sqrt(3)*sector.hex_size)

        # honor intel criteria's desire of course
        if intel_criteria.hex_loc is not None:
            target_loc = intel_criteria.hex_loc
        if intel_criteria.hex_dist is not None:
            target_dist = intel_criteria.hex_dist

        candidate_hexes:set[tuple[int, int]] = {(int(x[0]), int(x[1])) for x in util.hexes_within_pixel_dist(target_loc, target_dist, sector.hex_size)}

        # find hexes in the current sector we know about
        for i in character.intel_manager.intel(intel.SectorHexPartialCriteria(sector_id=sector_id, is_static=intel_criteria.is_static, hex_loc=target_loc, hex_dist=target_dist), intel.SectorHexIntel):
            hex_key = (int(i.hex_loc[0]), int(i.hex_loc[1]))
            if hex_key in candidate_hexes:
                candidate_hexes.remove(hex_key)

        # pick closest remaining candidate
        candidate = next(iter(sorted(candidate_hexes, key=lambda x: util.axial_distance(x, hex_loc))), None)
        if candidate is None:
            return None
        else:
            return np.array((float(candidate[0]), float(candidate[1])))

    def estimate_cost(self, character:core.Character, intel_criteria:intel.SectorHexPartialCriteria) -> Optional[tuple[bool, float]]:
        # passive => target hex is adjacent to the one we're in right now
        # cost = time to travel to center of target hex
        # target hex is closest one where a scan will produce new intel that
        # will match this partial criteria

        # we can't estimate cost if we don't know where the character is
        if character.location is None:
            return None
        if character.location.sector is None:
            return None

        sector = character.location.sector
        sector_id = sector.entity_id

        if intel_criteria.sector_id is None or intel_criteria.sector_id == sector_id:
            candidate = self._candidate_in_sector(character, intel_criteria, sector_id)
            if candidate is not None:
                candidate_coords = sector.get_coords_from_hex(candidate)
                loc = character.location.loc
                hex_loc = sector.get_hex_coords(loc)
                hex_dist = util.axial_distance(candidate, hex_loc)

                #TODO: what if we're not a captain? can we take action to travel to some location?
                # this behavior assumes we're a captain of a ship
                assert(isinstance(character.location, core.Ship))
                assert(character.location.captain == character)
                eta = movement.GoToLocation.compute_eta(character.location, candidate_coords)
                return (hex_dist <= 1, eta)

        # we've already tried to find a hex in the current sector, only
        # remaining candidates would be outside the current sector
        if intel_criteria.sector_id is not None and intel_criteria.sector_id == sector_id:
            return None

        #TODO: find a candidate in another sector
        return None

    def collect_intel(self, character:core.Character, intel_criteria:intel.SectorHexPartialCriteria) -> float:
        # we can't estimate cost if we don't know where the character is
        if character.location is None:
            raise ValueError(f'cannot collect intel for {character} not on any ship')
        if character.location.sector is None:
            raise ValueError(f'cannot collect intel for {character} not in any sector')

        sector = character.location.sector
        sector_id = sector.entity_id

        if intel_criteria.sector_id is None or intel_criteria.sector_id == sector_id:
            candidate = self._candidate_in_sector(character, intel_criteria, sector_id)
            if candidate is not None:
                #TODO: what if we're not a captain?
                # collect intel at this candidate
                assert(isinstance(character.location, core.Ship))
                assert(character.location.captain == character)
                candidate_coords = sector.get_coords_from_hex(candidate)
                explore_order = ocore.LocationExploreOrder(sector_id, candidate_coords, self.gamestate)
                character.location.prepend_order(explore_order)
                return self.gamestate.timestamp + explore_order.estimate_eta() * 1.2

        #TODO: find a candidate in another sector

        # if we have no candidates we should not have gotten called because we
        # would not have returned anything from estimate_cost.
        raise ValueError(f'no candidates to collect intel on')

class SectorEntityIntelGatherer(IntelGatherer[intel.SectorEntityPartialCriteria]):
    def estimate_cost(self, character:core.Character, intel_criteria:intel.SectorEntityPartialCriteria) -> Optional[tuple[bool, float]]:
        # we'll just ask for sector hex intel, that's free
        return (True, 0.0)

    def collect_intel(self, character:core.Character, intel_criteria:intel.SectorEntityPartialCriteria) -> float:
        sector_hex_criteria = intel.SectorHexPartialCriteria(sector_id=intel_criteria.sector_id, is_static=intel_criteria.is_static)
        character.intel_manager.register_intel_interest(sector_hex_criteria, source=intel_criteria)
        return 0.0

class EconAgentSectorEntityIntelGatherer(IntelGatherer[intel.EconAgentSectorEntityPartialCriteria]):
    def _find_candidate(self, character:core.Character, station_criteria:intel.StationIntelPartialCriteria) -> Optional[tuple[float, intel.StationIntel]]:
        #TODO: handle characters that aren't captains
        assert(isinstance(character.location, core.Ship))
        assert(character == character.location.captain)
        assert(character.location.sector)
        sector_id = character.location.sector.entity_id

        # find the closest one (number of jumps, distance)
        travel_cost = np.inf
        closest_station_intel:Optional[intel.StationIntel] = None
        for station_intel in character.intel_manager.intel(station_criteria, intel.StationIntel):
            # make sure we don't have econ agent intel for this station already
            # we're looking to create new intel
            #TODO: should this be a freshness thing?
            if character.intel_manager.get_intel(intel.EconAgentSectorEntityPartialCriteria(underlying_entity_id=station_intel.intel_entity_id), core.Intel):
                continue

            #TODO: handle stations out of sector
            if station_intel.sector_id != sector_id:
                continue
            station = self.gamestate.get_entity(station_intel.intel_entity_id, sector_entity.Station)
            eta = ocore.DockingOrder.compute_eta(character.location, station)
            if eta < travel_cost:
                closest_station_intel = station_intel
                travel_cost = eta

        if closest_station_intel:
            return (travel_cost, closest_station_intel)
        else:
            return None

    def estimate_cost(self, character:core.Character, intel_criteria:intel.EconAgentSectorEntityPartialCriteria) -> Optional[tuple[bool, float]]:
        # we'll need to find a matching sector entity to dock at
        station_criteria = intel.StationIntelPartialCriteria(cls=intel_criteria.underlying_entity_type, sector_id=intel_criteria.sector_id, resources=intel_criteria.sell_resources, inputs=intel_criteria.buy_resources)

        ret = self._find_candidate(character, station_criteria)
        if ret is not None:
            travel_cost, closest_station_intel = ret
            return (travel_cost < 45., travel_cost)

        # if we don't have one, we'll need to find one, but submitting a
        # request for more intel is free
        return (True, 0.0)

    def collect_intel(self, character:core.Character, intel_criteria:intel.EconAgentSectorEntityPartialCriteria) -> float:
        station_criteria = intel.StationIntelPartialCriteria(cls=intel_criteria.underlying_entity_type, sector_id=intel_criteria.sector_id, resources=intel_criteria.sell_resources, inputs=intel_criteria.buy_resources)

        ret = self._find_candidate(character, station_criteria)
        if ret is None:
            character.intel_manager.register_intel_interest(station_criteria, source=intel_criteria)
            return 0.0

        travel_cost, station_intel = ret
        station = self.gamestate.get_entity(station_intel.intel_entity_id, sector_entity.Station)

        assert(isinstance(character.location, core.Ship))
        assert(character == character.location.captain)
        assert(character.location.sector)
        assert(character.location.sector == station.sector)

        docking_order = ocore.DockingOrder.create_docking_order(station, self.gamestate)
        character.location.prepend_order(docking_order)
        return self.gamestate.timestamp + docking_order.estimate_eta() * 1.2

