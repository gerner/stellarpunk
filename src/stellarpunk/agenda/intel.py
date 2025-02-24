import enum
import uuid
import collections
import abc
from collections.abc import Iterable
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
        self._state = IntelCollectionAgendum.State.IDLE

        # some stats
        self._interests_advertised = 0
        self._interests_satisfied = 0
        self._immediate_interest_count = 0
        self._immediate_interests_satisfied = 0

        # interests we can actively work on
        self._interests:set[core.IntelMatchCriteria] = set()
        self._immediate_interest:Optional[core.IntelMatchCriteria] = None

        # source/depdnency many:many relationship, including the None source
        self._source_interests_by_dependency:collections.defaultdict[core.IntelMatchCriteria, set[Optional[core.IntelMatchCriteria]]] = collections.defaultdict(set)
        self._source_interests_by_source:collections.defaultdict[Optional[core.IntelMatchCriteria], set[core.IntelMatchCriteria]] = collections.defaultdict(set)

        # interests that would cause a dependency cycle and need to be rejected
        self._cycle_interests:set[core.IntelMatchCriteria] = set()

        self._preempted_primary:Optional[core.AbstractAgendum] = None

    def __str__(self) -> str:
        return f'{util.fullname(self)} {self._state.name} interests:{len(self._interests)} preempted_primary:{self._preempted_primary}'


    def sanity_check(self) -> None:
        for dependency, sources in self._source_interests_by_dependency.items():
            for source in sources:
                assert dependency in self._source_interests_by_source[source]
        for source, dependencies in self._source_interests_by_source.items():
            if source is None:
                continue
            for dependency in dependencies:
                assert source in self._source_interests_by_dependency[dependency]

        if self._preempted_primary:
            assert self.is_primary()
            assert self._state in (IntelCollectionAgendum.State.PASSIVE, IntelCollectionAgendum.State.ACTIVE)
            assert self.character.find_primary_agendum() == self
            assert not self._preempted_primary.is_primary()
        else:
            assert self._state in (IntelCollectionAgendum.State.IDLE, IntelCollectionAgendum.State.ACTIVE)

        if self._state in (IntelCollectionAgendum.State.PASSIVE, IntelCollectionAgendum.State.ACTIVE):
            assert self.is_primary()
            assert self.character.find_primary_agendum() == self

    # core.IntelManagerObserver

    @property
    def observer_id(self) -> uuid.UUID:
        return self.agenda_id

    def intel_desired(self, intel_manager:core.AbstractIntelManager, intel_criteria:core.IntelMatchCriteria, source:Optional[core.IntelMatchCriteria]=None) -> None:
        # this new desire is triggered because we actually want this source
        # kind of intel. We need to track that because looking for source
        # directly is not productive any more
        # once the dependency is achieved, we can start looking for the
        # source again
        # we explicitly model the None source for the purposes of explicitly
        # tracking if a piece of intel is needed independently and thus always
        # knowing if a piece of intel is still needed

        # one interest can be a dependency of several sources. once we
        # satisfy the dependency, all sources should be re-examined

        # a dependency can end up being a source of further dependencies

        # remove the source interest if we have it. we don't want to try to
        # get that before we get the dependent intel. but we'll keep track
        # to make sure we eventually get it or try fresh if we satisfy this
        # dependency without satisfying the source

        # immediate interest is this source, track the dependency instead
        if source:
            if util.detect_cycle(intel_criteria, self._source_interests_by_source, set((source,))): # type: ignore
                # adding intel_criteria from source would create a dependency
                # cycle
                # this needs to be resolved by rejecting intel_criteria and
                # related interests
                # we'll mark these here and reject them on a subsequent tick
                self._cycle_interests.add(intel_criteria)
                self._cycle_interests.add(source)
                if source == self._immediate_interest:
                    self.gamestate.schedule_agendum_immediate(self, jitter=1.0)
                return
            if source == self._immediate_interest:
                self._immediate_interest = intel_criteria
            if source in self._interests:
                self._interests.remove(source)
        self._source_interests_by_dependency[intel_criteria].add(source)
        self._source_interests_by_source[source].add(intel_criteria)

        # make note that we want to find such intel, if we haven't already
        assert(intel_manager == self.character.intel_manager)
        if intel_criteria not in self._source_interests_by_source:
            self._interests.add(intel_criteria)
            self._interests_advertised += 1

        if self._immediate_interest is not None and self._immediate_interest not in self._interests:
            # this interest must be a dependency of what we were just looking
            # for, but we already tried to get it and have some other interest
            # we need to satisfy for it
            assert self._state in (IntelCollectionAgendum.State.PASSIVE, IntelCollectionAgendum.State.ACTIVE)
            assert self._immediate_interest == intel_criteria
            #TODO: make sure this interest has some interest dependency path to self._interests
            assert len(self._interests) > 0

            # and just try gathering intel again
            #self._immediate_interest = None
            self.gamestate.schedule_agendum_immediate(self, jitter=1.0)
            return

        assert self._immediate_interest is None or self._immediate_interest in self._interests

        # if we're already passively or actively collecting intel, no sense
        # interrupting that, so wait for that to finish and we'll get an act
        # call when that should be complete. at that point we'll consider new
        # intel needs.
        if self._state == IntelCollectionAgendum.State.IDLE:
            # note: we might ask to be scheduled many times here if someone
            # registers several interests, but the schedule will dedupe
            self.gamestate.schedule_agendum_immediate(self, jitter=1.0)

    def _reject_interest(self, interest:core.IntelMatchCriteria, visited:Optional[set[core.IntelMatchCriteria]]=None) -> None:
        if visited is None:
            visited = set()
        if interest in visited:
            return
        visited.add(interest)
        # this is how we signal to intel consumers that we will not try
        # to collect this intel
        # someone else should try to take back primary agendum and do
        # something else.
        self.character.intel_manager.unregister_intel_interest(interest)
        for child in self._source_interests_by_dependency[interest].copy():
            if child is None:
                continue
            self._reject_interest(child, visited)
        self.logger.info(f'rejection dropping {interest}')
        self._remove_interest(interest)

    def _remove_interest(self, interest:core.IntelMatchCriteria) -> None:
        # prec: interest was an interest we were tracking with at least the None source, possibly others
        # postc: any sources that dependended on interest are directly tracked
        # postc: any depdendencies of only interest are removed
        # postc: interest is removed

        # this interest is no longer needed, pull all corresponding sources back
        # into core interest set and eliminate all dependencies
        # if any of those sources are satisfied, recursively remove them

        # if this has already been removed, just bail
        if interest not in self._source_interests_by_dependency:
            assert interest not in self._interests
            assert interest not in self._source_interests_by_source
            return

        # stop tracking directly if we are
        if interest in self._interests:
            self._interests.remove(interest)
            if interest == self._immediate_interest:
                self._immediate_interest = None

        # stop tracking dependencies if no other sources are counting on them
        if interest in self._source_interests_by_source:
            while self._source_interests_by_source[interest]:
                dependency = self._source_interests_by_source[interest].pop()
                self._source_interests_by_dependency[dependency].remove(interest)
                if not self._source_interests_by_dependency[dependency]:
                    self._remove_interest(dependency)
                # otherwise this dependency has some other source so keep it
            # nuke the record that we were ever a source
            del self._source_interests_by_source[interest]

        # start directly tracking sources
        # we definitely have a source, even if it's the None source
        assert interest in self._source_interests_by_dependency
        while self._source_interests_by_dependency[interest]:
            source = self._source_interests_by_dependency[interest].pop()
            if source is None:
                # we won't do anything with the None source except remove
                # interest as a dependency
                self._source_interests_by_source[None].remove(interest)
                if len(self._source_interests_by_source[None]) == 0:
                    del self._source_interests_by_source[None]
                continue
            # nuke all this sources dependencies since we're going to directly
            # track it again.
            while self._source_interests_by_source[source]:
                other_dependency = self._source_interests_by_source[source].pop()
                if other_dependency == interest:
                    continue
                self._remove_interest(other_dependency)
            del self._source_interests_by_source[source]
            # re-add it as a direct interest
            self._interests.add(source)

        # nuke the record that we were ever a dependency
        del self._source_interests_by_dependency[interest]

    def intel_added(self, intel_manager:core.AbstractIntelManager, intel_item:core.AbstractIntel) -> None:
        # prec: intel_item is known
        # postc: any interests, directly tracked or not, that match intel_item are removed
        # postc: any unneeded dependencies are removed
        # postc: any sources who have at least one satisfied dependencies are directly tracked
        # check if it matches our immediate interest now because we'll clear it
        # as part of the removal process below
        if self._immediate_interest and self._immediate_interest.matches(intel_item):
            matched_immediate = True
            self._immediate_interests_satisfied += 1
        else:
            matched_immediate = False

        matching_interests:list[core.IntelMatchCriteria] = []
        for interest in self._source_interests_by_source.keys():
            if interest is None:
                continue
            if interest.matches(intel_item):
                self._interests_satisfied += 1
                matching_interests.append(interest)

        # first remove all the matching source interests
        # this might move some matching sources to self._interests
        for interest in matching_interests:
            self._remove_interest(interest)

        for interest in self._interests:
            if interest.matches(intel_item):
                self._interests_satisfied += 1
                matching_interests.append(interest)

        # now remove matching direct interests
        # there cannot be any matching source interests at this point. we would
        # have matched them above.
        for interest in matching_interests:
            self._remove_interest(interest)


        # if we have no interests left
        if len(self._interests) == 0:
            # we must have cleared our immediate interest
            # either by matching it directly or making it redundant
            assert self._immediate_interest is None
            self._go_idle()

        # if we just gained intel that satisfies one of our immediate interests
        # either by matching it directly or making it redundant
        elif self._immediate_interest is None:
            # keep collecting more intel (passively or actively)
            if self._state in [IntelCollectionAgendum.State.PASSIVE, IntelCollectionAgendum.State.ACTIVE]:
                self.gamestate.schedule_agendum_immediate(self, jitter=1.0)
            else:
                # this must have been incidental intel that got collected
                # e.g. we docked at a station
                assert self._state == IntelCollectionAgendum.State.IDLE

        # otherwise this was an incidental added intel, not one we're actively
        # working on, so no need to reschedule ourselves

    # Agendum

    def _preempt_primary(self) -> bool:
        # we can be preempted, we'll just interrupt our current collection
        self._go_idle()
        return True

    def _unpause(self) -> None:
        self._go_idle()

    def _pause(self) -> None:
        if self._preempted_primary:
            self._restore_preempted()

    def _start(self) -> None:
        self.character.intel_manager.observe(self)
        self._go_idle()

    def _stop(self) -> None:
        if self._preempted_primary:
            self._restore_preempted()
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
            current_primary = self.character.find_primary_agendum()
            # there must be a current primary, otherwise we'd be in ACTIVE mode
            assert(current_primary is not None)

            # check if they'll let us preempt
            if not current_primary.preempt_primary():
                self._go_idle()
                return
            self._preempted_primary = current_primary
            current_primary.pause()
            self.make_primary()
            if isinstance(self.character.location, core.Ship):
                self.character.location.clear_orders()

        self._immediate_interest = cheapest_criteria
        self._immediate_interest_count += 1
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
            # else is directing primary character behavior.
            self.logger.info(f'{self.character} has intel interests that we cannot actively satisfy: {self._interests}')
            # clear out all intel interests, we will not try to collect them
            # if someone still wants them, something else needs to happen in
            # order to collect that intel.

            # do this in a while loop because we'll be pulling source interests
            # back into interests and we want to remove those too.
            while self._interests:
                interest = self._interests.pop()
                self._reject_interest(interest)

            assert len(self._source_interests_by_source) == 0
            assert len(self._source_interests_by_dependency) == 0
            self._go_idle()
            return

        self._immediate_interest = cheapest_criteria
        self._immediate_interest_count += 1
        self.logger.debug(f'{self.character} attempting to collect {cheapest_criteria}')
        next_ts = self._director.collect_intel(self.character, cheapest_criteria)
        if next_ts > 0:
            self.gamestate.schedule_agendum(next_ts, self, jitter=1.0)
        else:
            self.gamestate.schedule_agendum_immediate(self)

    def _restore_preempted(self) -> None:
        assert(self._preempted_primary)
        assert(self.is_primary())
        assert(self._state == IntelCollectionAgendum.State.PASSIVE)
        self.relinquish_primary()
        self._preempted_primary.make_primary()
        self._preempted_primary.unpause()
        self._preempted_primary = None

    def _go_idle(self) -> None:
        if self._preempted_primary:
            self._restore_preempted()
        elif self.is_primary():
            self.relinquish_primary()
        self._immediate_interest = None
        self._state = IntelCollectionAgendum.State.IDLE

        # we'll wake ourselves up if someone registers an interest, no need to
        # force a wakeup that will do nothing
        if len(self._interests) > 0:
            self.gamestate.schedule_agendum(self.gamestate.timestamp + self._idle_period, self, jitter=1.0)
        else:
            self.gamestate.unschedule_agendum(self)


    def act(self) -> None:
        if self.paused:
            return

        while self._cycle_interests:
            cycle_interest = self._cycle_interests.pop()
            # this cycle makes the interest unsatisfiable
            self.logger.info(f'{self.character} has intel interest {cycle_interest} that we cannot satisfy: (interest dependency cycle)')
            # bail on all the interests for which this is a dependency
            self._reject_interest(cycle_interest)

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
                primary_agendum = self.character.find_primary_agendum()
                if primary_agendum is None:
                    self.make_primary()
                    if isinstance(self.character.location, core.Ship):
                        self.character.location.clear_orders()
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
        # a list of match criteria type, intel gatherer pairs where each
        # gatherer can find intel matching crtieria of that type.
        # this is sorted so the most specific types come first (e.g.
        # AsteroidIntelPartialCriteria before SectorEntityPartialCriteria, the
        # first is a more specific subclass of the other)
        self._gatherers:list[tuple[Type[core.IntelMatchCriteria], IntelGatherer]] = []

    def _find_gatherer(self, klass:Type[core.IntelMatchCriteria]) -> Optional["IntelGatherer"]:
        for criteria_klass, gatherer in self._gatherers:
            if issubclass(klass, criteria_klass):
                return gatherer
        return None

    def initialize_gamestate(self, gamestate:core.Gamestate) -> None:
        for _, gatherer in self._gatherers:
            gatherer.initialize_gamestate(gamestate)

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

class SectorIntelGatherer(IntelGatherer[intel.SectorPartialCriteria]):
    def _find_candidate(self, character:core.Character, intel_criteria:intel.SectorPartialCriteria) -> Optional[uuid.UUID]:
        if intel_criteria.sector_id is not None and intel_criteria.jump_distance == 0:
            # check if we have that sector already
            sector = character.intel_manager.get_intel(intel.EntityIntelMatchCriteria(intel_criteria.sector_id), intel.SectorIntel)
            if sector is not None:
                #TODO: should we try to refresh it?
                return None
            return intel_criteria.sector_id

        universe_view = intel.UniverseView.create(character)

        if character.location is not None and character.location.sector is not None:
            sector_ids:Iterable[uuid.UUID] = universe_view.sector_ids_by_distance(character.location.sector.entity_id)
        else:
            sector_ids = universe_view.sector_idx_lookup.keys()

        for sector_id in sector_ids:
            sector = character.intel_manager.get_intel(intel.EntityIntelMatchCriteria(sector_id), intel.SectorIntel)
            if sector is not None:
                #TODO: should we try to refresh it?
                continue
            if intel_criteria.sector_id is not None:
                path = universe_view.compute_path(intel_criteria.sector_id, sector_id)
                if path is None:
                    # no known path to that sector
                    continue
                if len(path) > intel_criteria.jump_distance:
                    # sector is too far away from source sector
                    continue
            return sector_id

        # we don't know of any sectors that we haven't already explored
        return None

    def estimate_cost(self, character:core.Character, intel_criteria:intel.SectorPartialCriteria) -> Optional[tuple[bool, float]]:
        #TODO: what if we're not a captain?
        assert isinstance(character.location, core.Ship)
        sector_id = self._find_candidate(character, intel_criteria)
        if sector_id is not None:
            eta = ocore.LocationExploreOrder.compute_eta(character.location, sector_id, ocore.ZERO_VECTOR)
            return True, eta
        else:
            # we should search for a new gate, right? this might cause a
            # cycle in interests: we might end up asking for a new sector again
            # however, it can only cycle back to searching for a new sector
            #TODO: how to avoid this cycle? or detect it and then bail on the
            # chain of interests related to it?
            return False, 0.0

    def collect_intel(self, character:core.Character, intel_criteria:intel.SectorPartialCriteria) -> float:
        sector_id = self._find_candidate(character, intel_criteria)

        if sector_id is not None:
            #TODO: what if we're not a captain?
            assert isinstance(character.location, core.Ship)
            explore_order = ocore.LocationExploreOrder.create_order(character.location, self.gamestate, sector_id, ocore.ZERO_VECTOR)
            #TODO: should we keep track of this order and cancel it if necessary?
            character.location.prepend_order(explore_order)
            return self.gamestate.timestamp + explore_order.estimate_eta() * 1.2
        else:
            # we need to find a new gate
            gate_criteria = intel.SectorEntityPartialCriteria(cls=sector_entity.TravelGate, sector_id=intel_criteria.sector_id, jump_distance=intel_criteria.jump_distance)
            character.intel_manager.register_intel_interest(gate_criteria, source=intel_criteria)
            return 0.0


class SectorHexIntelGatherer(IntelGatherer[intel.SectorHexPartialCriteria]):
    def _candidate_in_sector(self, character:core.Character, intel_criteria:intel.SectorHexPartialCriteria, sector:intel.SectorIntel) -> Optional[npt.NDArray[np.float64]]:
        if character.location is not None and character.location.sector is not None and character.location.sector.entity_id == sector.intel_entity_id:
            target_loc = character.location.loc
        else:
            target_loc = ocore.ZERO_VECTOR

        target_dist = sector.radius*3.0
        hex_loc = sector.get_hex_coords(target_loc)
        # look for hex options in current sector

        # start looking close-ish to where we are, within a sector radius
        target_hex_loc = hex_loc
        target_hex_dist = target_dist / (np.sqrt(3)*sector.hex_size)

        # honor intel criteria's desire of course
        if intel_criteria.hex_loc is not None:
            target_hex_loc = intel_criteria.hex_loc
        if intel_criteria.hex_dist is not None:
            target_hex_dist = intel_criteria.hex_dist

        candidate_hexes:set[tuple[int, int]] = {(int(x[0]), int(x[1])) for x in util.hexes_within_pixel_dist(target_loc, target_dist, sector.hex_size)}

        # find hexes in the current sector we know about
        for i in character.intel_manager.intel(intel.SectorHexPartialCriteria(sector_id=sector.intel_entity_id, is_static=intel_criteria.is_static, hex_loc=target_hex_loc, hex_dist=target_hex_dist), intel.SectorHexIntel):
            hex_key = (int(i.hex_loc[0]), int(i.hex_loc[1]))
            if hex_key in candidate_hexes:
                candidate_hexes.remove(hex_key)

        # pick closest remaining candidate
        candidate = next(iter(sorted(candidate_hexes, key=lambda x: util.axial_distance(np.array(x), hex_loc))), None)
        if candidate is None:
            return None
        else:
            return np.array((float(candidate[0]), float(candidate[1])))

    def _find_candidate(self, character:core.Character, intel_criteria:intel.SectorHexPartialCriteria) -> Optional[tuple[intel.SectorIntel, npt.NDArray[np.float64]]]:
        if intel_criteria.sector_id is not None and intel_criteria.jump_distance == 0:
            # return whatever candidate is available in that sector, if any
            sector = character.intel_manager.get_intel(intel.EntityIntelMatchCriteria(intel_criteria.sector_id), intel.SectorIntel)
            if sector is None:
                return None
            candidate = self._candidate_in_sector(character, intel_criteria, sector)
            if candidate is None:
                return None
            return (sector, candidate)

        # search across sectors for an unexplored hex
        universe_view = intel.UniverseView.create(character)
        if character.location is not None and character.location.sector is not None:
            sectors:Iterable[intel.SectorIntel] = universe_view.sectors_by_distance(character.location.sector.entity_id)
        else:
            sectors = universe_view.sector_intels

        for sector in sectors:
            if intel_criteria.sector_id is not None:
                path = universe_view.compute_path(intel_criteria.sector_id, sector.intel_entity_id)
                if path is None:
                    # no known path to that sector
                    continue
                if len(path) > intel_criteria.jump_distance:
                    # sector is too far away from source sector
                    continue
            candidate = self._candidate_in_sector(character, intel_criteria, sector)
            if candidate is not None:
                return (sector, candidate)

        # no unexplored hex in any known sectors
        return None

    def estimate_cost(self, character:core.Character, intel_criteria:intel.SectorHexPartialCriteria) -> Optional[tuple[bool, float]]:
        # passive => target hex is adjacent to the one we're in right now
        # cost = time to travel to center of target hex
        # target hex is closest one where a scan will produce new intel that
        # will match this partial criteria

        #TODO: what if we're not captain?
        # we can't estimate cost if we don't know where the character is
        if character.location is None:
            return None
        if character.location.sector is None:
            return None
        assert isinstance(character.location, core.Ship)

        ret = self._find_candidate(character, intel_criteria)

        if ret is not None:
            sector, candidate = ret
            candidate_coords = sector.get_coords_from_hex(candidate)

            eta = ocore.LocationExploreOrder.compute_eta(character.location, sector.intel_entity_id, candidate_coords)
            return eta > 45.0, eta
        else:
            # no candidates to explore, we'll have to find a new sector
            # but submitting more intel is free
            return False, 0.0

    def collect_intel(self, character:core.Character, intel_criteria:intel.SectorHexPartialCriteria) -> float:
        # we can't estimate cost if we don't know where the character is
        if character.location is None:
            raise ValueError(f'cannot collect intel for {character} not on any ship')
        if character.location.sector is None:
            raise ValueError(f'cannot collect intel for {character} not in any sector')

        ret = self._find_candidate(character, intel_criteria)

        if ret is not None:
            sector, candidate = ret

            #TODO: what if we're not a captain?
            # collect intel at this candidate
            assert(isinstance(character.location, core.Ship))
            assert(character.location.captain == character)
            candidate_coords = sector.get_coords_from_hex(candidate)
            explore_order = ocore.LocationExploreOrder.create_order(character.location, self.gamestate, sector.intel_entity_id, candidate_coords)
            #TODO: should we keep track of this order and cancel it if necessary?
            character.location.prepend_order(explore_order)
            return self.gamestate.timestamp + explore_order.estimate_eta() * 1.2
        else:
            sector_criteria = intel.SectorPartialCriteria(sector_id=intel_criteria.sector_id, jump_distance=intel_criteria.jump_distance)
            character.intel_manager.register_intel_interest(sector_criteria, source=intel_criteria)
            return 0.0

class SectorEntityIntelGatherer(IntelGatherer[intel.SectorEntityPartialCriteria]):
    def estimate_cost(self, character:core.Character, intel_criteria:intel.SectorEntityPartialCriteria) -> Optional[tuple[bool, float]]:
        # we'll just ask for sector hex intel, that's free
        return (True, 0.0)

    def collect_intel(self, character:core.Character, intel_criteria:intel.SectorEntityPartialCriteria) -> float:
        sector_hex_criteria = intel.SectorHexPartialCriteria(sector_id=intel_criteria.sector_id, is_static=intel_criteria.is_static, jump_distance=intel_criteria.jump_distance)
        character.intel_manager.register_intel_interest(sector_hex_criteria, source=intel_criteria)
        return 0.0

class EconAgentSectorEntityIntelGatherer(IntelGatherer[intel.EconAgentSectorEntityPartialCriteria]):
    def _find_candidate(self, character:core.Character, station_criteria:intel.StationIntelPartialCriteria) -> Optional[tuple[float, intel.StationIntel]]:
        #TODO: handle characters that aren't captains
        assert(isinstance(character.location, core.Ship))
        assert(character == character.location.captain)
        assert(character.location.sector)

        # find the closest one (number of jumps, distance)
        if station_criteria.sector_id and station_criteria.jump_distance > 0:
            universe_view = intel.UniverseView.create(character)
        travel_cost = np.inf
        closest_station_intel:Optional[intel.StationIntel] = None
        for station_intel in character.intel_manager.intel(station_criteria, intel.StationIntel):
            # make sure we don't have econ agent intel for this station already
            # we're looking to create new intel
            #TODO: should this be a freshness thing?
            if character.intel_manager.get_intel(intel.EconAgentSectorEntityPartialCriteria(underlying_entity_id=station_intel.intel_entity_id), core.AbstractIntel):
                continue

            # we might get stations from intel manager that are in technically
            # closer than we have intel for (i.e. we might not know the
            # shortest path), we should reject them.
            if station_criteria.sector_id and station_criteria.jump_distance > 0:
                path = universe_view.compute_path(station_criteria.sector_id, station_intel.sector_id)
                if path is None:
                    # no known path to that sector
                    continue
                if len(path) > station_criteria.jump_distance:
                    # sector is too far away from source sector
                    continue

            if station_intel.sector_id == character.location.sector.entity_id:
                eta = ocore.DockingOrder.compute_eta(character.location, station_intel.loc)
            else:
                eta = ocore.NavigateOrder.compute_eta(character.location, station_intel.sector_id)
                eta += ocore.DockingOrder.compute_eta(character.location, station_intel.loc, starting_loc=ocore.ZERO_VECTOR)

            if eta < travel_cost:
                closest_station_intel = station_intel
                travel_cost = eta

        if closest_station_intel:
            return (travel_cost, closest_station_intel)
        else:
            return None

    def estimate_cost(self, character:core.Character, intel_criteria:intel.EconAgentSectorEntityPartialCriteria) -> Optional[tuple[bool, float]]:
        # we'll need to find a matching sector entity to dock at
        station_criteria = intel.StationIntelPartialCriteria(cls=intel_criteria.underlying_entity_type, sector_id=intel_criteria.sector_id, resources=intel_criteria.sell_resources, inputs=intel_criteria.buy_resources, jump_distance=intel_criteria.jump_distance)

        ret = self._find_candidate(character, station_criteria)
        if ret is not None:
            travel_cost, closest_station_intel = ret
            return (travel_cost > 45., travel_cost)

        # if we don't have one, we'll need to find one, but submitting a
        # request for more intel is free
        return (False, 0.0)

    def collect_intel(self, character:core.Character, intel_criteria:intel.EconAgentSectorEntityPartialCriteria) -> float:
        station_criteria = intel.StationIntelPartialCriteria(cls=intel_criteria.underlying_entity_type, sector_id=intel_criteria.sector_id, resources=intel_criteria.sell_resources, inputs=intel_criteria.buy_resources, jump_distance=intel_criteria.jump_distance)

        ret = self._find_candidate(character, station_criteria)
        if ret is None:
            character.intel_manager.register_intel_interest(station_criteria, source=intel_criteria)
            return 0.0

        travel_cost, station_intel = ret
        assert(isinstance(character.location, core.Ship))
        assert(character.location.sector)
        assert(character == character.location.captain)

        docking_order = ocore.DockingOrder.create_docking_order(character.location, self.gamestate, target_id=station_intel.intel_entity_id)

        if station_intel.sector_id == character.location.sector.entity_id:
            character.location.prepend_order(docking_order)
            eta = ocore.DockingOrder.compute_eta(character.location, station_intel.loc) * 1.2
        else:
            character.location.prepend_order(docking_order, begin=False)
            navigate_order = ocore.NavigateOrder.create_order(character.location, self.gamestate, station_intel.sector_id)
            character.location.prepend_order(navigate_order)
            eta = ocore.DockingOrder.compute_eta(character.location, station_intel.loc, starting_loc=ocore.ZERO_VECTOR) * 1.2
            eta += ocore.NavigateOrder.compute_eta(character.location, station_intel.sector_id)

        return self.gamestate.timestamp + eta
