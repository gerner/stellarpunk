""" Tests for intel collection agenda """

import uuid
from typing import Any, Optional

from stellarpunk import core, agenda, intel
from stellarpunk.core import sector_entity
from stellarpunk.agenda import intel as aintel

def test_no_interests(gamestate, generator, ship, intel_director, testui, simulator):
    # * does it stay idle if there's no interests
    # character captains ship, w/ intel collection agendum
    character = generator.spawn_character(ship)
    character.take_ownership(ship)
    ship.captain = character
    intel_agendum = aintel.IntelCollectionAgendum.create_agendum(character, intel_director, gamestate)
    character.add_agendum(intel_agendum)

    # first of all, we shouldn't even have the agendum be scheduled at this point (no interests)
    assert not gamestate.is_agendum_scheduled(intel_agendum)

    # but even if we do get scheduled, act should not change state, not take primary, not call estimate_cost or collect intel
    gamestate.schedule_agendum_immediate(intel_agendum)
    assert gamestate.is_agendum_scheduled(intel_agendum)

    # make sure we never left IDLE state, never became primary
    def tick_callback():
        character.intel_manager.sanity_check()
        assert intel_agendum._state == aintel.IntelCollectionAgendum.State.IDLE
        assert not intel_agendum.is_primary()

        if not gamestate.is_agendum_scheduled(intel_agendum):
            testui.done = True
    testui.tick_callback = tick_callback
    # just make sure we don't run forever
    testui.eta = 1.0

    # run simulator for some ticks until act gets called
    simulator.run()

def test_no_preempt(gamestate, generator, ship, intel_director, testui, simulator):
    # * does it stay idle if it cannot preempt
    # character captains ship, w/ intel collection agendum
    character = generator.spawn_character(ship)
    character.take_ownership(ship)
    ship.captain = character
    intel_agendum = aintel.IntelCollectionAgendum.create_agendum(character, intel_director, gamestate, idle_period=0.5)
    character.add_agendum(intel_agendum)

    class NoPreemptAgendum(agenda.Agendum):
        def __init__(self, *args:Any, **kwargs:Any) -> None:
            super().__init__(*args, **kwargs)
            self._tried_preempt_count = 0

        def _preempt_primary(self) -> bool:
            self._tried_preempt_count += 1
            return False

        def _start(self) -> None:
            assert(self.character.find_primary_agendum() is None)
            self.make_primary()

        def _stop(self) -> None:
            if self.is_primary():
                self.relinquish_primary()

    no_preempt_agendum = NoPreemptAgendum.create_agendum(character, gamestate)
    character.add_agendum(no_preempt_agendum)

    # advertise some intel interests
    character.intel_manager.register_intel_interest(intel.SectorHexPartialCriteria())

    # we should be scheduled since we advertised an interest
    assert gamestate.is_agendum_scheduled(intel_agendum)

    # make sure we never left IDLE state, never became primary
    def tick_callback():
        character.intel_manager.sanity_check()
        assert intel_agendum._state == aintel.IntelCollectionAgendum.State.IDLE
        assert not intel_agendum.is_primary()

        if no_preempt_agendum._tried_preempt_count > 0:
            testui.done = True

    testui.tick_callback = tick_callback
    # just make sure we don't run forever
    testui.eta = 2.0

    # run simulator for some ticks until act gets called
    assert no_preempt_agendum._tried_preempt_count == 0
    simulator.run()

def test_passive_collection(gamestate, generator, ship, intel_director, testui, simulator):
    # * does it go passive if there's already a primary and it can preempt
    # * does it preempt properly and return primary when done
    # * does it go back to idle if there's a primary and there's no passive intel

    # character captains ship, w/ intel collection agendum
    character = generator.spawn_character(ship)
    character.take_ownership(ship)
    ship.captain = character
    intel_agendum = aintel.IntelCollectionAgendum.create_agendum(character, intel_director, gamestate, idle_period=0.5)
    character.add_agendum(intel_agendum)

    class PreemptAgendum(core.IntelManagerObserver, agenda.Agendum):
        def __init__(self, *args:Any, **kwargs:Any) -> None:
            super().__init__(*args, **kwargs)
            self._preempt_count = 0
            self._pause_count = 0
            self._unpause_count = 0
            self._added_intels:list[core.AbstractIntel] = []

        @property
        def observer_id(self) -> uuid.UUID:
            return core.OBSERVER_ID_NULL

        def intel_added(self, intel_manager:core.AbstractIntelManager, intel:core.AbstractIntel) -> None:
            assert(intel_manager == character.intel_manager)
            self._added_intels.append(intel)

        def _preempt_primary(self) -> bool:
            self._preempt_count += 1
            return True

        def _pause(self) -> None:
            self._pause_count += 1

        def _unpause(self) -> None:
            self._unpause_count += 1
            assert(self.is_primary())

        def _start(self) -> None:
            assert(self.character.find_primary_agendum() is None)
            self.make_primary()
            self.character.intel_manager.observe(self)

        def _stop(self) -> None:
            if self.is_primary():
                self.relinquish_primary()
            self.character.intel_manager.unobserve(self)

    preempt_agendum = PreemptAgendum.create_agendum(character, gamestate)
    character.add_agendum(preempt_agendum)

    # advertise some intel interests
    criteria = intel.SectorHexPartialCriteria()
    character.intel_manager.register_intel_interest(criteria)
    assert len(intel_agendum._interests) == 1

    # we should be scheduled since we advertised an interest
    assert gamestate.is_agendum_scheduled(intel_agendum)

    # make sure we never left IDLE state, never became primary
    def tick_callback():
        character.intel_manager.sanity_check()
        if not preempt_agendum.is_primary():
            assert intel_agendum.is_primary()
            assert intel_agendum._state == aintel.IntelCollectionAgendum.State.PASSIVE
            assert preempt_agendum.paused
        else:
            assert not intel_agendum.is_primary()
            assert intel_agendum._state == aintel.IntelCollectionAgendum.State.IDLE
            assert not preempt_agendum.paused

        if preempt_agendum._unpause_count > 0:
            assert preempt_agendum._preempt_count > 0
            testui.done = True

    testui.tick_callback = tick_callback
    # just make sure we don't run forever
    testui.eta = 5.0

    # run simulator for some ticks until act gets called
    assert not intel_agendum.is_primary()
    assert preempt_agendum._preempt_count == 0
    assert preempt_agendum.is_primary()
    assert len(preempt_agendum._added_intels) == 0
    simulator.run()

    assert len(intel_agendum._interests) == 0
    assert not preempt_agendum.paused
    assert preempt_agendum.is_primary()
    assert len(preempt_agendum._added_intels) > 0
    satisfied = False
    for i in preempt_agendum._added_intels:
        if criteria.matches(i):
            satisfied = True
            break
    assert satisfied

def test_active_collection(gamestate, generator, ship, intel_director, testui, simulator):
    # * does it go active if there's not already a primary
    # * does it go back to idle when there's no intel
    # character captains ship, w/ intel collection agendum
    character = generator.spawn_character(ship)
    character.take_ownership(ship)
    ship.captain = character
    intel_agendum = aintel.IntelCollectionAgendum.create_agendum(character, intel_director, gamestate, idle_period=0.5)
    character.add_agendum(intel_agendum)

    # advertise some intel interests
    criteria = intel.SectorHexPartialCriteria()
    character.intel_manager.register_intel_interest(criteria)
    assert len(intel_agendum._interests) == 1

    # we should be scheduled since we advertised an interest
    assert gamestate.is_agendum_scheduled(intel_agendum)

    # make sure we never left IDLE state, never became primary
    saw_active = False
    def tick_callback():
        nonlocal saw_active
        character.intel_manager.sanity_check()
        if intel_agendum.is_primary():
            assert intel_agendum._state == aintel.IntelCollectionAgendum.State.ACTIVE
            saw_active = True
        else:
            assert intel_agendum._state == aintel.IntelCollectionAgendum.State.IDLE

        if len(character.intel_manager._intel) > 0 and intel_agendum._state == aintel.IntelCollectionAgendum.State.IDLE:
            testui.done = True

    testui.tick_callback = tick_callback
    # just make sure we don't run forever
    testui.eta = 5.0

    simulator.run()

    assert saw_active
    assert intel_agendum._state == aintel.IntelCollectionAgendum.State.IDLE
    assert not intel_agendum.is_primary()
    assert len(intel_agendum._interests) == 0

def test_intel_dependency_chain(gamestate, generator, ship, intel_director, testui, simulator):
    # if we adverstise some interest that implies some other intel that implies
    # some other intel and we satisfy the deepest dependency, but not either of the
    # sources, do both sources end up in base set of interests?

    character = generator.spawn_character(ship)
    character.take_ownership(ship)
    ship.captain = character
    intel_agendum = aintel.IntelCollectionAgendum.create_agendum(character, intel_director, gamestate, idle_period=0.5)
    character.add_agendum(intel_agendum)

    class InterestObserver(core.IntelManagerObserver):
        def __init__(self) -> None:
            super().__init__()
            self._interests:set[core.IntelMatchCriteria] = set()
            self._added_intels:list[core.AbstractIntel] = []

        @property
        def observer_id(self) -> uuid.UUID:
            return core.OBSERVER_ID_NULL

        def intel_desired(self, intel_manager:core.AbstractIntelManager, intel_criteria:core.IntelMatchCriteria, source:Optional[core.IntelMatchCriteria]) -> None:
            self._interests.add(intel_criteria)

        def intel_added(self, intel_manager:core.AbstractIntelManager, intel:core.AbstractIntel) -> None:
            assert intel_manager == character.intel_manager
            self._added_intels.append(intel)

    interest_observer = InterestObserver()
    character.intel_manager.observe(interest_observer)

    # advertise econ agent intel interest. this should trigger a sector entity
    # intel interest, which should trigger a sector hex interest.
    # those triggerings should push source interests out of the interest set
    # only the sector hex interest can be satisfied
    # once it is, at least one source interest should go back to the interest set
    # and intel agendum should be idle
    criteria = intel.EconAgentSectorEntityPartialCriteria(underlying_entity_type=sector_entity.Station)
    character.intel_manager.register_intel_interest(criteria)
    assert len(intel_agendum._interests) == 1

    # we should be scheduled since we advertised an interest
    assert gamestate.is_agendum_scheduled(intel_agendum)

    # stage:
    # 0. econ agent interest
    # 1. sector entity interest
    # 2. sector hex interest
    # 3. back sector entity interest with econ agent interest as a source
    saw_active = False
    saw_interest_ea = False
    saw_interest_se = False
    saw_interest_sh = False
    def tick_callback():
        nonlocal saw_active, saw_interest_ea, saw_interest_se, saw_interest_sh
        character.intel_manager.sanity_check()

        if intel_agendum._state == aintel.IntelCollectionAgendum.State.ACTIVE:
            saw_active = True
        if isinstance(next(iter(intel_agendum._interests)), intel.EconAgentSectorEntityPartialCriteria):
            assert len(intel_agendum._interests) == 1
            saw_interest_ea = True
        if isinstance(next(iter(intel_agendum._interests)), intel.SectorEntityPartialCriteria):
            assert len(intel_agendum._interests) == 1
            saw_interest_se = True
        if isinstance(next(iter(intel_agendum._interests)), intel.SectorHexPartialCriteria):
            assert len(intel_agendum._interests) == 1
            saw_interest_sh = True

        if intel_agendum._state == aintel.IntelCollectionAgendum.State.IDLE and len(interest_observer._added_intels) > 0:
            testui.done = True

    testui.tick_callback = tick_callback
    # just make sure we don't run forever
    testui.eta = 5.0

    simulator.run()

    # at the end we should have seen interests: econ agent, station, sector hex
    # we should still have that full chain of interests
    # the current interest should be unsatisfiable
    assert len(interest_observer._interests) == 3
    assert criteria in interest_observer._interests
    assert len(intel_agendum._interests) == 1
    assert next(iter(intel_agendum._interests)) in intel_agendum._source_interests_by_dependency
    assert len(intel_agendum._source_interests_by_source) == 2
    assert criteria in intel_agendum._source_interests_by_source

    # this is basically asking us to find sector hexes in a different sector
    # but there are no other sectors in this test, so this will never be satisfiable
    assert intel_director.estimate_cost(character, next(iter(intel_agendum._interests))) is None

