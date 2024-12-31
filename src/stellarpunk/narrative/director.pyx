# cython: boundscheck=False, wraparound=False, cdivision=True, infer_types=False, nonecheck=False

import sys
from typing import Tuple, Mapping, Dict, Any, Iterable, List

from libc.stdint cimport uint64_t
from libcpp cimport bool
from libcpp.unordered_map  cimport unordered_map
from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr, make_unique
from libcpp.utility cimport move
from cpython.ref cimport PyObject
from cython.operator cimport dereference, preincrement
import cython


cdef extern from "director.hpp":
    ctypedef unordered_map[uint64_t, uint64_t] cEventContext

    struct cEvent:
        uint64_t event_type
        cEventContext event_context
        unordered_map[uint64_t, cEventContext]* entity_context
        void* data

        cEvent(uint64_t et, cEventContext ec, unordered_map[uint64_t, cEventContext]* ent_c, void* d)

    cdef cppclass cIntRef:
        uint64_t value
        cIntRef()
        cIntRef(uint64_t v)

    cdef cppclass cFlagRef:
        uint64_t fact
        cFlagRef()
        cFlagRef(uint64_t f)

    cdef cppclass cEntityRef:
        uint64_t entity_fact
        uint64_t sub_fact
        cEntityRef()
        cEntityRef(uint64_t ef, uint64_t sf)

    cdef cppclass cCriteriaBase:
        pass

    cdef cppclass cCriteria[T, L, U](cCriteriaBase):
        T fact
        L low
        U high
        cCriteria()
        cCriteria(T f, uint64_t l, uint64_t h)

    cdef cppclass cCriteriaBuilder:
        cCriteriaBuilder()
        void addL[L](L l)
        void addF[F](F f)
        void addU[U](U u)
        unique_ptr[cCriteriaBase] build()

    cdef cppclass cActionTemplate:
        cActionTemplate()
        cActionTemplate(uint64_t aid, void* args)

    cdef struct cCharacterCandidate:
        cEventContext* character_context
        void* data

        cCharacterCandidate(cEventContext* cc, void* d)

    cdef struct cAction:
        uint64_t action_id;
        cCharacterCandidate character_candidate;
        void* args;

        cAction()

    cdef cppclass cRule:
        cRule()
        cRule(
            uint64_t et,
            uint64_t pri,
            vector[unique_ptr[cCriteriaBase]] &cri,
            vector[cActionTemplate] a
        )

        uint64_t get_priority()

    cdef cppclass cDirector:
        cDirector()
        cDirector(unordered_map[uint64_t, vector[unique_ptr[cRule]]] &r)
        vector[cAction] evaluate(cEvent* event, vector[cCharacterCandidate] character_candidates)


cdef class EventContext:
    cdef cEventContext* event_context;

    def set_flag(self, flag, value):
        dereference(self.event_context)[flag] = value

    def get_flag(self, flag):
        return dereference(self.event_context)[flag]

    def to_dict(self):
        cdef unordered_map[uint64_t, uint64_t].iterator itr = self.event_context.begin()
        d = {}
        while itr != self.event_context.end():
            d[dereference(itr).first] = dereference(itr).second
            preincrement(itr)
        return d


cdef class EntityStore:
    cdef unordered_map[uint64_t, cEventContext] entity_context

    def register_entity(self, entity_id) -> EventContext:
        if self.entity_context.count(entity_id) != 0:
            raise ValueError(f'entity with id {entity_id} already registered')

        self.entity_context[entity_id] = cEventContext()
        cdef EventContext event_context = EventContext()
        event_context.event_context = &(self.entity_context[entity_id])
        return event_context

    def unregister_entity(self, entity_id) -> None:
        if self.entity_context.count(entity_id) != 1:
            raise ValueError(f'no entity with id {entity_id} registered')

        self.entity_context.erase(<uint64_t?>entity_id)


cdef class IntRef:
    cdef cIntRef c_ref
    def __cinit__(self, value):
        self.c_ref = cIntRef(value)

    @property
    def value(self):
        return self.c_ref.value


cdef class FlagRef:
    cdef cFlagRef c_ref
    def __cinit__(self, flag):
        self.c_ref = cFlagRef(flag)

    @property
    def fact(self):
        return self.c_ref.fact


cdef class EntityRef:
    cdef cEntityRef c_ref
    def __cinit__(self, entity_fact, sub_fact):
        self.c_ref = cEntityRef(entity_fact, sub_fact)

    @property
    def entity_fact(self):
        return self.c_ref.entity_fact
    @property
    def sub_fact(self):
        return self.c_ref.sub_fact


cdef class CriteriaBuilder:
    cdef cCriteriaBuilder c_builder
    cdef vector[unique_ptr[cCriteriaBase]] c_criteria

    cdef public object last_low
    cdef public object last_fact
    cdef public object last_high

    def add_low(self, low):
        self.last_low = low
        if isinstance(low, IntRef):
            self.c_builder.addL((<IntRef?>low).c_ref)
        elif isinstance(low, FlagRef):
            self.c_builder.addL((<FlagRef?>low).c_ref)
        elif isinstance(low, EntityRef):
            self.c_builder.addL((<EntityRef?>low).c_ref)
        else:
            raise ValueError(f'only int, flag, entity refs are allowed, got {low.__class__}')

    def add_fact(self, fact):
        self.last_fact = fact
        if isinstance(fact, IntRef):
            self.c_builder.addF((<IntRef?>fact).c_ref)
        elif isinstance(fact, FlagRef):
            self.c_builder.addF((<FlagRef?>fact).c_ref)
        elif isinstance(fact, EntityRef):
            self.c_builder.addF((<EntityRef?>fact).c_ref)
        else:
            raise ValueError(f'only int, flag, entity refs are allowed, got {fact.__class__}')

    def add_high(self, high):
        self.last_high = high
        if isinstance(high, IntRef):
            self.c_builder.addU((<IntRef?>high).c_ref)
        elif isinstance(high, FlagRef):
            self.c_builder.addU((<FlagRef?>high).c_ref)
        elif isinstance(high, EntityRef):
            self.c_builder.addU((<EntityRef?>high).c_ref)
        else:
            raise ValueError(f'only int, flag, entity refs are allowed, got {high.__class__}')

    def build(self):
        self.c_criteria.push_back(self.c_builder.build())
        self.c_builder = cCriteriaBuilder()


cdef class ActionTemplate:
    cdef cActionTemplate action_template
    cdef object args

    def __cinit__(self, action_id, args):
        cdef PyObject* c_args = <PyObject*>args
        self.action_template = cActionTemplate(action_id, c_args)

        # we hang on to a reference of the arguments to keep them alive
        self.args = args

    def check_refcounts(self):
        print(f'actiontemplate.args: {sys.getrefcount(self.args)}')


cdef class Rule:
    cdef unique_ptr[cRule] c_rule
    cdef object actions

    def __cinit__(self, event_type, priority, criteria, actions=[]):
        cdef vector[cActionTemplate] c_actions
        cdef uint64_t c_event_type = event_type
        cdef uint64_t c_priority = priority

        for a in actions:
            c_actions.push_back((<ActionTemplate?>a).action_template)

        self.c_rule = make_unique[cRule](c_event_type, c_priority, (<CriteriaBuilder?>criteria).c_criteria, c_actions)

        # we hang on to action templates to keep them alive
        self.actions = list(actions)

    def check_refcounts(self):
        print(f'self.c_rule.actions {sys.getrefcount(self.actions)}')
        for action in self.actions:
            print(f'self.c_rule.actions[i] {sys.getrefcount(action)}')
            action.check_refcounts()

    def get_priority(self):
        return dereference(self.c_rule).get_priority();


cdef class Event:
    cdef cEvent event
    cdef public object event_type
    cdef public object event_context
    cdef public object entity_context
    cdef public object args

    def __cinit__(self, event_type: int, event_context: Mapping[int, int], entity_context: EntityStore, args: Any):

        self.event = cEvent()
        self.event.event_type = event_type
        self.event.entity_context = &((<EntityStore?>entity_context).entity_context)
        self.event.data = <PyObject *>args

        # populate the event context
        for k, v in event_context.items():
            self.event.event_context[k] = v

        # we hang on to several items
        self.event_type = event_type
        self.event_context = event_context
        self.entity_context = entity_context
        self.args = args


class CharacterCandidate:
    def __init__(self, character_context:EventContext, data:Any):
        self.character_context = character_context
        self.data = data


class Action:
    def __init__(self, action_id:int, character_candidate:CharacterCandidate, args:Any):
        self.action_id = action_id
        self.character_candidate = character_candidate
        self.args = args


cdef class Director:
    cdef cDirector director
    cdef object rules

    def __cinit__(self, rules:Mapping[int, Iterable[Rule]]):
        cdef unordered_map[uint64_t, vector[unique_ptr[cRule]]] c_rules;

        for k, v in rules.items():
            for r in v:
                c_rules[k].push_back(move((<Rule?>r).c_rule))

        self.director = cDirector(c_rules)

        # we hang on to the python wrapper for the rules to keep them alive
        self.rules = rules

    def check_refcounts(self):
        print(f'director.rules: {sys.getrefcount(self.rules)}')
        for k, v in self.rules.items():
            print(f'director.rules[k] {sys.getrefcount(v)}')
            for r in v:
                print(f'director.rules[k][i] {sys.getrefcount(r)}')
                r.check_refcounts()

    def evaluate(self, event:Event, character_candidates:Iterable[CharacterCandidate]) -> List[Action]:
        cdef vector[cCharacterCandidate] c_candidates
        #TODO: construct cCharacterCandidate instances from character_candidates
        for candidate in character_candidates:
            c_candidates.push_back(cCharacterCandidate(
                (<EventContext?>candidate.character_context).event_context,
                <PyObject*>candidate
            ))

        cdef vector[cAction] c_actions = self.director.evaluate(&((<Event?>event).event), c_candidates)
        actions = []
        cdef size_t i = 0
        while i < c_actions.size():
            character = <object>c_actions[i].character_candidate.data
            args = <object>c_actions[i].args
            actions.append(Action(
                c_actions[i].action_id,
                character,
                args
            ))
            i += 1
        return actions
