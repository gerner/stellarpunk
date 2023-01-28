# cython: boundscheck=False, wraparound=False, cdivision=True, infer_types=False, nonecheck=False

from libc.stdint cimport uint64_t
from libcpp cimport bool
from libcpp.unordered_map  cimport unordered_map
from libcpp.vector  cimport vector
from cpython.ref cimport PyObject
from cython.operator cimport dereference, preincrement

from typing import Tuple, Dict, Any, Iterable, List


cdef extern from "director.hpp":
    ctypedef unordered_map[uint64_t, uint64_t] cEventContext

    struct cEvent:
        uint64_t event_type
        cEventContext* event_context
        unordered_map[uint64_t, cEventContext*] entity_context
        void* data

        cEvent(uint64_t et, cEventContext* ec, unordered_map[uint64_t, cEventContext*] ent_c, void* data)

    cdef cppclass cFlagRef:
        uint64_t fact
        cFlagRef(uint64_t f)

    cdef cppclass cEntityRef:
        uint64_t entity_fact
        uint64_t sub_fact
        cEntityRef(uint64_t ef, uint64_t sf)

    cdef cppclass cCriteria[T]:
        T fact
        uint64_t low
        uint64_t high
        cCriteria()
        cCriteria(T f, uint64_t l, uint64_t h)

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
            vector[cCriteria[cFlagRef]] cri,
            vector[cCriteria[cEntityRef]] e_cri,
            vector[cActionTemplate] a
        )
        bool evaluate(cEvent &event)

    cdef cppclass cDirector:
        cDirector()
        cDirector(unordered_map[uint64_t, vector[cRule]] r)
        vector[cAction] evaluate(cEvent* event, vector[cCharacterCandidate] character_candidates)


cdef class EventContext:
    cdef cEventContext event_context;

    def set_flag(self, flag, value):
        self.event_context[flag] = value

    def get_flag(self, flag):
        return self.event_context[flag]


def context(c:Dict[int, int]) -> EventContext:
    e = EventContext()
    for k,v in c.items():
        e.set_flag(k,v)
    return e


cdef class FlagCriteria:
    cdef cCriteria[cFlagRef] criteria

    def __cinit__(self, f, l, h):
        self.criteria = cCriteria[cFlagRef](cFlagRef(f), l, h)

    @property
    def fact(self) -> int:
        return self.criteria.fact.fact

    @property
    def low(self) -> int:
        return self.criteria.low

    @property
    def high(self) -> int:
        return self.criteria.high


cdef class EntityCriteria:
    cdef cCriteria[cEntityRef] entity_criteria

    def __cinit__(self, ef, sf, l, h):
        self.entity_criteria = cCriteria[cEntityRef](cEntityRef(ef, sf), l, h)

    @property
    def entity_fact(self) -> int:
        return self.entity_criteria.fact.entity_fact

    @property
    def sub_fact(self) -> int:
        return self.entity_criteria.fact.sub_fact

    @property
    def low(self) -> int:
        return self.entity_criteria.low

    @property
    def high(self) -> int:
        return self.entity_criteria.high


cdef class ActionTemplate:
    cdef cActionTemplate action_template
    cdef object args

    def __cinit__(self, action_id, args):
        cdef PyObject* c_args = <PyObject*>args
        self.action_template = cActionTemplate(action_id, c_args)

        # we hang on to a reference of the arguments to keep them alive
        self.args = args


cdef class Rule:
    cdef cRule rule
    cdef object actions

    def __cinit__(self, event_type, priority, criteria=[], entity_criteria=[], actions=[]):
        cdef vector[cCriteria[cFlagRef]] c_criteria
        cdef vector[cCriteria[cEntityRef]] c_entity_criteria
        cdef vector[cActionTemplate] c_actions

        for c in criteria:
            c_criteria.push_back((<FlagCriteria?>c).criteria)

        for ec in entity_criteria:
            c_entity_criteria.push_back((<EntityCriteria?>ec).entity_criteria)

        for a in actions:
            c_actions.push_back((<ActionTemplate?>a).action_template)

        self.rule = cRule(event_type, priority, c_criteria, c_entity_criteria, c_actions)

        # we hang on to action templates to keep them alive
        self.actions = list(actions)


cdef class Event:
    cdef cEvent event
    cdef object args

    def __cinit__(self, event_type: int, event_context: EventContext, entity_context: Dict[int, EventContext], args: Any):
        cdef unordered_map[uint64_t, cEventContext*] c_entity_context
        cdef PyObject* c_args = <PyObject *>args

        for k, v in entity_context.items():
            c_entity_context[k] = &((<EventContext?>v).event_context)

        self.event = cEvent(event_type, &((<EventContext?>event_context).event_context), c_entity_context, c_args)

        # we hang on to event args to keep them alive
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

    def __cinit__(self, rules:Dict[int, Iterable[Rule]]):
        cdef unordered_map[uint64_t, vector[cRule]] c_rules;
        cdef vector[cRule] c_rule_list

        for k, v in rules.items():
            c_rule_list.clear()
            for r in v:
                c_rule_list.push_back((<Rule?>r).rule)
            c_rules[k] = c_rule_list

        self.director = cDirector(c_rules)

        # we hang on to the python wrapper for the rules to keep them alive
        self.rules = list(rules)

    def evaluate(self, event:Event, character_candidates:Iterable[CharacterCandidate]) -> List[Action]:
        cdef vector[cCharacterCandidate] c_candidates
        #TODO: construct cCharacterCandidate instances from character_candidates
        for candidate in character_candidates:
            c_candidates.push_back(cCharacterCandidate(
                &((<EventContext?>candidate.character_context).event_context),
                <PyObject*>candidate
            ))

        cdef vector[cAction] c_actions = self.director.evaluate(&((<Event?>event).event), c_candidates)
        #TODO: construct Action instances from actions
        actions = []
        cdef vector[cAction].iterator ptr = c_actions.begin()
        while ptr != c_actions.end():
            action = Action(dereference(ptr).action_id, <object>dereference(ptr).character_candidate.data, <object>dereference(ptr).args)
            actions.append(action)
            preincrement(ptr)
        return actions
