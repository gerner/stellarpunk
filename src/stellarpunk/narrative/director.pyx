from libc.stdint cimport uint64_t
from libcpp cimport bool
from libcpp.unordered_map  cimport unordered_map
from libcpp.vector  cimport vector


cdef extern from "director.hpp":
    ctypedef unordered_map[uint64_t, uint64_t] cEventContext

    struct cEvent:
        uint64_t event_type;
        cEventContext* character_context;
        cEventContext* event_context;
        unordered_map[uint64_t, cEventContext*] entity_context;

        cEvent()

    cdef cppclass cFlagCriteria:
        cFlagCriteria()
        cFlagCriteria(uint64_t f, uint64_t l, uint64_t h)

    cdef cppclass cEntityCriteria:
        cEntityCriteria()
        cEntityCriteria(uint64_t ef, uint64_t sf, uint64_t l, uint64_t h)

    cdef cppclass cRule:
        cRule()
        cRule(
            uint64_t rid,
            uint64_t et,
            uint64_t pri,
            vector[cFlagCriteria] cri,
            vector[cEntityCriteria] e_cri,
        )
        bool evaluate(cEvent &event)

    cdef cppclass cDirector:
        pass


cdef class FlagCriteria:
    cdef cFlagCriteria criteria

    def __cinit__(self, f, l, h):
        self.criteria = cFlagCriteria(f, l, h)


cdef class EntityCriteria:
    cdef cEntityCriteria entity_criteria

    def __cinit__(self, ef, sf, l, h):
        self.entity_criteria = cEntityCriteria(ef, sf, l, h)


cdef class Rule:
    cdef cRule rule

    def __cinit__(self, rule_id, event_type, priority, criteria=[], entity_criteria=[]):
        cdef vector[cFlagCriteria] c_criteria
        cdef vector[cEntityCriteria] c_entity_criteria
        cdef vector[uint64_t] c_actions

        for c in criteria:
            c_criteria.push_back((<FlagCriteria?>c).criteria)

        for ec in entity_criteria:
            c_entity_criteria.push_back((<EntityCriteria?>ec).entity_criteria)

        self.rule = cRule(rule_id, event_type, priority, c_criteria, c_entity_criteria)

        self.rule_id = rule_id


cdef class EventContext:
    cdef cEventContext event_context;

    def set_flag(self, flag, value):
        self.event_context[flag] = value

    def get_flag(self, flag):
        return self.event_context[flag]


cdef class Event:
    cdef cEvent event

    def __cinit__(self, event_type, event_context, entity_context):
        self.event = cEvent()
        self.event.event_type = event_type
        self.event.event_context = &((<EventContext?>event_context).event_context)

        for k, v in entity_context.items():
            self.event.entity_context[k] = &((<EventContext?>v).event_context)

        # character context gets handled separately


cdef class Director:
    cdef cDirector director

