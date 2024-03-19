from libcpp cimport utility
from libcpp.memory cimport unique_ptr

"""
TODO: figure out how to expose goap to python
cdef extern from "goap.hpp":
    cdef cppclass Goal:
        int value

    cdef cppclass Action:
        float c

    cdef cppclass PlanningMap:
        PlanningMap()

cdef extern from "astar.hpp":
    cdef cppclass AStarNode[State, Edge]:
        unique_ptr[State] state
        utility.pair[unique_ptr[Edge], const AStarNode*] parent

    cdef cppclass AStar[State, Edge, Map]:
        AStar()
        AStarNode* run_astar(Map* m)

cdef class GOAPPlanner:
    cdef PlanningMap planning_map

    def astar(self):
        cdef AStar[Goal, Action, PlanningMap] astar
        cdef const AStarNode[Goal, Action]* solution = astar.run_astar(&self.planning_map)
        if solution  == NULL:
            print("NULL")
        else:
            print("not NULL")
            while solution != NULL:
                print(f'{solution.state.get().value} -> {solution.parent.first.get().c}')
                solution = solution.parent.second
"""
