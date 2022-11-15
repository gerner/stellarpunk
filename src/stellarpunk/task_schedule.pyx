# cython: boundscheck=False, wraparound=False, cdivision=True, infer_types=False, nonecheck=False

from libcpp cimport bool
from libcpp.set cimport set
from cpython.ref cimport PyObject

cdef extern from "task_schedule.hpp":
    # note this class handles ref counting
    cdef cppclass cTaskSchedule:
        bool empty(double timestamp)
        void push(double timestamp, PyObject* task)
        PyObject* top()
        PyObject* pop()

cdef class TaskSchedule:
    cdef cTaskSchedule schedule
    cdef set[PyObject*] scheduled_tasks

    def empty(self, timestamp):
        return self.schedule.empty(timestamp)

    def is_task_scheduled(self, task):
        return self.scheduled_tasks.count(<PyObject *>task) > 0

    def push_task(self, timestamp, task):
        self.schedule.push(timestamp, <PyObject *>task)
        self.scheduled_tasks.insert(<PyObject *> task)

    def pop_current_tasks(self, timestamp):
        """ Pops and returns all the current tasks as of timestamp """

        cdef double c_timestamp = <double?> timestamp
        tasks = []
        while not self.schedule.empty(c_timestamp):
            self.scheduled_tasks.erase(self.schedule.top())
            tasks.append(<object>self.schedule.top())
            self.schedule.pop()

        return tasks

