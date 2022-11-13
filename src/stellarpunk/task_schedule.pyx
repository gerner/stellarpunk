# cython: boundscheck=False, wraparound=False, cdivision=True, infer_types=False, nonecheck=False

from libcpp cimport bool
from cpython.ref cimport PyObject

cdef extern from "task_schedule.hpp":
    cdef cppclass cTaskSchedule:
        bool empty(double timestamp)
        void push(double timestamp, PyObject* task)
        PyObject* top()
        PyObject* pop()

cdef class TaskSchedule:
    cdef cTaskSchedule schedule

    def empty(self, timestamp):
        return self.schedule.empty(timestamp)

    def push_task(self, timestamp, task):
        self.schedule.push(timestamp, <PyObject *>task)

    def pop_current_tasks(self, timestamp):
        """ Pops and returns all the current tasks as of timestamp """

        cdef double c_timestamp = <double?> timestamp
        tasks = []
        while not self.schedule.empty(c_timestamp):
            tasks.append(<object>self.schedule.top())
            self.schedule.pop()

        return tasks

