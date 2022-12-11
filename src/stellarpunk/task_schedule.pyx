# cython: boundscheck=False, wraparound=False, cdivision=True, infer_types=False, nonecheck=False

from libcpp cimport bool
from libcpp.set cimport set
from cpython.ref cimport PyObject

cdef extern from "task_schedule.hpp":
    # note this class handles ref counting
    cdef cppclass cTaskSchedule:
        size_t count(PyObject* task)
        bool empty(double timestamp)
        void push(double timestamp, PyObject* task)
        PyObject* top()
        PyObject* pop()
        size_t erase(PyObject *task)

cdef class TaskSchedule:
    cdef cTaskSchedule schedule

    def empty(self, timestamp):
        return self.schedule.empty(timestamp)

    def top(self):
        cdef PyObject *t = self.schedule.top()
        if t == NULL:
            return None
        else:
            return <object>t

    def is_task_scheduled(self, task):
        return self.schedule.count(<PyObject *>task) > 0

    def cancel_task(self, task):
        if self.schedule.count(<PyObject *>task) == 0:
            return
        self.schedule.erase(<PyObject *> task)

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
