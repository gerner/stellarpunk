# cython: boundscheck=False, wraparound=False, cdivision=True, infer_types=False, nonecheck=False

from libcpp cimport bool
from libcpp.set cimport set
from cpython.ref cimport PyObject
from cython.operator cimport dereference, preincrement

cdef extern from "task_schedule.hpp":
    cdef cppclass ScheduledTask:
        double timestamp
        PyObject* task

    # note this class handles ref counting
    cdef cppclass cTaskSchedule:
        size_t size()
        size_t count(PyObject* task)
        bool empty(double timestamp)
        void push(double timestamp, PyObject* task)
        PyObject* top()
        PyObject* pop()
        size_t erase(PyObject *task)
        set[ScheduledTask].iterator begin()
        set[ScheduledTask].iterator end()

cdef class TaskScheduleIterator:
    cdef set[ScheduledTask].iterator itr
    cdef set[ScheduledTask].iterator end

    def __next__(self):
        if self.itr == self.end:
            raise StopIteration()

        timestamp = dereference(self.itr).timestamp
        task = <object>dereference(self.itr).task
        preincrement(self.itr)
        return (timestamp, task)

cdef class TaskSchedule:
    cdef cTaskSchedule schedule

    def size(self):
        return self.schedule.size()

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

    def __iter__(self):
        cdef TaskScheduleIterator itr = TaskScheduleIterator()
        itr.itr = self.schedule.begin()
        itr.end = self.schedule.end()
        return itr
