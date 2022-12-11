#define PY_SSIZE_T_CLEAN
#include <cassert>
#include <unordered_map>
#include <set>
#include <Python.h>
#include <stdio.h>


struct ScheduledTask {
    double timestamp;
    PyObject* task;

    ScheduledTask(double ts, PyObject *t) {
        timestamp = ts;
        task = t;
    }
};

bool operator<(const ScheduledTask& lhs, const ScheduledTask& rhs) {
    if(lhs.timestamp == rhs.timestamp) {
        return lhs.task < rhs.task;
    } else {
        return lhs.timestamp < rhs.timestamp;
    }
}

class cTaskSchedule {
    private:
        std::set<ScheduledTask> task_schedule;
        std::unordered_map<PyObject *, double> tasks;

    public:
        ~cTaskSchedule() {
            // decrementing ref counts of everything in the schedule
            for(auto it = task_schedule.begin(); it != task_schedule.end();) {
                Py_XDECREF(it->task);
                it = task_schedule.erase(it);
            }
        }

        size_t count(PyObject *task) {
            return tasks.count(task);
        }

        bool empty(double timestamp) const {
            return task_schedule.empty() || task_schedule.begin()->timestamp > timestamp;
        }

        void push(double timestamp, PyObject* task) {
            Py_XINCREF(task);
            task_schedule.insert(ScheduledTask(timestamp, task));
            tasks[task] = timestamp;
        }

        PyObject* top() const {
            if(task_schedule.empty()) {
                return NULL;
            } else {
                return task_schedule.begin()->task;
            }
        }

        void pop() {
            PyObject *t = task_schedule.begin()->task;
            task_schedule.erase(task_schedule.begin());
            tasks.erase(t);
            Py_XDECREF(t);
        }

        size_t erase(PyObject *task) {
            assert(tasks.count(task) == 1);
            ScheduledTask t = ScheduledTask(tasks[task], task);
            assert(task_schedule.count(t) == 1);
            task_schedule.erase(t);
            tasks.erase(task);
            Py_XDECREF(task);
            return 1;
        }
};
