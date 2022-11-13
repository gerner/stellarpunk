#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <queue>


struct ScheduledTask {
    double timestamp;
    PyObject* task;

    ScheduledTask(double ts, PyObject *t) {
        timestamp = ts;
        task = t;
    }
};

bool operator<(const ScheduledTask& lhs, const ScheduledTask& rhs) {
    // lower timestamps have higher priority
    return lhs.timestamp > rhs.timestamp;
}

class cTaskSchedule {
    private:
        std::priority_queue<ScheduledTask> task_schedule;

    public:
        ~cTaskSchedule() {
            while(!task_schedule.empty()) {
                Py_XDECREF(task_schedule.top().task);
                task_schedule.pop();
            }
        }
        bool empty(double timestamp) const {
            return task_schedule.empty() || task_schedule.top().timestamp > timestamp;
        }

        void push(double timestamp, PyObject* task) {
            Py_XINCREF(task);
            task_schedule.push(ScheduledTask(timestamp, task));
        }

        PyObject* top() const {
            if(task_schedule.empty()) {
                return NULL;
            } else {
                return task_schedule.top().task;
            }
        }

        void pop() {
            Py_XDECREF(task_schedule.top().task);
            task_schedule.pop();
        }

};
