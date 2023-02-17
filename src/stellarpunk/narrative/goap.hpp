#include <utility>
#include <vector>
#include <string>

struct Goal {
    int value;
    Goal(int v) : value(v) {}
    friend bool operator<(const Goal& lhs, const Goal& rhs) {
        // TODO: less than
        return lhs.value < rhs.value;
    }

    std::string to_string() {
        return std::to_string(value);
    }
};

struct Action {
    float c;

    Action(float cost) : c(cost) {}

    float cost() const {
        return c;
    }
};

class PlanningMap {
    public:
        PlanningMap() {}
        /*PlanningMap(std::unique_ptr<Goal> starting_goal, std::unique_ptr<TBD> starting_state) {
        }*/
        ~PlanningMap() {}

        bool satisfied(Goal* goal) {
            // TODO: is given goal satisfied by the actual world state
            return goal->value == 42;
        }

        float heuristic_cost(Goal* goal) {
            return (float)(42 - goal->value);
        }

        std::pair<std::unique_ptr<Action>, std::unique_ptr<Goal> > initial_state() {
            return std::make_pair(std::make_unique<Action>(0.0), std::make_unique<Goal>(0));
        }

        std::vector<std::pair<std::unique_ptr<Action>, std::unique_ptr<Goal> > > neighbors(const Goal* goal) {
            std::vector<std::pair<std::unique_ptr<Action>, std::unique_ptr<Goal> > > neighbors;
            // TODO: find neighbors of state: actions and corresponding goals
            // that, if we took that action from that state, would result in
            // the given goal
            neighbors.push_back(std::make_pair(
                std::make_unique<Action>(1.0),
                std::make_unique<Goal>(goal->value+1)
            ));
            if(42 - goal->value > 5) {
                neighbors.push_back(std::make_pair(
                    std::make_unique<Action>(2.0),
                    std::make_unique<Goal>(goal->value+5)
                ));
            }
            return neighbors;
        }

    protected:
        std::unique_ptr<Goal> starting_goal;
        //std::unique_ptr<TBD> starting_state;
};
