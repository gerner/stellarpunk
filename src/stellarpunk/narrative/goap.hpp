#ifndef NARRATIVE_GOAP_H
#define NARRATIVE_GOAP_H

#include <utility>
#include <vector>
#include <string>
#include <memory>
#include <sstream>
#include <array>
#include <cassert>
#include <functional>
#include <bitset>

#include "director.hpp"

namespace stellarpunk {
namespace narrative {


const std::uint64_t k_POS_INF = std::numeric_limits<std::uint64_t>::max();

struct NonZeroDistance {
    float operator ()(const std::uint64_t& k, const std::uint64_t& d) const {
        return d > 0 ? 1 : 0;
    }
};

template <size_t N_CRITERIA>
struct Goal {
    std::bitset<N_CRITERIA> set_criteria_;
    std::array<cCriteria<cIntRef, cFlagRef, cIntRef>, N_CRITERIA> criteria_;

    Goal() {}

    template<class T>
    Goal(T &criteria) {
        for(auto &c : criteria) {
            if(c) {
                set_criteria_[c.key()] = true;
                criteria_[c.key()] = c;
            }
        }
    }

    bool satisfied(cEvent* state, cEventContext* character_context) const {
        for(int i=0; i<criteria_.size(); i++) {
            if(!set_criteria_[i]) {
                continue;
            }
            if(!criteria_[i].evaluate(state, character_context)) {
                //printf("failed criteria %lu\n", c.fact.fact);
                return false;
            }
        }

        //printf("passed all criteria\n");
        return true;
    }

    std::uint64_t low(std::uint64_t fact) const {
        if(set_criteria_[fact]) {
            return criteria_[fact].low.value;
        } else {
            return 0;
        }
    }

    std::uint64_t high(std::uint64_t fact) const {
        if(set_criteria_[fact]) {
            return criteria_[fact].high.value;
        } else {
            return k_POS_INF;
        }
    }

    void remove(std::uint64_t fact) {
        if(set_criteria_[fact]) {
            //TODO: do we always want to access the fact directly here?
            set_criteria_[fact] = false;
        }
    }

    void exactly(std::uint64_t fact, std::uint64_t amount) {
        if(set_criteria_[fact]) {
            criteria_[fact].low.value = amount;
            criteria_[fact].high.value = amount;
        } else {
            set_criteria_[fact] = true;
            criteria_[fact] = cCriteria<cIntRef, cFlagRef, cIntRef>(amount, fact, amount);
        }
    }

    void at_least(std::uint64_t fact, std::uint64_t amount) {
        if(set_criteria_[fact]) {
            if(criteria_[fact].low.value < amount) {
                criteria_[fact].low.value = amount;
            }
            if(criteria_[fact].high.value < amount) {
                criteria_[fact].high.value = amount;
            }
        } else {
            set_criteria_[fact] = true;
            criteria_[fact] = cCriteria<cIntRef, cFlagRef, cIntRef>(amount, fact, k_POS_INF);
        }
    }

    void inc(std::uint64_t fact, std::uint64_t amount) {
        if(set_criteria_[fact]) {
            criteria_[fact].low.value += amount;
            if(criteria_[fact].high.value == k_POS_INF) {
                return;
            } else if(k_POS_INF - criteria_[fact].high.value < amount) {
                criteria_[fact].high.value += amount;
            } else {
                //uh oh!
                criteria_[fact].high.value = k_POS_INF;
            }
        } else {
            set_criteria_[fact] = true;
            criteria_[fact] = cCriteria<cIntRef, cFlagRef, cIntRef>(amount, fact, k_POS_INF);
        }
    }

    void dec(std::uint64_t fact, std::uint64_t amount) {
        if(set_criteria_[fact]) {
            if(criteria_[fact].low.value > amount) {
                criteria_[fact].low.value -= amount;
            } else if(criteria_[fact].high.value < k_POS_INF) {
                criteria_[fact].low.value = 0;
            } else {
                set_criteria_[fact] = false;
                return;
            }

            if(criteria_[fact].high.value < k_POS_INF) {
                criteria_[fact].high.value -= amount;
            }
        } else {
            return;
        }
    }

    void add(cCriteria<cIntRef, cFlagRef, cIntRef> c) {
        set_criteria_[c.key()] = true;
        criteria_[c.key()] = c;
    }

    std::unique_ptr<Goal> clone() const {
        std::unique_ptr<Goal> goal = std::make_unique<Goal>();
        for(int i=0; i<criteria_.size(); i++) {
            if(set_criteria_[i]) {
                goal->add(criteria_[i]);
            }
        }

        return std::move(goal);
    }

    friend bool operator<(const Goal& lhs, const Goal& rhs) {
        for(int i=0; i<lhs.criteria_.size(); i++) {
            if(!lhs.set_criteria_[i] && !rhs.set_criteria_[i]) {
                // neither has the fact
                continue;
            } else if(!lhs.set_criteria_[i]) {
                // rhs has it but lhs does not
                return false;
            } else if (!rhs.set_criteria_[i]) {
                // lhs has it but rhs does not
                return true;
            } else {
                // they both have it
                if(lhs.criteria_[i].low.value < rhs.criteria_[i].low.value) {
                    return true;
                } else if(lhs.criteria_[i].low.value > rhs.criteria_[i].low.value) {
                    return false;
                } else if(lhs.criteria_[i].high.value < rhs.criteria_[i].high.value) {
                    return true;
                } else if(lhs.criteria_[i].high.value > rhs.criteria_[i].high.value) {
                    return false;
                } else {
                    //low and high are the same
                    continue;
                }
            }
        }
        // all present criteria were the same (and so lhs is not less)
        return false;
    }

    friend bool operator==(const Goal& lhs, const Goal& rhs) {
        for(int i=0; i<lhs.criteria_.size(); i++) {
            if(!lhs.set_criteria_[i] && !rhs.set_criteria_[i]) {
                // neither has the fact
                continue;
            } else if(!lhs.set_criteria_[i]) {
                // rhs has it but lhs does not
                return false;
            } else if (!rhs.set_criteria_[i]) {
                // lhs has it but rhs does not
                return false;
            } else {
                // they both have it
                if(lhs.criteria_[i].low.value == rhs.criteria_[i].low.value &&
                        lhs.criteria_[i].high.value == rhs.criteria_[i].high.value) {
                    continue;
                } else {
                    return false;
                }
            }
        }
        // all present criteria were the same (and so lhs is not less)
        return true;
    }
};

std::string to_string(const cCriteria<cIntRef, cFlagRef, cIntRef>& c, const char** fact_names) {
    std::ostringstream s;
    s << c.low.value << " <= ";
    if(fact_names) {
        s << fact_names[(size_t)c.fact.fact];
    } else {
        s << c.fact.fact;
    }

    if(c.high.value < k_POS_INF) {
        s << " <= " << c.high.value;
    }
    return s.str();
}

template<size_t N>
std::string to_string(const Goal<N>& g, const char** fact_names=NULL) {
    /*if(g.criteria_.empty()) {
        return std::string("[ ]");
    }*/
    // TODO: represent a goal
    std::ostringstream s;
    s << "[ ";
    for(int i=0; i<g.criteria_.size(); i++) {
        //s << to_string(*c.second) << ", ";
        if(g.set_criteria_[i]) {
            s << to_string(g.criteria_[i], fact_names) << ", ";
        }
    }
    s << "]";
    return s.str();
}


template <class Action, class Goal, class Distance=NonZeroDistance>
class ActionFactory {
    public:
        ActionFactory() {}
        virtual ~ActionFactory() {}

        virtual bool compatible(const Goal *desired_goal) = 0;
        virtual std::pair<std::unique_ptr<Action>, std::unique_ptr<Goal> > neighbor(const Goal* desired_goal) = 0;
};

template <class Action, class Goal, class Distance=NonZeroDistance>
class PlanningMap {
    public:
        PlanningMap() = delete;
        PlanningMap(
            Goal* starting_goal, cEvent* starting_state,
            cEventContext* character_context,
            const std::vector<ActionFactory<Action, Goal>*> action_factories
        )
            : starting_goal_(starting_goal),
              starting_state_(starting_state),
              character_context_(character_context),
              action_factories_(action_factories) { }
        ~PlanningMap() {}

        bool satisfied(Goal* goal) {
            // TODO: is given goal satisfied by the actual world state
            return goal->satisfied(starting_state_, character_context_);
        }

        float heuristic_cost(Goal* goal) {
            float distance = 0.0f;
            for(int i=0; i<goal->criteria_.size(); i++) {
                if(goal->set_criteria_[i]) {
                    std::uint64_t d = goal->criteria_[i].distance(starting_state_, character_context_);
                    distance += Distance{}(goal->criteria_[i].key(), d);
                }
            }
            return distance;
        }

        std::pair<std::unique_ptr<Action>, std::unique_ptr<Goal> > initial_state() {
            return {std::make_unique<Action>(), starting_goal_->clone()};
        }

        std::vector<std::pair<std::unique_ptr<Action>, std::unique_ptr<Goal>>> neighbors(const Goal* goal) {
            std::vector<std::pair<std::unique_ptr<Action>, std::unique_ptr<Goal>>> neighbors;
            for(auto& factory : action_factories_) {
                if(!factory->compatible(goal)) {
                    continue;
                }

                neighbors.emplace_back(factory->neighbor(goal));
            }
            return neighbors;
        }

    protected:
        Goal* starting_goal_;
        cEvent* starting_state_;
        cEventContext* character_context_;
        std::vector<ActionFactory<Action, Goal>*> action_factories_;
};

} // namespace narrative
} // namespace stellarpunk

template<size_t N>
struct std::hash<stellarpunk::narrative::Goal<N>>
{
    std::size_t operator()(const stellarpunk::narrative::Goal<N>& g) const noexcept
    {
        size_t result = 0;
        for(int i=0; i<g.criteria_.size(); i++) {
            if(g.set_criteria_[i]) {
                result = result * 31 + (size_t)g.criteria_[i].low.value;
                result = result * 31 + (size_t)g.criteria_[i].fact.fact;
                result = result * 31 + (size_t)g.criteria_[i].high.value;
            }
        }
        return result;
    }
};

#endif /* NARRATIVE_GOAP_H */
