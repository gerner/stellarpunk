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

#include "director.hpp"

namespace stellarpunk {
namespace narrative {


std::unique_ptr<cCriteria<cIntRef, cFlagRef, cIntRef>> clone_criteria(const cCriteria<cIntRef, cFlagRef, cIntRef> &cri) {
    return std::make_unique<cCriteria<cIntRef, cFlagRef, cIntRef>>(cri.low, cri.fact, cri.high);
}

const std::uint64_t k_POS_INF = std::numeric_limits<std::uint64_t>::max();

struct NonZeroDistance {
    float operator ()(const std::uint64_t& k, const std::uint64_t& d) const {
        return d > 0 ? 1 : 0;
    }
};

struct Goal {
    std::array<cCriteria<cIntRef, cFlagRef, cIntRef>, 6> criteria_;

    Goal() {}

    template<class T>
    Goal(T &criteria) {
        for(auto &c : criteria) {
            if(c) {
                criteria_[c.key()] = c;
            }
        }
    }

    bool satisfied(cEvent* state, cEventContext* character_context) const {
        for(const auto &c : criteria_) {
            if(!c) {
                continue;
            }
            if(!c.evaluate(state, character_context)) {
                //printf("failed criteria %lu\n", c.fact.fact);
                return false;
            }
        }

        //printf("passed all criteria\n");
        return true;
    }

    std::uint64_t low(std::uint64_t fact) const {
        if(criteria_[fact]) {
            return criteria_[fact].low.value;
        } else {
            return 0;
        }
    }

    std::uint64_t high(std::uint64_t fact) const {
        if(criteria_[fact]) {
            return criteria_[fact].high.value;
        } else {
            return k_POS_INF;
        }
    }

    void remove(std::uint64_t fact) {
        if(criteria_[fact]) {
            //TODO: do we always want to access the fact directly here?
            criteria_[fact].fact = 0;
        }
    }

    void exactly(std::uint64_t fact, std::uint64_t amount) {
        if(criteria_[fact]) {
            criteria_[fact].low.value = amount;
            criteria_[fact].high.value = amount;
        } else {
            criteria_[fact] = cCriteria<cIntRef, cFlagRef, cIntRef>(amount, fact, amount);
        }
    }

    void at_least(std::uint64_t fact, std::uint64_t amount) {
        if(criteria_[fact]) {
            if(criteria_[fact].low.value < amount) {
                criteria_[fact].low.value = amount;
            }
            if(criteria_[fact].high.value < amount) {
                criteria_[fact].high.value = amount;
            }
        } else {
            criteria_[fact] = cCriteria<cIntRef, cFlagRef, cIntRef>(amount, fact, k_POS_INF);
        }
    }

    void inc(std::uint64_t fact, std::uint64_t amount) {
        if(criteria_[fact]) {
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
            criteria_[fact] = cCriteria<cIntRef, cFlagRef, cIntRef>(amount, fact, k_POS_INF);
        }
    }

    void dec(std::uint64_t fact, std::uint64_t amount) {
        if(criteria_[fact]) {
            if(criteria_[fact].low.value > amount) {
                criteria_[fact].low.value -= amount;
            } else if(criteria_[fact].high.value < k_POS_INF) {
                criteria_[fact].low.value = 0;
            } else {
                criteria_[fact].fact = 0;
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
        criteria_[c.key()] = c;
    }

    std::unique_ptr<Goal> clone() const {
        std::unique_ptr<Goal> goal = std::make_unique<Goal>();
        for(const auto& c : criteria_) {
            if(c) {
                goal->add(c);
            }
        }

        return std::move(goal);
    }

    friend bool operator<(const Goal& lhs, const Goal& rhs) {
        for(int i=0; i<lhs.criteria_.size(); i++) {
            if(!lhs.criteria_[i] && !rhs.criteria_[i]) {
                // neither has the fact
                continue;
            } else if(!lhs.criteria_[i]) {
                // rhs has it but lhs does not
                return false;
            } else if (!rhs.criteria_[i]) {
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
            if(!lhs.criteria_[i] && !rhs.criteria_[i]) {
                // neither has the fact
                continue;
            } else if(!lhs.criteria_[i]) {
                // rhs has it but lhs does not
                return false;
            } else if (!rhs.criteria_[i]) {
                // lhs has it but rhs does not
                return false;
            } else {
                // they both have it
                if(lhs.criteria_[i].low.value < rhs.criteria_[i].low.value) {
                    return false;
                } else if(lhs.criteria_[i].low.value > rhs.criteria_[i].low.value) {
                    return false;
                } else if(lhs.criteria_[i].high.value < rhs.criteria_[i].high.value) {
                    return false;
                } else if(lhs.criteria_[i].high.value > rhs.criteria_[i].high.value) {
                    return false;
                } else {
                    //low and high are the same
                    continue;
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
std::string to_string(const Goal& g, const char** fact_names=NULL) {
    /*if(g.criteria_.empty()) {
        return std::string("[ ]");
    }*/
    // TODO: represent a goal
    std::ostringstream s;
    s << "[ ";
    for(const auto& c: g.criteria_) {
        //s << to_string(*c.second) << ", ";
        if(c) {
            s << to_string(c, fact_names) << ", ";
        }
    }
    s << "]";
    return s.str();
}


template <class Action, class Distance=NonZeroDistance>
class ActionFactory {
    public:
        ActionFactory() {}
        virtual ~ActionFactory() {}

        virtual bool compatible(const Goal *desired_goal) = 0;
        virtual std::pair<std::unique_ptr<Action>, std::unique_ptr<Goal> > neighbor(const Goal* desired_goal) = 0;
};

template <class Action, class Distance=NonZeroDistance>
class PlanningMap {
    public:
        PlanningMap() = delete;
        PlanningMap(
            Goal* starting_goal, cEvent* starting_state,
            cEventContext* character_context,
            const std::vector<ActionFactory<Action>*> action_factories
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
            for(auto &c : goal->criteria_) {
                if(c) {
                    std::uint64_t d = c.distance(starting_state_, character_context_);
                    distance += Distance{}(c.key(), d);
                }
            }
            return distance;
        }

        std::pair<std::unique_ptr<Action>, std::unique_ptr<Goal> > initial_state() {
            return {std::make_unique<Action>(), starting_goal_->clone()};
        }

        std::vector<std::pair<std::unique_ptr<Action>, std::unique_ptr<Goal> > > neighbors(const Goal* goal) {
            std::vector<std::pair<std::unique_ptr<Action>, std::unique_ptr<Goal> > > neighbors;
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
        std::vector<ActionFactory<Action>*> action_factories_;
};

} // namespace narrative
} // namespace stellarpunk

template<>
struct std::hash<stellarpunk::narrative::Goal>
{
    std::size_t operator()(const stellarpunk::narrative::Goal& g) const noexcept
    {
        size_t result = 0;
        for(const auto& cri : g.criteria_) {
            if(cri) {
                result = result * 31 + (size_t)cri.low.value;
                result = result * 31 + (size_t)cri.fact.fact;
                result = result * 31 + (size_t)cri.high.value;
            }
        }
        return result;
    }
};

#endif /* NARRATIVE_GOAP_H */
