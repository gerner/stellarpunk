#include <stdio.h>
#include <vector>
#include <memory>
#include <cstdint>
#include <limits>
#include <array>

#include "director.hpp"
#include "astar.hpp"
#include "goap.hpp"

using namespace stellarpunk;

const std::uint64_t POS_INF = std::numeric_limits<std::uint64_t>::max();

enum struct Location : std::uint64_t {
    k_invalid = 0ULL,
    k_none,
    k_woods,
    k_store
};

enum struct Fact : std::uint64_t {
    k_invalid = 0ULL,
    k_money,
    k_have_axe,
    k_forest,
    k_wood,
    k_location
};

const char* fact_names[] = {
    "INVALID",
    "money",
    "have_axe",
    "forest",
    "wood",
    "location"
};

struct FactDistance {
    float operator ()(const std::uint64_t& k, const std::uint64_t& d) const {
        switch(k) {
            case (std::uint64_t)Fact::k_money:
            case (std::uint64_t)Fact::k_forest:
                return d; // money, forest, wood cost 1:1
                break;
            case (std::uint64_t)Fact::k_wood:
                return d*0.7;
                break;
            case (std::uint64_t)Fact::k_have_axe:
                return d * 15; // axes cost 25 but can sell for 10
                break;
            case (std::uint64_t)Fact::k_location:
                return d>0 ? 1 : 0; // either at location or not
                break;
            default:
                assert(false); // shouldn't ever see anything else
        }
    }
};

enum struct ActionType : size_t {
    k_null = 0,
    k_grind_money,
    k_buy_axe,
    k_sell_axe,
    k_gather_wood,
    k_chop_wood,
    k_sell_wood,
    k_go_to
};

const char* action_names[] = {
    "NULL",
    "grind_money",
    "buy_axe",
    "sell_axe",
    "gather_wood",
    "chop_wood",
    "sell_wood",
    "go_to"
};

cCriteria<cIntRef, cFlagRef, cIntRef> make_criteria(
        std::uint64_t l,
        std::uint64_t f,
        std::uint64_t h) {
    return cCriteria<cIntRef, cFlagRef, cIntRef>(l, f, h);
}

struct Action {
    //TODO: TBD information about using this action later: type, parameters
    float cost_;
    ActionType action_type_;
    std::uint64_t action_param_;

    Action() : cost_(0.0f), action_type_(ActionType::k_null), action_param_(0) { }
    Action(float cost, ActionType at, std::uint64_t p) : cost_(cost), action_type_(at), action_param_(p) { }

    float cost() const {
        return cost_;
    }
};

std::string to_string(const Action& a) {
    std::ostringstream s;
    s << action_names[(size_t)a.action_type_] << "(" << a.action_param_ << ")";
    return s.str();

}

class BuyAxe : public narrative::ActionFactory<Action> {
    public:
    virtual bool compatible(const narrative::Goal *desired_goal) {
        // effect:
        //  get an axe
        //  spend 20 money
        //  will be at the store (but we need someone else to get us there)
        return (
            desired_goal->low((std::uint64_t)Fact::k_have_axe) > 0 ||
            desired_goal->high((std::uint64_t)Fact::k_money) < POS_INF
        ) && (
            desired_goal->low((std::uint64_t)Fact::k_location) == 0 ||
            desired_goal->low((std::uint64_t)Fact::k_location) == (std::uint64_t)Location::k_store
        );
    }

    virtual std::pair<std::unique_ptr<Action>, std::unique_ptr<narrative::Goal>> neighbor(
            const narrative::Goal* desired_goal) {
        std::unique_ptr<narrative::Goal> g = desired_goal->clone();
        g->exactly((std::uint64_t)Fact::k_location, (std::uint64_t)Location::k_store);
        g->inc((std::uint64_t)Fact::k_money, 20);
        g->dec((std::uint64_t)Fact::k_have_axe, 1);

        return {std::make_unique<Action>(20.0, ActionType::k_buy_axe, 1), std::move(g)};
    }
};

class SellAxe : public narrative::ActionFactory<Action> {
    public:
    virtual bool compatible(const narrative::Goal *desired_goal) {
        // effect:
        //  remove an axe
        //  get 10 money
        //  will be at the store (but we need someone else to get us there)
        return (
            desired_goal->high((std::uint64_t)Fact::k_have_axe) < POS_INF ||
            desired_goal->low((std::uint64_t)Fact::k_money) > 0
        ) && (
            desired_goal->low((std::uint64_t)Fact::k_location) == 0 ||
            desired_goal->low((std::uint64_t)Fact::k_location) == (std::uint64_t)Location::k_store
        );
    }

    virtual std::pair<std::unique_ptr<Action>, std::unique_ptr<narrative::Goal> > neighbor(
            const narrative::Goal* desired_goal) {
        std::unique_ptr<narrative::Goal> g = desired_goal->clone();
        g->exactly((std::uint64_t)Fact::k_location, (std::uint64_t)Location::k_store);
        g->dec((std::uint64_t)Fact::k_money, 10);
        g->inc((std::uint64_t)Fact::k_have_axe, 1);

        return { std::make_unique<Action>(10.0, ActionType::k_sell_axe, 1), std::move(g) };
    }
};

class GatherWood : public narrative::ActionFactory<Action> {
    public:
    GatherWood(std::uint64_t chop_amount=1, float chop_fraction=1.0) : chop_amount_(chop_amount), chop_fraction_(chop_fraction) { }

    virtual bool compatible(const narrative::Goal *desired_goal) {
        // effect:
        //  get wood
        //  remove forest
        //  will be at the woods (but we need someone else to get us there)
        return (
            amount_to_chop(desired_goal) > 0
        ) && (
            desired_goal->low((std::uint64_t)Fact::k_location) == 0 ||
            desired_goal->low((std::uint64_t)Fact::k_location) == (std::uint64_t)Location::k_woods
        );
    }

    virtual std::pair<std::unique_ptr<Action>, std::unique_ptr<narrative::Goal> > neighbor(
            const narrative::Goal* desired_goal) {

        std::uint64_t amount = amount_to_chop(desired_goal);

        std::unique_ptr<narrative::Goal> g = desired_goal->clone();
        g->exactly((std::uint64_t)Fact::k_location, (std::uint64_t)Location::k_woods);
        g->inc((std::uint64_t)Fact::k_forest, amount);
        g->dec((std::uint64_t)Fact::k_wood, amount);

        return { std::make_unique<Action>(10.0*amount, ActionType::k_gather_wood, amount), std::move(g) };
    }

    protected:
    std::uint64_t amount_to_chop(const narrative::Goal *desired_goal) const {
        std::uint64_t desired_wood = desired_goal->low((std::uint64_t)Fact::k_wood);
        if(desired_wood == 0) {
            return 0;
        } else if(chop_amount_ > 0) {
            return chop_amount_;
        } else {
            return desired_wood * chop_fraction_;
        }
    }

    std::uint64_t chop_amount_;
    float chop_fraction_;
};

class ChopWood : public narrative::ActionFactory<Action> {
    public:
    ChopWood(std::uint64_t chop_amount=1, float chop_fraction=1.0) : chop_amount_(chop_amount), chop_fraction_(chop_fraction) { }

    virtual bool compatible(const narrative::Goal *desired_goal) {
        // effect:
        //  get wood
        //  remove forest
        //  will be at the woods (but we need someone else to get us there)
        return (
            amount_to_chop(desired_goal) > 0
        ) && (
            desired_goal->low((std::uint64_t)Fact::k_location) == 0 ||
            desired_goal->low((std::uint64_t)Fact::k_location) == (std::uint64_t)Location::k_woods
        );
    }

    virtual std::pair<std::unique_ptr<Action>, std::unique_ptr<narrative::Goal> > neighbor(
            const narrative::Goal* desired_goal) {
        std::uint64_t amount = amount_to_chop(desired_goal);
        std::unique_ptr<narrative::Goal> g = desired_goal->clone();
        g->exactly((std::uint64_t)Fact::k_location, (std::uint64_t)Location::k_woods);
        g->inc((std::uint64_t)Fact::k_forest, amount);
        g->dec((std::uint64_t)Fact::k_wood, amount);
        g->at_least((std::uint64_t)Fact::k_have_axe, 1);

        return { std::make_unique<Action>(1.0*amount, ActionType::k_chop_wood, amount), std::move(g) };
    }

    protected:
    std::uint64_t amount_to_chop(const narrative::Goal *desired_goal) const {
        std::uint64_t desired_wood = desired_goal->low((std::uint64_t)Fact::k_wood);
        if(desired_wood == 0) {
            return 0;
        } else if(chop_amount_ > 0) {
            return chop_amount_;
        } else {
            return desired_wood * chop_fraction_;
        }
    }

    std::uint64_t chop_amount_;
    float chop_fraction_;
};

class SellWood : public narrative::ActionFactory<Action> {
    public:
    SellWood(std::uint64_t sell_amount=1, float sell_fraction=1.0) : sell_amount_(sell_amount), sell_fraction_(sell_fraction) { }

    virtual bool compatible(const narrative::Goal *desired_goal) {
        // effect:
        //  get money
        //  remove wood
        //  will be at the store (but we need someone else to get us there)
        return (
            amount_to_sell(desired_goal) > 0
        ) && (
            desired_goal->low((std::uint64_t)Fact::k_location) == 0 ||
            desired_goal->low((std::uint64_t)Fact::k_location) == (std::uint64_t)Location::k_store
        );
    }

    virtual std::pair<std::unique_ptr<Action>, std::unique_ptr<narrative::Goal> > neighbor(
            const narrative::Goal* desired_goal) {
        std::uint64_t amount = amount_to_sell(desired_goal);
        std::unique_ptr<narrative::Goal> g = desired_goal->clone();
        g->exactly((std::uint64_t)Fact::k_location, (std::uint64_t)Location::k_store);
        g->dec((std::uint64_t)Fact::k_money, 1);
        g->inc((std::uint64_t)Fact::k_wood, 1);

        return { std::make_unique<Action>(1.0*amount, ActionType::k_sell_wood, amount), std::move(g) };
    }
    protected:

    std::uint64_t amount_to_sell(const narrative::Goal *desired_goal) const {
        std::uint64_t desired_money = desired_goal->low((std::uint64_t)Fact::k_money);
        if(desired_money == 0) {
            return 0;
        } else if(sell_amount_ > 0) {
            return sell_amount_;
        } else {
            return desired_money * sell_fraction_;
        }
    }

    std::uint64_t sell_amount_;
    float sell_fraction_;
};

class GoTo : public narrative::ActionFactory<Action> {
    public:
    virtual bool compatible(const narrative::Goal *desired_goal) {
        // effect:
        //  set location to whatever the goal is
        return desired_goal->low((std::uint64_t)Fact::k_location) > 0;
    }

    virtual std::pair<std::unique_ptr<Action>, std::unique_ptr<narrative::Goal> > neighbor(
            const narrative::Goal* desired_goal) {
        std::unique_ptr<narrative::Goal> g = desired_goal->clone();
        std::uint64_t location = desired_goal->low((std::uint64_t)Fact::k_location);
        g->remove((std::uint64_t)Fact::k_location);

        return { std::make_unique<Action>(5.0, ActionType::k_go_to, location), std::move(g) };
    }
};

class GrindMoney : public narrative::ActionFactory<Action> {
    public:
    virtual bool compatible(const narrative::Goal *desired_goal) {
        return desired_goal->low((std::uint64_t)Fact::k_money) > 0;
    }

    virtual std::pair<std::unique_ptr<Action>, std::unique_ptr<narrative::Goal> > neighbor(
            const narrative::Goal* desired_goal) {
        std::unique_ptr<narrative::Goal> g = desired_goal->clone();
        g->dec((std::uint64_t)Fact::k_money, 1);

        return { std::make_unique<Action>(1.0, ActionType::k_grind_money, 1), std::move(g) };
    }
};

int main(int argc, char** argv) {
    printf("hello world!\n");

    // initial state
    cEvent event;
    cEventContext character_context;
    character_context[(std::uint64_t)Fact::k_money] = 40;
    character_context[(std::uint64_t)Fact::k_forest] = 100;

    // desired goal
    std::array<cCriteria<cIntRef, cFlagRef, cIntRef>, 6> cri;
    cri[(std::uint64_t)Fact::k_money] = make_criteria(
        50, (std::uint64_t)Fact::k_money, POS_INF
    );
    narrative::Goal starting_goal = narrative::Goal(cri);

    // action factories
    GrindMoney af_gm;
    BuyAxe af_ba;
    SellAxe af_sa;
    GatherWood af_gw_one;
    GatherWood af_gw_all(10);
    ChopWood af_cw_one;
    ChopWood af_cw_all(10);
    SellWood af_sw_one;
    SellWood af_sw_all(10);
    GoTo af_gt;

    narrative::PlanningMap<Action, FactDistance> map(
        &starting_goal,
        &event,
        &character_context,
        {
            &af_ba,
            &af_sa,
            &af_gw_one,
            //&af_gw_all,
            &af_cw_one,
            //&af_cw_all,
            &af_sw_one,
            //&af_sw_all,
            &af_gt
        }
    );
    narrative::AStar<narrative::Goal, Action, narrative::PlanningMap<Action, FactDistance>> astar;

    const narrative::AStarNode<narrative::Goal, Action>* solution;
    for(int i=0; i < 1000; i++) {
        solution = astar.run_astar(&map);
    }
    printf("open_set: %zu closed_set %zu\n", astar.open_set.size(), astar.closed_set.size());
    for(size_t i=0; i < astar.k_cnt_LEN; i++) {
        printf("%zu\t%lu\n", i, astar.counters_[i]);
    }

    if(solution == NULL) {
        printf("no solution found\n");
    } else {
        printf("solution cost %f\n", solution->g_score);
        while(solution != NULL) {
            printf(
                "%s %s with g_score: %f f_score: %f via %f\n",
                to_string(*solution->parent.first).c_str(),
                narrative::to_string(*solution->state, fact_names).c_str(),
                solution->g_score,
                solution->f_score,
                solution->parent.first->cost_
            );
            solution = solution->parent.second;
        }
    }
}
