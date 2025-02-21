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
const float FIXED_COST = 1e-4;

// Facts that describe the world

enum struct Fact : std::uint64_t {
    k_invalid = 0ULL,
    k_money,
    k_have_axe,
    k_forest,
    k_wood,
    k_location,
    k_COUNT
};

const char* fact_names[] = {
    "INVALID",
    "money",
    "have_axe",
    "forest",
    "wood",
    "location"
};

// location is a non-scalar fact with named states

enum struct Location : std::uint64_t {
    k_invalid = 0ULL,
    k_none,
    k_woods,
    k_store
};

const char* location_names[] = {
    "INVALID",
    "NONE",
    "woods",
    "store"
};

// heuristic estimate of the cost to move d points for one fact
// this must be a lower bound of the cost for A* to work

struct FactDistance {
    float operator ()(const std::uint64_t& k, float& d) const {
        switch(k) {
            case (std::uint64_t)Fact::k_forest:
                return 0.0;
            case (std::uint64_t)Fact::k_money:
            case (std::uint64_t)Fact::k_wood:
                // money, forest, wood cost 1 each
                return d>0?d:0.0;//std::abs(d);
            case (std::uint64_t)Fact::k_have_axe:
                // axes cost 20 but can sell for 10
                return d>0?d*10:-d;//std::abs(d);//*10;
            case (std::uint64_t)Fact::k_location:
                // either at location or not, going to a location costs 5
                return d == 0 ? 0.0 : 5.0;
            default:
                assert(false); // shouldn't ever see anything else
        }
    }
};

cCriteria<cIntRef, cFlagRef, cIntRef> make_criteria(
        std::uint64_t l,
        std::uint64_t f,
        std::uint64_t h) {
    return cCriteria<cIntRef, cFlagRef, cIntRef>(l, f, h);
}

// actions that can be taken

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

// logic that implements reasoning about and taking those actions
// these are "factories" that can create instances of actions taking us from
// state to another

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

using Goal = narrative::Goal<(size_t)Fact::k_COUNT>;

class BuyAxe : public narrative::ActionFactory<Action, Goal> {
    public:
    virtual bool compatible(const Goal *desired_goal) const {
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

    virtual std::pair<std::unique_ptr<Action>, std::unique_ptr<Goal>> neighbor(
            const Goal* desired_goal) const {
        std::unique_ptr<Goal> g = desired_goal->clone();
        g->exactly((std::uint64_t)Fact::k_location, (std::uint64_t)Location::k_store);
        g->inc((std::uint64_t)Fact::k_money, 20);
        g->dec((std::uint64_t)Fact::k_have_axe, 1);

        return {std::make_unique<Action>(1.0, ActionType::k_buy_axe, 1), std::move(g)};
    }
};

class SellAxe : public narrative::ActionFactory<Action, Goal> {
    public:
    virtual bool compatible(const Goal *desired_goal) const {
        // effect:
        //  remove an axe
        //  get 10 money
        //  will be at the store (but we need someone else to get us there)
        return (
            desired_goal->high((std::uint64_t)Fact::k_have_axe) < POS_INF ||
            (desired_goal->low((std::uint64_t)Fact::k_money) >= 0 && desired_goal->high((std::uint64_t)Fact::k_money) >= 10)
        ) && (
            desired_goal->low((std::uint64_t)Fact::k_location) == 0 ||
            desired_goal->low((std::uint64_t)Fact::k_location) == (std::uint64_t)Location::k_store
        );
    }

    virtual std::pair<std::unique_ptr<Action>, std::unique_ptr<Goal> > neighbor(
            const Goal* desired_goal) const {
        std::unique_ptr<Goal> g = desired_goal->clone();
        g->exactly((std::uint64_t)Fact::k_location, (std::uint64_t)Location::k_store);
        g->dec((std::uint64_t)Fact::k_money, 10);
        g->inc((std::uint64_t)Fact::k_have_axe, 1);

        return { std::make_unique<Action>(1.0, ActionType::k_sell_axe, 1), std::move(g) };
    }
};

class GatherWood : public narrative::ActionFactory<Action, Goal> {
    public:
    GatherWood(std::uint64_t chop_amount=1, float chop_fraction=1.0) : chop_amount_(chop_amount), chop_fraction_(chop_fraction) { }

    virtual bool compatible(const Goal *desired_goal) const {
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

    virtual std::pair<std::unique_ptr<Action>, std::unique_ptr<Goal> > neighbor(
            const Goal* desired_goal) const {

        std::uint64_t amount = amount_to_chop(desired_goal);

        std::unique_ptr<Goal> g = desired_goal->clone();
        g->exactly((std::uint64_t)Fact::k_location, (std::uint64_t)Location::k_woods);
        g->inc((std::uint64_t)Fact::k_forest, amount);
        g->dec((std::uint64_t)Fact::k_wood, amount);

        return { std::make_unique<Action>(10.0*amount+FIXED_COST, ActionType::k_gather_wood, amount), std::move(g) };
    }

    protected:
    std::uint64_t amount_to_chop(const Goal *desired_goal) const {
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

class ChopWood : public narrative::ActionFactory<Action, Goal> {
    public:
    ChopWood(std::uint64_t chop_amount=1, float chop_fraction=1.0) : chop_amount_(chop_amount), chop_fraction_(chop_fraction) { }

    virtual bool compatible(const Goal *desired_goal) const {
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

    virtual std::pair<std::unique_ptr<Action>, std::unique_ptr<Goal> > neighbor(
            const Goal* desired_goal) const {

        std::uint64_t amount = amount_to_chop(desired_goal);

        std::unique_ptr<Goal> g = desired_goal->clone();
        g->exactly((std::uint64_t)Fact::k_location, (std::uint64_t)Location::k_woods);
        g->inc((std::uint64_t)Fact::k_forest, amount);
        g->dec((std::uint64_t)Fact::k_wood, amount);
        g->at_least((std::uint64_t)Fact::k_have_axe, 1);

        return { std::make_unique<Action>(1.0*amount+FIXED_COST, ActionType::k_chop_wood, amount), std::move(g) };
    }

    protected:
    std::uint64_t amount_to_chop(const Goal *desired_goal) const {
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

class SellWood : public narrative::ActionFactory<Action, Goal> {
    public:
    SellWood(std::uint64_t sell_amount=1, float sell_fraction=1.0) : sell_amount_(sell_amount), sell_fraction_(sell_fraction) { }

    virtual bool compatible(const Goal *desired_goal) const {
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

    virtual std::pair<std::unique_ptr<Action>, std::unique_ptr<Goal> > neighbor(
            const Goal* desired_goal) const {
        std::uint64_t amount = amount_to_sell(desired_goal);
        std::unique_ptr<Goal> g = desired_goal->clone();
        g->exactly((std::uint64_t)Fact::k_location, (std::uint64_t)Location::k_store);
        g->dec((std::uint64_t)Fact::k_money, amount);
        g->inc((std::uint64_t)Fact::k_wood, amount);

        return { std::make_unique<Action>(1.0*amount+FIXED_COST, ActionType::k_sell_wood, amount), std::move(g) };
    }
    protected:

    std::uint64_t amount_to_sell(const Goal *desired_goal) const {
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

class GoTo : public narrative::ActionFactory<Action, Goal> {
    public:
    virtual bool compatible(const Goal *desired_goal) const {
        // effect:
        //  set location to whatever the goal is
        return desired_goal->low((std::uint64_t)Fact::k_location) > 0;
    }

    virtual std::pair<std::unique_ptr<Action>, std::unique_ptr<Goal> > neighbor(
            const Goal* desired_goal) const {
        std::unique_ptr<Goal> g = desired_goal->clone();
        std::uint64_t location = desired_goal->low((std::uint64_t)Fact::k_location);
        g->remove((std::uint64_t)Fact::k_location);

        return { std::make_unique<Action>(5.0, ActionType::k_go_to, location), std::move(g) };
    }
};

class GrindMoney : public narrative::ActionFactory<Action, Goal> {
    public:
    virtual bool compatible(const Goal *desired_goal) const {
        return desired_goal->low((std::uint64_t)Fact::k_money) > 0;
    }

    virtual std::pair<std::unique_ptr<Action>, std::unique_ptr<Goal> > neighbor(
            const Goal* desired_goal) const {
        std::unique_ptr<Goal> g = desired_goal->clone();
        g->dec((std::uint64_t)Fact::k_money, 1);

        return { std::make_unique<Action>(1.0, ActionType::k_grind_money, 1), std::move(g) };
    }
};

std::string to_string(const Action& a) {
    std::ostringstream s;
    s << action_names[(size_t)a.action_type_];
    switch(a.action_type_) {
        case ActionType::k_go_to:
            s << "(" << location_names[a.action_param_] << ")";
            break;
        default:
            s << "(" << a.action_param_ << ")";
    }
    return s.str();

}

int main(int argc, char** argv) {
    printf("hello world!\n");

    /*std::unordered_map<std::uint64_t, std::uint64_t> test_map;
    printf("bucket count: %zu\n", test_map.bucket_count());
    for(int i = 1; i < 50; i++) {
        test_map[i] = 0;
    }
    printf("bucket count: %zu\n", test_map.bucket_count());
    return 0;*/

    // initial state
    cEvent event;
    cEventContext character_context;
    character_context[(std::uint64_t)Fact::k_money] = 0;
    character_context[(std::uint64_t)Fact::k_forest] = 100;

    // desired goal
    std::array<cCriteria<cIntRef, cFlagRef, cIntRef>, (std::uint64_t)Fact::k_COUNT> cri;
    cri[(std::uint64_t)Fact::k_money] = make_criteria(
        50, (std::uint64_t)Fact::k_money, 50//POS_INF
    );
    /*cri[(std::uint64_t)Fact::k_forest] = make_criteria(
        101, (std::uint64_t)Fact::k_forest, POS_INF
    );*/
    Goal starting_goal = Goal(cri);

    // action factories
    GrindMoney af_gm;
    BuyAxe af_ba;
    SellAxe af_sa;
    GatherWood af_gw_one;
    GatherWood af_gw_half(0,0.5);
    GatherWood af_gw_all(0);
    ChopWood af_cw_one;
    ChopWood af_cw_half(0,0.5);
    ChopWood af_cw_all(0);
    SellWood af_sw_one;
    SellWood af_sw_half(0,0.5);
    SellWood af_sw_all(0);
    GoTo af_gt;

    narrative::PlanningMap<Action, Goal, FactDistance> map(
        &starting_goal,
        &event,
        &character_context,
        {
            &af_ba,
            &af_sa,
            &af_gw_one,
            &af_gw_half,
            //&af_gw_all,
            &af_cw_one,
            &af_cw_half,
            //&af_cw_all,
            &af_sw_one,
            &af_sw_half,
            //&af_sw_all,
            &af_gt
        }
    );
    narrative::AStar<Goal, Action, narrative::PlanningMap<Action, Goal, FactDistance>, narrative::WeightedAStarFScore<2.5f>> astar;

    const narrative::AStarNode<Goal, Action>* solution;
    for(int i=0; i < 1; i++) {
        solution = astar.run_astar(&map);
    }
    printf("open_set: %zu closed_set %zu\n", astar.open_set.size(), astar.closed_set.size());
    printf("counters:\n");
    for(size_t i=0; i < astar.k_cnt_LEN; i++) {
        printf("\t%s\t%lu\n", astar.k_counter_names[i], astar.counters_[i]);
    }

    if(solution == NULL) {
        printf("no solution found\n");
    } else {
        printf("solution cost %f\n", solution->g_score);
        narrative::print_solution(solution, fact_names);
    }

    return 0;
}
