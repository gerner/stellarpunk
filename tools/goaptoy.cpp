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

const char* location_names[] = {
    "INVALID",
    "NONE",
    "woods",
    "store"
};

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

struct FactDistance {
    float operator ()(const std::uint64_t& k, const std::uint64_t& d) const {
        float dist;
        switch(k) {
            case (std::uint64_t)Fact::k_money:
            case (std::uint64_t)Fact::k_forest:
            case (std::uint64_t)Fact::k_wood:
                // money, forest, wood cost 1 each
                dist = d;
                break;
            case (std::uint64_t)Fact::k_have_axe:
                // axes cost 20 but can sell for 10
                dist = d * 20;
                break;
            case (std::uint64_t)Fact::k_location:
                // either at location or not, going to a location costs 5
                dist = d>0 ? 5.0 : 0.0;
                break;
            default:
                assert(false); // shouldn't ever see anything else
        }

        return dist;
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

using Goal = narrative::Goal<(size_t)Fact::k_COUNT>;

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

class BuyAxe : public narrative::ActionFactory<Action, Goal> {
    public:
    virtual bool compatible(const Goal *desired_goal) {
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
            const Goal* desired_goal) {
        std::unique_ptr<Goal> g = desired_goal->clone();
        g->exactly((std::uint64_t)Fact::k_location, (std::uint64_t)Location::k_store);
        g->inc((std::uint64_t)Fact::k_money, 20);
        g->dec((std::uint64_t)Fact::k_have_axe, 1);

        return {std::make_unique<Action>(20.0, ActionType::k_buy_axe, 1), std::move(g)};
    }
};

class SellAxe : public narrative::ActionFactory<Action, Goal> {
    public:
    virtual bool compatible(const Goal *desired_goal) {
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

    virtual std::pair<std::unique_ptr<Action>, std::unique_ptr<Goal> > neighbor(
            const Goal* desired_goal) {
        std::unique_ptr<Goal> g = desired_goal->clone();
        g->exactly((std::uint64_t)Fact::k_location, (std::uint64_t)Location::k_store);
        g->dec((std::uint64_t)Fact::k_money, 10);
        g->inc((std::uint64_t)Fact::k_have_axe, 1);

        return { std::make_unique<Action>(10.0, ActionType::k_sell_axe, 1), std::move(g) };
    }
};

class GatherWood : public narrative::ActionFactory<Action, Goal> {
    public:
    GatherWood(std::uint64_t chop_amount=1, float chop_fraction=1.0) : chop_amount_(chop_amount), chop_fraction_(chop_fraction) { }

    virtual bool compatible(const Goal *desired_goal) {
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
            const Goal* desired_goal) {

        std::uint64_t amount = amount_to_chop(desired_goal);

        std::unique_ptr<Goal> g = desired_goal->clone();
        g->exactly((std::uint64_t)Fact::k_location, (std::uint64_t)Location::k_woods);
        g->inc((std::uint64_t)Fact::k_forest, amount);
        g->dec((std::uint64_t)Fact::k_wood, amount);

        return { std::make_unique<Action>(10.0*amount+1.0, ActionType::k_gather_wood, amount), std::move(g) };
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

    virtual bool compatible(const Goal *desired_goal) {
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
            const Goal* desired_goal) {
        std::uint64_t amount = amount_to_chop(desired_goal);
        std::unique_ptr<Goal> g = desired_goal->clone();
        g->exactly((std::uint64_t)Fact::k_location, (std::uint64_t)Location::k_woods);
        g->inc((std::uint64_t)Fact::k_forest, amount);
        g->dec((std::uint64_t)Fact::k_wood, amount);
        g->at_least((std::uint64_t)Fact::k_have_axe, 1);

        return { std::make_unique<Action>(1.0*amount+1.0, ActionType::k_chop_wood, amount), std::move(g) };
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

    virtual bool compatible(const Goal *desired_goal) {
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
            const Goal* desired_goal) {
        std::uint64_t amount = amount_to_sell(desired_goal);
        std::unique_ptr<Goal> g = desired_goal->clone();
        g->exactly((std::uint64_t)Fact::k_location, (std::uint64_t)Location::k_store);
        g->dec((std::uint64_t)Fact::k_money, amount);
        g->inc((std::uint64_t)Fact::k_wood, amount);

        return { std::make_unique<Action>(1.0*amount+1.0, ActionType::k_sell_wood, amount), std::move(g) };
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
    virtual bool compatible(const Goal *desired_goal) {
        // effect:
        //  set location to whatever the goal is
        return desired_goal->low((std::uint64_t)Fact::k_location) > 0;
    }

    virtual std::pair<std::unique_ptr<Action>, std::unique_ptr<Goal> > neighbor(
            const Goal* desired_goal) {
        std::unique_ptr<Goal> g = desired_goal->clone();
        std::uint64_t location = desired_goal->low((std::uint64_t)Fact::k_location);
        g->remove((std::uint64_t)Fact::k_location);

        return { std::make_unique<Action>(5.0, ActionType::k_go_to, location), std::move(g) };
    }
};

class GrindMoney : public narrative::ActionFactory<Action, Goal> {
    public:
    virtual bool compatible(const Goal *desired_goal) {
        return desired_goal->low((std::uint64_t)Fact::k_money) > 0;
    }

    virtual std::pair<std::unique_ptr<Action>, std::unique_ptr<Goal> > neighbor(
            const Goal* desired_goal) {
        std::unique_ptr<Goal> g = desired_goal->clone();
        g->dec((std::uint64_t)Fact::k_money, 1);

        return { std::make_unique<Action>(1.0, ActionType::k_grind_money, 1), std::move(g) };
    }
};

std::string to_string(const narrative::AStarNode<Goal, Action>* solution) {
    std::ostringstream s;
    s << to_string(*solution->parent.first) << " ";
    s << narrative::to_string(*solution->state, fact_names) << " ";
    s << "h_score: " << solution->f_score - solution->g_score << " ";
    s << "g_score: " << solution->g_score << " ";
    s << "f_score: " << solution->f_score << " ";
    s << "cost: " << solution->parent.first->cost_;

    return s.str();
}

void print_solution(const narrative::AStarNode<Goal, Action>* solution) {
    while(solution != NULL) {
        printf("%s\n", to_string(solution).c_str());
        solution = solution->parent.second;
    }
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
    character_context[(std::uint64_t)Fact::k_money] = 39;
    character_context[(std::uint64_t)Fact::k_forest] = 100;

    // desired goal
    std::array<cCriteria<cIntRef, cFlagRef, cIntRef>, 6> cri;
    cri[(std::uint64_t)Fact::k_money] = make_criteria(
        50, (std::uint64_t)Fact::k_money, POS_INF
    );
    Goal starting_goal = Goal(cri);

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

    narrative::PlanningMap<Action, Goal, FactDistance> map(
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
    narrative::AStar<Goal, Action, narrative::PlanningMap<Action, Goal, FactDistance>> astar;

    const narrative::AStarNode<Goal, Action>* solution;
    for(int i=0; i < 1000; i++) {
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
        print_solution(solution);
    }

    return 0;
}
