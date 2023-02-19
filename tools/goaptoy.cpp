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

std::unique_ptr<cCriteria<cIntRef, cFlagRef, cIntRef>> make_criteria(
        std::uint64_t l,
        std::uint64_t f,
        std::uint64_t h) {
    return std::make_unique<cCriteria<cIntRef, cFlagRef, cIntRef>>(l, f, h);
}

class BuyAxe : public narrative::ActionFactory {
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

    virtual std::pair<std::unique_ptr<narrative::Action>, std::unique_ptr<narrative::Goal> > neighbor(
            const narrative::Goal* desired_goal) {
        std::unique_ptr<narrative::Goal> g = desired_goal->clone();
        g->exactly((std::uint64_t)Fact::k_location, (std::uint64_t)Location::k_store);
        g->inc((std::uint64_t)Fact::k_money, 20);
        g->dec((std::uint64_t)Fact::k_have_axe, 1);

        return {std::make_unique<narrative::Action>(20.0), std::move(g)};
    }
};

class SellAxe : public narrative::ActionFactory {
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

    virtual std::pair<std::unique_ptr<narrative::Action>, std::unique_ptr<narrative::Goal> > neighbor(
            const narrative::Goal* desired_goal) {
        std::unique_ptr<narrative::Goal> g = desired_goal->clone();
        g->exactly((std::uint64_t)Fact::k_location, (std::uint64_t)Location::k_store);
        g->dec((std::uint64_t)Fact::k_money, 10);
        g->inc((std::uint64_t)Fact::k_have_axe, 1);

        return std::make_pair(
            std::make_unique<narrative::Action>(10.0),
            std::move(g)
        );
    }
};

class ChopWood : public narrative::ActionFactory {
    public:
    virtual bool compatible(const narrative::Goal *desired_goal) {
        // effect:
        //  get wood
        //  remove forest
        //  will be at the woods (but we need someone else to get us there)
        return (
            desired_goal->high((std::uint64_t)Fact::k_forest) < POS_INF ||
            desired_goal->low((std::uint64_t)Fact::k_wood) > 0
        ) && (
            desired_goal->low((std::uint64_t)Fact::k_location) == 0 ||
            desired_goal->low((std::uint64_t)Fact::k_location) == (std::uint64_t)Location::k_woods
        );
    }

    virtual std::pair<std::unique_ptr<narrative::Action>, std::unique_ptr<narrative::Goal> > neighbor(
            const narrative::Goal* desired_goal) {
        std::unique_ptr<narrative::Goal> g = desired_goal->clone();
        g->exactly((std::uint64_t)Fact::k_location, (std::uint64_t)Location::k_woods);
        g->dec((std::uint64_t)Fact::k_wood, 10);
        g->at_least((std::uint64_t)Fact::k_have_axe, 1);

        return std::make_pair(
            std::make_unique<narrative::Action>(1.0),
            std::move(g)
        );
    }
};

class SellWood : public narrative::ActionFactory {
    public:
    virtual bool compatible(const narrative::Goal *desired_goal) {
        // effect:
        //  get money
        //  remove wood
        //  will be at the store (but we need someone else to get us there)
        return (
            desired_goal->high((std::uint64_t)Fact::k_wood) < POS_INF ||
            desired_goal->low((std::uint64_t)Fact::k_money) > 0
        ) && (
            desired_goal->low((std::uint64_t)Fact::k_location) == 0 ||
            desired_goal->low((std::uint64_t)Fact::k_location) == (std::uint64_t)Location::k_store
        );
    }

    virtual std::pair<std::unique_ptr<narrative::Action>, std::unique_ptr<narrative::Goal> > neighbor(
            const narrative::Goal* desired_goal) {
        std::unique_ptr<narrative::Goal> g = desired_goal->clone();
        g->exactly((std::uint64_t)Fact::k_location, (std::uint64_t)Location::k_store);
        g->dec((std::uint64_t)Fact::k_money, 1);
        g->inc((std::uint64_t)Fact::k_wood, 1);

        return std::make_pair(
            std::make_unique<narrative::Action>(1.0),
            std::move(g)
        );
    }
};

class GoTo : public narrative::ActionFactory {
    public:
    virtual bool compatible(const narrative::Goal *desired_goal) {
        // effect:
        //  set location to whatever the goal is
        return desired_goal->low((std::uint64_t)Fact::k_location) > 0;
    }

    virtual std::pair<std::unique_ptr<narrative::Action>, std::unique_ptr<narrative::Goal> > neighbor(
            const narrative::Goal* desired_goal) {
        std::unique_ptr<narrative::Goal> g = desired_goal->clone();
        g->remove((std::uint64_t)Fact::k_location);

        return std::make_pair(
            std::make_unique<narrative::Action>(5.0),
            std::move(g)
        );
    }
};

class GrindMoneyFactory : public narrative::ActionFactory {
    public:
    virtual bool compatible(const narrative::Goal *desired_goal) {
        return desired_goal->low((std::uint64_t)Fact::k_money) > 0;
    }

    virtual std::pair<std::unique_ptr<narrative::Action>, std::unique_ptr<narrative::Goal> > neighbor(
            const narrative::Goal* desired_goal) {
        std::unique_ptr<narrative::Goal> g = desired_goal->clone();
        g->dec((std::uint64_t)Fact::k_money, 1);

        return std::make_pair(
            std::make_unique<narrative::Action>(1.0),
            std::move(g)
        );
    }
};

int main(int argc, char** argv) {
    printf("hello world!\n");

    std::array<std::unique_ptr<cCriteria<cIntRef, cFlagRef, cIntRef>>, 6> cri;
    cri[(std::uint64_t)Fact::k_money] = make_criteria(50, (std::uint64_t)Fact::k_money, POS_INF);
    narrative::Goal starting_goal = narrative::Goal(cri);

    std::vector<std::unique_ptr<narrative::ActionFactory>> action_factories;
    //action_factories.emplace_back(std::make_unique<GrindMoneyFactory>());
    action_factories.emplace_back(std::make_unique<BuyAxe>());
    action_factories.emplace_back(std::make_unique<SellAxe>());
    action_factories.emplace_back(std::make_unique<ChopWood>());
    action_factories.emplace_back(std::make_unique<SellWood>());
    action_factories.emplace_back(std::make_unique<GoTo>());

    cEvent event;
    cEventContext character_context;
    character_context[(std::uint64_t)Fact::k_money] = 40;

    narrative::PlanningMap map(
        &starting_goal,
        &event,
        &character_context,
        &action_factories
    );
    narrative::AStar<narrative::Goal, narrative::Action, narrative::PlanningMap> astar;
    const narrative::AStarNode<narrative::Goal, narrative::Action>* solution;
    for(int i=0; i< 1000; i++) {
        solution = astar.run_astar(&map);
    }
    printf("open_set: %zu closed_set %zu\n", astar.open_set.size(), astar.closed_set.size());
    if(solution == NULL) {
        printf("no solution found\n");
    } else {
        while(solution != NULL) {
            printf(
                "%s with g_score: %f f_score: %f via %f\n",
                narrative::to_string(*solution->state).c_str(),
                solution->g_score,
                solution->f_score,
                solution->parent.first->cost_
            );
            solution = solution->parent.second;
        }
    }
}
