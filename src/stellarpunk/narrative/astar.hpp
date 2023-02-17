#include <set>
#include <map>
#include <limits>
#include <string>
#include <stdio.h>

template <class State, class Edge>
struct AStarNode {
    std::unique_ptr<State> state;

    float g_score;
    float f_score;
    std::pair<std::unique_ptr<Edge>, const AStarNode<State, Edge>*> parent;

    AStarNode(std::unique_ptr<State> s, float g, float f, std::pair<std::unique_ptr<Edge>, const AStarNode<State, Edge>*> p)
        : state(std::move(s)),
          g_score(g),
          f_score(f),
          parent(std::move(p.first), p.second) {
    }

    friend bool operator<(const AStarNode& lhs, const AStarNode& rhs) {
        return std::tie(lhs.f_score, lhs.g_score, *lhs.state)
            < std::tie(rhs.f_score, rhs.g_score, *rhs.state);
    }
};

template<class T>
struct ptr_less
{
    bool operator()(const T* lhs, const T* rhs) const
    {
        return *lhs < *rhs;
    }
};

template <class State, class Edge, class Map>
class AStar {
    public:
        AStar() {}
        ~AStar() {}

        const AStarNode<State, Edge>* run_astar(Map* map) {
            // drop any state from prior calls to run_astar
            open_map.clear();
            open_set.clear();
            closed_map.clear();
            closed_set.clear();

            // create an AStarNode for the initial state and add to open set

            std::pair<std::unique_ptr<Edge>, std::unique_ptr<State>> initial_state = map->initial_state();
            open_set.emplace(
                std::move(initial_state.second),
                0.0f,
                map->heuristic_cost(initial_state.second.get()),
                std::make_pair(std::move(initial_state.first), (const AStarNode<State, Edge>*)NULL)
            );

            while(!open_set.empty()) {
                // move the cheapest element from open_set to the closed_set
                const AStarNode<State, Edge>* current = &(*closed_set.insert(open_set.extract(open_set.begin())).position);
                printf("considering node %s with g_score: %f f_score: %f\n", current->state->to_string().c_str(), current->g_score, current->f_score);
                // move from open map to closed map
                open_map.erase(current->state.get());
                closed_map[current->state.get()] = current;

                // check if this is the desired state, if so, we're done
                if(map->satisfied(current->state.get())) {
                    return current;
                }

                for(std::pair<std::unique_ptr<Edge>, std::unique_ptr<State>> &neighbor : map->neighbors(current->state.get())) {
                    // compute the path cost to reach the neighbor state via neighbor edge
                    float tentative_g_score = current->g_score + neighbor.first->cost();
                    // check if neighbor leads to a destination we've seen before
                    // if so, we'll re-use the node from there
                    auto open_itr = open_set.end();
                    auto closed_itr = closed_set.end();
                    const AStarNode<State, Edge>* neighbor_ptr;
                    if(closed_map.count(neighbor.second.get()) > 0) {
                        neighbor_ptr = closed_map[neighbor.second.get()];
                        closed_itr = closed_set.find(*neighbor_ptr);
                    } else if(open_map.count(neighbor.second.get()) > 0) {
                        neighbor_ptr = open_map[neighbor.second.get()];
                        open_itr = open_set.find(*neighbor_ptr);
                    } else {
                        neighbor_ptr = NULL;
                    }

                    // if we've already seen this state and the new score is no
                    // better than we've seen before, skip this neighbor
                    if(neighbor_ptr != NULL && tentative_g_score >= neighbor_ptr->g_score) {
                        continue;
                    }

                    // if we've already seen this state (and we know this new
                    // one is better) we'll dump the old and take the new one
                    if(closed_itr != closed_set.end()) {
                        closed_set.erase(closed_itr);
                        closed_map.erase(neighbor_ptr->state.get());
                    } else if(open_itr != open_set.end()) {
                        open_set.erase(open_itr);
                        open_map.erase(neighbor_ptr->state.get());
                    }

                    // make a new AStarNode for this neighboring state
                    State* neighbor_state = neighbor.second.get();
                    neighbor_ptr = &(*open_set.emplace(
                        std::move(neighbor.second),
                        tentative_g_score,
                        tentative_g_score + map->heuristic_cost(neighbor_state),
                        std::make_pair(std::move(neighbor.first), current)
                    ).first);
                    open_map[neighbor_ptr->state.get()] = neighbor_ptr;
                }
            }
            return NULL;
        }

    protected:
        std::set<AStarNode<State, Edge> > open_set;
        std::map<State*, const AStarNode<State, Edge>*, ptr_less<State> > open_map;
        std::set<AStarNode<State, Edge> > closed_set;
        std::map<State*, const AStarNode<State, Edge>*, ptr_less<State> > closed_map;
};
