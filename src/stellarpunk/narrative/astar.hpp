#ifndef NARRATIVE_ASTAR_H
#define NARRATIVE_ASTAR_H

#include <set>
#include <map>
#include <limits>
#include <string>
#include <memory>
#include <stdio.h>

namespace stellarpunk {
namespace narrative {

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

template<class T>
struct ptr_hash
{
    size_t operator()(const T* t) const
    {
        return std::hash<T>{}(*t);
    }
};

template<class T>
struct ptr_equal_to
{
    bool operator()(const T* lhs, const T* rhs) const
    {
        return *lhs == *rhs;
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
            auto emplace_result = open_set.emplace(
                std::move(initial_state.second),
                0.0f,
                map->heuristic_cost(initial_state.second.get()),
                std::make_pair(std::move(initial_state.first), (const AStarNode<State, Edge>*)NULL)
            );
            open_map.emplace(std::make_pair(
                emplace_result.first->state.get(), emplace_result.first
            ));

            while(!open_set.empty()) {
                // move the cheapest element from open_set to the closed_set
                auto current = closed_set.insert(open_set.extract(open_set.begin())).position;
                //printf("considering node %s with g_score: %f f_score: %f\n", to_string(*current->state).c_str(), current->g_score, current->f_score);
                // move from open map to closed map
                open_map.erase(current->state.get());
                closed_map[current->state.get()] = current;

                // check if this is the desired state, if so, we're done
                if(map->satisfied(current->state.get())) {
                    return &*current;
                }

                for(std::pair<std::unique_ptr<Edge>, std::unique_ptr<State>> &neighbor : map->neighbors(current->state.get())) {
                    // compute the path cost to reach the neighbor state via neighbor edge
                    float tentative_g_score = current->g_score + neighbor.first->cost();
                    // check if neighbor leads to a destination we've seen before
                    // if so, we'll re-use the node from there
                    typename NodeMap::iterator open_map_itr;
                    typename NodeContainer::iterator open_itr;
                    typename NodeMap::iterator closed_map_itr;
                    typename NodeContainer::iterator closed_itr;
                    const AStarNode<State, Edge>* neighbor_ptr;

                    if((closed_map_itr = closed_map.find(neighbor.second.get())) != closed_map.end()) {
                        closed_itr = closed_map_itr->second;
                        neighbor_ptr = &*closed_itr;
                    } else if((open_map_itr = open_map.find(neighbor.second.get())) != open_map.end()) {
                        closed_itr = closed_set.end();
                        open_itr = open_map_itr->second;
                        neighbor_ptr = &*open_itr;
                    } else {
                        closed_itr = closed_set.end();
                        open_itr = open_set.end();
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
                        closed_map.erase(neighbor_ptr->state.get());
                        auto nh = closed_set.extract(closed_itr);
                        State* neighbor_state = nh.value().state.get();
                        nh.value().g_score = tentative_g_score;
                        nh.value().f_score = tentative_g_score + map->heuristic_cost(neighbor_state);
                        nh.value().parent = {std::move(neighbor.first), &*current};
                        auto insert_result = open_set.insert(std::move(nh));
                        open_map.emplace(std::make_pair(
                            insert_result.position->state.get(), insert_result.position
                        ));
                    } else if(open_itr != open_set.end()) {
                        open_map.erase(neighbor_ptr->state.get());
                        auto nh = open_set.extract(open_itr);
                        State* neighbor_state = nh.value().state.get();
                        nh.value().g_score = tentative_g_score;
                        nh.value().f_score = tentative_g_score + map->heuristic_cost(neighbor_state);
                        nh.value().parent = {std::move(neighbor.first), &*current};
                        auto insert_result = open_set.insert(std::move(nh));
                        open_map.emplace(std::make_pair(
                            insert_result.position->state.get(), insert_result.position
                        ));
                    } else {
                        // make a new AStarNode for this neighboring state
                        State* neighbor_state = neighbor.second.get();
                        emplace_result = open_set.emplace(
                            std::move(neighbor.second),
                            tentative_g_score,
                            tentative_g_score + map->heuristic_cost(neighbor_state),
                            std::make_pair(std::move(neighbor.first), &*current)
                        );
                        open_map.emplace(std::make_pair(
                            emplace_result.first->state.get(), emplace_result.first
                        ));
                    }
                }
            }
            return NULL;
        }

        using NodeContainer = std::set<AStarNode<State, Edge>>;
        using NodeMap = std::unordered_map<State*, typename NodeContainer::iterator, ptr_hash<State>, ptr_equal_to<State>>;

        NodeContainer open_set;
        NodeMap open_map;
        NodeContainer closed_set;
        NodeMap closed_map;
};

} // namespace narrative
} // namespace stellarpunk

#endif /* NARRATIVE_ASTAR_H */
