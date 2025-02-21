#ifndef NARRATIVE_ASTAR_H
#define NARRATIVE_ASTAR_H

#include <sstream>
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
    float h_score;
    float f_score;
    std::pair<std::unique_ptr<Edge>, const AStarNode<State, Edge>*> parent;

    AStarNode(std::unique_ptr<State> s, float g, float h, float f, std::pair<std::unique_ptr<Edge>, const AStarNode<State, Edge>*> p)
        : state(std::move(s)),
          g_score(g),
          h_score(h),
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

struct AStarFScore {
    float operator()(float g_score, float h_score) const {
        return g_score + h_score;
    }
};
struct DjikstraFScore {
    float operator()(float g_score, float h_score) const {
        return g_score;
    }
};
struct GreedyFScore {
    float operator()(float g_score, float h_score) const {
        return h_score;
    }
};
template<float W=2.5f>
struct WeightedAStarFScore {
    float operator()(float g_score, float h_score) const {
        // weighted A*
        //const float w = 5.0;
        return g_score + W*h_score;
        /*
        // pxWD and pxWU
        const float w = 1.5;
        if(g_score < h_score) {
            return g_score + h_score;
            //return (g_score + (2*w - 1) * h_score) / w;
        } else {
            return (g_score + (2*w - 1) * h_score) / w;
            //return g_score + h_score;
        }*/
    }
};

template <class State, class Edge, class Map, class FScore=WeightedAStarFScore<> >
class AStar {
    public:
        enum CounterKeys : size_t {
            k_cnt_steps = 0,
            k_cnt_neighbors,
            k_cnt_n_closed,
            k_cnt_n_open,
            k_cnt_n_no_improvement,
            k_cnt_n_closed_reopen,
            k_cnt_n_open_reopen,
            k_cnt_n_back,
            k_cnt_LEN
        };

        static constexpr float k_improvement_epsilon = 1.0e-1;

        const char* k_counter_names[k_cnt_LEN] {
            "steps",
            "neighbors",
            "nbr_closed",
            "nbr_open",
            "nbr_no_imprv",
            "nbr_c_reopen",
            "nbr_o_reopen",
            "nbr_back",
        };

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
            float h_score = map->heuristic_cost(initial_state.second.get());
            auto emplace_result = open_set.emplace(
                std::move(initial_state.second),
                0.0f,
                h_score,
                FScore()(0.0f, h_score),
                std::make_pair(std::move(initial_state.first), (const AStarNode<State, Edge>*)NULL)
            );
            open_map.emplace(std::make_pair(
                emplace_result.first->state.get(), emplace_result.first
            ));

            // some debug state
            /*float best_h = std::numeric_limits<float>::infinity();
            float best_g = std::numeric_limits<float>::infinity();*/

            int steps_since_gain = 0;
            float best_h = std::numeric_limits<float>::infinity();

            while(!open_set.empty()) {
                counters_[k_cnt_steps]++;
                // move the cheapest element from open_set to the closed_set
                auto current = closed_set.insert(open_set.extract(open_set.begin())).position;

                // a couple of debug logging options
                //print_solution(&*current);
                //printf("STEP(%*zu, %*zu): %s\n", 5, open_set.size(), 5, closed_set.size(), to_string(&*current).c_str());

                // move from open map to closed map
                open_map.erase(current->state.get());
                closed_map[current->state.get()] = current;

                // some debug logging
                /*if(current->f_score-current->g_score < best_h || (
                    current->f_score-current->g_score ==  best_h &&
                    current->g_score < best_g
                )) {
                    best_h = current->f_score-current->g_score;
                    best_g = current->g_score;
                    printf("best_h: %f best_g: %f state: %s\n", best_h, best_g, to_string(*current->state).c_str());
                }*/

                // check if this is the desired state, if so, we're done
                if(map->satisfied(current->state.get())) {
                    return &*current;
                }

                /*if(current->h_score < best_h) {
                    steps_since_gain = 0;
                    best_h = current->h_score;
                } else if(steps_since_gain > max_steps_since_gain_) {
                    return NULL;
                } else {
                    steps_since_gain++;
                }*/

                for(std::pair<std::unique_ptr<Edge>, std::unique_ptr<State>> &neighbor : map->neighbors(current->state.get())) {
                    counters_[k_cnt_neighbors]++;
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
                        counters_[k_cnt_n_closed]++;
                    } else if((open_map_itr = open_map.find(neighbor.second.get())) != open_map.end()) {
                        closed_itr = closed_set.end();
                        open_itr = open_map_itr->second;
                        neighbor_ptr = &*open_itr;
                        counters_[k_cnt_n_open]++;
                    } else {
                        closed_itr = closed_set.end();
                        open_itr = open_set.end();
                        neighbor_ptr = NULL;
                    }

                    // if we've already seen this state and the new score is no
                    // better than we've seen before, skip this neighbor
                    if(neighbor_ptr != NULL && tentative_g_score + k_improvement_epsilon >= neighbor_ptr->g_score ) {
                        counters_[k_cnt_n_no_improvement]++;
                        continue;
                    }/* else if(neighbor_ptr != NULL) {
                        printf("%f\n", neighbor_ptr->g_score - tentative_g_score);
                    }*/

                    h_score = map->heuristic_cost(neighbor.second.get());
                    // if we've already seen this state (and we know this new
                    // one is better) we'll dump the old and take the new one
                    if(closed_itr != closed_set.end()) {
                        closed_map.erase(neighbor_ptr->state.get());
                        auto nh = closed_set.extract(closed_itr);
                        State* neighbor_state = nh.value().state.get();
                        nh.value().g_score = tentative_g_score;
                        nh.value().h_score = h_score;
                        nh.value().f_score = FScore()(tentative_g_score, h_score);
                        nh.value().parent = {std::move(neighbor.first), &*current};
                        auto insert_result = open_set.insert(std::move(nh));
                        open_map.emplace(std::make_pair(
                            insert_result.position->state.get(), insert_result.position
                        ));
                        counters_[k_cnt_n_closed_reopen]++;
                    } else if(open_itr != open_set.end()) {
                        open_map.erase(neighbor_ptr->state.get());
                        auto nh = open_set.extract(open_itr);
                        State* neighbor_state = nh.value().state.get();
                        nh.value().g_score = tentative_g_score;
                        nh.value().h_score = h_score;
                        nh.value().f_score = FScore()(tentative_g_score, h_score);
                        nh.value().parent = {std::move(neighbor.first), &*current};
                        auto insert_result = open_set.insert(std::move(nh));
                        open_map.emplace(std::make_pair(
                            insert_result.position->state.get(), insert_result.position
                        ));
                        counters_[k_cnt_n_open_reopen]++;
                    } else {
                        // make a new AStarNode for this neighboring state
                        State* neighbor_state = neighbor.second.get();
                        emplace_result = open_set.emplace(
                            std::move(neighbor.second),
                            tentative_g_score,
                            h_score,
                            FScore()(tentative_g_score, h_score),
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

        int max_steps_since_gain_ = 1000;

        std::uint64_t counters_[k_cnt_LEN] = { 0 };
};

template <class State, class Edge>
std::string to_string(const stellarpunk::narrative::AStarNode<State, Edge>* solution, const char** fact_names=NULL) {
    std::ostringstream s;
    s << to_string(*solution->parent.first) << std::string(" ");
    s << to_string(*solution->state, fact_names) << " ";
    s << "h_score: " << solution->h_score << " ";
    s << "g_score: " << solution->g_score << " ";
    s << "f_score: " << solution->f_score << " ";
    s << "cost: " << solution->parent.first->cost();

    return s.str();
}

template <class State, class Edge>
void print_solution(const narrative::AStarNode<State, Edge>* solution, const char** fact_names=NULL) {
    while(solution != NULL) {
        printf("%s\n", to_string(solution).c_str());
        solution = solution->parent.second;
    }
}

} // namespace narrative
} // namespace stellarpunk

#endif /* NARRATIVE_ASTAR_H */
