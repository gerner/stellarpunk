import sys
import logging
import enum
import collections
import math
import time
import pdb
from dataclasses import dataclass
from typing import Sequence, Tuple, List, Mapping, MutableMapping, Any, Set, Collection, Iterator, Optional
import sortedcontainers # type: ignore

ZERO = 0
POS_INF = (1<<64)-1

class Location(enum.IntEnum):
    invalid = -1
    none = 0
    woods = enum.auto()
    store = enum.auto()

class ContextKeys(enum.IntEnum):
    invalid = -1
    money = enum.auto()
    have_axe = enum.auto()
    forest = enum.auto()
    wood = enum.auto()
    location = enum.auto()

class State:
    def __init__(self, values:Mapping[ContextKeys, int]) -> None:
        self.values = values

    def __str__(self) -> str:
        return str([f'{k.name} = {v}' for k,v in self.values.items()])

    def __repr(self) -> str:
        return repr(self.values)

@dataclass(order=True, eq=True, unsafe_hash=True)
class Bound:
    key: ContextKeys
    low: int
    high: int

    def inc(self, amount: int) -> None:
        assert amount > 0
        self.low = self.low + amount
        if self.high < POS_INF:
            self.high += amount

    def atleast(self, amount:int) -> None:
        assert amount > 0
        if self.low < amount:
            self.low = amount
        if self.high < amount:
            self.high = amount

    def dec(self, amount: int) -> None:
        assert amount > 0
        self.low = max(self.low - amount, ZERO)

    def set(self, amount:int) -> None:
        self.low = amount
        self.high = amount

    def nontrivial(self) -> bool:
        return self.low > ZERO or self.high < POS_INF

    def satisfies(self, other: "Bound") -> bool:
        """ True iff any state that satisfies us satisfies other """
        if self.key != other.key:
            return False
        # we need to be at least as narrow as other
        return self.low >= other.low and self.high <= other.high

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if self.low > 0:
            l = f'{self.low} <= '
        else:
            l = ""

        if self.high < POS_INF:
            h = f' <= {self.high}'
        else:
            h = ""

        return f'{l}{self.key.name}{h}'


EMPTY_BOUND = Bound(ContextKeys.invalid, ZERO, POS_INF)


def distance(k: ContextKeys, delta: int) -> float:
    if k == ContextKeys.have_axe:
        if delta > 0:
            return 20*delta
        else:
            return 10*-delta
    elif k == ContextKeys.wood:
        return abs(delta)
    elif k == ContextKeys.money:
        if delta > 0:
            return delta
        else:
            return max(25, -delta)
    elif k == ContextKeys.location:
        return 5 if abs(delta) > 0 else 0
    else:
        return 1

class Goal:
    def __init__(self, bounds:Mapping[ContextKeys, Bound]) -> None:
        #assert all(x.low <= x.high for x in bounds.values())
        self.bounds = {k:v for k,v in bounds.items() if v.nontrivial()}
        self.goal_value = math.inf
        self.fitness_value = math.inf

    def delta(self, state:State) -> float:
        d = 0.
        for key, bound in self.bounds.items():
            if state.values[key] < bound.low:
                d += distance(key, bound.low - state.values[key])
            elif state.values[key] > bound.high:
                d += distance(key, bound.high - state.values[key])
            # else we're already in bound
        return d

    def satisfies(self, other: "Goal") -> bool:
        for k in other.bounds:
            if k not in self.bounds:
                return False
            if not self.bounds[k].satisfies(other.bounds[k]):
                return False

        # it's ok for us to have extra conditions
        return True

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return str(list(self.bounds.values()))

    def __eq__(self, other:Any) -> bool:
        if not isinstance(other, Goal):
            return False
        return self.bounds == other.bounds

    def __lt__(self, other:Any) -> bool:
        if not isinstance(other, Goal):
            raise ValueError(f'both items must be goals')
        for k in ContextKeys:
            if k in self.bounds:
                if k in other.bounds:
                    if self.bounds[k].low < other.bounds[k].low:
                        return True
                    elif self.bounds[k].high < other.bounds[k].high:
                        return True
                    # else bounds for k are equal, consider next
                else:
                    return False
            elif k in other.bounds:
                return True
            # else neigher has k, consider next
        # they must be equal
        return False

    def __hash__(self) -> int:
        h = 0
        # always iterate over these (and compute hash) in same order
        for k in ContextKeys:
            if k in self.bounds:
                h = 31 * h + hash(self.bounds[k])

        return h


@dataclass
class Action:
    name: str
    cost: float

class ActionFactory:
    def _copy_bounds(self, goal: Goal) -> MutableMapping[ContextKeys, Bound]:
        return {k: Bound(k, b.low, b.high) for k, b in goal.bounds.items()}

    def compatible(self, goal:Goal) -> bool:
        return False

    def neighbor(self, goal:Goal) -> Tuple[Action, Goal]:
        # add precondition to goal
        # backout post condition
        return (Action("hi", 5.), Goal({}))

class BuyAxe(ActionFactory):
    def compatible(self, goal:Goal) -> bool:
        return (
            (
                goal.bounds.get(ContextKeys.have_axe, EMPTY_BOUND).low > 0 or
                goal.bounds.get(ContextKeys.money, EMPTY_BOUND).high < POS_INF
            ) and
            goal.bounds.get(ContextKeys.location, EMPTY_BOUND).low in [0, Location.store]
        )

    def neighbor(self, goal:Goal) -> Tuple[Action, Goal]:
        new_bounds = self._copy_bounds(goal)

        #prec: location = store
        if ContextKeys.location in new_bounds:
            new_bounds[ContextKeys.location].set(Location.store)
        else:
            new_bounds[ContextKeys.location] = Bound(ContextKeys.location, Location.store, Location.store)

        # prec: 20 <= money <= POS_INF
        # postc: money -= 20
        if ContextKeys.money not in goal.bounds:
            new_bounds[ContextKeys.money] = Bound(ContextKeys.money, 20, POS_INF)
        else:
            new_bounds[ContextKeys.money].inc(20)

        # postc: has_axe += 1
        if ContextKeys.have_axe in goal.bounds:
            new_bounds[ContextKeys.have_axe].dec(1)
        else:
            # goal didn't care about an axe, doesn't change the goal
            pass

        # slight bias to keep the axe around (so it's cost is slighly more than
        # the value we get for selling it
        return Action("buy_axe", 20.), Goal(new_bounds)

class SellWood(ActionFactory):
    def __init__(self, sell_one: bool = False, sell_fraction: float = 1.0) -> None:
        self.sell_one = sell_one
        self.sell_fraction = sell_fraction

    def compatible(self, goal:Goal) -> bool:
        return (
            (
                goal.bounds.get(ContextKeys.money, EMPTY_BOUND).low > 0 or
                goal.bounds.get(ContextKeys.wood, EMPTY_BOUND).high < POS_INF
            ) and
            int(goal.bounds.get(ContextKeys.money, EMPTY_BOUND).low * self.sell_fraction) > 0 and
            goal.bounds.get(ContextKeys.location, EMPTY_BOUND).low in [0, Location.store]
        )

    def neighbor(self, goal:Goal) -> Tuple[Action, Goal]:
        new_bounds = self._copy_bounds(goal)

        #prec: location = store
        if ContextKeys.location in new_bounds:
            new_bounds[ContextKeys.location].set(Location.store)
        else:
            new_bounds[ContextKeys.location] = Bound(ContextKeys.location, Location.store, Location.store)

        if self.sell_one:
            amount_to_sell = 1
        else:
            amount_to_sell = int(goal.bounds.get(ContextKeys.money, EMPTY_BOUND).low * self.sell_fraction)

        # prec: 1 <= wood <= POS_INF
        # postc: wood -= 1
        if ContextKeys.wood not in new_bounds:
            new_bounds[ContextKeys.wood] = Bound(ContextKeys.wood, amount_to_sell, POS_INF)
        else:
            new_bounds[ContextKeys.wood].inc(amount_to_sell)

        # postc: money += 1
        if ContextKeys.money in new_bounds:
            new_bounds[ContextKeys.money].dec(amount_to_sell)
        else:
            # goal didn't care about money, doesn't change the goal
            pass

        return Action(f'sell_wood({amount_to_sell})', amount_to_sell), Goal(new_bounds)

class ChopWood(ActionFactory):
    def __init__(self, chop_one: bool = False, chop_fraction: float = 1.0) -> None:
        self.chop_one = chop_one
        self.chop_fraction = chop_fraction

    def compatible(self, goal:Goal) -> bool:
        return (
            (
                goal.bounds.get(ContextKeys.wood, EMPTY_BOUND).low > 0 or
                goal.bounds.get(ContextKeys.forest, EMPTY_BOUND).high < POS_INF
            ) and
            int(goal.bounds.get(ContextKeys.wood, EMPTY_BOUND).low * self.chop_fraction) > 0 and
            goal.bounds.get(ContextKeys.location, EMPTY_BOUND).low in [0, Location.woods]
        )

    def neighbor(self, goal:Goal) -> Tuple[Action, Goal]:
        new_bounds = self._copy_bounds(goal)

        #prec: location = woods
        if ContextKeys.location in new_bounds:
            new_bounds[ContextKeys.location].set(Location.woods)
        else:
            new_bounds[ContextKeys.location] = Bound(ContextKeys.location, Location.woods, Location.woods)

        # prec: 1 <= have_axe < POS_INF
        if ContextKeys.have_axe not in new_bounds:
            new_bounds[ContextKeys.have_axe] = Bound(ContextKeys.have_axe, 1, POS_INF)
        else:
            new_bounds[ContextKeys.have_axe].atleast(1)

        if self.chop_one:
            amount_to_chop = 1
        else:
            amount_to_chop = int(goal.bounds.get(ContextKeys.wood, EMPTY_BOUND).low * self.chop_fraction)

        # prec: 1 <= forest < POS_INF
        # postc: forest -= 1
        if ContextKeys.forest not in new_bounds:
            new_bounds[ContextKeys.forest] = Bound(ContextKeys.forest, amount_to_chop, POS_INF)
        else:
            new_bounds[ContextKeys.forest].inc(amount_to_chop)

        # postc: wood += 1
        if ContextKeys.wood in new_bounds:
            new_bounds[ContextKeys.wood].dec(amount_to_chop)
        else:
            # goal didn't care about wood, no change
            pass

        return Action(f'chop_wood({amount_to_chop})', amount_to_chop), Goal(new_bounds)

class GatherWood(ActionFactory):
    def __init__(self, gather_one: bool = False, gather_fraction: float = 1.0) -> None:
        self.gather_one = gather_one
        self.gather_fraction = gather_fraction

    def compatible(self, goal:Goal) -> bool:
        return (
            (
                goal.bounds.get(ContextKeys.wood, EMPTY_BOUND).low > 0 or
                goal.bounds.get(ContextKeys.forest, EMPTY_BOUND).high < POS_INF
            ) and
            int(goal.bounds.get(ContextKeys.wood, EMPTY_BOUND).low * self.gather_fraction) > 0 and
            goal.bounds.get(ContextKeys.location, EMPTY_BOUND).low in [0, Location.woods]
        )

    def neighbor(self, goal:Goal) -> Tuple[Action, Goal]:
        new_bounds = self._copy_bounds(goal)

        #prec: location = woods
        if ContextKeys.location in new_bounds:
            new_bounds[ContextKeys.location].set(Location.woods)
        else:
            new_bounds[ContextKeys.location] = Bound(ContextKeys.location, Location.woods, Location.woods)

        if self.gather_one:
            amount_to_gather = 1
        else:
            amount_to_gather = int(goal.bounds.get(ContextKeys.wood, EMPTY_BOUND).low * self.gather_fraction)

        # prec: 1 <= forest < POS_INF
        # postc: forest -= 1
        if ContextKeys.forest not in new_bounds:
            new_bounds[ContextKeys.forest] = Bound(ContextKeys.forest, amount_to_gather, POS_INF)
        else:
            new_bounds[ContextKeys.forest].inc(amount_to_gather)

        # postc: wood += 1
        if ContextKeys.wood in new_bounds:
            new_bounds[ContextKeys.wood].dec(amount_to_gather)
        else:
            # goal didn't care about wood, no change
            pass

        return Action("gather wood", amount_to_gather*10.), Goal(new_bounds)

class SellAxe(ActionFactory):
    def compatible(self, goal:Goal) -> bool:
        return (
            (
                goal.bounds.get(ContextKeys.money, EMPTY_BOUND).low > 0 or
                goal.bounds.get(ContextKeys.have_axe, EMPTY_BOUND).high < POS_INF
            ) and
            goal.bounds.get(ContextKeys.location, EMPTY_BOUND).low in [0, Location.store]
        )

    def neighbor(self, goal:Goal) -> Tuple[Action, Goal]:
        new_bounds = self._copy_bounds(goal)

        #prec: location = store
        if ContextKeys.location in new_bounds:
            new_bounds[ContextKeys.location].set(Location.store)
        else:
            new_bounds[ContextKeys.location] = Bound(ContextKeys.location, Location.store, Location.store)

        # prec: 1 <= have_axe < POS_INF
        # post: have_axe -= 1
        if ContextKeys.have_axe not in new_bounds:
            new_bounds[ContextKeys.have_axe] = Bound(ContextKeys.have_axe, 1, POS_INF)
        else:
            new_bounds[ContextKeys.have_axe].inc(1)

        # postc: money += 10
        if ContextKeys.money in new_bounds:
            new_bounds[ContextKeys.money].dec(10)
        else:
            # goal didn't care about money, doesn't change the goal
            pass

        return Action("sell axe", 10.), Goal(new_bounds)


class GoTo(ActionFactory):
    def compatible(self, goal:Goal) -> bool:
        return goal.bounds.get(ContextKeys.location, EMPTY_BOUND).low > 0

    def neighbor(self, goal:Goal) -> Tuple[Action, Goal]:
        new_bounds = self._copy_bounds(goal)

        # postc: location is whatever the goal wants
        del new_bounds[ContextKeys.location]

        return Action(f'goto({Location(goal.bounds[ContextKeys.location].low).name})', 5.), Goal(new_bounds)

action_factories:List[ActionFactory] = [
    BuyAxe(),
    SellWood(sell_fraction=1.0),
    SellWood(sell_fraction=0.5),
    SellWood(sell_one=True),
    ChopWood(chop_fraction=1.0),
    ChopWood(chop_fraction=0.5),
    ChopWood(chop_one=True),
    GatherWood(gather_fraction=1.0),
    GatherWood(gather_fraction=0.5),
    GatherWood(gather_one=True),
    SellAxe(),
    GoTo(),
]


@dataclass(order=True, eq=True, unsafe_hash=True)
class FitnessAndGoal:
    fitness: float
    goal: Goal


class GoalQueue:
    def __init__(self) -> None:
        self.queue = sortedcontainers.SortedSet(key=lambda x: -x.fitness)
        # mapping from a generic goal (with any fitness value) to the one
        # that's actually in the queue
        self.goals: MutableMapping[Goal, FitnessAndGoal] = {}

    def __len__(self) -> int:
        return len(self.goals)

    def __contains__(self, g:Goal) -> bool:
        return g in self.goals

    def __getitem__(self, g:Goal) -> Goal:
        return self.goals[g].goal

    def __iter__(self) -> Iterator[Goal]:
        for fg in self.queue:
            yield fg.goal

    def add(self, goal:Goal) -> None:
        f = FitnessAndGoal(goal.fitness_value, goal)
        self.goals[goal] = f
        self.queue.add(f)

    def remove(self, goal:Goal) -> None:
        f = self.goals[goal]
        self.queue.remove(f)
        del self.goals[goal]

    def pop(self) -> Goal:
        f = self.queue.pop()
        del self.goals[f.goal]
        return f.goal

def heuristic_cost(current_state: Goal, initial_state: State) -> float:
    # return estimate of cost between current_state and intial_state
    return current_state.delta(initial_state)

def choose_cheapest(open_set: GoalQueue) -> Goal:
    # return the state with lowest path cost + heuristic cost to initial_state
    return open_set.pop()
    #cheapest_cost = math.inf
    #cheapest:Goal = None # type: ignore[assignment]
    #for state in open_set:
    #    if state.fitness_value < cheapest_cost:
    #        cheapest_cost = state.fitness_value
    #        cheapest = state
    #open_set.remove(cheapest)
    #return cheapest

def reconstruct_path(came_from: Mapping[Goal, Tuple[Goal, Action]], current: Goal) -> Sequence[Tuple[Goal, Action]]:
    # return the sequence of edges from goal state to this state
    total_path = [(current, Action("initial", 0.))]
    while current in came_from:
        current, edge = came_from[current]
        total_path.append((current, edge))
    return total_path

def get_neighbors(state: Goal) -> Sequence[Tuple[Action, Goal]]:
    # return a set of edges from state
    neighbors:List[Tuple[Action, Goal]] = []
    for f in action_factories:
        if f.compatible(state):
            neighbors.append(f.neighbor(state))
    return neighbors

def cost(edge: Action) -> float:
    # return the cost of traversing edge
    return edge.cost


class CKeys(enum.IntEnum):
    in_open_set = 0
    in_closed_set = enum.auto()
    no_improvement = enum.auto()
    neighbor_options = enum.auto()

COUNTERS = [0] * len(CKeys)


def astar(goal_state: Goal, initial_state: State) -> Sequence[Tuple[Goal, Action]]:
    open_set = GoalQueue()
    goal_state.goal_value = 0
    goal_state.fitness_value = heuristic_cost(goal_state, initial_state)
    open_set.add(goal_state)

    closed_set: MutableMapping[Goal, Goal] = {}

    came_from: MutableMapping[Goal, Tuple[Goal, Action]] = {}

    global COUNTERS

    best_distance = math.inf
    best_goal_value = math.inf
    best_goal:Optional[Goal] = None
    while len(open_set) > 0:
        current = choose_cheapest(open_set)
        logging.debug(f'considering {current} with f_score: {current.fitness_value}')
        closed_set[current] = current

        current_distance = current.delta(initial_state)
        if current_distance == 0.:
            logging.debug(f'compatible with initial state')
            return reconstruct_path(came_from, current)

        if current_distance < best_distance or current_distance == best_distance and current.goal_value < best_goal_value:
            logging.info(f'closest goal: {current} cost: {current.goal_value} fitness: {current.fitness_value} distance: {current_distance}')
            logging.info(f'closed_set: {len(closed_set)} open_set: {len(open_set)}')
            best_distance = current_distance
            best_goal_value = current.goal_value
            best_goal = current

        for edge, destination in get_neighbors(current):
            #TODO: shouldn't we check if we have a shorter path to destination and re-open it if so?
            COUNTERS[CKeys.neighbor_options] += 1
            if destination in closed_set:
                COUNTERS[CKeys.in_closed_set] += 1
                destination = closed_set[destination]
            elif destination in open_set:
                COUNTERS[CKeys.in_open_set] += 1
                destination = open_set[destination]

            tentative_g_score = current.goal_value + cost(edge)
            if tentative_g_score < destination.goal_value:
                came_from[destination] = (current, edge)
                destination.goal_value = tentative_g_score
                destination.fitness_value = tentative_g_score + heuristic_cost(destination, initial_state)
                logging.debug(f'adding {destination} via {edge.name} to open set with g_score {destination.goal_value} f_score {destination.fitness_value}')
                if destination in open_set:
                    # destination came from open set, but we have a cheaper way
                    # to get there. before we modify it, let's remove it
                    open_set.remove(destination)
                    open_set.add(destination)
                else:
                    if destination in closed_set:
                        del closed_set[destination]
                    open_set.add(destination)
            else:
                COUNTERS[CKeys.no_improvement] += 1

    raise Exception("ohnoes")

if __name__ == "__main__":
    try:
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)

        goal = Goal({
            ContextKeys.money: Bound(ContextKeys.money, 50, POS_INF),
            ContextKeys.have_axe: Bound(ContextKeys.have_axe, 1, POS_INF),
            #ContextKeys.forest: Bound(ContextKeys.forest, ZERO, 50),
        })
        initial_state = State({
            ContextKeys.forest: 100,
            ContextKeys.have_axe: 0,
            ContextKeys.wood: 0,
            ContextKeys.money: 40,
            ContextKeys.location: 0,
        })

        logging.info(f'looking for a solution for: {goal}')
        logging.info(f'starting conditions: {initial_state}')

        starttime = time.perf_counter()
        #import tqdm
        #for i in tqdm.tqdm(range(30)):
        #    solution = astar(goal, initial_state)
        solution = astar(goal, initial_state)
        endtime = time.perf_counter()

        solution_cost = sum(x[1].cost for x in solution)
        logging.info(f'solution of length {len(solution)} of cost {solution_cost} found in {endtime-starttime:.2f}s')
        counter_str = "\n".join(f'{k.name}:\t{COUNTERS[k]}' for k in CKeys)
        logging.info(f'counters:\n{counter_str}')
        for goal, action in solution:
            print(f'{action.name} {action.cost} {goal} cost: {goal.goal_value} distance: {goal.delta(initial_state)}')
    except Exception as e:
        logging.error(f'handling exception {e}')
        pdb.post_mortem(sys.exc_info()[2])
