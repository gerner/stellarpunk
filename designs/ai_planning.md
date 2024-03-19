# Use-Case: NPCs doing Quests

## Sample Quests

 * Tutorial event chain:
   * player gets a message from tutorial guy to come see him -> dialog -> start
   (or skip) the tutorial.
   * player goes to gather resources
   * player trades resources at the tutorial station
 * Take passenger from one station to another
   * simple case: just get them there
   * complex case: something happens on the way, need to do some other goal
 * Build a station in a sector
 * Deliver goods to a station

## Tutorial Quest

What if we want any NPC to do this event chain? We could put some goals to that
NPC to accomplish the relevant tasks in the chain: go see the tutorial guy,
gather enough of the relevant resource, trade the resource with the right
station. We express each of these steps as some state. In fact, we have already
done that in our event system as the followup events!. And if we can do that,
we should be able to build a plan of actions our AI can follow to accomplish
those quest tasks.

Suppose that our AI can do things like:

* Action: Talk with a given NPC
  * parameters: `who`
  * prec: `player.location=$who.location`
  * postc: `talk.character=$who`
* Action: Go to a given sector entity
  * parameters: `where`
  * prec: `player.location.is_ship`
  * postc: `player.location=$where`
* Action: Scout for resources (or scout for a given resource)
  * parameters: `resource`
  * prec: `player.location.is_ship`
  * postc: `resource_intel.resource=$resource`
* Action: Mine a resource of given resource
  * parameters: `resource, amount`
  * prec: `resource_intel.$resource=$resource, player.location=resource_intel.location`
  * postc: `cargo.$resource=$amount`
* Action: Trade a given amount of a given resource at a given station
  * parameter: `resource, where, amount`
  * prec: `cargo.$resource=$amount, player.location=$where`
  * postc: `trade.location=$where, trade.resource=$resource, trade.amount=$amount`

If we can express our goals like so:

* Goal: Talk to tutorial guy
  * parameters: `who=$tutorial_guy`
  * goal: `talk.character=$who`
* Goal: Gather tutorial amount of tutorial resource
  * parameters: `resource=$resource, amount=$amount`
  * goal: `cargo.$resource=$amount`
* Goal: Trade tutorial amount of tutorial reosource with tutorial station
  * parameters: `where=$where, amount=$amount, resource=$resource`
  * goal: `trade.amount=$amount, trade.resource=$resource, trade.location=$where`

For each of those goals we should be able to assemble a sequence of actions
that accomplishes the desired goal using a technique like Goal Oriented Action
Planning (GOAP).

In fact, we could just jump to that last goal (of wanting to trade the resource
to the station) and come up with the relevant plan. The tutorial (which is
strictly structured in the above steps) is sort of spoon feeding you a coarse
plan.

# Goal Oriented Action Planning

Key Concepts:

 * World State: a set of variables describing the world. Can use context and
   entity contexts (a la events)
 * Goal: expressed as a desired world state (this is a subset of the actual
   world state)
 * Action: transformation from one world state to another, with some
   precondition predicates on the world state
 * Plan: a sequence of actions to reach the goal such that each action is valid
   in the context of the world state at the point in the plan that action
   falls.

## Actions

Good actions should be modular/reusable with clear behavior we can implement in
game code. For instance, we already have a rich navigational capability, so
offering a planning action of `GoToLocation` makes a lot of sense. Perhaps
there are variants on how that's parameterized: near some sector entity, near
some specific location, etc.

## Plan Search

Actions cannot be nodes in our search. We're searching through states and
actions provide adjacency in that state space. For instance, consider the
following setup:

Actions:

* Buy Axe
  * prec: `money >= $20`
  * postc: `money -= $20, have_axe = 1`
  * cost: 1
* Sell Wood
  * prec: `wood >= 1`
  * postc: `money += 1, wood -= 1`
  * cost: 1
* Chop Wood
  * prec: `forest >= 1, have_axe = 1`
  * postc: `forest -= 1, wood += 1`
  * cost: 1
* Gather Wood
  * prec: `forest >= 1`
  * postc: `forest -=1, wood += 1` <--- same effect as Chop Wood!
  * cost: 10

Goal: `money >= 30`
Initial State: `forest = 1000, have_axe = 0, money = 25, wood = 0`

A reasonable plan here is:

 1. Buy Axe (`money -> 5`, `have_axe -> 1`)
 2. Chop Wood (`wood -> 1`)
 3. ...
 26. Chop Wood (`wood -> 25`)
 27. Sell Wood (`wood -> 24`, `money -> 6`)
 28. ...
 51. Sell Wood (`wood -> 0`, `money -> 30`)

Notice, we repeat the Chop Wood and Sell Wood actions. What's different is the
context we're in at each of those.

constructing the plan (from goal toward initial state):
Add Sell Wood (goal: `money >= 29, wood >= 1`)
Add Sell Wood (goal: `money >= 28, wood >= 2`)
...
Add Sell Wood (goal: `money >= 0, wood >= 30`)
Add Chop Wood (goal: `money >= 0, wood >= 29, has_axe >= 1`)
Add Chop Wood (goal: `money >= 0, wood >= 28, has_axe >= 1`)
...
Add Chop Wood (goal: `money >= 0, wood >= 0, has_axe >= 1`)
Add Buy Axe (goal: `money >= 20, wood >= 0, has_axe >= 0`) <--- compatible with initial state

alternatively:
Add Buy Axe (goal: money >= 50) --> worse than before, unlikely we'll expand this
