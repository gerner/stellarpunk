Missions
========

These are directions created by one character for another to fulfill. This
might be an employer/employee relationship, this might be a mission on a job
board. When a character, the poster, has a desire/need, they can create one of
these tasks and _somehow_ get another character, the agent, to do it.

## Examples

For example, a station manager might need resources so they get someone else to
collect them (outside of standard trading). In this case the agent is operating
on behalf of the original task poster. They mine or buy resources (should not
really matter, right?) and transfer them to the poster's station. The agent
might get paid as part of the task, but they aren't trading the resources to
the station directly.

For example, a pirate lord wants to steal cargo and have it delivered to a
trader. The poster indicates a target of some kind and the cargo in question
and the destination trader (ship?, station?) to deliver it to.

For example, a merchant baron has some cargo and wants it delivered. Some agent
is directed to (optionally) collect the cargo from some place (ship? station?)
and sell it to some other. In this case the agent's trade happens through the
poster's account.

For example, the tutorial missions could all be tasks: mining, trading, etc.
What happens if the player (or any character) doesn't follow tasks from their
employer? When/how do we determine this has happened? What are the
conseqeuences? Has the character now stolen the ship? Do bounty hunters or
police come after them?

Player character works for some character or org. When game starts they get
assigned the tutorial mining task. That includes enough information for the
player to do it. Maybe there's followup tasks. This looks like the tutorial
event chain. The key differences are that the tasks are materialized to be
interacted with by the player, they have explicit rewards, the tasks can be
failed. Events here are generic. Tasks are specialized.

## Details

* Tasks come with a payment of some form, but also improve relations with the
  poster.
* Re-use IntelMatchCriteria to specify some elements of the job (e.g. the
  destination station to deliver goods)
* Tasks have completion criteria to determine when complete. This could be in
  terms of event flags so we can use AI planning (e.g. GOAP) to complete the
  task.
* Tasks have failure criteria. This could be a time limit or some
  sector/location limit.
* Multiple tasks can be undertaken simultaneously. They can be completed or
  failed simultaneously.
* Tasks can be assigned to subordinates (should be coordinated between
  supervisor/subordinate)
* Characters can seek freelance tasks according to some logic
* Tasks have a type or template and instances of those tasks
    * Task template defines broad criteria, how it can be presented to the
      player (or other characters), etc.
    * Task templates are parameterized (e.g. which resource to mine, how much
      to mine, which station to sell it to)
    * Task instances fill in template parameters
    * There can be many instances of a template simultaneously or over time

### Characterizing a Task

* Tasks have a poster character (or faction?) and an agent (character)
* Tasks have a lifecycle (see below)
* Tasks have a reward (e.g. money, poster reputation)
* Tasks have failure criteria defining how/when a task fails
* Tasks have success criteria defining how/when a task succeeds

### Lifecycle

* Open - created by the poster, but yet to be assigned to any agent.
* Assigned - assigned to the agent, has not met either failure or success criteria.
* Fulfilled - task has not met failure criteria and has met success criteria.
  task has not yet been turned into the poster, rewards have not yet been
  dispensed.
* Failed - an assigned task has met the failure criteria.
* Completed - task has been fulfilled and has been turned in and rewards have
  been dispensed.
* Cancelled - the task (assigned or not) is no longer relevant for some reason,
  perhaps the poster cancelled it.

# Strawman

Two characters: poster, agent. Poster decides to create a task. Task gets
assigned to agent. Agent plans to carry out the task.

Task has a goal (success criteria). Task has failure criteria. Agent ignores
failure criteria (for now).

Agent uses success criteria to come up with a strategy to fulfill the task,
given a set of actions they can take (e.g. traveling to a location, mining an
asteroid, trading with a station, etc.)

