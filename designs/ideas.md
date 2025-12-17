Ideas
=====

# Misc Ideas

 * Characters have a nemesis
 * Characters plot to overthrow leaders of factions

# Projects

 - [ ] sensors in cython
 - [ ] universeview and/or all of intel in cython
 - [ ] fuel and/or lifesupport resources + NPC logic to refill it somehow
 - [x] multi-sector NPC travel, mining, trading
 - [x] make mining and trading not use intel instead of SectorEntities and
       EconAgents as input params directly (and maybe decide when we can go
       from intel/sensor image to the actual entity or tell the player it's
       toast)
 - [ ] non-captain characters (crew, passengers seeking travel, reporters)
 - [ ] character/faction relationships inlcluding a framework to hurt/improve
       relations
 - [ ] mission system with NPCs posting and  doing missions

# Stuff for Player to Do

* Mine and sell resources
* Buy/Sell goods trying to make a profit
* Transport characters to desired locations. Why?
    * Characters want to gain intel
    * Crew need to get from home to work their tour on a station or other craft
    * Characters specifically created to travel
* Collect and sell intel, aka explore
* Hunt and kill (or capture?) a specific bounty (a character)
* Patrol a sector for another character (or org?)
* Ambitions a la Battle Brothers

# News

BBS style information where you get imperfect info.

Player wants deltas as events unfold. How to visualize this in a strategic
layer?

Heatmap of universe-wide traffic: thicker, brighter links with high traffic.
Holes or dark areas indicate low traffic for some reason.

# Combat

* Different kinds of torpedoes a la highfleet or cold waters (e.g. own sensors, home on enemy sensors, different search patterns)
* Boost or afterburners to dodge projectiles a la highfleet

# Comms Chatter

Have a broadcast channel in a sector (or localized in some radius in a sector?)
Characters can broadcast a message (hopefully contextually relevant)
This broadcast triggers an event that gets received by other characters and the
player.
Characters can respond to that event in lots of ways. Perhaps with another
broadcast.
Maybe someone calls for help, or announces a special deal on a good, or asks
for directions and other characters react to that with help or directions, or
deliver goods. This might affect trade or relationships.

But maybe this just creates back-and-forth irrelevant chatter. Someone asks for
help, someone else says, "good luck buddy, you're on your own." Or someone else
says, "Anyone know where I can unload a batch of base metals?" and someone else
says, "I've been trying to unload base metals for four cycles. No luck around
here." And then someone else responds, "I heard you can find good deals in
sector XYZ" or something. Maybe this is real info, but it doesn't have to be to
create a sense of life.

# Crew

Every ship has multiple characters, beyond just the captain. Each has a job on
the ship which affects the ship's performance (mining, trading, sensors,
weapons, etc.) Their relationships with each other are very important because
it adds content and affects their ability to do their jobs.

Player sees dialog between characters, even if it's just flavor. But maybe
these are also interactions that affect relationships, give missions or
bonsues.

Player could be a crew member and not just the captain!

# Missions

* Characters have needs that the player (or any other character) can satisfy:
    * Mining raw resources for production
    * Buying goods for production
    * Deliver goods from one owned station to another
    * Ferry passengers/crew
    * Gather intel
* Defend a sector/sector entity
* Patrol a sector and enforce "laws" (e.g. no mining, no trading certain goods,
  defend from pirate attacks)
* Do piracy (e.g. factions warring on each other, you're a privateer)
* Bounty hunting
* Ferry a character, or goods, or intel/news to a location that a player has
  never been to promote exploration, or through dangerous space

# Characters can do Quests and AI Planning

Consider the tutorial event chain:

 * player gets a message from tutorial guy to come see him -> dialog -> start
   (or skip) the tutorial.
 * player goes to gather resources
 * player trades resources at the tutorial station

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
  * desire: `talk.character=$who`
* Goal: Gather tutorial amount of tutorial resource
  * parameters: `resource=$resource, amount=$amount`
  * desire: `cargo.$resource=$amount`
* Goal: Trade tutorial amount of tutorial reosource with tutorial station
  * parameters: `where=$where, amount=$amount, resource=$resource`
  * desire: `trade.amount=$amount, trade.resource=$resource, trade.location=$where`

For each of those goals we should be able to assemble a sequence of actions
that accomplishes the desired goal using a technique like Goal Oriented Action
Planning (GOAP).

In fact, we could just jump to that last goal (of wanting to trade the resource
to the station) and come up with the relevant plan. The tutorial (which is
strictly structured in the above steps) is sort of spoon feeding you a coarse
plan.

## Issues

Things like "trade", "talk", seem like instances that might co-exist with other
instances of the same type. Do we model that? how do we select one particular
trade and make it clear that we are talking about that same instance when we
say things like `trade.resource=$foo` and `trade.amount=$bar`? Will we ever
want to have two instances at the same time? Notationally we could do something
like `trade1 = select(trade)`, `trade2 = select(trade)`.

There's probably some weirdness about the  parameter binding I'm doing above.

# Visual Effects/UI

* show crafts travel lanes in universe view, perhaps also implies that we model
  travel from one sector to another.
*

# Managing a Reactor

Lots of power, on the order of dozens of GW, are required to achive the desired
acceleration (that's fun) for masses that are even approaching realistic
(several metric tons, although just the orbiter is 110 tons and most large
rockets that can enter LEO are 1000s of tons). At launch, Starship generates
100GW of power on launch for a 2nd stage dry mass of 100 tons.

A 2 ton vehicle generating 200kN of thrust which only needs 2.5 tons of
propellant for 5 minutes of constant thrust (implying an $I_sp$ of 60,000
seconds, comparable to SOTA ion and plasma engines) needs to generate nearly
60GW of exhaust energy. In a chemical rocket that energy comes from the
propellant itself (i.e. combustion). In an electric rocket (e.g. an ion or
plasma rocket) that energy needs to be supplied to the rocket engine. You could imagine some kind of hybrid where the fuel provides energy

Systems to manage:
* magnetic confinement (superconducting, zero energy cost once established and maintained at cryogenic levels)
* crygenic cooling systems
* plasma heating to reach fusion temperatures (injecting high energy fuel, Pulsed Power Electrical Network generates EM waves tuned at resonant frequency, direct energy "ohmic heating" through conductive plasma, like a toaster
* fuel injection: crygenically frozen pellet injection system and gas injection system
* closed-loop fuel recycling (since only a small portion of injected fuel is consumed): exhaust collection, fusion "ash" byproduct removed to prevent poisoning the plasma, separation and isotope refinement (to maintain fuel mix ratio)
* energy generation: traditional coolant loop: primary coolant to secondary heat exchange, steam production, steam turbines, generators, condensers




# Sources

## AI Planning

### Applying Goal Oriented Action Planning to Games
* [paper](https://alumni.media.mit.edu/~jorkin/GOAP_draft_AIWisdom2_2003.pdf)

### Three States and a Plan: The A.I. of F.E.A.R.
* [paper](http://alumni.media.mit.edu/~jorkin/gdc2006_orkin_jeff_fear.pdf)

### Building the AI of F.E.A.R. with Goal Oriented Action Planning | AI 101
* [video](https://www.youtube.com/watch?v=PaOLBOuyswI)

### Exploring HTN Planners through Example
* [chapter](http://www.gameaipro.com/GameAIPro/GameAIPro_Chapter12_Exploring_HTN_Planners_through_Example.pdf)

## Events, Storytelling, Narrative Systems

### Monster Train feedback system, "press F8 to give feedback":
* Roguelike Celebration talk by Brian Cronin of Shiny Shoe
* [video](https://youtu.be/qO3CIpP62Q0?t=959)
* zero friction, no un/pw, no separate website.
* collect username, build version, "run id", log file, save file (start of
  battle), screenshot
* take consistent action

### Monster Train action graph for processing complex interactions:
* Roguelike Celebration talk by Brian Cronin of Shiny Shoe
* [video](https://youtu.be/qO3CIpP62Q0?t=844)
* make control flow and effects explicit with a DSL. "cards are a programming
  language"

### Hades dialog priority system
* People Making Games interviews with Greg Kasavin and Amir Rao of Supergiant Games
* [video](https://www.youtube.com/watch?v=bwdYL0KFA_U)
* in any situation perhaps many dialogs are relevant, so prioritize them and
  execute the highest priority one.

### Thoughts on General Purpose Quality-Based Narrative
* Blog post by Bruno Dias
* [article](https://brunodias.dev/2017/05/30/an-ideal-qbn-system.html)

### AI-driven Dynamic Dialog through Fuzzy Pattern Matching
* GDC talk on key-value fact querying rule database for dialog
* [video](https://www.youtube.com/watch?v=tAbBID3N64A)

### Ship Names
* https://www.dco.uscg.mil/Our-Organization/Assistant-Commandant-for-Prevention-Policy-CG-5P/Inspections-Compliance-CG-5PC-/Office-of-Investigations-Casualty-Analysis/Merchant-Vessels-of-the-United-States/

### geonames
* https://geonames.org/
