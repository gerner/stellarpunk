# Stellarpunk

A space sim of trading, stealth naval combat and interplanetary economic
development.

# Pillars of Gameplay

Three sets of mechanics form the core of the gameplay:

* Spaceflight
* Trading, production chain, economy
* Characters dynamics

Spaceflight includes transportation, navigation, sensing, combat and the
physics and mechanics to support those things. The space sim gameplay includes
features you might have seen in games like Escape Velocity. The aesthetic is
grimey and low-fidelity. You interact with the environment not through a 1:1
visual interface, but the kinds of scopes and consoles a space operator running
on a budget might use. Here we draw inspiration from '70s and '80s style
computer UIs. Think Alien's Nostromo. We also draw inspiration from submarine
naval combat simulators like Cold Waters. This is the "bazookas and
flashlights' style combat seen in games like Objects in Space.

Trading is the economic simulation including modelling a production chain,
sourcing raw resources, sinking ultimate goods, combinging input resources into
manufactured goods, setting prices, forming a market and various economic
indicators and interactions with other systems in the game. The universe is
filled with rational (sometimes not so rational) agents trying to maximize
profits with limited resources. The AI, playing many competing agents, should
be able to form a successful economy, even with adverse factors like piracy,
resource shortages, etc. The player is invited to make a profit in this
challenging landscape, outdoing the competition, building an industrial empire.

Character dynamics drive the motivations of and relationships between
characters, including the one character the player plays as. Here we draw
inspiration from grand strategy games like the Crusader Kings franchise. The AI
plays the game in the same way as the player, through characters who have their
own goals, limited information, and relationships. Characters might work for
one another, might lead or follow political factions. The universe is filled
with many different kinds of jobs like trader, factory manager, company leader,
government official, political leader, police officer, pirate, rebel. Each
Character has a limited scope of action and takes advantages of relationships
to expand that scope: subordinate, superior, partner, ally, rival. The player
takes part in this web of relationships, just like each AI character, to
accumulate power.

# Plan

## Spaceflight

- [x] physics model
- [x] basic spaceflight simulation
- [x] ship navigation/autopilot
- [x] collision avoidance
- [x] universe vs sectors
- [x] inter-sector transport
- [x] mining
- [x] cargo transport and cargo transportation
- [x] basic player flight controls
- [x] player can mine and trade
- [ ] docking experience
- [x] basic combat (think submarines or "bazookas and flashlights")
- [x] sensor
- [ ] different kinds of ships?

### Player Mining/Trading

- [x] player directed mining
- [x] UI improvements for ship status view
- [x] production chain product names
- [ ] comms and player interaction
- [ ] handle cargo transfer for owned/directed ship (no cash transfer)
- [ ] player directed trading
- [x] discovering asteroids/stations
- [ ] multi-sector trading/mining

### Intel System

- [x] data model for intel, knowing POIs, knowing prices
- [x] intel lifecycle: created and added to a character's knowledge, expires or
      somehow goes away
- [x] intel used by miners and traders
- [x] intel integrated into sensors for identification and physical params
- [ ] buying/selling intel
- [ ] intel is available in intel forums
- [x] behavior for gathering intel (captain)
- [ ] behavior for gathering intel (non-captain)
- [ ] behavior for distributing intel
- [ ] intel manager interface for player to see intel
- [x] intel serialization for save/load

## News System

- [ ] events can be recorded creating news items
- [ ] stories are created from news items for the player to read
- [ ] news is available in news forums
- [ ] news interface for managing and reading stories

### Station Experience

- [x] buy/sell goods
- [ ] meet other characters (sense of activitiy, content/interaction opp)
- [ ] trigger for events (on docking, on leave, maybe other actions)

### Combat

- [x] basic sensors
- [x] basic weapons
- [ ] damage model
- [x] combat AI
- [ ] what does losing combat mean?

## Trading

- [x] basic economic simulation (PoC)
- [ ] integrating simulation into rest of game
- [x] trading/cargo transport + manufacturing facilities interaction
- [ ] economic indicators hooked into news generation

### Economic Simulation

- [x] price setting strategy that facilitates full production chain
- [ ] agents deciding to exit a market and enter a new one
- [x] optimize price setting to maximize profit
- [x] figuring out roughly correct scales for number of each producer
- [x] adding/removing agents

## Character Dynamics

- [x] character framework hooked into each ships, stations, etc.
- [ ] characters have jobs
- [ ] lots of characters including those that aren't ship/station operators
- [ ] basic character relationship framework
- [ ] basic/short-term missions and employment
- [ ] long-term subordinate/employment system, including independent balance
      and depdendent budget
- [ ] basic background, attributes/traits, overall goals
- [ ] character motivation/agenda setting/AI
- [ ] factions/political organizations
- [ ] experiment with character death and taking over some other character
      (including suggesting a character to take over)

## Other Stuff

- [ ] news generation
- [ ] TBD progression (agenda experience? questlets?)
- [ ] content: storylet system

### World Generation

- [x] basic universe layout (including transit paths)
- [x] basic sectors layout (asteroids, stations, ships)
- [ ] good distribution of resources/production facilities
- [ ] history recording important events
- [ ] integrating relevant history/bios for characters/universe objects
- [ ] characters start with some intel/knowledge of sector/universe

