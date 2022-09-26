# Stellarpunk

A space sim of trading, stealth naval combat and interplanetary economic
development.

# Pillars of Gameplay

Three sets of mechanics form the core of the gameplay:

* Spaceflight
* Trading,
* Characters Dynamics

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
- [ ] basic combat (think submarines or "bazookas and flashlights")
- [ ] consistent sensor model between combat and non-combat situations
- [ ] docking experience
- [ ] different kinds of ships?

### Combat

- [ ] basic sensors
- [ ] basic weapons and damage model
- [ ] combat AI
- [ ] what does losing combat mean?

## Trading

- [ ] basic economic simulation (PoC)
- [ ] integrating simulation into rest of game
- [ ] trading/cargo transport + manufacturing facilities interaction

### Economic Simulation

- [x] price setting strategy that facilitates full production chain
- [ ] optimize price setting to maximize profit
- [ ] figuring out roughly correct scales for number of each producer
- [ ] agents deciding to exit a market and enter a new one
- [ ] adding/removing agents

## Character Dynamics

- [ ] character framework hooked into ships, stations, etc.
- [ ] basic character relationship framework
- [ ] basic/short-term missions and employment
- [ ] long-term subordinate/employment system, including independent balance and depdendent budget
- [ ] basic background, attributes/traits, overall goals
- [ ] character motivation/agenda setting/AI
- [ ] factions/political organizations

## Other Stuff

- [ ] news generation

### World Generation

- [x] basic universe layout (including transit paths)
- [x] basic sectors layout (asteroids, stations, ships)
- [ ] good distribution of resources/production facilities
- [ ] history recording important events
- [ ] integrating relevant history/bios for characters/universe objects

