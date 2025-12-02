Character Dynamics
==================

# Goals

- [ ] Characters have relationships with any other character
- [ ] Those relationships change character behaviors
- [ ] Relationships act as a form of progression, unlocking mechanics and
      opportunities for gameplay
- [ ] Faction/member relationships
- [ ] Employer/employee relationships
- [ ] Organizations (company, faction) are things characters interact with
- [ ] Employees/faction-members have behavior on behalf of employer/faction
      (aka jobs and jobs direct action)
- [ ] Not all characters are captains
- [ ] Relationships are impacted by actions
- [ ] Relationships are impacted by background (is this just a compressed set
      of actions?)
- [ ] Characters have missions/tasks they want performed and post, other characters take on those tasks/missions and accomplish them

It goes without saying, the player character participates in all of this.

# Types of Relationships

## Character-Character

Every character can have a relationship with every other character. This greatly influences how the characters interact. A good relationship might cause a character to overlook trespasses, give better prices in trades, or come to the aid of another character. Poor relationships might cause the opposite.

Relationships can be represented on a single numeric scale. Higher values are
better than lower values. The scale might be 100 to -100.

There are many factors that impact the relationship score. Past interactions
should figure in greatly. Even simple prior interactions like conducting trade
between two characters or docking at a station managed by a character can
improve relations. Multiple instances of an interaction should have a reduced
impact (positive or negative).

### Relationship Factors

 * Conducting trade between characters (i.e. econ agents)
 * Conducting trade between captains (i.e. ship/station regardless of owner)
 * Docking at a station
 * Attacking (owner/owner and captain/captain)
 * Combat support (TODO: combat needs to make friend/foe more explicit)
 * Performing a task for (either employer/employee, or freelance)

### Relationship Effects

 * Share tasks/mission opportunities
 * Accept tasks/missions
 * Docking access (certainly owner/owner or owner/captain, but station
   captain/captain too)
 * Trade prices (certainly owner/owner or owner/captain, but station
   captain/captain too)
 * Trade prioritization (plan to trade with friendly characters)
 * Travel gate access
 * Intel sharing
 * Combat support
 * Combat aggression

## Factions

Factions group characters together and act somewhat like a government. They own
and control ships, stations, travel gates, sectors. Relationships with them
dictate what characters can do with those things: enter a sector, dock at a
station, use a travel gate.

Factions might peacefully coexist within space. Factions have relationships
with each other that influence how members of each faction interact with
characters of other factions.

Characters can belong to a faction, whose relationships influences their
behavior. They might have independent relationships with many factions which
further influence their behavior, creating diversity in behavior across
characters in the same faction.

### Questions

 * Do factions have a single leader? how is succession handled?
 * How is a faction relationship different than a relationship with the leader?
 * How is a faction different than an employer?
 * Can characters belong to multiple factions? (what if the factions have
   negative relations?)

### Characterizing a Faction

 * name
 * assets
 * members
 * leader(s)
 * relationships with other factions and with characters

A faction member's relationship with a character is affected by, but separate
from the faction relationship with that character.

## Employer/employee:

One character (or faction?) is an employer, conducting some kind of business.
They have suborinate employees that perform tasks to aid in that busines.
Employees are compensated for performing the tasks. Employees might have access
to assets (ships, stations) owned by the employer for use in performing those
tasks. Employees must perform the tasks if they want to remain employed. If
they leave the employer, they must return the assets, or else they're stealing
the asset and are a criminal.

Kinds of jobs:
 * Crew on ship or station: just need to be there for the ship/station to
   function properly. tasks are just to stay as crew
 * Free trade captain/mining captain. tasks are to make trades
 * Station captain: tasks are to make production (what about docking operations?)
 * Refinery miner captain: tasks are to mine resources for a refinery
 * Station supply trader: tasks are to buy input resources for a station
 * Station demand trader: tasks are to sell output products for a station

# Missions/Tasks

See `designs/missions.md` for details.

