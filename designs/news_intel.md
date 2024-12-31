News and Intel
==============

# Intel

Intel manager keeps track of known points of interest (stations, asteroid
clusters, entire sectors in a universe context, etc.), prices (which might be
out of date), inventories (stations and asteroids)

This information can be discovered, collected, distributed, bought and sold.

## Uses of Intel

Characters use intel in planning. This ties into sensors and how characters
should not have direct access to entities, etc. Instead they can get sensor
images which may or may not be identified. When planning trading or mining, a
character should not directly reach into the set of all stations or asteroids.
Instead they should rely on the intel they have to make that trade. This
creates interesting dynamics where one trader might have access to better intel
about profitable trade routes. A miner might not know where to find the best
resources.

The player can be very strategic here about exploration. This can motivate a
lot of gameplay: exploration is valuable directly to the player to find trade
routes or mining fields, intel gained during exploration is also valuable
itself as something that can be bought and sold to other characters. Different
characters have different needs and values for acquiring new intel. This also
makes inter-sector travel especially valuable since a trader might not be
motivated to travel to a different sector, so they might not naturally acquire
intel about entities in that sector the way they might while travelling within
a sector.

## Intel Mechanics

* Intel and sensors: sensor identification is tied to intel so known objects
  can be quickly identified and known physical parameters can be used from past
  experience.
* Intel used for planning, especially trading/mining: asteroid and station
  existence and location is intel. prices are intel too, although they might be
  outdated.
* Intel can be naturally acquired: seeing an asteroid or a field of asteroids
  adds intel about those resources. Similarly for stations and prices learned
  by visiting a station.
* Intel can be shared, bought or sold: One character with intel can share that
  with another and can be compensated for that. A character need not personally
  witness intel to gain it and used it. E.g. characters can explore as a
  profitable occupation.
* Intel are entities. Two Characters that have the same piece of intel refer to
  the same entity.

Questions:
* Do you need to dock at a station to learn prices? Will characters be
  motivated to dock at a station just to gather that information?
* If knowledge of an asteroid field is created, is that a new piece of intel?
  should we reference an existing piece of intel about that asteroid field?
  What about intel that is time bound, like prices?
* How does intel age and become invalid?
* How fine grained should intel be? Should each asteroid be its own piece of
  intel, or the resource field in general?
* Can there be false intel?

### Sensor Integration

* Identifying a sector entity should trigger creating some intel about it,
  specifically where it was last seen.
* Having intel about a sector entity should enhance what we know about it once
  it is identified
    * What resource(s) are bought/sold by a station at what prices (last seen)
    * Who's the captain and TBD properties like allegances/relationships
    * Other TBD parameters (perhaps combat parameters like armament)

How do we gain detailed intel about the sector entity? docking =? resources bought/sold and prices. Comms => captain? maybe just getting a strong enough sensor reading?

# News

News is how the player can observe the universe as a whole. This reveals and
describes game dynamics like how the economy is operating how important
characters or organizations are behaving or interacting. News records events
and creates a story describing that event. The player can consume these as game
content and also to strategize their gameplay. Characters can potentially
consume these as intel they can use in their behavior planning.

* Some pieces of news (and corresponding intel) like this might get shared with
  some public news forum for anyone to retrieve, as long as they can access
  that forum.
* Some pieces of news will not be publically shared this way. It probably makes
  sense for the player to differentiate between written content related to news
  like this and content related to private intel.
* A news story can indirectly reference other pieces of intel, including
  information about all of them. For instance, we should be able to write a
  news story about an entire field of asteroids. Or we should be able to write
  a news story about an attack by one ship, captained by a particular Character
  on another ship, captained by a different Character, leading to some piece of
  cargo being left behind in the battle.

Ideas:
* News could also be used to distribute information (i.e. Intel).
* News could be distributed by Characters (i.e. courier)
* Publicaly (Re)Discovering an asteroid field is a news event
* Reporter could be an occupation for a Character who gathers news items by
  observing them and then sharing them with a public forum. The news forum
  should compensate them.

Questions:
* Are News and Intel the same thing?
* What is an "event" (or is it anything about which News/Intel exists?)
* What's a news forum? is that an organization? where does their money come
  from? Subscriptions characters can make to get access to the forum?

# Strawman Implementation
* A piece of intel might automatically be created according to some trigger (is
  this plugged into the event system?)
* No or minimal effort to dedupe this intel with other intel. E.g. maybe if a
  piece of intel is permanent with limited scope we can search for a matching,
  existing piece of itel. but otherwise don't worry about duplication. This is
  especially not a problem if intel is likely to expire.
* Pieces of intel might be redundant or partially matching where a Character
  can prefer one piece to another and assign some value to replacing an
  existing partial match. E.g. price information for a resource at a station
  that is newer than some existing price information for the same resource at
  the same station might be preferred, replacing that existing piece of intel.
* If a piece of intel is transferred from one Character to another, they both
  refer to the same object, sharing it.
* Intel should be immutable or any changes should make sense to be shared by
  anyone referring to that intel, even if separated, without any communication
  channel.
* Each Character has an intel manager that keeps track of their intel. The
  Character, and their agenda can consult this intel manager for relevant
  information: the locations of known asteroids, prices of goods at stations,
  bounties on ships/characters, etc.
* The model is that this information is "learned" from the acquired intel and
  is a function of the intel in the manager. So if the intel goes away, that
  learned knowledge should change.

## Knowledge
Stuff we might want to know that is a function of the intel we have:
* SectorEntities: locations, properties, basically access to the actual
  SectorEntity. If this is a ship or other ephemeral/dynamic object this might
  not be valid for long or might be incorrect.
* SectorEntities matching some criteria: asteroids of a particular resource
  near a point, stations selling a particular resource near a point
* Characters: known characters, but not some dynamic properties, like their
  location
* Where to find, or to start looking for a particular character
* Prices:
    * sales: resource, amount desired, sector entity selling
    * buys: resource, amount available, sector entity selling
* Level of danger in a sector (probability of combat/harassment?)
* Route from one sector to another, or just the edges to/from a sector so we
  can compute a route.

## Kinds of Intel

* Last known location of a Character: including when they were seen there
* Last known location of a SectorEntity: including when they were seen there
* Bounties: TBD, points at a character and offers some amount of money to go
  get them. Perhaps accompanied by last known location.  (is this "accompanied
  by" manifested or incidental?)
* Certain events: like battles between ships, trades between a ship and a
  station.
    * Perhaps a Character can draw some conclusion from this which becomes
      a different piece of intel, like "this sector is dangerous" or "this
      station generally has good trades for some resource"
    * Maybe this is what we use to generate news stories. Maybe these ARE news
      stories.
* Conclusions or rules of thumb: this sector is dangerous, this sector has a
  lot of some particular resource. Do we materialize this or is this inferred
  from knowledge questions above?

