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
* News is a function of one or more "related" pieces of intel.
    * For instance, several intels about asteroids that are all within a
      certain distance could create a story about a rich resource field.
    * For instance, a sell price much below average, or buy price much above,
      or a rare resource for sale could create a story about a great trade
      opportunity. Especially if this is part of a pattern.
    * A sequence in a short period of time of "combat events" (how is this
      discovered?) in a sector could create a story about priacy in a sector.
    * "Claiming a bounty" (how is this discovered?) on a pirate could create a
      story about a notorious pirate being captured/killed.
* News is created by Reporters. News items are attributed to the reporter.
    * News cannot be sold by anyone but the author of the news item.
    * News can be freely shared by anyone. There's incentive for an
      organization to do so.
* News can be distributed by Characters.
    * A Reporter can sell their story to different news forums that don't
      already have the news story (do they always buy?)
    * A Courier for a news network can distribute news from one forum to
      another (e.g. across sector boundaries)
* News stories carry the related intel with them. Acquiring the news story gets
  all the related intel. (this is the value to AI)
* News forums are connected into news networks. News is not necessarily shared
  across a news network crossing sector boundaries (sector boundaries are a big
  deal.)
* News stories in one news forum in a sector are shared by all related forums
  in that sector. i.e. instant communication within a sector is possible.

Ideas:
* Perhaps every sector has a single news network. Perhaps every
  social/political org has a single news network. (e.g. different stations
  owned by different orgs might have different news networks, but all the
  stations owned by one org have the same news network)

Questions:
* Are News and Intel the same thing? No: a news story is related to zero or
  more pieces of intel and is generally a function of intel. But they are
  separate
* What is an "event" (or is it anything about which News/Intel exists?)
* What's a news forum? is that an organization? where does their money come
  from? Subscriptions characters can make to get access to the forum?

## Types of stories

All of these should check that we haven't seen a "recent" "related" news story.

* Resource fields: a bunch of asteroid intels close to each other
* Trade opportunities: a particularly high buy offer for a resource or a
  particularly low sell offer or a sell offer for a "rare" resource.
* Economic pattern: a pattern of low or high prices for a resource. Or a
  general lack of a resource at stations.
* Economic summary: simply publishing average buy/sell prices and amounts in a
  sector.
* Combat: simply observing combat, especially if a craft is destroyed: who's
  the victim, who's the perpetrator? was an LEO involved? was cargo stolen?
* Piracy pattern: a set of priacy events in a sector over a short period of
  time, especially if there's a common perp or cargo involved, or no/rarely LEO
  involved.

# Implementation
* A piece of intel might automatically be created according to some trigger
* Intel created via events, independent from other events. E.g. identifying a
  target on sensors creates intel about that entity if such intel doesn't
  already exist.
* Intel is deduped against other "matching" intel if the existing intel is
  "better" (fresher). E.g. maybe if a piece of intel is permanent with limited
  scope we can search for a matching, existing piece of itel.
* Pieces of intel might be redundant or partially matching where a Character
  can prefer one piece to another and assign some value to replacing an
  existing partial match. E.g. price information for a resource at a station
  that is newer than some existing price information for the same resource at
  the same station might be preferred, replacing that existing piece of intel.
* If a piece of intel is transferred from one Character to another, they both
  refer to the same object, sharing it. Intel are Entities.
* Intel should be immutable or any changes should make sense to be shared by
  anyone referring to that intel, even if separated, without any communication
  channel.
* Each Character has an intel manager that keeps track of their intel. The
  Character, and their agenda can consult this intel manager for relevant
  information: the locations of known asteroids, prices of goods at stations,
  bounties on ships/characters, etc.
* The model is that this information is "learned" from the acquired intel and
  is a function of the intel in the manager. So if the intel goes away, that
  learned knowledge should change. The "knowledge" is entirely ephemeral and
  code should consult the IntelManager whenever it needs such information.

## Use Cases

### Choose an asteroid to mine
Pass over all known asteroids looking at what resource it is, how much was
available last we saw, and where it is, distance and time it'll take to get
there

### Choose a station to sell to
Pass over buy offers (buyer, resource, price, max amount)
for each also get corresponding intel about station
consider price, amount, distance/time it'll take to get there

### Newly Identify a SensorImage
if a sensor image is identified we now know about the location and some
parameters of the entity. For instance we know what resource an asteroid has or
what resource a statio produces. We don't know buy/sell prices for the station.

### Docking at a Station
Suddenly we know a great deal about the economic status of the station:
resources, prices, amounts, etc. We also know about people at the station and
other properties of that station we might want in the future.

### Attach prevoius intel to a SensorImage
if a sensor image is identified we can pull in whatever information we know
about that object.

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

