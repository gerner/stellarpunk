Saving and Loading Games
========================

# Stuff To Load and Save

* Simulator:
    * simple fields
    * gamestate (see elsewhere for details)
    * last colliders
* Gamestate:
    * simple fields/references
    * event manager (see elsewhere for details)
    * generator (see elsewhere for details)
    * all entities (each entity needs logic to save it)
    * entity contexts (will be saved by entities themselves)
    * production chain
    * sectors
    * sector edges
    * sector entities (ref, these are entities)
    * ships including orders (ref, these are entities)
    * econ agents (ref, these are entities)
    * effect schedule (effects owned by sector)
    * order schedule (orders owned by sector entities)
    * agenda schedule (agenda owned by characters)
    * task schedule (and all tasks)
    * starfields
    * entity destroy list
* ScheduledTask
    * only subclass is TimedOrderTask at the moment
    * `Gamestate._task_schedule` is the only place references are held (unlike
      other task schedules)
* Entity
    * simple fields and refs
    * event context
    * lots of subclasses
* UniverseGenerator: (we will not actually save this)
    * random state (UG owns it, gamestate has a reference)
    * portraits (should we just reload from config?)
    * station sprites (should we just reload from config?)
    * sector name models (should we just reload from config? will need to know
      which cultures are active)
    * observers (references)
* EventManager:
    * event queue
    * action schedule
    * error check: code registers events, context keys, actions. double check
      against prior registrations. new registrations are ok, conflicts are not.
    * actual events and such should be reloaded from config
* Sector (an entity)
    * simple fields/references
    * entities (at least which are in this sector as actual entities are owned
      by gamestate)
    * physics space (see elsewhere for details)
    * effects (see elsewhere for details)
    * collision observers (just that they are observing)
    * sector weather regions (and associated location index)
* SectorEntity (an entity)
    * simple fields/references
    * physics body/shape (at least a reference, who owns it?)
    * history?
    * cargo
    * captain (a reference, captain is a Character)
    * observer references
    * sensor settings
    * lots of subclasses
* Ship (a sector entity)
    * simple fields
    * orders (see elsewhere for details)
    * default order callable
* Asset: Planet, Station, Ship
    * owner (ref to a character)
* Effect
    * simple fields/references
    * observers
    * other specifics from subclasses
    * lots of subclasses
* Order
    * simple fields/references
    * parent/child order references
    * observers
    * other specifics from subclasses
    * lots of subclasses
* Character (an entity)
    * simple fields/references
    * portrait (a reference? portrait sprites should live elsewhere?)
    * agenda (see elsewhere for details)
    * observers (references)
* Player (an entity)
    * player econ agent (should this actually live on gamestate like any
      other?)
    * messages (see elsewhere for details)
* EconAgent (an entity)
    * simple fields
    * lots of subclasses
* Message (an entity)
    * simple fields and references
* Agenda
    * simple fields and references
    * lots of subclasses
* Physics Space
    * all the bodies and shapes (maybe should by owned by SectorEntity/others)
    * collision handler (there's more than one!)
* SectorWeatherRegion
    * simple fields
* SensorSettings
    * simple fields
* SensorImage
    * TBD
* config: do we want to reload from stock config or do we want to save/load the
  already parsed configs? (settings, dialogs)
* TBD ui stuff? (do we back out of all UI to save? what about stuff like being
  part way through a dialog?)
    * InterfaceManager: nothing really...
    * Interface: nothing really...

# Serialization Architecture

Could have every object responsible for serializing itself. e.g.
`Gamestate.save()` calls `Entity.save()` for each entity, each is subclassed
with some `_save()` method. Those call out to save whatever internal objects
they've got. This has a benefit that we get some static type checking to make
sure we implement `_save()` on subclasses. But this spreads serialization logic
all over the place (e.g. if we want to change data formats we gotta touch
everything.)

Could have a whole serialization system that parallels the objects. So there's
special code for `Gamestate` and each kind of `Entity` and so on. These would
need intimate knowledge of each class they're serializing (or get support from
the objects to get the necessary internal data out).

## Serialization System

`SaveGame` class handles saving the game. It holds any relevant state (e.g.
where saves should go). Have some kind of plugins or modules that are
responsible for saving different kinds of things. Each piece of save logic
knows how to save its own object, but when it wants to save an internal complex
object, it'll call out to some other piece of independent save logic to save
that object. For example, ship saving logic knows how to save a ship, and knows
it needs to save the ship's orders, but it defers to order saving logic to save
each order.

This creates a tree of saving logic that progressively, recursively saves
each field, dispatching to appropriate class specific logic to save each
object when we run into a field that represents strong ownership of the
associated object.

# Issues

## Object References
We have a global, authoritative owner for entities in the gamestate and every
entity has a unique id. So anyone else can just save the id. But other things
(e.g. Order, Effect, Agendum, SensorImage) do not and there might be references
floating around (e.g. task schedules in gamestate or various SensorImages
floating around combat stuff).

We could deal with this by creating global registries and identiifers for these
things too. That's a lot of extra ids to manage and it's not so natural to have
global registries of these objects the way it is for entities (e.g. narrative
director needs to be able to reference a global store of entities which adds
extra motivation for such a global registry).

ScheduledTask: could have a separate registry in Gamestate for these. But we
could just let them live in the task schedule as authoritative. If it's not in
the task schedule, we will not save it. Right now (2024-12-19) the only
ScheduledTask is combat.TimedOrderTask and the only long-lived reference to
those is in the task schedule.

SensorImage: could live on SensorSettings which could keep track of all sensor
images a particular ship has targeted and do deduplication. Maybe this is a
weakref style registry so we don't have to be careful about notifying the
registry when we're done with SensorImage objects?

## Dynamic vs Static State

Some stuff, like production chain, or sector layout in the universe, doesn't
change after initial generation. Other things might only change rarely. Should
these get saved over and over again? We could save these once and just
reference that on every subsequent save. Could that be generalized into a delta
save format where everything (at some granularity) can reference another save
instead of being serialized again?

## Configuration
Lots of state is loaded from config. Should we save that and load what we saved
or should we reload the config when you load the game? These could go out of
sync if the game configs are updated after game start (e.g. development, game
patches).

Configuration Items:
* name models
* events
* dialogs
* character portraits and other sprites
* other config settings in config.toml

Decision: config, sprites, events, etc. all get loaded from whatever is in the
code. Save games can store some sanity checking stuff to make sure they are
compatible, but whatever is in code is authoritative. If there's a conflict,
the save is invalid. Or we'll need to come up with some kind of resolution
(e.g. migrate old save to new)

## Initialization Methods
These exist all over and might assume we're starting fresh. Should we still
run these when loading? before or after loading? Some are called before the
universe is generated, some after. So that might be tricky.

Maybe loading is an alternative to universe generation so we should do all the
loading instead of universe generation?

Decision: separate global, one-time-per-process initialize from
one-time-per-gamestate. New pattern has `pre_initialize` (one-time-per-process)
and `initialize_gamestate`. Some stuff still has `initialize`
(one-time-per-process) where it doesn't care about gamestate.

## Physics Object Ownership?
Who owns the physics object? The object needs a reference to the SectorEntity
and vice versa. Might make sense for the SectorEntity to be responsible for
storing/loading the physics object? puts a dependency for understanding
physics on the SectorEntity save/load logic.

Decision: SectorEntity owns the phys object. SectorSaver will add
SectorEntities to its space once entities are loaded.

## Observers
There's observers everywhere and the nature of the observer is highly variable
(e.g. orders parts of UI, etc.) So how can these be saved in a uniform way?

Observables:
* Perspective (not saved and should only be observed by UI stuff)
* UniverseGenerator (not saved, but the observers might)
* Effect
* Order
* SectorEntity:
  * `entity_migrated`
  * `entity_destroyed`
  * `entity_targeted`
* Character
* Sector Collisions

Irrelevant Observers:
* Simulator: UniverseGenerator universe generated or loaded (not neither saved)
* Starfield: Perspective. irrelevant (Starfields are UI only. StarfieldLayer is
  saved (?) but doesn't observe)
* UniverseView: Perspective. irrelevant
* StartupView: UniverseGenerator. irrelevant

UI that observe:
* PilotView: tricky because we set up a lot of state for the player's ship that
  exists whie the view is open.
  * Perspective: irrelevant
  * SectorEnttiy: player ship in case ship migrates so we can restart view
  * PlayerControlOrder: watch if it completes/cancels so we can clear state
  * DockingOrder: watch for when it completes so we can do station UI
* InterfaceManager
  * UniverseGenerator: irrelevant
  * Character: watches Player Character in case they are destroyed to handle
    player died UX

ScheduledTasks that observe:
* TimedOrderTask: Order. cancels task if watched order changes state

Agenda that observe:
* EntityOperatorAgendum (aka EOA): SectorEntity. operated craft so we stop
  agenda when craft destroyed
* CaptainAgendum (EOA):
  * Order: formulates a threat response when at risk and pauses agenda.
    unpauses when threat response completes/cancels
  * SectorEntity: if targeted, starts or updated threat response
* MiningAgendum (EOA): Order. watches mining/trading orders to decide next
* TradingAgendum (EOA): Order. watches buy/sell to decide what to do next

Orders that observe:
* MineOrder: Order, Effect. watches to schedule self to know what to do next
* TransferCargo (Order): Order, Effect. watches to schedule self for next step
* DisembarkToEntity (Order): Order. watches to schedule self
* TravelThroughGate (Order): Order, Effect. watches to change state and
  schedule self
* MissileOrder: collision. watches its target to know if it should damage/kill
  both

Effects that observe
* TransferCargoEffect: SectorEntity source/target of transfer in case destroyed
* PointDefenseEffect: SectorEntity. watches if source ship destroyed/migrates
  and cancels effect

Misc stuff that observe:
* SensorImage: SectorEntity. source ship in case destroyed/migrated to clear
  sensor image (unobserve target). target to know if we need to keep track of
  target any more.
* ThreatTracker: SectorEntity. referenced by PointDefenseEffect, might create
  one. used by FleeOrder. watches if ship is targeted to add targeter as a
  threat
* EntityOrderWatch: Order, SectorEntity. cancels an order when entity
  migrates/destroyed. Used by lots of orders that "target" some SectorEntity in
  case that target becomes invalid (migrates or is destroyed)

## PilotView
We set up some state while the view is open (e.g. PlayerControlOrder or
DockingOrder with observers so we can do stuff with them interactively). But
the view isn't saved, so we can't restore that state perfectly. And the view
isn't strictly bound to the ship anyway, so whatever we're doing with the ship
should still work even if the view closes.

Suppose the player initiates a docking order while in PilotView and saves the
game, then quits. If they load that save later, should we restore the state so
that when docking completes we trigger the station UI?

## Mid-Dialog State
Dialog mangager (stellarpunk.events.core.DialogManager) will update context as
the dialog unfolds. However, the dialog manager stores state about where we are
in the dialog and it might be challenging to reload the dialog state properly
in the middle of the dialog.

Perhaps we should refactor things so that any flags get set at the end of the
dialog, so the dialog itself does not modify state during the dialog. Then we
can make flag setting atomic with respect to the dialog.

