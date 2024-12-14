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
* Entity
    * simple fields and refs
    * event context
    * lots of subclasses
* UniverseGenerator:
    * random state (UG owns it, gamestate has a reference)
    * portraits (should we just reload from config?)
    * station sprites (should we just reload from config?)
    * sector name models (should we just reload from config? will need to know which cultures are active)
    * observers (references)
* EventManager:
    * simple fields/references
    * action schedule
    * do we want to reload events or just get from config?
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
    * player econ agent (should this actually live on gamestate like any other?)
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

## Initialization Methods
These exist all over and might assume we're starting fresh. Should we still
run these when loading? before or after loading? Some are called before the
universe is generated, some after. So that might be tricky.

Maybe loading is an alternative to universe generation so we should do all the
loading instead of universe generation?

## Physics Object Ownership?
Who owns the physics object? The object needs a reference to the SectorEntity
and vice versa. Might make sense for the SectorEntity to be responsible for
storing/loading the physics object? puts a dependency for understanding
physics on the SectorEntity save/load logic.

## Observers
There's observers everywhere and the nature of the observer is highly variable
(e.g. orders parts of UI, etc.) So how can these be saved in a uniform way?

## Mid-Dialog State
Dialog mangager (stellarpunk.events.core.DialogManager) will update context as
the dialog unfolds. However, the dialog manager stores state about where we are
in the dialog and it might be challenging to reload the dialog state properly
in the middle of the dialog.

Perhaps we should refactor things so that any flags get set at the end of the
dialog, so the dialog itself does not modify state during the dialog. Then we
can make flag setting atomic with respect to the dialog.

