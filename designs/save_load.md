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
    * entity contexts
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

# Issues

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
universe is generated, some after so that might be tricky.

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
can make flag setting atomic.

