Saving and Loading Games
========================

# Stuff To Load and Save
* gamestate:
    * simple fields/references
    * event manager (see elsewhere for details)
    * random state
    * all entities (each entity needs logic to save it)
    * entity contexts
    * production chain
    * sectors
    * sector edges
    * sector entities (these are entities)
    * ships including orders (these are entities)
    * econ agents (these are entities)
    * effect schedule (effects owned by sector)
    * order schedule (orders owned by sector entities)
    * agenda schedule (agenda owned by characters)
    * task schedule (and all tasks)
    * starfields
    * entity destroy list
* event manager:
    * simple fields/references
    * action schedule
    * do we want to reload events?
* sim:
    * 
    * gamestate (see elsewhere for details)
    * last colliders
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
    * other specifics from subclasses
* Ship (a sector entity)
    * simple fields
    * orders (see elsewhere for details)
    * default order callable
* Effect
    * simple fields/references
    * observers
    * other specifics from subclasses
* Order
    * simple fields/references
    * parent/child order references
    * observers
    * other specifics from subclasses
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

# Issues

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

