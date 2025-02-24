Navigation
==========

## Multi-Sector Navigation

* have intel on known sectors
* have intel on known travel gates
* combine knowledge of gates and sectors to build a sector graph
* use sector graph to plan route
* Travel order has a sector id and location and figures out how to get a
  character there using TravelThroughGate orders
* Mining and Trading are multi-sector away, but have a "home" sector and a max
  distance in terms of jumps needed. They won't consider asteroids/stations in
  sectors out of range.


## Ideas

### Collision Avoidance

* project threats ahead of current threat toward the current threat, coalesce
  to help with flapping threats?
* switch to A\* in dense areas of static threats?
* CBDR rework: both ships should do something that will work if the other does
  nothing, but if both follow the protocol, collision should be avoided.
* other ideas to simultaneously avoid all nearby threats?

