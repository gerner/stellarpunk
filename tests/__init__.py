import functools
import os
import json
import uuid
from typing import Optional, List, Tuple

import numpy as np
import cymunk # type: ignore

from stellarpunk import core, sim, orders, interface, util
from stellarpunk.orders import steering

def write_history(func):
    """ Decorator that writes sector history to file when an exception is
    raised in a test. """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        sector = kwargs["sector"]
        gamestate = kwargs["gamestate"]
        wrote=False
        try:
            return func(*args, **kwargs)
        except Exception as e:
            core.write_history_to_file(sector, f'/tmp/stellarpunk_test.{func.__name__}.history.gz', now=gamestate.timestamp)
            wrote=True
            raise
        finally:
            if not wrote and os.environ.get("WRITE_HIST"):
                core.write_history_to_file(sector, f'/tmp/stellarpunk_test.{func.__name__}.history.gz', now=gamestate.timestamp)
    return wrapper

def nearest_neighbor(sector:core.Sector, entity:core.SectorEntity) -> Tuple[Optional[core.SectorEntity], float]:
    neighbor_distance:np.float64 = np.inf # type: ignore
    neighbor = None
    for hit in sector.spatial_point(entity.loc):
        if hit == entity:
            continue
        d = np.linalg.norm(entity.loc - hit.loc) - hit.radius - entity.radius
        if d < neighbor_distance:
            neighbor_distance = d
            neighbor = hit
    return neighbor, neighbor_distance # type: ignore

def ship_from_history(history_entry, generator, sector):
    x, y = history_entry["loc"]
    v = history_entry["v"]
    w = history_entry["av"]
    theta = history_entry["a"]
    ship = generator.spawn_ship(sector, x, y, v=v, w=w, theta=theta, entity_id=uuid.UUID(history_entry["eid"]))
    ship.name = history_entry["eid"]
    # some histories have force as a vector, some as a dict
    f = history_entry.get("f", (0., 0.))
    if isinstance(f, dict):
        f = [f["x"], f["y"]]
    ship.phys.force = cymunk.vec2d.Vec2d(f)
    ship.phys.torque = history_entry.get("t", 0.)
    return ship

def station_from_history(history_entry, generator, sector):
    x, y = history_entry["loc"]
    station = generator.spawn_station(sector, x, y, 0, entity_id=uuid.UUID(history_entry["eid"]))
    station.name = history_entry["eid"]
    return station

def asteroid_from_history(history_entry, generator, sector):
    x, y = history_entry["loc"]
    asteroid = generator.spawn_asteroid(sector, x, y, 0, 1, entity_id=uuid.UUID(history_entry["eid"]))
    asteroid.name = history_entry["eid"]
    return asteroid

def planet_from_history(history_entry, generator, sector):
    x, y = history_entry["loc"]
    planet = generator.spawn_planet(sector, x, y, entity_id=uuid.UUID(history_entry["eid"]))
    planet.name = history_entry["eid"]
    return planet

def order_from_history(history_entry:dict, ship:core.Ship, gamestate:core.Gamestate, load_ct:bool=True):
    assert ship.sector
    order_type = history_entry["o"]["o"]
    if order_type in ("stellarpunk.orders.GoToLocation", "stellarpunk.orders.movement.GoToLocation"):
        arrival_distance = history_entry["o"].get("ad", 1.5e3)
        min_distance = history_entry["o"].get("md", None)
        gorder = orders.GoToLocation(np.array(history_entry["o"]["t_loc"]), ship, gamestate, arrival_distance=arrival_distance, min_distance=min_distance)
        gorder.neighborhood_density = history_entry["o"].get("nd", 0.)

        if "_ncts" in history_entry["o"]:
            gorder._next_compute_ts = history_entry["o"]["_ncts"] - history_entry["ts"]
            gorder._desired_velocity = np.array(history_entry["o"]["_dv"])
            gorder.nearest_neighbor_dist = history_entry["o"]["nnd"]
            gorder.neighborhood_density = history_entry["o"]["nd"]

        if load_ct and "ct" in history_entry["o"]:
            gorder.collision_threat = ship.sector.entities[uuid.UUID(history_entry["o"]["ct"])]
            gorder.collision_threat_time = history_entry["o"]["ct_ts"] - history_entry["ts"]
            gorder.collision_coalesced_neighbors.extend(
                    next(ship.sector.spatial_point(np.array(x), 100)) for x in history_entry["o"]["ct_cn"]
            )
            gorder.collision_threat_loc = np.array(history_entry["o"]["ct_cloc"])
            gorder.collision_threat_radius = history_entry["o"]["ct_cradius"]
            gorder.cannot_avoid_collision = history_entry["o"]["cac"]
            gorder.cannot_avoid_collision_hold = history_entry["o"]["cach"]
            gorder.collision_cbdr = history_entry["o"]["cbdr"]

        if "msc" in history_entry["o"]:
            gorder.max_speed_cap = history_entry["o"]["msc"]
            gorder.max_speed_cap_ts = history_entry["o"]["msc_ts"] - history_entry["ts"]
            gorder.max_speed_cap_alpha = history_entry["o"]["msc_a"]

        order:core.Order=gorder
    elif order_type in ("stellarpunk.orders.core.TransferCargo", "stellarpunk.orders.core.MineOrder", "stellarpunk.orders.core.HarvestOrder", "stellarpunk.orders.movement.WaitOrder"):
        # in these cases we'll just give a null order so they just stay exactly
        # where they are, without collision avoidance or any other steering.
        order = core.Order(ship, gamestate)
    else:
        raise ValueError(f'can not load {history_entry["o"]["o"]}')
    ship.prepend_order(order)

    return order

def history_from_file(fname, generator, sector, gamestate, load_ct:bool=True):
    entities = {}

    with open(fname, "rt") as f:
        # hang on to orders so we can add them after all the entities are added
        order_entries = []
        for line in f:
            entry = json.loads(line)
            if entry["p"] == "STA":
                entities[entry["eid"]] = station_from_history(entry, generator, sector)
            elif entry["p"] == "AST":
                entities[entry["eid"]] = asteroid_from_history(entry, generator, sector)
            elif entry["p"] == "GAT":
                entities[entry["eid"]] = asteroid_from_history(entry, generator, sector)
            elif entry["p"] == "PLT":
                entities[entry["eid"]] = planet_from_history(entry, generator, sector)
            elif entry["p"] == "SHP":
                entities[entry["eid"]] = ship = ship_from_history(entry, generator, sector)
                order_entries.append((entry, ship))
            else:
                raise ValueError(f'unknown prefix {entry["p"]}')
        for entry, ship in order_entries:
            order_from_history(entry, ship, gamestate, load_ct)
    return entities

class MonitoringUI(interface.AbstractInterface):
    def __init__(self, gamestate:core.Gamestate, sector:core.Sector) -> None:
        self.gamestate = gamestate
        self.sector = sector
        self.margin = 2e2
        self.min_neighbor_dist = np.inf

        self.agenda:List[core.Agendum] = []
        self.orders:List[core.Order] = []
        self.cannot_stop_orders:List[orders.GoToLocation] = []
        self.cannot_avoid_collision_orders:List[steering.AbstractSteeringOrder] = []
        self.margin_neighbors:List[core.SectorEntity] = []
        self.eta = np.inf
        self.max_timestamp = np.inf

        self.collisions:List[tuple[core.SectorEntity, core.SectorEntity, Tuple[float, float], float]] = []
        self.complete_orders:List[core.Order] = []

        self.done = False

        self.order_eta_error_factor = 1.0

    def collision_detected(self, entity_a:core.SectorEntity, entity_b:core.SectorEntity, impulse:Tuple[float, float], ke:float) -> None:
        self.collisions.append((entity_a, entity_b, impulse, ke))

    def order_complete(self, order:core.Order) -> None:
        self.complete_orders.append(order)
        if len(self.orders) > 0 and len(set(self.orders) - set(self.complete_orders)) == 0:
            self.done = True

    def tick(self, timeout:float, dt:float) -> None:

        assert not self.collisions, f'collided! {self.collisions[0][0].entity_id} and {self.collisions[0][1].entity_id}'

        if self.eta < np.inf:
            assert self.gamestate.timestamp < self.eta, f'exceeded set eta (still running: {[x.ship.entity_id for x in self.orders if x not in self.complete_orders]}, {self.agenda})'
        else:
            assert self.gamestate.timestamp < max(map(lambda x: x.init_eta, self.orders))*self.order_eta_error_factor, "exceeded max eta over all orders"

        for x in self.cannot_stop_orders:
            assert not x.cannot_stop, f'cannot stop ({x.ship.entity_id})'
        for x in self.cannot_avoid_collision_orders: # type: ignore
            assert not x.cannot_avoid_collision, f'cannot avoid collision ({x.ship.entity_id})'
        for margin_neighbor in self.margin_neighbors:
            neighbor, neighbor_dist = nearest_neighbor(self.sector, margin_neighbor)
            assert neighbor_dist >= self.margin - steering.VELOCITY_EPS, f'violated margin ({margin_neighbor.entity_id})'
            if neighbor_dist < self.min_neighbor_dist:
                self.min_neighbor_dist = neighbor_dist

        #TODO: is there an event that can handle this?
        for agendum in self.agenda:
            if agendum.is_complete():
                self.done = True

        if self.done:
            self.gamestate.quit()

        if self.gamestate.timestamp > self.max_timestamp:
            self.gamestate.quit()
