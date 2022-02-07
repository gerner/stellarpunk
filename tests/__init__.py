import functools
import os
import json
from typing import Optional, List, Tuple

import numpy as np

from stellarpunk import core, sim, orders, interface

def write_history(func):
    """ Decorator that writes sector history to file when an exception is
    raised in a test. """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        sector = kwargs["sector"]
        wrote=False
        try:
            return func(*args, **kwargs)
        except Exception as e:
            core.write_history_to_file(sector, f'/tmp/stellarpunk_test.{func.__name__}.history.gz')
            wrote=True
            raise
        finally:
            if not wrote and os.environ.get("WRITE_HIST"):
                core.write_history_to_file(sector, f'/tmp/stellarpunk_test.{func.__name__}.history.gz')
    return wrapper

def nearest_neighbor(sector:core.Sector, entity:core.SectorEntity) -> Tuple[Optional[core.SectorEntity], float]:
    neighbor_distance = np.inf
    neighbor = None
    for hit in sector.spatial_point(entity.loc):
        if hit == entity:
            continue
        d = np.linalg.norm(entity.loc - hit.loc) - hit.radius - entity.radius
        if d < neighbor_distance:
            neighbor_distance = d
            neighbor = hit
    return neighbor, neighbor_distance

def ship_from_history(history_entry, generator, sector):
    x, y = history_entry["loc"]
    v = history_entry["v"]
    w = history_entry["av"]
    theta = history_entry["a"]
    ship = generator.spawn_ship(sector, x, y, v=v, w=w, theta=theta)
    ship.name = history_entry["eid"]
    ship.phys.force = history_entry.get("f", (0., 0.))
    ship.phys.torque = history_entry.get("t", 0.)
    return ship

def station_from_history(history_entry, generator, sector):
    x, y = history_entry["loc"]
    station = generator.spawn_station(sector, x, y, 0)
    station.name = history_entry["eid"]
    return station

def asteroid_from_history(history_entry, generator, sector):
    x, y = history_entry["loc"]
    asteroid = generator.spawn_asteroid(sector, x, y, 0, 1)
    asteroid.name = history_entry["eid"]
    return asteroid

def order_from_history(history_entry, ship, gamestate):
    if history_entry["o"]["o"] != "stellarpunk.orders.GoToLocation":
        raise ValueError(f'can only support stellarpunk.orders.GoToLocation, not {history_entry["o"]["o"]}')

    arrival_distance = history_entry["o"].get("ad", 1.5e3)
    min_distance = history_entry["o"].get("md", None)
    order = orders.GoToLocation(np.array(history_entry["o"]["t_loc"]), ship, gamestate, arrival_distance=arrival_distance, min_distance=min_distance)
    order.neighborhood_density = history_entry["o"].get("nd", 0.)
    ship.orders.append(order)
    return order

def history_from_file(fname, generator, sector, gamestate):
    entities = {}
    with open(fname, "rt") as f:
        for line in f:
            entry = json.loads(line)
            if entry["p"] == "STA":
                entities[entry["eid"]] = station_from_history(entry, generator, sector)
            elif entry["p"] == "AST":
                entities[entry["eid"]] = asteroid_from_history(entry, generator, sector)
            elif entry["p"] == "SHP":
                entities[entry["eid"]] = ship = ship_from_history(entry, generator, sector)
                order_from_history(entry, ship, gamestate)
            else:
                raise ValueError(f'unknown prefix {entry["p"]}')
    return entities

class MonitoringUI(interface.AbstractInterface):
    def __init__(self, gamestate:core.Gamestate, sector:core.Sector) -> None:
        self.gamestate = gamestate
        self.sector = sector
        self.margin = 500.
        self.min_neighbor_dist = np.inf

        self.orders:List[core.Order] = []
        self.cannot_stop_orders:List[orders.GoToLocation] = []
        self.cannot_avoid_collision_orders:List[orders.AbstractSteeringOrder] = []
        self.margin_neighbors:List[core.SectorEntity] = []
        self.eta = np.inf

        self.collisions:List[tuple[core.SectorEntity, core.SectorEntity, float, float]] = []

    def collision_detected(self, entity_a:core.SectorEntity, entity_b:core.SectorEntity, impulse:float, ke:float) -> None:
        self.collisions.append((entity_a, entity_b, impulse, ke))

    def tick(self, timeout:float) -> None:
        assert not self.collisions

        assert self.gamestate.timestamp < self.eta

        assert all(map(lambda x: not x.cannot_stop, self.cannot_stop_orders))
        assert all(map(lambda x: not x.cannot_avoid_collision, self.cannot_avoid_collision_orders))
        for margin_neighbor in self.margin_neighbors:
            neighbor, neighbor_dist = nearest_neighbor(self.sector, margin_neighbor)
            assert neighbor_dist >= self.margin - orders.VELOCITY_EPS
            if neighbor_dist < self.min_neighbor_dist:
                self.min_neighbor_dist = neighbor_dist

        if all(map(lambda x: x.is_complete(), self.orders)):
            self.gamestate.quit()

