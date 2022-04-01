import functools
import os
import json
from typing import Optional, List, Tuple

import numpy as np

from stellarpunk import core, sim, orders, interface
from stellarpunk.orders import steering

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
    order_type = history_entry["o"]["o"]
    if order_type in ("stellarpunk.orders.GoToLocation", "stellarpunk.orders.movement.GoToLocation"):
        arrival_distance = history_entry["o"].get("ad", 1.5e3)
        min_distance = history_entry["o"].get("md", None)
        order = orders.GoToLocation(np.array(history_entry["o"]["t_loc"]), ship, gamestate, arrival_distance=arrival_distance, min_distance=min_distance)
        order.neighborhood_density = history_entry["o"].get("nd", 0.)
    elif order_type in ("stellarpunk.orders.core.TransferCargo, stellarpunk.orders.core.HarvestOrder"):
        # in these cases we'll just give a null order so they just stay exactly
        # where they are, without collision avoidance or any other steering.
        order = core.Order(ship, gamestate)
    else:
        raise ValueError(f'can not load {history_entry["o"]["o"]}')

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
        self.margin = 2e2
        self.min_neighbor_dist = np.inf

        self.orders:List[core.Order] = []
        self.cannot_stop_orders:List[orders.GoToLocation] = []
        self.cannot_avoid_collision_orders:List[steering.AbstractSteeringOrder] = []
        self.margin_neighbors:List[core.SectorEntity] = []
        self.eta = np.inf

        self.collisions:List[tuple[core.SectorEntity, core.SectorEntity, Tuple[float, float], float]] = []
        self.complete_orders:List[core.Order] = []

        self.done = False

    def collision_detected(self, entity_a:core.SectorEntity, entity_b:core.SectorEntity, impulse:Tuple[float, float], ke:float) -> None:
        self.collisions.append((entity_a, entity_b, impulse, ke))

    def order_complete(self, order:core.Order) -> None:
        self.complete_orders.append(order)
        if len(set(self.orders) - set(self.complete_orders)) == 0:
            self.done = True

    def tick(self, timeout:float) -> None:

        assert not self.collisions

        if self.eta < np.inf:
            assert self.gamestate.timestamp < self.eta
        else:
            assert self.gamestate.timestamp < max(map(lambda x: x.init_eta, self.orders))

        assert all(map(lambda x: not x.cannot_stop, self.cannot_stop_orders))
        assert all(map(lambda x: not x.cannot_avoid_collision, self.cannot_avoid_collision_orders))
        for margin_neighbor in self.margin_neighbors:
            neighbor, neighbor_dist = nearest_neighbor(self.sector, margin_neighbor)
            assert neighbor_dist >= self.margin - steering.VELOCITY_EPS
            if neighbor_dist < self.min_neighbor_dist:
                self.min_neighbor_dist = neighbor_dist

        if self.done:
            self.gamestate.quit()
