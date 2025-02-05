import functools
import os
import json
import uuid
from typing import Optional, List, Tuple, Union, Any, Callable
from dataclasses import dataclass
import logging

import numpy as np
import numpy.typing as npt
import cymunk # type: ignore

from stellarpunk import core, sim, orders, interface, util, generate, intel
from stellarpunk.core import sector_entity
from stellarpunk.orders import steering

def write_history(func):
    """ Decorator that writes sector history to file when an exception is
    raised in a test. """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        sector = kwargs["sector"]
        #gamestate = kwargs["gamestate"]
        sector_id = sector.entity_id
        wrote=False
        try:
            return func(*args, **kwargs)
        except Exception as e:
            gamestate = core.Gamestate.gamestate
            sector = gamestate.get_entity(sector_id, core.Sector)
            core.write_history_to_file(sector, f'/tmp/stellarpunk_test.{func.__name__}.history.gz', now=gamestate.timestamp)
            wrote=True
            raise
        finally:
            if not wrote and os.environ.get("WRITE_HIST"):
                gamestate = core.Gamestate.gamestate
                sector = gamestate.get_entity(sector_id, core.Sector)
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
        gorder = orders.GoToLocation.create_go_to_location(np.array(history_entry["o"]["t_loc"]), ship, gamestate, arrival_distance=arrival_distance, min_distance=min_distance)
        #if "nn" in history_entry["o"]:
        #    gorder.neighbor_analyzer.set_nearest_neighbors(history_entry["o"]["nn"])

        if "_ncts" in history_entry["o"]:
            gorder._next_compute_ts = history_entry["o"]["_ncts"] - history_entry["ts"]
            gorder._desired_velocity = cymunk.Vec2d(*history_entry["o"]["_dv"])

        gorder.neighbor_analyzer.set_telemetry(history_entry["o"], history_entry["ts"])

        order:core.Order=gorder
    elif order_type in ("stellarpunk.orders.core.TransferCargo", "stellarpunk.orders.core.MineOrder", "stellarpunk.orders.core.HarvestOrder", "stellarpunk.orders.movement.WaitOrder"):
        # in these cases we'll just give a null order so they just stay exactly
        # where they are, without collision avoidance or any other steering.
        order = core.NullOrder.create_null_order(ship, gamestate)
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

def add_sector_intel(detector:core.CrewedSectorEntity, sector:core.Sector, character:core.Character, gamestate:core.Gamestate) -> None:
    #TODO: fresh/expires
    intel.add_sector_intel(sector, character, gamestate)
    images = sector.sensor_manager.scan(detector)
    intel.add_sector_scan_intel(detector, sector, character, gamestate)
    for entity in sector.entities.values():
        if isinstance(entity, sector_entity.Asteroid):
            intel.add_asteroid_intel(entity, character, gamestate)
        elif isinstance(entity, sector_entity.Station):
            intel.add_station_intel(entity, character, gamestate)
            if entity.entity_id in gamestate.econ_agents:
                intel.add_econ_agent_intel(gamestate.econ_agents[entity.entity_id], character, gamestate)
        else:
            intel.add_sector_entity_intel(entity, character, gamestate)

@dataclass
class LoggedTransaction:
    diff:float
    product_id:int
    buyer:int
    seller:int
    price:float
    sale_amount:float
    ticks:float

class MonitoringEconDataLogger(core.AbstractEconDataLogger):
    def __init__(self) -> None:
        self.transactions:List[LoggedTransaction] = []

    def transact(self, diff:float, product_id:int, buyer:int, seller:int, price:float, sale_amount:float, ticks:Optional[Union[int,float]]) -> None:
        assert isinstance(ticks, float)
        self.transactions.append(LoggedTransaction(diff, product_id, buyer, seller, price, sale_amount, ticks))

    def log_econ(self,
            ticks:float,
            inventory:npt.NDArray[np.float64],
            balance:npt.NDArray[np.float64],
            buy_prices:npt.NDArray[np.float64],
            buy_budget:npt.NDArray[np.float64],
            sell_prices:npt.NDArray[np.float64],
            max_buy_prices:npt.NDArray[np.float64],
            min_sell_prices:npt.NDArray[np.float64],
            cannot_buy_ticks:npt.NDArray[np.int64],
            cannot_sell_ticks:npt.NDArray[np.int64],
    ) -> None:
        pass

class MonitoringUI(core.OrderObserver, generate.UniverseGeneratorObserver, interface.AbstractInterface):
    def __init__(self, sector:core.Sector, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(util.fullname(self))
        self.sector = sector
        self.margin = 2e2
        self.min_neighbor_dist = np.inf

        self.agenda:List[core.AbstractAgendum] = []
        self.orders:List[core.Order] = []
        self.cannot_stop_orders:List[orders.GoToLocation] = []
        self.cannot_avoid_collision_orders:List[steering.AbstractSteeringOrder] = []
        self.margin_neighbors:List[core.SectorEntity] = []
        self.eta = np.inf
        self.max_timestamp = np.inf

        self.collisions:List[tuple[core.SectorEntity, core.SectorEntity, Tuple[float, float], float]] = []
        self.collisions_allowed=False
        self.complete_orders:List[core.Order] = []


        self.done = False

        self.order_eta_error_factor = 1.0

        self._viewscreen = interface.BasicCanvas(0, 0, 0, 0, self.aspect_ratio)

        self.last_status_message = ""

        self.tick_callback:Optional[Callable[[], None]] = None

        self.generator.observe(self)

    def universe_loaded(self, gamestate:core.Gamestate) -> None:
        self.logger.info("universe loaded")
        gamestate.econ_logger = self.gamestate.econ_logger
        self.gamestate = gamestate
        assert(gamestate.player)
        assert(gamestate.player.character)
        assert(gamestate.player == self.player)

        self.sector = gamestate.get_entity(self.sector.entity_id, core.Sector)

        old_agenda = self.agenda.copy()
        self.agenda.clear()
        for agendum in old_agenda:
            self.agenda.append(gamestate.agenda[agendum.agenda_id])
        old_orders = self.orders.copy()
        self.orders.clear()
        for order in old_orders:
            if order.is_complete() and order.order_id not in gamestate.orders:
                self.orders.append(order)
            else:
                new_order = gamestate.get_order(order.order_id, core.Order) # type: ignore
                self.orders.append(new_order)
                new_order.observe(self)

        old_cannot_stop_orders = self.cannot_stop_orders.copy()
        self.cannot_stop_orders.clear()
        for gtl_order in old_cannot_stop_orders:
            if gtl_order.is_complete() and gtl_order.order_id not in gamestate.orders:
                self.cannot_stop_orders.append(gtl_order)
            else:
                self.cannot_stop_orders.append(gamestate.get_order(gtl_order.order_id, orders.GoToLocation))
        old_cannot_avoid_collision_orders = self.cannot_avoid_collision_orders.copy()
        self.cannot_avoid_collision_orders.clear()
        for as_order in old_cannot_avoid_collision_orders:
            if as_order.is_complete() and as_order.order_id not in gamestate.orders:
                self.cannot_avoid_collision_orders.append(as_order)
            else:
                self.cannot_avoid_collision_orders.append(gamestate.get_order(gtl_order.order_id, steering.AbstractSteeringOrder)) # type: ignore
        old_margin_neighbors = self.margin_neighbors.copy()
        self.margin_neighbors.clear()
        for sector_entity in old_margin_neighbors:
            self.margin_neighbors.append(gamestate.get_entity(sector_entity.entity_id, core.SectorEntity))

        old_collisions = self.collisions.copy()
        self.collisions.clear()
        for old_se_a, old_se_b, x, y in old_collisions:
            se_a = gamestate.get_entity(old_se_a.entity_id, core.SectorEntity)
            se_b = gamestate.get_entity(old_se_b.entity_id, core.SectorEntity)
            self.collisions.append((se_a, se_b, x, y))

        old_complete_orders = self.complete_orders.copy()
        self.complete_orders.clear()
        for order in old_complete_orders:
            if order.is_complete() and order.order_id not in gamestate.orders:
                self.complete_orders.append(order)
            else:
                self.complete_orders.append(gamestate.get_order(order.order_id, core.Order)) # type: ignore

    @property
    def player(self) -> core.Player:
        return self.gamestate.player

    @property
    def aspect_ratio(self) -> float:
        return 2.0

    @property
    def viewscreen(self) -> interface.BasicCanvas:
        return self._viewscreen

    def status_message(self, message:str="", attr:int=0, cursor:bool=False) -> None:
        self.last_status_message = message

    def newpad(
            self,
            pad_lines:int, pad_cols:int,
            height:int, width:int,
            y:int, x:int,
            aspect_ratio:float) -> interface.BasicCanvas:
        return interface.BasicCanvas(height, width, y, x, aspect_ratio)

    def add_order(self, order:core.Order) -> None:
        if order in self.orders:
            return
        self.orders.append(order)
        order.observe(self)

    def collision_detected(self, entity_a:core.SectorEntity, entity_b:core.SectorEntity, impulse:Tuple[float, float], ke:float) -> None:
        self.collisions.append((entity_a, entity_b, impulse, ke))

    @property
    def observer_id(self) -> uuid.UUID:
        return core.OBSERVER_ID_NULL

    def order_cancel(self, order:core.Order) -> None:
        self.order_complete(order)

    def order_complete(self, order:core.Order) -> None:
        self.logger.debug(f'{order} complete at {self.gamestate.timestamp}')
        self.complete_orders.append(order)
        if len(self.orders) > 0 and len(set(self.orders) - set(self.complete_orders)) == 0:
            self.done = True

    def first_tick(self) -> None:
        for o in self.orders:
            o.observe(self)

    def tick(self, timeout:float, dt:float) -> None:
        if not self.collisions_allowed:
            assert not self.collisions, f'collided! {self.collisions[0][0].entity_id} and {self.collisions[0][1].entity_id}'

        if self.eta < np.inf:
            assert self.gamestate.timestamp < self.eta, f'exceeded set eta (still running: {[(x.ship.entity_id, x) for x in self.orders if x not in self.complete_orders]}, {self.agenda})'
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
            self.runtime.quit()

        if self.gamestate.timestamp > self.max_timestamp:
            self.runtime.quit()

        if self.tick_callback:
            self.tick_callback()

class MonitoringSimulator(sim.Simulator):
    def __init__(self, generator:generate.UniverseGenerator, testui:MonitoringUI) -> None:
        super().__init__(generator, testui, ticks_per_hist_sample=1)
        self.testui = testui

    def run(self) -> None:
        assert self.gamestate == self.generator.gamestate
        assert self.gamestate == self.testui.gamestate
        self.testui.first_tick()
        super().run()
