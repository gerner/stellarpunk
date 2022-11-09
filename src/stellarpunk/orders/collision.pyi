from typing import Tuple, List

import cymunk # type: ignore

class NeighborAnalyzer:
    def __init__(self, space:cymunk.Space, body:cymunk.Body) -> None: ...

    def add_neighbor_shape(self, shape:cymunk.Shape) -> None: ...
    def coalesced_neighbor_locations(self) -> List[Tuple[float, float]]: ...

    def analyze_neighbors(
            self,
            max_distance:float,
            ship_radius:float,
            margin:float,
            neighborhood_radius:float,
            maximum_acceleration:float,
            ) -> Tuple[
                cymunk.Body,
                float,
                cymunk.Vec2d,
                cymunk.Vec2d,
                float,
                int,
                int,
                int,
                float,
                cymunk.Vec2d,
                cymunk.Vec2d,
                cymunk.Body,
                float,
                float,
                int,
                int,
            ]: ...

def torque_for_angle(target_angle:float, angle:float, w:float, moment:float, max_torque:float, dt:float) -> float: ...
def force_for_delta_velocity(dv:cymunk.Vec2d, mass:float, max_thrust:float, dt:float) -> cymunk.Vec2d: ...

def rotate_to(
        body:cymunk.Body, target_angle:float, dt:float,
        max_torque:float) -> float: ...

def find_target_v(
        body:cymunk.Body,
        target_location:cymunk.Vec2d, arrival_distance:float, min_distance:float,
        max_acceleration:float, max_angular_acceleration:float, max_speed: float,
        dt: float, safety_factor:float) -> Tuple[cymunk.Vec2d, float, float, bool, float]: ...

def accelerate_to(
        body:cymunk.Body, target_velocity:cymunk.Vec2d, dt:float,
        max_speed:float, max_torque:float, max_thrust:float, max_fine_thrust:float) -> float: ...
