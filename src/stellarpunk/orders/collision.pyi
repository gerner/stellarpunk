from typing import Tuple, List, TypeVar, Generic, Sequence

import cymunk # type: ignore

def make_enclosing_circle(
        c1:cymunk.Vec2d, r1:float,
        c2:cymunk.Vec2d, r2:float) -> Tuple[cymunk.Vec2d, float]: ...

def collision_dv(entity_pos:cymunk.Vec2d, entity_vel:cymunk.Vec2d, pos:cymunk.Vec2d, vel:cymunk.Vec2d, margin:float, v_d:cymunk.Vec2d, cbdr:bool, cbdr_bias:float, delta_v_budget:float) -> cymunk.Vec2d: ...

class Navigator:
    def __init__(
            self, space:cymunk.Space, body:cymunk.Body,
            radius:float,
            max_thrust:float, max_torque:float, max_speed:float,
            base_margin:float,
            ) -> None: ...

    def set_location_params(self,
            target_location:cymunk.Vec2d,
            arrival_radius:float, min_radius:float) -> None: ...
    def add_neighbor_shape(self, shape:cymunk.Shape) -> None: ...
    def coalesced_neighbor_locations(self) -> List[Tuple[float, float]]: ...
    def cbdr_history_summary(self) -> List[Tuple[float,float]]: ...
    def get_cannot_avoid_collision_hold(self) -> bool: ...
    def set_cannot_avoid_collision_hold(self, cach:bool) -> None: ...
    def get_margin(self) -> float: ...
    def get_collision_margin_histeresis(self) -> float: ...

    def find_target_v(self,
            max_speed: float,
            dt: float, safety_factor:float,
            ) -> Tuple[cymunk.Vec2d, float, float, bool, float]: ...

    def analyze_neighbors(
            self,
            current_timestamp:float,
            max_distance:float,
            neighborhood_radius:float,
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

    def collision_dv(self,
            current_timestamp:float,
            desired_direction:cymunk.Vec2d,
            ) -> Tuple[cymunk.Vec2d, bool, bool]: ...

def torque_for_angle(target_angle:float, angle:float, w:float, moment:float, max_torque:float, dt:float) -> float: ...
def force_for_delta_velocity(dv:cymunk.Vec2d, mass:float, max_thrust:float, dt:float) -> cymunk.Vec2d: ...

def rotation_time(
        delta_angle:float, angular_velocity:float,
        max_angular_acceleration:float) -> float: ...

def rotate_to(
        body:cymunk.Body, target_angle:float, dt:float,
        max_torque:float) -> float: ...


def accelerate_to(
        body:cymunk.Body, target_velocity:cymunk.Vec2d, dt:float,
        max_speed:float, max_torque:float, max_thrust:float, max_fine_thrust:float) -> float: ...

def migrate_threat_location(
        inref_loc:cymunk.Vec2d, inref_radius:float,
        inold_loc:cymunk.Vec2d, inold_radius:float,
        innew_loc:cymunk.Vec2d, innew_radius:float) -> Tuple[cymunk.Vec2d, float]: ...
