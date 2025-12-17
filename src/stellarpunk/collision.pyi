from typing import Tuple, List, Dict, Any, Optional

import cymunk # type: ignore

def make_enclosing_circle(
        c1:cymunk.Vec2d, r1:float,
        c2:cymunk.Vec2d, r2:float) -> Tuple[cymunk.Vec2d, float]: ...

def collision_dv(entity_pos:cymunk.Vec2d, entity_vel:cymunk.Vec2d, pos:cymunk.Vec2d, vel:cymunk.Vec2d, margin:float, v_d:cymunk.Vec2d, cbdr:bool, cbdr_bias:float, delta_v_budget:float) -> cymunk.Vec2d: ...

def find_target_v(body:cymunk.Body, target_location:cymunk.Vec2d, arrival_distance:float, min_distance:float, max_acceleration:float, max_angular_acceleration:float, max_speed:float, dt:float, safety_factor:float, final_speed:float) -> Tuple[cymunk.Vec2d, float, float, bool, float]: ...

def find_intercept_v(body:cymunk.Body, target_loc:cymunk.Vec2d, target_v:cymunk.Vec2d, arrival_distance:float, max_acceleration:float, max_angular_acceleration:float, max_speed:float, dt:float, final_speed:float) -> Tuple[cymunk.Vec2d, float, float, cymunk.Vec2d]: ...

def find_intercept_heading(start_loc:cymunk.Vec2d, start_v:cymunk.Vec2d, target_loc:cymunk.Vec2d, target_v:cymunk.Vec2d, muzzle_velocity:float) -> Tuple[float, cymunk.Vec2d, float]: ...

def compute_neighborhood_center(body:cymunk.Body, neighborhood_radius:float, margin:float) -> cymunk.Vec2d: ...
def compute_sensor_cone(course:cymunk.Vec2d, neighborhood_radius:float, margin:float, neighborhood_loc:cymunk.Vec2d, radius:float) -> Tuple[cymunk.Vec2d, cymunk.Vec2d, cymunk.Vec2d, cymunk.Vec2d]: ...

class RocketModel:
    def __init__(self, body:cymunk.Body, i_sp:float) -> None: ...
    def get_i_sp(self) -> float: ...
    def set_i_sp(self, i_sp:float) -> None: ...
    def get_propellant(self) -> float: ...
    def set_propellant(self, propellant:float) -> None: ...
    def adjust_propellant(self, delta:float) -> None: ...
    def get_thrust(self) -> float: ...
    def set_thrust(self, thrust:float) -> None: ...

def tick(dt:float) -> None: ...

class NeighborAnalysisParameters:
    """ carries analysis parameters for serialization. """
    neighborhood_radius:float
    threat_count:int
    neighborhood_size:int
    nearest_neighborhood_dist:float
    cannot_avoid_collision:bool
    coalesced_threat_count:int

    threat_shape:Optional[cymunk.Shape]
    minimum_separation:float
    current_threat_loc:tuple[float, float]
    threat_velocity:tuple[float, float]
    detection_timestamp:float
    approach_time:float
    threat_loc:tuple[float, float]
    threat_radius:float

class NavigatorParameters:
    """ carries navigator parameters for serialization. """
    radius:float
    max_thrust:float
    max_torque:float
    max_acceleration:float
    max_angular_acceleration:float
    worst_case_rot_time:float

    base_neighborhood_radius:float
    neighborhood_radius:float
    full_neighborhood_radius_period:float
    full_neighborhood_radius_ts:float

    base_max_speed:float
    max_speed:float

    max_speed_cap:float
    max_speed_cap_ts:float
    max_speed_cap_alpha:float
    min_max_speed:float
    max_speed_cap_max_expiration:float

    base_margin:float
    margin:float

    target_location:tuple[float,float]
    arrival_radius:float
    min_radius:float

    last_threat_id:int
    collision_margin_histeresis:float
    cannot_avoid_collision_hold = False
    collision_cbdr:bool

    analysis:NeighborAnalysisParameters
    prior_threats:list[cymunk.Shape]


class Navigator:
    def __init__(
            self, space:cymunk.Space, body:cymunk.Body,
            radius:float, max_thrust:float, max_torque:float,
            max_speed:float,
            base_margin:float,
            base_neighborhood_radius:float,
            ) -> None: ...

    def get_navigator_parameters(self) -> NavigatorParameters: ...
    def set_navigator_parameters(self, params:NavigatorParameters) -> None: ...

    def set_location_params(self,
            target_location:cymunk.Vec2d,
            arrival_radius:float, min_radius:float) -> None: ...
    def add_neighbor_shape(self, shape:cymunk.Shape) -> None: ...
    #def set_neighbor_params(self, neighborhood_size:float, nearest_neighbor_dist:float) -> None: ...
    def get_collision_margin(self) -> float: ...
    def coalesced_neighbor_locations(self) -> List[Tuple[float, float]]: ...
    def cbdr_history_summary(self) -> List[Tuple[float,float]]: ...
    def get_cannot_avoid_collision_hold(self) -> bool: ...
    def set_cannot_avoid_collision_hold(self, cach:bool) -> None: ...
    def get_threat_count(self) -> int: ...
    def get_cannot_avoid_collision(self) -> bool: ...
    def get_collision_cbdr(self) -> bool: ...
    def get_num_neighbors(self) -> int: ...
    def get_nearest_neighbor_dist(self) -> float: ...
    def get_margin(self) -> float: ...
    def get_collision_margin_histeresis(self) -> float: ...
    def get_neighborhood_radius(self) -> float: ...
    def get_max_speed(self) -> float: ...
    def get_max_speed_cap(self) -> float: ...
    #def set_max_speed_cap_params(self, max_speed_cap:float, max_speed_cap_ts:float, max_speed_cap_alpha:float) -> None: ...

    def get_telemetry(self) -> Dict[str, Any]: ...
    def set_telemetry(self, telemetry:Dict[str, Any], timestamp:float) -> None: ...

    def prepare_analysis(self, timestamp:float) -> None: ...

    def find_target_v(self,
            dt: float, safety_factor:float,
            ) -> Tuple[cymunk.Vec2d, float, float, bool, float]: ...

    def analyze_neighbor(self, target_loc:cymunk.Vec2d, target_velocity:cymunk.Vec2d, margin:float, max_distance:float
            ) -> Tuple[
                    float,
                    float,
                    cymunk.Vec2d,
                    cymunk.Vec2d,
                    float,
                    cymunk.Vec2d,
                    float
            ]: ...

    def analyze_neighbors(
            self,
            current_timestamp:float,
            max_distance:float,
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
        body:cymunk.Body, rocket_model:RocketModel, target_velocity:cymunk.Vec2d, dt:float,
        max_speed:float, max_torque:float, max_thrust:float, max_fine_thrust:float,
        sensor_settings:Any, timestamp:float) -> float: ...

def migrate_threat_location(
        inref_loc:cymunk.Vec2d, inref_radius:float,
        inold_loc:cymunk.Vec2d, inold_radius:float,
        innew_loc:cymunk.Vec2d, innew_radius:float) -> Tuple[cymunk.Vec2d, float]: ...
