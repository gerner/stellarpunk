from typing import Tuple, List

import cymunk # type: ignore

def analyze_neighbors(
        body:cymunk.Body,
        space:cymunk.Space,
        max_distance:float,
        ship_radius:float,
        margin:float,
        neighborhood_loc:cymunk.Vec2d,
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
            List[cymunk.Body],
        ]: ...
