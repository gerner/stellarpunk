import numpy as np

from stellarpunk import util

def test_interpolate():
    assert util.interpolate(0, 0, 1, 1, 0.5) == 0.5
    assert util.interpolate(1, 2, 2, 1.5, 3) == 1.0

def test_normalize_angle_shortest():
    assert np.isclose(abs(util.normalize_angle(np.pi - (np.pi + 0.1), shortest=True)), 0.1)
    assert np.isclose(abs(util.normalize_angle(np.pi - (-np.pi + 0.1), shortest=True)), 0.1)

def test_segments_intersect():
    assert util.segments_intersect(
            (0., 0., 1., 0.),
            (0.5, 0.5, 0.5, -0.5)
    ) == (0.5, 0.)

    assert util.segments_intersect(
            (0., 0., -1., 0.),
            (0.5, 0.5, 0.5, -0.5)
    ) is None

    assert util.segments_intersect(
            (0., 0., 1., 1.),
            (0., 1., 1., 0.)
    ) == (0.5, 0.5)
    assert util.segments_intersect(
            (1., 1., 0., 0.),
            (0., 1., 1., 0.)
    ) == (0.5, 0.5)

    assert util.segments_intersect(
            (0., 0., 1., 1.),
            (1., 0., 2., 1.)
    ) is None

def test_segment_intersects_rect():
    # entirely inside
    assert util.segment_intersects_rect(
            (0.5, 0.5, 1.5, 1.5),
            (0.0, 0.0, 2.0, 2.0)
    ) == (0.5, 0.5, 1.5, 1.5)

    # intersects each side
    assert util.segment_intersects_rect(
            (-0.5, 0.5, 1.0, 1.0),
            (0.0, 0.0, 2.0, 2.0)
    )

    assert util.segment_intersects_rect(
            (0.5, 0.5, 3.0, 1.0),
            (0.0, 0.0, 2.0, 2.0)
    )
    assert util.segment_intersects_rect(
            (0.5, 0.5, 1.0, 3.0),
            (0.0, 0.0, 2.0, 2.0)
    )
    assert util.segment_intersects_rect(
            (0.5, 0.5, 1.0, -3.0),
            (0.0, 0.0, 2.0, 2.0)
    )

    # across horizontally
    assert util.segment_intersects_rect(
            (-0.5, 0.5, 3.0, 1.0),
            (0.0, 0.0, 2.0, 2.0)
    )
    # across vertically
    assert util.segment_intersects_rect(
            (0.5, -0.5, 1.0, 3.0),
            (0.0, 0.0, 2.0, 2.0)
    )
    # crossing just a corner
    assert util.segment_intersects_rect(
            (-0.5, 0.5, 2.25, 2.25),
            (0.0, 0.0, 2.0, 2.0)
    )

    # non-intersecting
    assert not util.segment_intersects_rect(
            (0.5, -1.5, 4.0, 0.5),
            (0.0, 0.0, 2.0, 2.0)
    )
