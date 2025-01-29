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

def test_hex_to_pixel():
    assert all(np.array((0.,0.)) == util.pointy_hex_to_pixel(np.array((0., 0.)), 100.))
    assert util.both_isclose(
            np.array((np.sqrt(3)*100.,0.)),
            util.pointy_hex_to_pixel(np.array((1., 0.)), 100.)
    )
    assert util.both_isclose(
            np.array((2.*np.sqrt(3)*100.,0.)),
            util.pointy_hex_to_pixel(np.array((2., 0.)), 100.)
    )
    assert util.both_isclose(
            np.array((-10.*np.sqrt(3)*100.,0.)),
            util.pointy_hex_to_pixel(np.array((-10., 0.)), 100.)
    )

    assert util.both_isclose(
            np.array((np.sqrt(3)*100./2., 3./2.*100.)),
            util.pointy_hex_to_pixel(np.array((0., 1.)), 100.)
    )
    assert util.both_isclose(
            np.array((5.*np.sqrt(3)*100./2., 5.*3./2*100.)),
            util.pointy_hex_to_pixel(np.array((0., 5.)), 100.)
    )
    assert util.both_isclose(
            np.array((-15.*np.sqrt(3)*100./2., -15.*3./2*100.)),
            util.pointy_hex_to_pixel(np.array((0., -15.)), 100.)
    )


    assert util.both_isclose(
            np.array((-105.*np.sqrt(3)*100. + 25.*np.sqrt(3)/2.*100., 25.*3./2.*100.)),
            util.pointy_hex_to_pixel(np.array((-105., 25.)), 100.)
    )

def test_pixel_to_hex():
    assert all(np.array((0.,0.)) == util.pixel_to_pointy_hex(np.array((0., 0.)), 100.))

    assert util.both_isclose(
            util.pixel_to_pointy_hex(np.array((np.sqrt(3)*100.,0.)), 100.),
            np.array((1., 0.))
    )
    assert util.both_isclose(
            util.pixel_to_pointy_hex(np.array((2.*np.sqrt(3)*100.,0.)), 100.),
            np.array((2., 0.))
    )
    assert util.both_isclose(
            util.pixel_to_pointy_hex(np.array((-10.*np.sqrt(3)*100.,0.)), 100.),
            np.array((-10., 0.))
    )

    assert util.both_isclose(
            util.pixel_to_pointy_hex(np.array((np.sqrt(3)*100./2., 3./2.*100.)), 100.),
            np.array((0., 1.))
    )
    assert util.both_isclose(
            util.pixel_to_pointy_hex(np.array((5.*np.sqrt(3)*100./2., 5.*3./2*100.)), 100.),
            np.array((0., 5.))
    )
    assert util.both_isclose(
            util.pixel_to_pointy_hex(np.array((-15.*np.sqrt(3)*100./2., -15.*3./2*100.)), 100.),
            np.array((0., -15.))
    )


    assert util.both_isclose(
            util.pixel_to_pointy_hex(np.array((-105.*np.sqrt(3)*100. + 25.*np.sqrt(3)/2.*100., 25.*3./2.*100.)), 100.),
            np.array((-105., 25.))
    )
