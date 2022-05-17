from stellarpunk import util

def test_interpolate():
    assert util.interpolate(0, 0, 1, 1, 0.5) == 0.5
    assert util.interpolate(1, 2, 2, 1.5, 3) == 1.0
