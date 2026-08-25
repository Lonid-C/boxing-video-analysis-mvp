import numpy as np

from boxing_mvp.tracker import IdentityAssigner


def detection(track_id, x, red=0.0, blue=0.0, score=1000.0):
    return track_id, float(x), float(score), float(red), float(blue)


def test_assigns_by_color_and_keeps_referee_unknown():
    assigner = IdentityAssigner()
    labels = assigner.assign_frame([
        detection(10, 100, red=.6), detection(20, 500, blue=.5), detection(30, 300),
    ])
    assert set(labels.values()) == {"unknown"}
    labels = assigner.assign_frame([
        detection(10, 100, red=.6), detection(20, 500, blue=.5), detection(30, 300),
    ])
    assert labels == {10: "red", 20: "blue", 30: "unknown"}
    labels = assigner.assign_frame([
        detection(10, 450, red=.5), detection(20, 150, blue=.5), detection(30, 300),
    ])
    assert labels == {10: "red", 20: "blue", 30: "unknown"}


def test_does_not_guess_without_color_evidence():
    assigner = IdentityAssigner()
    labels = assigner.assign_frame([detection(1, 100), detection(2, 500), detection(3, 300)])
    assert set(labels.values()) == {"unknown"}


def test_reassociates_replacement_track_after_grace_period():
    assigner = IdentityAssigner(reassign_after=2, min_color_samples=1)
    assigner.assign_frame([detection(1, 100, red=.5), detection(2, 500, blue=.5)])
    assert assigner.assign_frame([detection(2, 500, blue=.5), detection(3, 105, red=.5)])[3] == "unknown"
    assigner.assign_frame([detection(2, 500, blue=.5), detection(3, 106, red=.5)])
    labels = assigner.assign_frame([detection(2, 500, blue=.5), detection(3, 107, red=.5)])
    assert labels[2] == "blue" and labels[3] == "red"
    labels = assigner.assign_frame([
        detection(1, 100, red=.5), detection(2, 500, blue=.5), detection(3, 107, red=.5),
    ])
    assert labels[1] == "unknown" and labels[3] == "red"


def test_color_evidence_separates_red_blue_and_white():
    red = np.zeros((100, 100, 3), dtype=np.uint8); red[:] = (0, 0, 255)
    blue = np.zeros((100, 100, 3), dtype=np.uint8); blue[:] = (255, 0, 0)
    white = np.full((100, 100, 3), 255, dtype=np.uint8)
    assert IdentityAssigner.color_evidence(red, (0, 0, 100, 100))[0] > .9
    assert IdentityAssigner.color_evidence(blue, (0, 0, 100, 100))[1] > .9
    assert IdentityAssigner.color_evidence(white, (0, 0, 100, 100)) == (0.0, 0.0)


def test_releases_identity_after_sustained_opposite_color():
    assigner = IdentityAssigner(min_color_samples=1, conflict_release_after=2)
    assert assigner.assign_frame([detection(1, 100, red=.8)])[1] == "red"
    assigner.assign_frame([detection(1, 100, blue=.9)])
    assert assigner.assign_frame([detection(1, 100, blue=.9)])[1] == "unknown"
