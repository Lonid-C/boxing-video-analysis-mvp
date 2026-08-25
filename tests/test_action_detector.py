from boxing_mvp.action_detector import _FighterState


def test_cv_wrist_labels_are_anatomical_not_stance_guesses():
    assert set(_FighterState().hands) == {"left", "right"}
