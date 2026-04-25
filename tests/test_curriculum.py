from env.curriculum import CurriculumController


def test_promotion_fires_at_threshold():
    ctrl = CurriculumController(start_tier="easy")
    for _ in range(10):
        result = ctrl.after_episode(5.0)
    assert ctrl.get_tier() == "medium"
    assert len(ctrl.promotion_log) == 1
    assert ctrl.promotion_log[0][1] == "medium"


def test_no_premature_promotion():
    ctrl = CurriculumController(start_tier="easy")
    for _ in range(5):
        ctrl.after_episode(5.0)
    assert ctrl.get_tier() == "easy"


def test_demotion_on_collapse():
    ctrl = CurriculumController(start_tier="easy")
    # Promote to medium
    for _ in range(10):
        ctrl.after_episode(5.0)
    assert ctrl.get_tier() == "medium"
    # Collapse performance — avg 0.5, well below 4.0 * 0.5 = 2.0
    for _ in range(10):
        ctrl.after_episode(0.5)
    assert ctrl.get_tier() == "easy"


def test_history_tracking():
    ctrl = CurriculumController(start_tier="easy")
    # 10 high rewards → promotion
    for _ in range(10):
        ctrl.after_episode(5.0)
    # 10 more on medium
    for _ in range(10):
        ctrl.after_episode(4.0)
    history = ctrl.get_history()
    assert len(history) == 20
    assert any(t == "easy" for _, t, _ in history)
    assert any(t == "medium" for _, t, _ in history)
    assert len(ctrl.promotion_log) >= 1
