from env.action_history import ActionPatternDetector


def test_no_cycle_on_empty():
    d = ActionPatternDetector()
    assert d.detect() == (False, 0)


def test_detect_period_2():
    d = ActionPatternDetector(window=40, min_repeats=3, p_max=10)
    for _ in range(20):
        d.push(1)
        d.push(2)
    is_cyclic, period = d.detect()
    assert is_cyclic
    assert period == 2


def test_detect_period_4():
    d = ActionPatternDetector(window=40, min_repeats=3, p_max=10)
    pattern = [3, 4, 5, 6]
    for _ in range(10):
        for a in pattern:
            d.push(a)
    is_cyclic, period = d.detect()
    assert is_cyclic
    assert period == 4


def test_no_cycle_on_progressive_sequence():
    d = ActionPatternDetector(window=40, min_repeats=3, p_max=10)
    for i in range(40):
        d.push(i % 11)  # secuencia casi no periódica
    is_cyclic, _ = d.detect()
    assert not is_cyclic


def test_reset():
    d = ActionPatternDetector()
    for _ in range(20):
        d.push(1); d.push(2)
    assert d.detect()[0] is True
    d.reset()
    assert d.detect() == (False, 0)
