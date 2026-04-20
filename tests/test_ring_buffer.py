from utils.ring_buffer import RingBuffer


def test_mean_empty():
    b = RingBuffer(5)
    assert b.mean() == 0.0
    assert len(b) == 0


def test_mean_before_full():
    b = RingBuffer(100)
    for v in [1.0, 2.0, 3.0]:
        b.append(v)
    assert b.mean() == 2.0
    assert len(b) == 3


def test_rolling_after_full():
    b = RingBuffer(3)
    for v in [10, 20, 30, 40, 50]:
        b.append(v)
    assert len(b) == 3
    assert b.last() == 50
    assert b.mean() == (30 + 40 + 50) / 3


def test_extend():
    b = RingBuffer(4)
    b.extend([1, 2, 3, 4, 5])
    assert b.values() == [2.0, 3.0, 4.0, 5.0]
