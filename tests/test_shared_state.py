import threading
import time

from configs.schema import RewardConfig
from training.shared_state import SharedState


def test_snapshot_consistency():
    s = SharedState()
    snap = s.snapshot()
    assert snap["timesteps"] == 0
    assert snap["mode"] == "idle"
    assert snap["paused"] is False


def test_pause_resume_events():
    s = SharedState()
    assert not s.pause_event.is_set()
    s.request_pause()
    assert s.pause_event.is_set()
    assert not s.resume_event.is_set()
    s.request_resume()
    assert not s.pause_event.is_set()
    assert s.resume_event.is_set()


def test_reward_config_mutation_visible_from_other_thread():
    s = SharedState(reward_config=RewardConfig())
    observed = []

    def reader():
        for _ in range(50):
            observed.append(s.reward_config.forward_reward_coef)
            time.sleep(0.002)

    t = threading.Thread(target=reader)
    t.start()
    time.sleep(0.01)
    with s.lock:
        s.reward_config.forward_reward_coef = 99.0
    t.join()
    assert 99.0 in observed


def test_concurrent_writes_do_not_corrupt():
    s = SharedState()

    def writer(n):
        for _ in range(200):
            with s.lock:
                s.timesteps += 1
                s.episodes += n

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(1, 5)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert s.timesteps == 800
    assert s.episodes == (1 + 2 + 3 + 4) * 200


def test_push_trajectory_preserves_limit():
    s = SharedState()
    for i in range(60):
        with s.lock:
            s.current_trajectory = [(i, i)]
        s.push_trajectory()
    assert len(s.trajectories) == 50
