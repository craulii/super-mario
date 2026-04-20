import json
from pathlib import Path

from env.death_map import DeathLocationTracker


def test_record_and_count(tmp_path: Path):
    d = DeathLocationTracker(tmp_path / "dm.json", bucket_size=32)
    d.record_death(100)
    d.record_death(105)   # mismo bucket (100//32 == 105//32 == 3)
    d.record_death(2000)  # otro bucket
    assert d.count_at(100) == 2
    assert d.count_at(2000) == 1


def test_persistence(tmp_path: Path):
    path = tmp_path / "dm.json"
    d1 = DeathLocationTracker(path, bucket_size=32)
    d1.record_death(500)
    d1.record_death(500)
    d1.save()

    d2 = DeathLocationTracker(path, bucket_size=32)
    d2.load()
    assert d2.count_at(500) == 2


def test_reset(tmp_path: Path):
    d = DeathLocationTracker(tmp_path / "dm.json")
    d.record_death(10)
    d.reset()
    assert d.count_at(10) == 0


def test_load_missing_file(tmp_path: Path):
    d = DeathLocationTracker(tmp_path / "missing.json")
    d.load()  # no debería fallar
    assert d.count_at(0) == 0


def test_load_corrupt(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text("not json")
    d = DeathLocationTracker(path)
    d.load()
    assert d.count_at(0) == 0
