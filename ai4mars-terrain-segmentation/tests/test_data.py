import numpy as np
import pytest

from terrain_seg.config import IGNORE
from terrain_seg.data import DataError, cache, load_label, split_by_position, training_samples
from terrain_seg.data import test_samples as load_test_samples


def test_finds_training_and_test_samples_with_masks():
    tr = training_samples()
    assert len(tr) == 40 and all(s.mxy and s.rng for s in tr)
    te = load_test_samples()
    assert {k: len(v) for k, v in te.items()} == {"min1": 8, "min2": 8, "min3": 8}
    assert not {s.stem for s in tr} & {s.stem for s in te["min2"]}


def test_rover_and_far_range_pixels_are_ignored():
    y = load_label(training_samples()[0])
    assert (y[-6:] == IGNORE).all() and (y[:4] == IGNORE).all() and (y[:, :3] == IGNORE).all()
    assert set(np.unique(y[6:-6, 3:]).tolist()) <= {0, 1, 2, 3}


def test_validation_holds_out_whole_rover_positions():
    tr, va = split_by_position(training_samples(), 0.25, seed=0)
    assert va and not {s.position for s in tr} & {s.position for s in va}
    assert len(tr) + len(va) == 40


def test_cache_is_resized_and_reused(tmp_path):
    s = training_samples()[:5]
    x, y = cache(s, 16, root=tmp_path)
    assert x.shape == (5, 16, 16) and y.shape == (5, 16, 16) and x.dtype == np.uint8
    files = sorted(tmp_path.iterdir())
    cache(s, 16, root=tmp_path)
    assert sorted(tmp_path.iterdir()) == files


def test_bad_label_values_are_rejected(tmp_path):
    from PIL import Image

    s = training_samples()[0]
    bad = tmp_path / "bad.png"
    Image.fromarray(np.full((8, 8), 7, np.uint8)).save(bad)
    s.label = bad
    with pytest.raises(DataError, match="unexpected label values"):
        load_label(s)
