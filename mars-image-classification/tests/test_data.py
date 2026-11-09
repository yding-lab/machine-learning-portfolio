import pytest
import torch

from mars_classifier.data import DataError, MarsImages, class_names, load_splits, read_split


def test_reads_official_split_files_and_metadata():
    names = class_names()
    assert names[0] == "ground" and names[4] == "sun"
    items = read_split("train")
    assert len(items) == 48 and all(3 <= i.sol <= 180 for i in items)
    assert {i.instrument for i in items} == {"MAHLI (close-up)", "Mastcam left", "Mastcam right"}


def test_splits_must_be_in_sol_order(tmp_path):
    from conftest import make_fake_dataset

    make_fake_dataset(tmp_path)
    f = tmp_path / "msl-images" / "test-calibrated-shuffled.txt"
    lines = f.read_text().splitlines()
    first, label = lines[0].split()
    src = tmp_path / "msl-images" / first
    early = src.with_name("0010" + src.name[4:])
    src.rename(early)
    lines[0] = f"calibrated/{early.name} {label}"
    f.write_text("\n".join(lines) + "\n")
    with pytest.raises(DataError, match="sol order"):
        load_splits(tmp_path)


def test_missing_image_is_an_error(tmp_path):
    from conftest import make_fake_dataset

    make_fake_dataset(tmp_path)
    next((tmp_path / "msl-images" / "calibrated").glob("*.JPG")).unlink()
    with pytest.raises(DataError, match="missing"):
        load_splits(tmp_path)


def test_greyscale_images_become_three_channel_tensors():
    items = read_split("val")
    grey = [i for i in items if i.path.name[8:11] == "000"][:1]
    x, y = MarsImages(grey, {l: l for l in range(5)}, 32, train=True)[0]
    assert x.shape == (3, 32, 32) and y.dtype == torch.int64
