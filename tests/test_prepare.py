"""การรวม dataset: map class, แบ่ง split ตามภาพต้นฉบับ, ไม่มีสำเนาซ้ำใน valid/test"""

import pytest
import yaml

from src.data.prepare import assign_split, class_id_map, group_key, prepare_dataset, remap_label
from tests.conftest import card_image

CLASSES = ["corner wear", "edge wear", "scratch", "crease"]


def test_class_id_map_renames_and_drops():
    m = class_id_map(["Card", "corner-wear", "Crease", "perfect edge"],
                     {"Card": None, "corner-wear": "corner wear", "crease": "crease", "perfect edge": None}, CLASSES)
    assert m == {0: None, 1: 0, 2: 3, 3: None}


def test_class_id_map_rejects_unknown_class():
    with pytest.raises(ValueError, match="dent"):
        class_id_map(["dent"], {"crease": "crease"}, CLASSES)


def test_remap_label_keeps_coords_and_drops_none():
    text = "0 0.5 0.5 0.9 0.9\n1 0.1 0.2 0.05 0.05\n"
    assert remap_label(text, {0: None, 1: 3}) == "3 0.1 0.2 0.05 0.05\n"
    assert remap_label(text, {0: None, 1: None}) == ""


def test_augmented_copies_share_group_and_split():
    a, b = "card001_jpg.rf.aaaa", "card001_jpg.rf.bbbb"
    assert group_key(a) == group_key(b) == "card001_jpg"
    fr = {"valid": 0.1, "test": 0.1}
    assert assign_split(group_key(a), fr, 42) == assign_split(group_key(b), fr, 42)
    assert assign_split("x", fr, 42) == assign_split("x", fr, 42)  # ทำซ้ำได้ผลเดิม


def _make_source(root, names, stems_by_split, label="0 0.5 0.5 0.2 0.2\n"):
    for split, stems in stems_by_split.items():
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for i, stem in enumerate(stems):
            card_image(i).save(root / split / "images" / f"{stem}.jpg")
            (root / split / "labels" / f"{stem}.txt").write_text(label)
    (root / "data.yaml").write_text(yaml.safe_dump({"nc": len(names), "names": names}))


def test_prepare_merges_sources(tmp_path, params):
    a, b = tmp_path / "a", tmp_path / "b"
    _make_source(a, ["Scratch"], {"train": ["t1"], "valid": ["v1"], "test": ["s1"]})
    # 30 ภาพต้นฉบับ x 3 สำเนา augment
    stems = [f"img{i:02d}_jpg.rf.{c}" for i in range(30) for c in "xyz"]
    _make_source(b, ["crease"], {"train": stems, "valid": [], "test": []}, label="0 0.3 0.3 0.1 0.1\n")
    params["data"].update(
        processed_dir=str(tmp_path / "out"), classes=CLASSES,
        sources=[
            {"name": "a", "resplit": False, "class_map": {"Scratch": "scratch"}},
            {"name": "b", "resplit": {"valid": 0.2, "test": 0.2}, "class_map": {"crease": "crease"}},
        ],
    )
    out = prepare_dataset({"a": a, "b": b}, params)

    assert yaml.safe_load((out / "data.yaml").read_text())["names"] == CLASSES
    assert (out / "train" / "labels" / "a__t1.txt").read_text() == "2 0.5 0.5 0.2 0.2\n"  # scratch -> id 2
    eval_b = [p.stem for s in ("valid", "test") for p in (out / s / "images").glob("b__*")]
    groups = [group_key(s.split("__", 1)[1]) for s in eval_b]
    assert eval_b and len(groups) == len(set(groups))  # valid/test: ภาพเดียวต่อภาพต้นฉบับ
    train_groups = {group_key(p.stem.split("__", 1)[1]) for p in (out / "train" / "images").glob("b__*")}
    assert not train_groups & set(groups)  # ไม่มีภาพต้นฉบับเดียวกันข้าม train กับ valid/test
