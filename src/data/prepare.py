"""รวม raw dataset ทุก source -> data/processed ด้วย transform ชุดเดียวกับตอน serving

- map ชื่อ class ของแต่ละ source ให้เป็นชุดกลาง (data.classes) ตาม class_map; class ที่ map เป็น null ถูกตัดทิ้ง
- resplit: false -> ใช้ train/valid/test ของ Roboflow ตามเดิม
  resplit: {valid: x, test: y} -> แบ่งใหม่ตาม "ภาพต้นฉบับ" (ไฟล์ augment ของ Roboflow ชื่อขึ้นต้นเหมือนกันก่อน ".rf.")
  ด้วย hash ของชื่อ + seed -> ภาพจากต้นฉบับเดียวกันอยู่ split เดียวกันเสมอ (กัน leakage) และแบ่งซ้ำได้ผลเดิม
  valid/test เก็บแค่ภาพเดียวต่อภาพต้นฉบับ
- ชื่อไฟล์ขึ้นต้นด้วยชื่อ source (เช่น hkdefect__xxx.jpg) -> วัดผลแยกตาม source ได้
- drift_augment: ตอน retrain เพราะเจอ data drift จะเพิ่มสำเนาภาพที่แปลงแบบเดียวกับ drift เข้า train set

รันเอง: python -m src.data.prepare
"""

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import yaml

from src.config import load_params, normalize_name
from src.data.ingest import compute_data_hash, raw_path
from src.data.validate import IMAGE_EXTS
from src.features.transform import load_image
from src.monitoring.drift_sim import PHOTOMETRIC_DRIFTS

SOURCE_SEP = "__"


def read_names(dataset_dir: Path) -> list[str]:
    names = yaml.safe_load((dataset_dir / "data.yaml").read_text(encoding="utf-8"))["names"]
    return list(names.values()) if isinstance(names, dict) else list(names)


def class_id_map(raw_names: list[str], class_map: dict, classes: list[str]) -> dict[int, int | None]:
    """id ใน source -> id ในชุดกลาง (None = ตัดทิ้ง); class ที่ไม่ได้ระบุใน class_map ถือว่าผิด config"""
    target = {normalize_name(c): i for i, c in enumerate(classes)}
    mapping = {normalize_name(k): (normalize_name(v) if v else None) for k, v in class_map.items()}
    out = {}
    for i, name in enumerate(raw_names):
        key = normalize_name(name)
        if key not in mapping:
            raise ValueError(f"class '{name}' ไม่มีใน class_map ของ source นี้ -> เพิ่มใน params.yaml (หรือ map เป็น null)")
        out[i] = target[mapping[key]] if mapping[key] else None
    return out


def remap_label(text: str, id_map: dict[int, int | None]) -> str:
    lines = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        new_id = id_map.get(int(parts[0]))
        if new_id is not None:
            lines.append(" ".join([str(new_id), *parts[1:]]))
    return "\n".join(lines) + ("\n" if lines else "")


def group_key(stem: str) -> str:
    """ไฟล์ที่ Roboflow augment จากภาพเดียวกันมีชื่อ <ต้นฉบับ>.rf.<hash>"""
    return stem.split(".rf.")[0]


def assign_split(group: str, fractions: dict, seed: int) -> str:
    u = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if u < fractions["test"]:
        return "test"
    if u < fractions["test"] + fractions["valid"]:
        return "valid"
    return "train"


def collect(raw_dir: Path, source: dict, splits: list[str], seed: int) -> list[tuple[str, Path, Path]]:
    """[(split ปลายทาง, path ภาพ, path label)]"""
    items, seen_eval_groups = [], set()
    for split in splits:
        for img in sorted((raw_dir / split / "images").glob("*")):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            target = split
            if source.get("resplit"):
                group = group_key(img.stem)
                target = assign_split(group, source["resplit"], seed)
                # valid/test: เก็บภาพเดียวต่อภาพต้นฉบับ (สำเนา augment ของ Roboflow ทำให้ผลวัดเอียงและเกินจริง)
                if target != "train":
                    if group in seen_eval_groups:
                        continue
                    seen_eval_groups.add(group)
            items.append((target, img, raw_dir / split / "labels" / f"{img.stem}.txt"))
    return items


def prepare_dataset(raw_dirs: dict[str, Path] | None = None, params: dict | None = None,
                    drift_augment: list[str] | None = None) -> Path:
    params = params or load_params()
    data_cfg = params["data"]
    raw_dirs = raw_dirs or {s["name"]: raw_path(params, s) for s in data_cfg["sources"]}
    out_dir = Path(data_cfg["processed_dir"])
    max_side = params["transform"]["max_side"]
    quality = params["transform"]["jpeg_quality"]
    classes = data_cfg["classes"]
    drift_augment = drift_augment or []
    for kind in drift_augment:
        if kind not in PHOTOMETRIC_DRIFTS:
            raise ValueError(f"augment '{kind}' ไม่รองรับ (ใช้ได้เฉพาะที่ไม่เปลี่ยนตำแหน่ง bbox: {list(PHOTOMETRIC_DRIFTS)})")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in data_cfg["splits"]:
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    counts: Counter = Counter()
    raw_versions = {}
    for source in data_cfg["sources"]:
        name, raw_dir = source["name"], Path(raw_dirs[source["name"]])
        id_map = class_id_map(read_names(raw_dir), source["class_map"], classes)
        version_file = raw_dir / "data_version.json"
        raw_versions[name] = json.loads(version_file.read_text())["sha256_16"] if version_file.exists() else "unknown"

        for split, img_path, label_src in collect(raw_dir, source, data_cfg["splits"], params["seed"]):
            label = remap_label(label_src.read_text(), id_map) if label_src.exists() else ""
            img = load_image(img_path, max_side=max_side)  # <- ฟังก์ชันเดียวกับ app/main.py
            variants = [("", img)]
            if split == "train":
                variants += [(f"__{k}", PHOTOMETRIC_DRIFTS[k](img)) for k in drift_augment]
            for suffix, variant in variants:
                stem = f"{name}{SOURCE_SEP}{img_path.stem}{suffix}"
                variant.save(out_dir / split / "images" / f"{stem}.jpg", quality=quality)
                (out_dir / split / "labels" / f"{stem}.txt").write_text(label)
            counts[(split, name)] += len(variants)

    # ไม่ใส่ "path": absolute path ของเครื่องที่ prepare (เช่น C:\...) ใช้ใน container ไม่ได้
    # -> ultralytics ใช้โฟลเดอร์ของ data.yaml แทน (ผู้เรียกต้องส่ง data.yaml เป็น absolute path)
    data_yaml = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(classes),
        "names": classes,
    }
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, allow_unicode=True), encoding="utf-8")

    data_hash, n_files = compute_data_hash(out_dir)
    version = {
        "raw_version": ",".join(f"{k}:{v}" for k, v in raw_versions.items()),
        "raw_versions": raw_versions,
        "processed_version": data_hash,
        "drift_augment": drift_augment,
        "n_files": n_files,
        "images": {f"{split}/{src}": n for (split, src), n in sorted(counts.items())},
    }
    (out_dir / "data_version.json").write_text(json.dumps(version, indent=2), encoding="utf-8")
    print(f"✅ processed data @ {out_dir} | version {data_hash} | augment={drift_augment}")
    for key, n in version["images"].items():
        print(f"   {key}: {n}")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drift-augment", nargs="*", default=[])
    args = parser.parse_args()
    prepare_dataset(None, load_params(), args.drift_augment)
