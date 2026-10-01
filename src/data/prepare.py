"""แปลง raw dataset -> data/processed ด้วย transform ชุดเดียวกับตอน serving

- ใช้ split train/valid/test ตามที่ Roboflow version นั้นกำหนด (ตายตัว -> ทำซ้ำได้เหมือนเดิม)
- เขียน data.yaml ใหม่ด้วย path แบบ absolute (data.yaml ของ Roboflow ใช้ ../train ซึ่งมักพัง)
- drift_augment: ตอน retrain เพราะเจอ data drift จะเพิ่มสำเนาภาพที่แปลงแบบเดียวกับ drift เข้า train set

รันเอง: python -m src.data.prepare
"""

import argparse
import json
import shutil
from pathlib import Path

import yaml

from src.config import load_params
from src.data.ingest import compute_data_hash
from src.data.validate import IMAGE_EXTS
from src.features.transform import load_image
from src.monitoring.drift_sim import PHOTOMETRIC_DRIFTS


def prepare_dataset(raw_dir: str | Path, params: dict | None = None, drift_augment: list[str] | None = None) -> Path:
    params = params or load_params()
    raw_dir = Path(raw_dir)
    out_dir = Path(params["data"]["processed_dir"])
    max_side = params["transform"]["max_side"]
    quality = params["transform"]["jpeg_quality"]
    drift_augment = drift_augment or []
    for kind in drift_augment:
        if kind not in PHOTOMETRIC_DRIFTS:
            raise ValueError(f"augment '{kind}' ไม่รองรับ (ใช้ได้เฉพาะที่ไม่เปลี่ยนตำแหน่ง bbox: {list(PHOTOMETRIC_DRIFTS)})")

    if out_dir.exists():
        shutil.rmtree(out_dir)

    raw_yaml = yaml.safe_load((raw_dir / "data.yaml").read_text(encoding="utf-8"))
    for split in params["data"]["splits"]:
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        for img_path in sorted((raw_dir / split / "images").glob("*")):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            label_src = raw_dir / split / "labels" / f"{img_path.stem}.txt"
            img = load_image(img_path, max_side=max_side)  # <- ฟังก์ชันเดียวกับ app/main.py
            variants = [("", img)]
            if split == "train":
                variants += [(f"__{k}", PHOTOMETRIC_DRIFTS[k](img)) for k in drift_augment]
            for suffix, variant in variants:
                variant.save(out_dir / split / "images" / f"{img_path.stem}{suffix}.jpg", quality=quality)
                label_dst = out_dir / split / "labels" / f"{img_path.stem}{suffix}.txt"
                if label_src.exists():
                    shutil.copy(label_src, label_dst)

    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": raw_yaml["nc"],
        "names": raw_yaml["names"],
    }
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, allow_unicode=True), encoding="utf-8")

    raw_version_file = raw_dir / "data_version.json"
    raw_version = json.loads(raw_version_file.read_text()) if raw_version_file.exists() else {}
    data_hash, n_files = compute_data_hash(out_dir)
    version = {
        "raw_version": raw_version.get("sha256_16", "unknown"),
        "processed_version": data_hash,
        "drift_augment": drift_augment,
        "n_files": n_files,
    }
    (out_dir / "data_version.json").write_text(json.dumps(version, indent=2), encoding="utf-8")
    print(f"✅ processed data @ {out_dir} | version {data_hash} | augment={drift_augment}")
    return out_dir


if __name__ == "__main__":
    params = load_params()
    parser = argparse.ArgumentParser()
    default_raw = Path(params["data"]["raw_dir"]) / f"v{params['data']['roboflow_version']}"
    parser.add_argument("--raw", default=str(default_raw))
    parser.add_argument("--drift-augment", nargs="*", default=[])
    args = parser.parse_args()
    prepare_dataset(args.raw, params, args.drift_augment)
