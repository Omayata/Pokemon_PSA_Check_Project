"""Data validation: schema ของ dataset (ตอนเทรน) และ schema ของภาพขาเข้า (ตอนให้บริการ)

ถ้าพบข้อมูลเสีย -> ส่ง alert และ raise DataValidationError เพื่อหยุด pipeline / ปฏิเสธ request

รันเอง: python -m src.data.validate --dataset data/processed
"""

import argparse
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from src.alerts import send_alert
from src.config import load_params, normalize_name

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_ERRORS_SHOWN = 50


class DataValidationError(ValueError):
    def __init__(self, message: str, errors: list[str]):
        super().__init__(message)
        self.errors = errors


# ---------------------------------------------------------------- serving


def validate_upload(data: bytes, schema: dict) -> None:
    """ตรวจภาพที่ผู้ใช้อัปโหลดก่อนส่งเข้าโมเดล"""
    errors = []
    if not data:
        raise DataValidationError("empty file", ["empty_file"])
    if len(data) > schema["max_upload_mb"] * 1024 * 1024:
        raise DataValidationError("file too large", [f"file_too_large: > {schema['max_upload_mb']} MB"])
    try:
        with Image.open(io.BytesIO(data)) as img:
            fmt = img.format
            img.verify()  # ตรวจโครงสร้างไฟล์
        with Image.open(io.BytesIO(data)) as img:
            img.load()  # ตรวจว่า decode ได้ทั้งไฟล์ (จับไฟล์ที่ถูกตัดกลางทาง)
            width, height = img.size
            gray = img.convert("L").resize((64, 64))
            pixel_std = float(np.asarray(gray, dtype=np.float32).std())
    except Exception as e:  # PIL โยน exception ได้หลายแบบ
        raise DataValidationError("not a valid image", [f"cannot_decode: {type(e).__name__}"]) from e

    if fmt not in schema["allowed_formats"]:
        errors.append(f"format_not_allowed: {fmt}")
    if min(width, height) < schema["min_image_side"]:
        errors.append(f"image_too_small: {width}x{height} < {schema['min_image_side']}")
    lo, hi = schema["aspect_ratio_range"]
    if not lo <= width / height <= hi:
        errors.append(f"aspect_ratio_out_of_range: {width / height:.2f}")
    if pixel_std < schema["min_pixel_std"]:
        errors.append(f"blank_image: pixel_std={pixel_std:.2f}")
    if errors:
        raise DataValidationError("upload failed schema validation", errors)


# ---------------------------------------------------------------- training


def _file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def validate_dataset(dataset_dir: str | Path, schema: dict, alert: bool = True) -> dict:
    """ตรวจ dataset แบบ YOLO (data.yaml + {split}/images + {split}/labels)

    คืน report (dict) ถ้าผ่าน / raise DataValidationError ถ้าไม่ผ่าน
    """
    dataset_dir = Path(dataset_dir)
    errors: list[str] = []
    warnings: list[str] = []
    report: dict = {"dataset": str(dataset_dir), "splits": {}, "class_counts": {}}

    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        errors.append("missing data.yaml")
        return _finish(report, errors, warnings, alert)

    names = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))["names"]
    names = list(names.values()) if isinstance(names, dict) else list(names)
    got = sorted(normalize_name(n) for n in names)
    expected = sorted(normalize_name(n) for n in schema["expected_classes"])
    if got != expected:
        errors.append(f"class names mismatch: got {got}, expected {expected}")
    num_classes = len(names)
    report["classes"] = names

    seen_hashes: dict[str, str] = {}
    for split, min_images in schema["min_images_per_split"].items():
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"
        images = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS) if img_dir.exists() else []
        report["splits"][split] = len(images)
        if len(images) < min_images:
            errors.append(f"{split}: only {len(images)} images (< {min_images})")

        class_counter: Counter = Counter()
        for img_path in images:
            # 1) ภาพเปิดได้และขนาดพอ
            try:
                with Image.open(img_path) as im:
                    im.verify()
                with Image.open(img_path) as im:
                    im.load()
                    if min(im.size) < schema["min_image_side"]:
                        errors.append(f"{split}/{img_path.name}: too small {im.size}")
            except Exception as e:
                errors.append(f"{split}/{img_path.name}: corrupt image ({type(e).__name__})")
                continue

            # 2) ห้ามมีภาพซ้ำข้าม split (data leakage)
            h = _file_md5(img_path)
            if h in seen_hashes and seen_hashes[h] != split:
                errors.append(f"leakage: {split}/{img_path.name} duplicates an image in {seen_hashes[h]}")
            seen_hashes.setdefault(h, split)

            # 3) label ถูกรูปแบบ: "class cx cy w h" ค่าอยู่ใน [0,1]
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                errors.append(f"{split}/{img_path.name}: missing label file")
                continue
            for i, line in enumerate(lbl_path.read_text().splitlines(), start=1):
                parts = line.split()
                if not parts:
                    continue
                is_polygon = len(parts) > 5 and len(parts) % 2 == 1
                if len(parts) != 5 and not is_polygon:
                    errors.append(f"{split}/{lbl_path.name}:{i}: expected 5 values, got {len(parts)}")
                    continue
                if is_polygon:
                    # Roboflow บางครั้ง export เป็น polygon -> ultralytics แปลงเป็น box ให้เอง แต่แจ้งเตือนไว้
                    warnings.append(f"{split}/{lbl_path.name}:{i}: polygon label")
                try:
                    cls = int(parts[0])
                    coords = [float(v) for v in parts[1:]]
                except ValueError:
                    errors.append(f"{split}/{lbl_path.name}:{i}: non-numeric value")
                    continue
                if not 0 <= cls < num_classes:
                    errors.append(f"{split}/{lbl_path.name}:{i}: class id {cls} out of range [0,{num_classes})")
                    continue
                bad_box = not is_polygon and (coords[2] <= 0 or coords[3] <= 0)
                if any(not 0.0 <= v <= 1.0 for v in coords) or bad_box:
                    errors.append(f"{split}/{lbl_path.name}:{i}: bbox out of range {coords[:4]}")
                    continue
                class_counter[names[cls]] += 1
        report["class_counts"][split] = dict(class_counter)

    # 4) ทุก class ต้องมีตัวอย่างพอใน train
    train_counts = report["class_counts"].get("train", {})
    for name in names:
        n = train_counts.get(name, 0)
        if n < schema["min_instances_per_class"]:
            errors.append(f"train: class '{name}' has only {n} instances (< {schema['min_instances_per_class']})")

    return _finish(report, errors, warnings, alert)


def _finish(report: dict, errors: list[str], warnings: list[str], alert: bool) -> dict:
    report["n_errors"] = len(errors)
    report["errors"] = errors[:MAX_ERRORS_SHOWN]
    report["n_warnings"] = len(warnings)
    report["warnings"] = warnings[:MAX_ERRORS_SHOWN]
    report["passed"] = not errors
    if errors:
        if alert:
            send_alert("Data validation failed - pipeline stopped", {"n_errors": len(errors), "errors": errors[:10]},
                       severity="critical")
        raise DataValidationError(f"dataset validation failed with {len(errors)} error(s)", errors)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a YOLO dataset against the schema in params.yaml")
    parser.add_argument("--dataset", default=None, help="default: data.processed_dir จาก params.yaml")
    parser.add_argument("--report", default="reports/data_validation.json")
    args = parser.parse_args()

    params = load_params()
    dataset = args.dataset or params["data"]["processed_dir"]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    try:
        report = validate_dataset(dataset, params["schema"])
        print(f"✅ Data validation passed: {report['splits']}")
    except DataValidationError as e:
        report = {"passed": False, "n_errors": len(e.errors), "errors": e.errors[:MAX_ERRORS_SHOWN]}
        print(f"❌ {e}")
        for err in e.errors[:20]:
            print("   -", err)
        sys.exit(1)
    finally:
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
