"""ประเมินโมเดล: mAP ต่อ class, latency p50/p95 ของโมเดล และสร้าง reference stats สำหรับตรวจ drift"""

import time
from pathlib import Path

import numpy as np

from src.config import normalize_name
from src.data.validate import IMAGE_EXTS
from src.features.transform import image_stats, load_image

MAX_REFERENCE_SAMPLES = 500


def list_images(split_dir: Path) -> list[Path]:
    return sorted(p for p in (split_dir / "images").glob("*") if p.suffix.lower() in IMAGE_EXTS)


def evaluate_detector(weights: str | Path, data_yaml: str | Path, split: str, params: dict) -> dict:
    """คืน metrics: map50, map50_95, precision, recall และ ap50_/recall_ ราย class"""
    from ultralytics import YOLO

    m = YOLO(str(weights)).val(
        data=str(data_yaml),
        split=split,
        imgsz=params["train"]["imgsz"],
        device=params["train"]["device"],
        project="runs/val",
        name=split,
        exist_ok=True,
        plots=False,
        verbose=False,
    )
    out = {
        "map50": float(m.box.map50),
        "map50_95": float(m.box.map),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
    }
    for i, class_idx in enumerate(m.box.ap_class_index):
        name = normalize_name(m.names[int(class_idx)]).replace(" ", "_")
        out[f"ap50_{name}"] = float(m.box.ap50[i])
        out[f"recall_{name}"] = float(m.box.r[i])
    return out


def measure_latency(predictor, image_paths: list[Path], max_side: int, n: int = 100) -> dict:
    """latency ของ transform + Stage 1 + Stage 2 ต่อภาพ (ไม่รวม network)"""
    paths = image_paths[:n]
    if not paths:
        return {"latency_p50_ms": float("nan"), "latency_p95_ms": float("nan")}
    for p in paths[:3]:  # warm-up
        predictor.predict(load_image(p, max_side=max_side))
    times = []
    for p in paths:
        t0 = time.perf_counter()
        predictor.predict(load_image(p, max_side=max_side))
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "latency_p50_ms": float(np.percentile(times, 50)),
        "latency_p95_ms": float(np.percentile(times, 95)),
    }


def build_reference_stats(train_images: list[Path], valid_images: list[Path], predictor, max_side: int) -> dict:
    """distribution อ้างอิงของ input (จาก train) และ output (จาก valid) สำหรับเทียบกับ production"""
    data: dict[str, list[float]] = {}
    for p in train_images[:MAX_REFERENCE_SAMPLES]:
        for k, v in image_stats(load_image(p, max_side=max_side)).items():
            data.setdefault(k, []).append(v)
    prediction: dict[str, list[float]] = {"score": [], "n_defects": []}
    for p in valid_images[:MAX_REFERENCE_SAMPLES]:
        result = predictor.predict(load_image(p, max_side=max_side))
        prediction["score"].append(result["score"])
        prediction["n_defects"].append(result["n_defects"])
    return {"data": data, "prediction": prediction}
