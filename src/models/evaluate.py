"""ประเมินโมเดล: (1) ระดับภาพ good/defective = ตัวชี้วัดหลัก (2) mAP ต่อ class ของ detector (ข้อมูลประกอบ)
(3) latency p50/p95 ของโมเดล และ (4) reference stats สำหรับตรวจ drift
"""

import time
from pathlib import Path

import numpy as np
import yaml

from src.config import normalize_name, resolve_device
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
        device=resolve_device(params["train"]["device"]),
        workers=params["train"]["workers"],  # ค่า default = 8 -> เปิด process ค้างหลายสิบตัวบน Windows
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


def image_labels(split_dir: Path, defect_classes: list[str]) -> tuple[list[Path], np.ndarray]:
    """label ระดับภาพจาก label ของ YOLO: มีกรอบตำหนิอย่างน้อย 1 กรอบ = defective (1) ไม่งั้น good (0)"""
    names = yaml.safe_load((split_dir.parent / "data.yaml").read_text(encoding="utf-8"))["names"]
    names = list(names.values()) if isinstance(names, dict) else list(names)
    wanted = {normalize_name(c) for c in defect_classes}
    defect_ids = {i for i, n in enumerate(names) if normalize_name(n) in wanted}
    paths = list_images(split_dir)
    y = []
    for p in paths:
        lbl = split_dir / "labels" / f"{p.stem}.txt"
        ids = {int(line.split()[0]) for line in lbl.read_text().splitlines() if line.strip()} if lbl.exists() else set()
        y.append(int(bool(ids & defect_ids)))
    return paths, np.array(y)


def predict_images(predictor, paths: list[Path], max_side: int) -> list[dict]:
    return [predictor.predict(load_image(p, max_side=max_side)) for p in paths]


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """ROC-AUC = โอกาสที่ภาพ defective ได้คะแนนสูงกว่าภาพ good (Mann-Whitney, เสมอนับครึ่ง)"""
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(greater / (len(pos) * len(neg)))


def binary_metrics(y: np.ndarray, s: np.ndarray, threshold: float) -> dict:
    pred = s >= threshold
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    return {
        "img_accuracy": (tp + tn) / max(len(y), 1),
        "img_recall": recall,
        "img_precision": precision,
        "img_specificity": tn / max(tn + fp, 1),
        "img_f1": 2 * precision * recall / max(precision + recall, 1e-9),
        "img_roc_auc": roc_auc(y, s),
        "img_tp": tp, "img_fp": fp, "img_fn": fn, "img_tn": tn,
    }


def tune_threshold(y: np.ndarray, s: np.ndarray) -> float:
    """threshold ที่ให้ F1 สูงสุด (ถ้าเท่ากันเลือกค่าต่ำกว่า = recall สูงกว่า) -> ใช้กับ valid set เท่านั้น"""
    candidates = np.round(np.arange(0.05, 0.96, 0.01), 2)
    best = max(candidates, key=lambda t: (round(binary_metrics(y, s, t)["img_f1"], 6), -t))
    return float(best)


def build_reference_stats(train_images: list[Path], valid_results: list[dict], max_side: int) -> dict:
    """distribution อ้างอิงของ input (จาก train) และ output (ผลทำนายบน valid) สำหรับเทียบกับ production"""
    data: dict[str, list[float]] = {}
    for p in train_images[:MAX_REFERENCE_SAMPLES]:
        for k, v in image_stats(load_image(p, max_side=max_side)).items():
            data.setdefault(k, []).append(v)
    prediction = {
        "defect_probability": [r["defect_probability"] for r in valid_results],
        "n_defects": [r["n_defects"] for r in valid_results],
    }
    return {"data": data, "prediction": prediction}
