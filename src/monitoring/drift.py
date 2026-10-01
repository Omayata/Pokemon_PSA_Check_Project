"""ตรวจ Data Drift และ Concept Drift จาก log ของการให้บริการ

Data drift    = distribution ของ input เปลี่ยน (P(x) เปลี่ยน) เช่น ภาพมืดลง เบลอขึ้น
                -> เทียบ image stats ของ request ล่าสุดกับ reference (train set) ด้วย PSI + KS test
Concept drift = ความสัมพันธ์ input->ผลลัพธ์เปลี่ยน (P(y|x) เปลี่ยน) เช่น มาตรฐาน "การ์ดสภาพดี" เข้มขึ้น
                -> input หน้าตาเหมือนเดิม แต่ผลที่ทายตรงกับผลตรวจจริงโดยคน (/feedback) น้อยลง
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

from src.alerts import send_alert
from src.monitoring.drift_sim import FEATURE_TO_AUGMENT


def psi(reference: list[float], current: list[float], bins: int = 10) -> float:
    """Population Stability Index: <0.1 เหมือนเดิม, 0.1-0.2 เริ่มเปลี่ยน, >0.2 เปลี่ยนมาก"""
    ref, cur = np.asarray(reference, dtype=float), np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # ค่าใน reference แทบไม่ต่างกัน
        lo, hi = min(ref.min(), cur.min()), max(ref.max(), cur.max())
        edges = np.linspace(lo, hi + 1e-9, bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.clip(np.histogram(ref, edges)[0] / len(ref), 1e-4, None)
    c = np.clip(np.histogram(cur, edges)[0] / len(cur), 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _compare(ref: list[float], cur: list[float], cfg: dict) -> dict:
    p = psi(ref, cur)
    pvalue = float(ks_2samp(ref, cur).pvalue)
    return {
        "psi": round(p, 4),
        "ks_pvalue": round(pvalue, 6),
        "ref_mean": round(float(np.mean(ref)), 3),
        "cur_mean": round(float(np.mean(cur)), 3),
        "direction": "up" if np.mean(cur) > np.mean(ref) else "down",
        "drift": bool(p > cfg["psi_threshold"] and pvalue < cfg["ks_pvalue"]),
    }


def detect_data_drift(reference: dict, records: list[dict], cfg: dict) -> dict:
    window = records[-cfg["window_size"]:]
    if len(window) < cfg["min_samples"]:
        return {"status": "insufficient_data", "n": len(window), "detected": False}
    features = {
        f: _compare(reference["data"][f], [r["input"][f] for r in window], cfg) for f in cfg["data_features"]
    }
    prediction = {
        f: _compare(reference["prediction"][f], [r["prediction"][f] for r in window], cfg)
        for f in cfg["prediction_features"]
        if reference.get("prediction", {}).get(f)
    }
    drifted = [f for f, v in features.items() if v["drift"]]
    return {
        "status": "ok",
        "n": len(window),
        "detected": bool(drifted),
        "drifted_features": drifted,
        "features": features,
        "prediction_drift": prediction,  # สัญญาณเตือนล่วงหน้า (ไม่ต้องรอ label)
    }


def detect_concept_drift(records: list[dict], feedback: list[dict], cfg: dict) -> dict:
    """เทียบอัตราที่ทาย good/defective ตรงกับผลตรวจจริง: ช่วงแรกหลัง deploy (baseline) vs ช่วงล่าสุด"""
    pred_by_id = {r["request_id"]: r["prediction"]["verdict"] for r in records}
    pairs = [(pred_by_id[fb["request_id"]], fb["true_label"]) for fb in feedback if fb["request_id"] in pred_by_id]
    if len(pairs) < 2 * cfg["min_samples"]:
        return {"status": "insufficient_feedback", "n": len(pairs), "detected": False}
    w = min(cfg["window_size"], len(pairs) // 2)
    baseline = float(np.mean([p == t for p, t in pairs[:w]]))
    current = float(np.mean([p == t for p, t in pairs[-w:]]))
    return {
        "status": "ok",
        "n": len(pairs),
        "baseline_agreement": round(baseline, 3),
        "current_agreement": round(current, 3),
        "drop": round(baseline - current, 3),
        "detected": bool(baseline - current > cfg["concept_agreement_drop"]),
    }


def check_drift(reference: dict, params: dict, log_dir: str | Path, model_version: str | None = None) -> dict:
    cfg = params["monitoring"]
    log_dir = Path(log_dir)
    records = read_jsonl(log_dir / "predictions.jsonl")
    if model_version is not None:  # reference เป็นของโมเดลตัวที่ใช้อยู่ -> เทียบกับ request ของตัวนี้เท่านั้น
        records = [r for r in records if str(r.get("model_version")) == str(model_version)]
    feedback = read_jsonl(log_dir / "feedback.jsonl")

    data = detect_data_drift(reference, records, cfg)
    concept = detect_concept_drift(records, feedback, cfg)
    augment = sorted({
        FEATURE_TO_AUGMENT[(f, data["features"][f]["direction"])]
        for f in data.get("drifted_features", [])
        if (f, data["features"][f]["direction"]) in FEATURE_TO_AUGMENT
    })
    retrain = (data["detected"] and params["retrain"]["on_data_drift"]) or (
        concept["detected"] and params["retrain"]["on_concept_drift"]
    )
    report = {
        "model_version": model_version,
        "data_drift": data,
        "concept_drift": concept,
        "retrain_recommended": bool(retrain),
        "suggested_augment": augment,
    }
    if data["detected"]:
        send_alert("Data drift detected", {"features": data["drifted_features"],
                                           "psi": {f: data["features"][f]["psi"] for f in data["drifted_features"]}})
    if concept["detected"]:
        send_alert("Concept drift detected", {k: concept[k] for k in ("baseline_agreement", "current_agreement")},
                   severity="critical")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/drift_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
