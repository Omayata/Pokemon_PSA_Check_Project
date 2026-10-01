"""Gating metrics: เกณฑ์ที่โมเดลต้องผ่านทุกข้อก่อนขึ้น production (ไม่มี dependency หนัก ใช้ใน CI ได้)

ตรวจบน test set ระดับภาพ (good / defective) + latency + ขนาดโมเดล + ห้ามแย่กว่า champion
"""

CHAMPION_METRIC = "test_img_f1"  # ชื่อ metric ใน MLflow ที่ใช้เทียบกับ champion


def check_gates(candidate: dict, champion_f1: float | None, gates: dict) -> dict:
    """candidate: ผลจาก train_experiment() / reevaluate_experiment() -> คืนผลตรวจทีละเกณฑ์"""
    test = candidate["test"]
    checks = {
        "min_recall_defect": (test["img_recall"], ">=", gates["min_recall_defect"]),
        "min_precision_defect": (test["img_precision"], ">=", gates["min_precision_defect"]),
        "min_roc_auc": (test["img_roc_auc"], ">=", gates["min_roc_auc"]),
        "max_p95_latency_ms": (candidate["latency_p95_ms"], "<=", gates["max_p95_latency_ms"]),
        "max_model_size_mb": (candidate["model_size_mb"], "<=", gates["max_model_size_mb"]),
    }
    if champion_f1 is not None:
        checks["no_regression_vs_champion"] = (
            test["img_f1"], ">=", round(champion_f1 - gates["max_regression_vs_champion"], 4)
        )
    results = {
        k: {"value": round(v, 4), "op": op, "threshold": t, "passed": v >= t if op == ">=" else v <= t}
        for k, (v, op, t) in checks.items()
    }
    return {"passed": all(r["passed"] for r in results.values()), "checks": results}
