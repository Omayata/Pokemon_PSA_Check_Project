"""Gating metrics: เกณฑ์ที่โมเดลต้องผ่านทุกข้อก่อนขึ้น production (ไม่มี dependency หนัก ใช้ใน CI ได้)

ตรวจบน test set ระดับภาพ (good / defective) + latency + ขนาดโมเดล + ห้ามแย่กว่า champion
"""

CHAMPION_METRIC = "test_img_f1"  # ชื่อ metric ใน MLflow ที่ใช้เทียบกับ champion
TEST_VERSION_TAG = "test_data_version"  # tag ใน MLflow: hash ของ test set ที่ใช้วัด CHAMPION_METRIC


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


def select_best(results: list[dict], train_cfg: dict) -> dict:
    """เลือกจาก valid set (ไม่แตะ test): ตัวที่คะแนนห่างจากอันดับ 1 ไม่เกิน tolerance ถือว่าเสมอกัน
    -> เลือกโมเดลที่เล็กที่สุด (มักไม่ overfit กับ valid set ขนาดเล็ก + เร็วกว่า)
    -> ขนาดเท่ากันเอา val tiebreak (AUC) สูงกว่า: F1 ต่างกันไม่ถึง tolerance = noise แต่ AUC วัดการแยก good/defective ทุก threshold
    (ไม่ใช้ latency เป็นตัวตัดสิน เพราะวัดบน CPU แล้วแกว่งระหว่างรอบ -> ผลการเลือกจะไม่คงที่)"""
    metric, tiebreak = train_cfg["selection_metric"], train_cfg["selection_tiebreak"]
    top = max(r["val"][metric] for r in results)
    tied = [r for r in results if r["val"][metric] >= top - train_cfg["selection_tolerance"]]
    return min(tied, key=lambda r: (round(r["model_size_mb"]), -r["val"][tiebreak], -r["val"][metric]))
