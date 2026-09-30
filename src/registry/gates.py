"""Gating metrics: เกณฑ์ที่โมเดลต้องผ่านทุกข้อก่อนขึ้น production (ไม่มี dependency หนัก ใช้ใน CI ได้)"""


def check_gates(candidate: dict, champion_map50: float | None, gates: dict) -> dict:
    """candidate: ผลจาก train_experiment() -> คืนผลตรวจทีละเกณฑ์"""
    test = candidate["test"]
    checks = {
        "min_map50": (test["map50"], ">=", gates["min_map50"]),
        "min_recall_scratch": (test.get("recall_scratch", 0.0), ">=", gates["min_recall_scratch"]),
        "max_p95_latency_ms": (candidate["latency_p95_ms"], "<=", gates["max_p95_latency_ms"]),
        "max_model_size_mb": (candidate["model_size_mb"], "<=", gates["max_model_size_mb"]),
    }
    if champion_map50 is not None:
        checks["no_regression_vs_champion"] = (
            test["map50"], ">=", round(champion_map50 - gates["max_regression_vs_champion"], 4)
        )
    results = {
        k: {"value": round(v, 4), "op": op, "threshold": t, "passed": v >= t if op == ">=" else v <= t}
        for k, (v, op, t) in checks.items()
    }
    return {"passed": all(r["passed"] for r in results.values()), "checks": results}
