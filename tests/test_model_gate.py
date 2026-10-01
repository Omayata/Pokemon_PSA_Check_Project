from src.registry.gates import check_gates, select_best


def candidate(recall=0.95, precision=0.97, auc=0.95, f1=0.96, p95=100.0, size=6.0):
    return {"run_id": "r", "name": "x",
            "test": {"img_recall": recall, "img_precision": precision, "img_roc_auc": auc, "img_f1": f1},
            "latency_p95_ms": p95, "model_size_mb": size}


def test_good_model_passes(params):
    assert check_gates(candidate(), None, params["gates"])["passed"]


def test_each_gate_can_fail(params):
    g = params["gates"]
    assert not check_gates(candidate(recall=g["min_recall_defect"] - 0.01), None, g)["passed"]
    assert not check_gates(candidate(precision=g["min_precision_defect"] - 0.01), None, g)["passed"]
    assert not check_gates(candidate(auc=g["min_roc_auc"] - 0.01), None, g)["passed"]
    assert not check_gates(candidate(p95=g["max_p95_latency_ms"] + 1), None, g)["passed"]
    assert not check_gates(candidate(size=g["max_model_size_mb"] + 1), None, g)["passed"]


def test_regression_vs_champion_blocks(params):
    decision = check_gates(candidate(f1=0.80), champion_f1=0.95, gates=params["gates"])
    assert not decision["passed"]
    assert not decision["checks"]["no_regression_vs_champion"]["passed"]


def experiment(name, val_f1, val_auc, size=6.0):
    return {"name": name, "val": {"img_f1": val_f1, "img_roc_auc": val_auc}, "model_size_mb": size}


def test_select_best_tie_broken_by_val_auc(params):
    # F1 ต่างกันไม่ถึง tolerance = เสมอ -> ตัดสินด้วย val AUC (ไม่ใช่ F1 ที่สูงกว่านิดเดียว)
    results = [experiment("noaug", 0.9533, 0.9526), experiment("aug", 0.9524, 0.9535)]
    assert select_best(results, params["train"])["name"] == "aug"


def test_select_best_prefers_smaller_then_clear_winner(params):
    tol = params["train"]["selection_tolerance"]
    assert select_best([experiment("n", 0.95, 0.90), experiment("s", 0.955, 0.99, size=22.0)],
                       params["train"])["name"] == "n"
    # ชนะเกิน tolerance -> ไม่นับว่าเสมอ แม้ AUC ต่ำกว่า
    assert select_best([experiment("a", 0.95, 0.99), experiment("b", 0.95 + 2 * tol, 0.90)],
                       params["train"])["name"] == "b"


def test_ci_gate_script_exit_codes(tmp_path, monkeypatch):
    """scripts/check_model_gate.py ที่ CI เรียก: ผ่าน = 0, ไม่ผ่าน = 1, ไม่มีไฟล์ = 0 (หรือ 1 ถ้า --strict)"""
    import json
    import sys

    from scripts import check_model_gate

    def run(*args):
        monkeypatch.setattr(sys, "argv", ["check_model_gate.py", *args])
        return check_model_gate.main()

    good, bad = tmp_path / "good.json", tmp_path / "bad.json"
    good.write_text(json.dumps(candidate()))
    bad.write_text(json.dumps(candidate(precision=0.5)))
    assert run("--metrics", str(good)) == 0
    assert run("--metrics", str(bad)) == 1
    assert run("--metrics", str(tmp_path / "missing.json")) == 0
    assert run("--metrics", str(tmp_path / "missing.json"), "--strict") == 1
