from src.registry.gates import check_gates


def candidate(map50=0.7, recall_scratch=0.6, p95=100.0, size=6.0):
    return {"run_id": "r", "name": "x", "test": {"map50": map50, "recall_scratch": recall_scratch},
            "latency_p95_ms": p95, "model_size_mb": size}


def test_good_model_passes(params):
    assert check_gates(candidate(), None, params["gates"])["passed"]


def test_each_gate_can_fail(params):
    g = params["gates"]
    assert not check_gates(candidate(map50=g["min_map50"] - 0.01), None, g)["passed"]
    assert not check_gates(candidate(recall_scratch=0.0), None, g)["passed"]
    assert not check_gates(candidate(p95=g["max_p95_latency_ms"] + 1), None, g)["passed"]
    assert not check_gates(candidate(size=g["max_model_size_mb"] + 1), None, g)["passed"]


def test_regression_vs_champion_blocks(params):
    decision = check_gates(candidate(map50=0.70), champion_map50=0.80, gates=params["gates"])
    assert not decision["passed"]
    assert not decision["checks"]["no_regression_vs_champion"]["passed"]
