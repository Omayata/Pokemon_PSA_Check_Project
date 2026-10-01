"""Pipeline แบบ DAG (Prefect) สั่งรันทั้งกระบวนการได้ด้วยคำสั่งเดียว

training:  ingest -> validate_raw -> prepare -> validate_processed -> train x N -> select_best
           -> gate_and_register -> deploy (API reload)
monitor:   check_drift -> (ถ้าเจอ drift ตามนโยบาย) -> training(drift_augment=...)

python -m pipelines.flow train [--quick]
python -m pipelines.flow monitor [--no-retrain]
"""

import argparse
import json
import os
from pathlib import Path

import httpx
from prefect import flow, get_run_logger, task

from src.config import load_params
from src.data.ingest import download
from src.data.prepare import prepare_dataset
from src.data.validate import validate_dataset
from src.models.train import train_experiment
from src.registry.promote import gate_and_register, reload_api


@task(name="ingest")
def ingest_task(params: dict) -> str:
    return str(download(params))


@task(name="validate-data")
def validate_task(dataset_dir: str, params: dict, stage: str) -> dict:
    report = validate_dataset(dataset_dir, params["schema"])  # ไม่ผ่าน -> raise -> flow หยุด + alert
    Path("reports").mkdir(exist_ok=True)
    Path(f"reports/data_validation_{stage}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


@task(name="prepare")
def prepare_task(raw_dir: str, params: dict, drift_augment: list[str]) -> str:
    return str(prepare_dataset(raw_dir, params, drift_augment))


@task(name="train")
def train_task(exp: dict, data_dir: str, params: dict, quick: bool) -> dict:
    return train_experiment(exp, data_dir, params, quick)


@task(name="select-best")
def select_best_task(results: list[dict], params: dict) -> dict:
    metric = params["train"]["selection_metric"]
    best = max(results, key=lambda r: r["val"][metric])  # เลือกจาก valid set (ไม่แตะ test)
    comparison = [
        {"name": r["name"], "run_id": r["run_id"], f"val_{metric}": round(r["val"][metric], 4),
         "test_map50": round(r["test"]["map50"], 4), "latency_p95_ms": round(r["latency_p95_ms"], 1),
         "model_size_mb": round(r["model_size_mb"], 1)}
        for r in results
    ]
    Path("reports").mkdir(exist_ok=True)
    Path("reports/experiment_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    Path("reports/candidate_metrics.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    get_run_logger().info("best experiment: %s", best["name"])
    return best


@task(name="gate-and-register")
def gate_task(best: dict, params: dict) -> dict:
    return gate_and_register(best, params)


@task(name="deploy")
def deploy_task(decision: dict) -> None:
    if decision["passed"]:
        reload_api()
    else:
        get_run_logger().warning("candidate rejected -> ไม่ deploy, champion เดิมให้บริการต่อ")


@flow(name="psa-training-pipeline")
def training_pipeline(quick: bool = False, drift_augment: list[str] | None = None) -> dict:
    params = load_params()
    raw_dir = ingest_task(params)
    validate_task(raw_dir, params, "raw")
    data_dir = prepare_task(raw_dir, params, drift_augment or [])
    validate_task(data_dir, params, "processed")
    results = [train_task(exp, data_dir, params, quick) for exp in params["train"]["experiments"]]
    best = select_best_task(results, params)
    decision = gate_task(best, params)
    deploy_task(decision)
    return decision


@task(name="check-drift")
def check_drift_task() -> dict:
    api_url = os.getenv("PSA_API_URL", "http://localhost:8000")
    r = httpx.get(f"{api_url}/monitoring/drift", timeout=60)
    r.raise_for_status()
    return r.json()


@flow(name="psa-drift-monitor")
def drift_monitor(auto_retrain: bool = True, quick: bool = False) -> dict:
    logger = get_run_logger()
    report = check_drift_task()
    logger.info("data drift=%s concept drift=%s", report["data_drift"]["detected"], report["concept_drift"]["detected"])
    if report["retrain_recommended"] and auto_retrain:
        logger.warning("retrain policy triggered -> augment=%s", report["suggested_augment"])
        report["retrain_decision"] = training_pipeline(quick=quick, drift_augment=report["suggested_augment"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--quick", action="store_true", help="เทรนแค่ quick_epochs เพื่อทดสอบ pipeline")
    m = sub.add_parser("monitor")
    m.add_argument("--no-retrain", action="store_true")
    m.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.cmd == "train":
        training_pipeline(quick=args.quick)
    else:
        drift_monitor(auto_retrain=not args.no_retrain, quick=args.quick)
