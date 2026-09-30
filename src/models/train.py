"""เทรน 1 experiment แล้วบันทึกลง MLflow ครบ 6 อย่าง:
โค้ด (git commit) / ข้อมูล (data version hash) / hyperparameters / metrics / artifacts / environment

รันเอง: python -m src.models.train --experiment baseline_yolov8n_noaug [--quick]
"""

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import mlflow
import yaml

from src.config import DEFAULT_PARAMS, ROOT, load_params
from src.models.evaluate import build_reference_stats, evaluate_detector, list_images, measure_latency
from src.models.predictor import Predictor, PSAGraderPyfunc


def git_commit() -> str:
    if os.getenv("GIT_COMMIT"):
        return os.environ["GIT_COMMIT"]
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        return f"{sha}{'-dirty' if dirty else ''}"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def train_experiment(exp: dict, data_dir: str | Path, params: dict | None = None, quick: bool = False) -> dict:
    import torch
    import ultralytics
    from ultralytics import YOLO, settings

    params = params or load_params()
    tp = params["train"]
    data_dir = Path(data_dir)
    data_yaml = data_dir / "data.yaml"
    data_version = json.loads((data_dir / "data_version.json").read_text())
    epochs = tp["quick_epochs"] if quick else exp["epochs"]

    settings.update({"mlflow": False})  # ปิด callback ของ ultralytics เราบันทึก MLflow เอง
    mlflow.set_experiment(tp["mlflow_experiment"])

    with mlflow.start_run(run_name=exp["name"]) as run:
        # 1) code version + 6) environment
        mlflow.set_tags({
            "git_commit": git_commit(),
            "data_version": data_version["processed_version"],
            "raw_data_version": data_version["raw_version"],
            "drift_augment": ",".join(data_version.get("drift_augment", [])) or "none",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "quick": str(quick),
        })
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
        mlflow.log_text(freeze, "environment/pip_freeze.txt")
        mlflow.log_artifact(str(DEFAULT_PARAMS), "config")
        mlflow.log_dict(data_version, "data/data_version.json")

        # 3) hyperparameters
        hp = {
            "model": exp["model"], "epochs": epochs, "batch": exp["batch"], "lr0": exp["lr0"],
            "imgsz": tp["imgsz"], "seed": params["seed"], "patience": tp["patience"],
        }
        mlflow.log_params({**hp, **{f"aug_{k}": v for k, v in exp.get("augment", {}).items()}})

        model = YOLO(exp["model"])
        model.train(
            data=str(data_yaml), epochs=epochs, batch=exp["batch"], lr0=exp["lr0"], imgsz=tp["imgsz"],
            patience=tp["patience"], device=tp["device"], workers=tp["workers"],
            seed=params["seed"], deterministic=True,
            project="runs/train", name=exp["name"], exist_ok=True, verbose=False,
            **exp.get("augment", {}),
        )
        save_dir = Path(model.trainer.save_dir)
        best = save_dir / "weights" / "best.pt"

        # 4) metrics: valid ใช้เลือกโมเดล / test ใช้ gate
        val_metrics = evaluate_detector(best, data_yaml, "val", params)
        test_metrics = evaluate_detector(best, data_yaml, "test", params)
        predictor = Predictor(best, params)
        max_side = params["transform"]["max_side"]
        latency = measure_latency(predictor, list_images(data_dir / "test"), max_side)
        size_mb = best.stat().st_size / 1024 / 1024
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_metrics({**latency, "model_size_mb": size_mb})

        # 5) artifacts: กราฟ/ผลการเทรน + reference stats + โมเดล (pyfunc ที่รวม config)
        for f in save_dir.glob("*"):
            if f.suffix in {".png", ".jpg", ".csv", ".yaml"}:
                mlflow.log_artifact(str(f), "training")
        reference = build_reference_stats(
            list_images(data_dir / "train"), list_images(data_dir / "valid"), predictor, max_side
        )
        ref_path = save_dir / "reference_stats.json"
        ref_path.write_text(json.dumps(reference), encoding="utf-8")
        params_path = save_dir / "params_snapshot.yaml"
        params_path.write_text(yaml.safe_dump(params, allow_unicode=True), encoding="utf-8")
        mlflow.pyfunc.log_model(
            "model",
            python_model=PSAGraderPyfunc(),
            artifacts={"weights": str(best), "params": str(params_path), "reference_stats": str(ref_path)},
            code_paths=[str(ROOT / "src")],
            pip_requirements=str(ROOT / "requirements.txt"),
        )

        result = {
            "run_id": run.info.run_id,
            "name": exp["name"],
            "val": val_metrics,
            "test": test_metrics,
            **latency,
            "model_size_mb": size_mb,
        }
        print(f"✅ {exp['name']}: val map50={val_metrics['map50']:.3f} test map50={test_metrics['map50']:.3f} "
              f"p95={latency['latency_p95_ms']:.0f}ms run={run.info.run_id}")
        return result


if __name__ == "__main__":
    params = load_params()
    names = [e["name"] for e in params["train"]["experiments"]]
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=names, default=names[0])
    parser.add_argument("--data", default=params["data"]["processed_dir"])
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    exp = next(e for e in params["train"]["experiments"] if e["name"] == args.experiment)
    train_experiment(exp, args.data, params, args.quick)
