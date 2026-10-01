"""เทรน (หรือประเมินซ้ำ) 1 experiment แล้วบันทึกลง MLflow ครบ 6 อย่าง:
โค้ด (git commit) / ข้อมูล (data version hash) / hyperparameters / metrics / artifacts / environment

python -m src.models.train --experiment baseline_yolov8n_noaug [--quick]
python -m src.models.train --experiment yolov8n_aug --reevaluate   # ใช้ weights เดิมใน runs/train ไม่เทรนใหม่
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import mlflow
import numpy as np
import yaml

from src.config import DEFAULT_PARAMS, ROOT, load_params, resolve_device
from src.models.evaluate import (
    binary_metrics,
    build_reference_stats,
    evaluate_detector,
    image_labels,
    measure_latency,
    predict_images,
    tune_threshold,
)
from src.models.predictor import CardConditionPyfunc, Predictor


def git_commit() -> str:
    if os.getenv("GIT_COMMIT"):
        return os.environ["GIT_COMMIT"]
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        return f"{sha}{'-dirty' if dirty else ''}"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _log_context(exp: dict, params: dict, data_dir: Path, epochs: int, extra_tags: dict) -> None:
    """1) code version, 2) data version, 3) hyperparameters, 6) environment"""
    import torch
    import ultralytics

    tp = params["train"]
    data_version = json.loads((data_dir / "data_version.json").read_text())
    mlflow.set_tags({
        "git_commit": git_commit(),
        "data_version": data_version["processed_version"],
        "raw_data_version": data_version["raw_version"],
        "drift_augment": ",".join(data_version.get("drift_augment", [])) or "none",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "train_device": resolve_device(tp["device"]),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        **extra_tags,
    })
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    mlflow.log_text(freeze, "environment/pip_freeze.txt")
    mlflow.log_artifact(str(DEFAULT_PARAMS), "config")
    mlflow.log_dict(data_version, "data/data_version.json")
    mlflow.log_params({
        "model": exp["model"], "epochs": epochs, "batch": exp["batch"], "lr0": exp["lr0"],
        "imgsz": tp["imgsz"], "seed": params["seed"], "patience": tp["patience"],
        **{f"aug_{k}": v for k, v in exp.get("augment", {}).items()},
    })


def _evaluate_and_log(exp: dict, save_dir: Path, data_dir: Path, params: dict, run) -> dict:
    """4) metrics + 5) artifacts สำหรับ weights ใน save_dir (ใช้ทั้งหลังเทรนและตอนประเมินซ้ำ)"""
    params = json.loads(json.dumps(params))  # copy: threshold ที่เลือกได้จะเก็บไปกับโมเดลตัวนี้เท่านั้น
    best = save_dir / "weights" / "best.pt"
    max_side = params["transform"]["max_side"]
    defect_classes = params["classifier"]["defect_classes"]

    # ตัวชี้วัดหลัก: จำแนกระดับภาพ good/defective -> เลือก threshold บน valid, รายงานผลบน test
    predictor = Predictor(best, params)
    val_paths, y_val = image_labels(data_dir / "valid", defect_classes)
    val_results = predict_images(predictor, val_paths, max_side)
    s_val = np.array([r["defect_probability"] for r in val_results])
    threshold = tune_threshold(y_val, s_val)
    params["classifier"]["threshold"] = threshold
    test_paths, y_test = image_labels(data_dir / "test", defect_classes)
    s_test = np.array([r["defect_probability"] for r in predict_images(predictor, test_paths, max_side)])
    val_metrics = binary_metrics(y_val, s_val, threshold)
    test_metrics = binary_metrics(y_test, s_test, threshold)

    # ข้อมูลประกอบ: คุณภาพการหาตำแหน่งตำหนิของ detector
    data_yaml = data_dir / "data.yaml"
    val_metrics.update(evaluate_detector(best, data_yaml, "val", params))
    test_metrics.update(evaluate_detector(best, data_yaml, "test", params))

    latency = measure_latency(predictor, test_paths, max_side)
    size_mb = best.stat().st_size / 1024 / 1024
    mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
    mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
    mlflow.log_metrics({**latency, "model_size_mb": size_mb, "threshold": threshold})

    for f in save_dir.glob("*"):
        if f.suffix in {".png", ".jpg", ".csv", ".yaml"} and f.name != "params_snapshot.yaml":
            mlflow.log_artifact(str(f), "training")
    reference = build_reference_stats(val_paths, val_results, max_side)
    ref_path = save_dir / "reference_stats.json"
    ref_path.write_text(json.dumps(reference), encoding="utf-8")
    params_path = save_dir / "params_snapshot.yaml"
    params_path.write_text(yaml.safe_dump(params, allow_unicode=True), encoding="utf-8")
    mlflow.pyfunc.log_model(
        "model",
        python_model=CardConditionPyfunc(),
        artifacts={"weights": str(best), "params": str(params_path), "reference_stats": str(ref_path)},
        code_paths=[str(ROOT / "src")],
        pip_requirements=str(ROOT / "requirements.txt"),
    )

    result = {
        "run_id": run.info.run_id,
        "name": exp["name"],
        "threshold": threshold,
        "val": val_metrics,
        "test": test_metrics,
        **latency,
        "model_size_mb": size_mb,
    }
    print(f"✅ {exp['name']}: threshold={threshold:.2f} val F1={val_metrics['img_f1']:.3f} "
          f"test F1={test_metrics['img_f1']:.3f} recall={test_metrics['img_recall']:.3f} "
          f"precision={test_metrics['img_precision']:.3f} AUC={test_metrics['img_roc_auc']:.3f} "
          f"p95={latency['latency_p95_ms']:.0f}ms run={run.info.run_id}")
    return result


def train_experiment(exp: dict, data_dir: str | Path, params: dict | None = None, quick: bool = False) -> dict:
    from ultralytics import YOLO, settings

    params = params or load_params()
    tp = params["train"]
    data_dir = Path(data_dir)
    epochs = tp["quick_epochs"] if quick else exp["epochs"]

    settings.update({"mlflow": False})  # ปิด callback ของ ultralytics เราบันทึก MLflow เอง
    mlflow.set_experiment(tp["mlflow_experiment"])
    with mlflow.start_run(run_name=exp["name"]) as run:
        _log_context(exp, params, data_dir, epochs, {"quick": str(quick)})

        run_dir = Path("runs/train") / exp["name"]
        if run_dir.exists():  # ultralytics ต่อท้าย results.csv เดิม -> ลบของรอบก่อนทิ้ง
            shutil.rmtree(run_dir)
        model = YOLO(exp["model"])
        model.train(
            data=str(data_dir / "data.yaml"), epochs=epochs, batch=exp["batch"], lr0=exp["lr0"], imgsz=tp["imgsz"],
            patience=tp["patience"], device=resolve_device(tp["device"]), workers=tp["workers"],
            seed=params["seed"], deterministic=True,
            project="runs/train", name=exp["name"], exist_ok=True, verbose=False,
            **exp.get("augment", {}),
        )
        mlflow.log_param("batch_actual", model.trainer.batch_size)  # batch -1 = auto
        return _evaluate_and_log(exp, Path(model.trainer.save_dir), data_dir, params, run)


def reevaluate_experiment(exp: dict, data_dir: str | Path, params: dict | None = None) -> dict:
    """ประเมิน weights ที่เทรนไว้แล้วใน runs/train/<name> ด้วยตัวชี้วัด/config ปัจจุบัน (ไม่เทรนใหม่)"""
    params = params or load_params()
    data_dir = Path(data_dir)
    save_dir = Path("runs/train") / exp["name"]
    if not (save_dir / "weights" / "best.pt").exists():
        raise FileNotFoundError(f"ไม่พบ {save_dir / 'weights' / 'best.pt'} -> ต้องเทรนก่อน")
    args = yaml.safe_load((save_dir / "args.yaml").read_text(encoding="utf-8"))
    if args.get("imgsz") != params["train"]["imgsz"]:
        raise ValueError(f"weights เทรนที่ imgsz={args.get('imgsz')} แต่ config คือ {params['train']['imgsz']}")

    mlflow.set_experiment(params["train"]["mlflow_experiment"])
    with mlflow.start_run(run_name=f"{exp['name']}-reeval") as run:
        _log_context(exp, params, data_dir, int(args.get("epochs", exp["epochs"])), {"reevaluated": "true"})
        mlflow.log_param("batch_actual", args.get("batch"))
        return _evaluate_and_log(exp, save_dir, data_dir, params, run)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    params = load_params()
    names = [e["name"] for e in params["train"]["experiments"]]
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=names, default=names[0])
    parser.add_argument("--data", default=params["data"]["processed_dir"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--reevaluate", action="store_true", help="ใช้ weights เดิม ไม่เทรนใหม่")
    args = parser.parse_args()
    exp = next(e for e in params["train"]["experiments"] if e["name"] == args.experiment)
    if args.reevaluate:
        reevaluate_experiment(exp, args.data, params)
    else:
        train_experiment(exp, args.data, params, args.quick)
