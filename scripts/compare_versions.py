"""เทียบหลาย model version ใน registry บนข้อมูลชุดเดียวกัน (fair comparison)

ตัวเลขที่ log ตอนเทรนของแต่ละ version วัดบน test set คนละรุ่น (data_version ต่างกัน) จึงเทียบกันตรง ๆ ไม่ได้
สคริปต์นี้ประเมินทุก version ใหม่ด้วยข้อมูลเดียวกัน:
  (1) test set ปัจจุบัน (data/processed/test)
  (2) การ์ดสภาพดีจากภายนอก 110 ใบ (data/external/mikegee_clean, ดู scripts/eval_clean_cards.py) -> วัด false positive
แต่ละ version ใช้ imgsz / max_side / threshold ของตัวเอง (จาก params_snapshot.yaml ที่เก็บมากับโมเดล)

python scripts/compare_versions.py --versions 8 9 11
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_params, resolve_device  # noqa: E402
from src.models.evaluate import (  # noqa: E402
    binary_metrics,
    image_labels,
    measure_latency,
    per_source_metrics,
    predict_images,
)

CLEAN_CARDS = Path("data/external/mikegee_clean")


def version_info(db: Path, artifacts: Path, model_name: str, version: int) -> dict:
    """อ่านจาก MLflow backend (sqlite) โดยตรง -> ไม่ต้องเปิด MLflow server"""
    con = sqlite3.connect(db)
    run_id = con.execute(
        "SELECT run_id FROM model_versions WHERE name = ? AND version = ?", (model_name, version)
    ).fetchone()[0]
    run_name = con.execute("SELECT name FROM runs WHERE run_uuid = ?", (run_id,)).fetchone()[0]
    tags = dict(con.execute(
        "SELECT key, value FROM model_version_tags WHERE name = ? AND version = ?", (model_name, version)
    ).fetchall())
    run_params = dict(con.execute(
        "SELECT key, value FROM params WHERE run_uuid = ? AND key = 'data_version'", (run_id,)
    ).fetchall())
    art = artifacts / "1" / run_id / "artifacts" / "model" / "artifacts"
    return {
        "version": version, "run_id": run_id, "run_name": run_name, "status": tags.get("status"),
        "trained_on_data_version": run_params.get("data_version"),
        "weights": next(art.glob("*.pt")), "params": yaml.safe_load(
            (art / "params_snapshot.yaml").read_text(encoding="utf-8")),
    }


def main() -> int:
    from src.models.predictor import Predictor

    current = load_params()
    parser = argparse.ArgumentParser()
    parser.add_argument("--versions", type=int, nargs="+", required=True)
    parser.add_argument("--db", default="mlflow_data/mlflow.db")
    parser.add_argument("--artifacts", default="mlflow_data/artifacts")
    parser.add_argument("--data", default=current["data"]["processed_dir"])
    parser.add_argument("--report", default="reports/version_comparison.json")
    parser.add_argument("--serving-threshold", action="store_true",
                        help="ใช้ serving.threshold ของ API (ค่าที่ผู้ใช้เห็นจริง) แทน threshold ที่ tune มากับแต่ละ version")
    args = parser.parse_args()

    data_dir = Path(args.data)
    data_version = json.loads((data_dir / "data_version.json").read_text(encoding="utf-8"))
    # label จริงตาม class ชุดปัจจุบัน -> ทุก version ถูกตัดสินด้วยเกณฑ์เดียวกัน
    test_paths, y_test = image_labels(data_dir / "test", current["classifier"]["defect_classes"])
    clean_paths = sorted(CLEAN_CARDS.glob("*")) if CLEAN_CARDS.exists() else []

    rows = []
    for v in args.versions:
        info = version_info(Path(args.db), Path(args.artifacts), current["registry"]["model_name"], v)
        params = info["params"]
        params["serving"]["device"] = resolve_device(params["train"]["device"])
        if args.serving_threshold and current["serving"].get("threshold") is not None:
            params["classifier"]["threshold"] = current["serving"]["threshold"]
        max_side, threshold = params["transform"]["max_side"], params["classifier"]["threshold"]
        predictor = Predictor(info["weights"], params)
        print(f"v{v} ({info['run_name']}, imgsz {params['train']['imgsz']}, threshold {threshold}) ...")

        s_test = np.array([r["defect_probability"] for r in predict_images(predictor, test_paths, max_side)])
        test = binary_metrics(y_test, s_test, threshold)
        row = {
            "version": v, "run_name": info["run_name"], "run_id": info["run_id"], "status": info["status"],
            "trained_on_data_version": info["trained_on_data_version"],
            "imgsz": params["train"]["imgsz"], "threshold": threshold,
            **{f"test_{k}": v_ for k, v_ in test.items()},
            **{f"test_{k}": v_ for k, v_ in per_source_metrics(test_paths, y_test, s_test, threshold).items()},
            **measure_latency(predictor, test_paths, max_side),
        }
        if clean_paths:
            s_clean = np.array([r["defect_probability"] for r in predict_images(predictor, clean_paths, max_side)])
            row["clean_cards_n"] = len(clean_paths)
            row["clean_cards_false_positive_rate"] = float((s_clean >= threshold).mean())
        rows.append(row)

    report = {
        "evaluated_on_data_version": data_version.get("processed_version"),
        "n_test": len(test_paths), "n_test_good": int((y_test == 0).sum()),
        "versions": rows,
    }
    Path(args.report).parent.mkdir(exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    cols = ["test_img_f1", "test_img_recall", "test_img_precision", "test_img_specificity",
            "test_img_roc_auc", "clean_cards_false_positive_rate", "latency_p95_ms"]
    short = ["F1", "recall", "precision", "specificity", "ROC-AUC", "FP การ์ดดี", "p95 ms"]
    print(f"\ntest set {len(test_paths)} ภาพ (การ์ดดี {report['n_test_good']}) + การ์ดดีภายนอก {len(clean_paths)} ใบ")
    print(f"{'version':<28}" + "".join(f"{h:>12}" for h in short))
    for r in rows:
        name = f"v{r['version']} {r['run_name']} ({r['imgsz']})"
        print(f"{name:<28}" + "".join(f"{r.get(c, float('nan')):>12.3f}" for c in cols))
    print(f"\nบันทึกที่ {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
