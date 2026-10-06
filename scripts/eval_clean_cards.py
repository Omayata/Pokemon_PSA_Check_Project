"""วัด false positive บนการ์ดสภาพสมบูรณ์จากภายนอก: โมเดลทักการ์ดดีว่ามีตำหนิบ่อยแค่ไหน

ใช้ภาพ clean_card_negative จาก Roboflow mike-gee/card_grading_defects (CC BY 4.0) เป็นชุดทดสอบเท่านั้น ไม่ใช้เทรน:
ภาพเป็น scan ดิจิทัล (ไม่ใช่รูปถ่าย) และ label ตำหนิของ dataset นี้ไม่น่าเชื่อถือ แต่ "การ์ดดี" ใช้ได้
-> วัด specificity บนการ์ดที่ไม่อยู่ใน dataset ที่เทรน (test set เรามีการ์ดดีแค่ 30 ใบ)

python scripts/eval_clean_cards.py [--weights runs/train/<exp>/weights/best.pt] [--thresholds 0.15 0.20]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import httpx
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import resolve_device  # noqa: E402
from src.features.transform import load_image  # noqa: E402

WORKSPACE, PROJECT, PREFIX = "mike-gee", "card_grading_defects", "clean"
CACHE = Path("data/external/mikegee_clean")


def download(api_key: str) -> list[Path]:
    """ดาวน์โหลดครั้งเดียวแล้ว cache ไว้ (ภาพที่ชื่อขึ้นต้นด้วย clean = การ์ดไม่มีตำหนิ)"""
    CACHE.mkdir(parents=True, exist_ok=True)
    url = f"https://api.roboflow.com/{WORKSPACE}/{PROJECT}/search"
    items, offset = [], 0
    while True:
        r = httpx.post(url, params={"api_key": api_key}, timeout=60,
                       json={"limit": 250, "offset": offset, "fields": ["name", "url"]}).json()
        batch = r.get("results", [])
        items += [x for x in batch if x["name"].lower().startswith(PREFIX)]
        offset += len(batch)
        if not batch or offset >= r.get("total", 0):
            break
    paths = []
    for x in items:
        path = CACHE / x["name"]
        if not path.exists():
            path.write_bytes(httpx.get(x["url"], timeout=60, follow_redirects=True).content)
        paths.append(path)
    return sorted(paths)


def main() -> int:
    from dotenv import load_dotenv

    from src.models.predictor import Predictor

    load_dotenv(".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/train/yolov8n_aug/weights/best.pt")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.15, 0.20])
    parser.add_argument("--report", default="reports/clean_cards_eval.json")
    args = parser.parse_args()

    cached = sorted(CACHE.glob("*")) if CACHE.exists() else []
    paths = cached or download(os.environ["ROBOFLOW_API_KEY"])
    # ใช้ config ที่บันทึกมากับ weights ตัวนี้ (imgsz/max_side/threshold ตรงกับตอนเทรน)
    weights = Path(args.weights)
    params = yaml.safe_load((weights.parents[1] / "params_snapshot.yaml").read_text(encoding="utf-8"))
    params["serving"]["device"] = resolve_device(params["train"]["device"])
    predictor = Predictor(weights, params)

    results = []
    for p in paths:
        r = predictor.predict(load_image(p, max_side=params["transform"]["max_side"]))
        top = max(r["defects"], key=lambda d: d["confidence"])["cls"] if r["defects"] else None
        results.append({"file": p.name, "defect_probability": r["defect_probability"], "top_defect": top})

    probs = [r["defect_probability"] for r in results]
    summary = {
        "weights": str(weights), "n_clean_cards": len(results), "imgsz": params["train"]["imgsz"],
        "tuned_threshold": params["classifier"]["threshold"],
        "false_positive_rate": {str(t): round(sum(p >= t for p in probs) / len(probs), 3) for t in args.thresholds},
        "worst": sorted(results, key=lambda r: -r["defect_probability"])[:10],
    }
    Path(args.report).parent.mkdir(exist_ok=True)
    Path(args.report).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"การ์ดสภาพดี {len(results)} ใบ | weights {weights} (imgsz {summary['imgsz']})")
    for t, fpr in summary["false_positive_rate"].items():
        print(f"  threshold {t}: ทักผิดว่ามีตำหนิ {fpr:.1%} -> specificity {1 - fpr:.1%}")
    print("  สูงสุด:", ", ".join(f"{r['file']} {r['defect_probability']:.2f} ({r['top_defect']})"
                              for r in summary["worst"][:5]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
