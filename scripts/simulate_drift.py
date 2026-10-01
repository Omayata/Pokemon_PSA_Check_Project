"""จำลอง drift แล้วส่งเข้า API จริง เพื่อสาธิตว่าระบบเฝ้าระวังตรวจจับได้และแยกแยะชนิดได้

Data drift:    python scripts/simulate_drift.py data --kind dark --n 100
               -> input มืดลง -> PSI ของ brightness สูง -> data drift
Concept drift: python scripts/simulate_drift.py concept --n 120
               -> ส่งภาพปกติ (input ไม่เปลี่ยน) แต่ feedback ช่วงหลังบอกว่าการ์ดที่ทายว่า good จริง ๆ มีตำหนิ
                  (มาตรฐานเข้มขึ้น เช่น ตำหนิแบบใหม่ที่โมเดลไม่รู้จัก)
               -> agreement ลด -> concept drift (โดย data drift ไม่ขึ้น)
Normal:        python scripts/simulate_drift.py normal --n 100
"""

import argparse
import io
import json
import random
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_params  # noqa: E402
from src.features.transform import load_image  # noqa: E402
from src.monitoring.drift_sim import ALL_DRIFTS  # noqa: E402


def to_jpeg(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def send(client: httpx.Client, url: str, name: str, data: bytes) -> dict | None:
    r = client.post(f"{url}/predict", files={"file": (name, data, "image/jpeg")})
    return r.json() if r.status_code == 200 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["normal", "data", "concept"])
    parser.add_argument("--kind", choices=list(ALL_DRIFTS), default="dark")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--images", default="data/processed/test/images")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    labels = load_params()["classifier"]["labels"]  # [good, defective]
    files = sorted(Path(args.images).glob("*.jpg"))
    client = httpx.Client(timeout=60)

    for i in range(args.n):
        path = files[i % len(files)]
        img = load_image(path)
        if args.mode == "data":
            img = ALL_DRIFTS[args.kind](img)
        result = send(client, args.url, path.name, to_jpeg(img))
        if result is None or args.mode != "concept":
            continue
        # concept drift: ครึ่งแรกคนตรวจเห็นด้วยกับโมเดล ~90%
        # ครึ่งหลังมาตรฐานเข้มขึ้น: การ์ดที่โมเดลว่า good คนตรวจบอกว่า defective ~90%
        pred = result["verdict"]
        if i < args.n // 2:
            true = pred if random.random() < 0.9 else labels[1 - labels.index(pred)]
        else:
            true = "defective" if pred == "good" and random.random() < 0.9 else pred
        client.post(f"{args.url}/feedback", json={"request_id": result["request_id"], "true_label": true})

    report = client.get(f"{args.url}/monitoring/drift").json()
    print(json.dumps({k: report[k] for k in ("data_drift", "concept_drift", "retrain_recommended",
                                             "suggested_augment")}, indent=2))


if __name__ == "__main__":
    main()
