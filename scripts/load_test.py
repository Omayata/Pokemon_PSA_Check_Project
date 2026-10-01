"""Load test: วัด latency p50/p95/p99 และ throughput ของ API จริง แล้วเทียบกับ SLO ใน params.yaml

python scripts/load_test.py --images data/processed/test/images --n 200 --concurrency 8
ผลลัพธ์: reports/load_test.json (exit 1 ถ้าผิด SLO)
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_params  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--images", default="data/processed/test/images")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    files = sorted(p for p in Path(args.images).glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not files:
        print(f"ไม่พบภาพใน {args.images}")
        return 1
    payloads = [(f.name, f.read_bytes()) for f in files]
    client = httpx.Client(timeout=30)

    def one(i: int) -> tuple[float, int]:
        name, data = payloads[i % len(payloads)]
        t0 = time.perf_counter()
        try:
            status = client.post(f"{args.url}/predict", files={"file": (name, data, "image/jpeg")}).status_code
        except httpx.HTTPError:
            status = 0
        return (time.perf_counter() - t0) * 1000, status

    for i in range(3):  # warm-up
        one(i)
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(one, range(args.n)))
    elapsed = time.perf_counter() - t_start

    lat = np.array([r[0] for r in results])
    errors = sum(1 for _, s in results if s != 200)
    slo = load_params()["slo"]
    report = {
        "n_requests": args.n,
        "concurrency": args.concurrency,
        "p50_ms": round(float(np.percentile(lat, 50)), 1),
        "p95_ms": round(float(np.percentile(lat, 95)), 1),
        "p99_ms": round(float(np.percentile(lat, 99)), 1),
        "throughput_rps": round(args.n / elapsed, 2),
        "error_rate": round(errors / args.n, 4),
    }
    report["slo"] = slo
    report["slo_check"] = {
        "p50": report["p50_ms"] <= slo["p50_latency_ms"],
        "p95": report["p95_ms"] <= slo["p95_latency_ms"],
        "throughput": report["throughput_rps"] >= slo["min_throughput_rps"],
        "error_rate": report["error_rate"] <= slo["max_error_rate"],
    }
    report["slo_met"] = all(report["slo_check"].values())
    Path("reports").mkdir(exist_ok=True)
    Path("reports/load_test.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["slo_met"] else 1


if __name__ == "__main__":
    sys.exit(main())
