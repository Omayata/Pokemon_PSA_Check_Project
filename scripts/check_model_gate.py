"""CI: ตรวจเกณฑ์คุณภาพโมเดลจาก reports/candidate_metrics.json (สร้างโดย pipeline และ commit มากับ PR)

python scripts/check_model_gate.py [--metrics reports/candidate_metrics.json] [--strict]
exit 1 ถ้าไม่ผ่าน gate -> CI แดง -> merge ไม่ได้
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_params  # noqa: E402
from src.registry.gates import check_gates  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="reports/candidate_metrics.json")
    parser.add_argument("--strict", action="store_true", help="ไม่มีไฟล์ metrics = fail")
    args = parser.parse_args()

    path = Path(args.metrics)
    if not path.exists():
        print(f"⚠️  ไม่พบ {path} (ยังไม่ได้รัน pipeline แล้ว commit ผล)")
        return 1 if args.strict else 0

    candidate = json.loads(path.read_text(encoding="utf-8"))
    decision = check_gates(candidate, champion_f1=None, gates=load_params()["gates"])
    print(f"Model: {candidate['name']} (run {candidate['run_id']})")
    for name, r in decision["checks"].items():
        mark = "✅" if r["passed"] else "❌"
        print(f"  {mark} {name}: {r['value']} {r['op']} {r['threshold']}")
    print("GATE PASSED" if decision["passed"] else "GATE FAILED")
    return 0 if decision["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
