"""Model Registry (MLflow): ด่านตรวจก่อนอนุมัติ, promote, rollback

สถานะของแต่ละ version เก็บใน tag "status": approved / rejected / rolled_back
alias "champion" = ตัวที่ production ใช้, alias "previous" = ตัวก่อนหน้า (ไว้ rollback)

python -m src.registry.promote list
python -m src.registry.promote rollback [--to-version N]
"""

import argparse
import json
import os
from pathlib import Path

import httpx
from mlflow import MlflowClient, register_model
from mlflow.exceptions import MlflowException

from src.alerts import send_alert
from src.config import load_params
from src.registry.gates import CHAMPION_METRIC, TEST_VERSION_TAG, check_gates


def _alias_version(client: MlflowClient, name: str, alias: str):
    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException:
        return None


def gate_and_register(candidate: dict, params: dict | None = None) -> dict:
    params = params or load_params()
    reg = params["registry"]
    client = MlflowClient()

    champion = _alias_version(client, reg["model_name"], reg["champion_alias"])
    # เทียบกับ champion เฉพาะเมื่อวัดบน test set ชุดเดียวกัน: test set เปลี่ยน (เช่น เพิ่ม dataset) -> F1 คนละสเกล
    # champion รุ่นเก่าที่ไม่มี metric/tag นี้ -> ไม่เทียบ (เกณฑ์ขั้นต่ำข้ออื่นยังบังคับใช้ตามปกติ)
    champion_f1 = None
    if champion:
        champion_run = client.get_run(champion.run_id)
        candidate_test = client.get_run(candidate["run_id"]).data.tags.get(TEST_VERSION_TAG)
        if candidate_test and champion_run.data.tags.get(TEST_VERSION_TAG) == candidate_test:
            champion_f1 = champion_run.data.metrics.get(CHAMPION_METRIC)
    decision = check_gates(candidate, champion_f1, params["gates"])
    decision.update({"candidate_run_id": candidate["run_id"], "candidate_name": candidate["name"],
                     "champion_version": champion.version if champion else None,
                     "compared_with_champion": champion_f1 is not None})

    # ลงทะเบียนทุกตัว (ทั้งผ่านและไม่ผ่าน) เพื่อให้เห็นประวัติใน registry
    mv = register_model(f"runs:/{candidate['run_id']}/model", reg["model_name"])
    status = "approved" if decision["passed"] else "rejected"
    client.set_model_version_tag(reg["model_name"], mv.version, "status", status)
    client.set_model_version_tag(reg["model_name"], mv.version, "gate_report", json.dumps(decision["checks"])[:5000])
    decision["registered_version"] = mv.version

    if decision["passed"]:
        if champion:
            client.set_registered_model_alias(reg["model_name"], reg["previous_alias"], champion.version)
        client.set_registered_model_alias(reg["model_name"], reg["champion_alias"], mv.version)
        print(f"✅ version {mv.version} ผ่าน gate -> เป็น champion")
    else:
        failed = [k for k, r in decision["checks"].items() if not r["passed"]]
        send_alert("Model rejected by gate", {"version": mv.version, "failed": failed})
        print(f"❌ version {mv.version} ไม่ผ่าน gate: {failed} (champion เดิมยังให้บริการต่อ)")

    Path("reports").mkdir(exist_ok=True)
    Path("reports/gate_report.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return decision


def rollback(to_version: str | None = None, params: dict | None = None) -> str:
    params = params or load_params()
    reg = params["registry"]
    client = MlflowClient()
    current = _alias_version(client, reg["model_name"], reg["champion_alias"])
    if to_version is None:
        previous = _alias_version(client, reg["model_name"], reg["previous_alias"])
        if previous is None:
            raise RuntimeError("ไม่มี alias 'previous' ให้ย้อนกลับ ระบุ --to-version แทน")
        to_version = previous.version

    client.set_registered_model_alias(reg["model_name"], reg["champion_alias"], to_version)
    if current and str(current.version) != str(to_version):
        client.set_model_version_tag(reg["model_name"], current.version, "status", "rolled_back")
    send_alert("Model rolled back", {"from": current.version if current else None, "to": to_version}, "info")
    print(f"↩️  champion: v{current.version if current else '-'} -> v{to_version}")
    reload_api()
    return str(to_version)


def reload_api() -> None:
    url = os.getenv("PSA_API_URL", "http://localhost:8000")
    try:
        r = httpx.post(f"{url}/admin/reload", timeout=120)
        print(f"🔄 API reload: {r.status_code} {r.text}")
    except httpx.HTTPError as e:
        print(f"⚠️  เรียก API reload ไม่ได้ ({e}) - API จะโหลด champion ใหม่ตอน restart")


def list_versions(params: dict | None = None) -> None:
    params = params or load_params()
    name = params["registry"]["model_name"]
    client = MlflowClient()
    # search_model_versions ไม่คืน aliases -> อ่านจาก registered model แทน
    aliases: dict[str, list[str]] = {}
    for alias, version in client.get_registered_model(name).aliases.items():
        aliases.setdefault(str(version), []).append(alias)
    print(f"{'ver':>4} {'status':<12} {'aliases':<20} {'test_F1':>8} {'test_map50':>10}  run")
    for mv in sorted(client.search_model_versions(f"name='{name}'"), key=lambda v: int(v.version)):
        metrics = client.get_run(mv.run_id).data.metrics
        f1, m = metrics.get(CHAMPION_METRIC, float("nan")), metrics.get("test_map50", float("nan"))
        tags = ",".join(aliases.get(str(mv.version), []))
        print(f"{str(mv.version):>4} {mv.tags.get('status', '-'):<12} {tags:<20} {f1:>8.3f} {m:>10.3f}  {mv.run_id}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    rb = sub.add_parser("rollback")
    rb.add_argument("--to-version", default=None)
    args = parser.parse_args()
    if args.cmd == "list":
        list_versions()
    else:
        rollback(args.to_version)
