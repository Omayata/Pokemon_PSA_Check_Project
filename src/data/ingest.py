"""ดาวน์โหลดทุก dataset ใน data.sources จาก Roboflow (pin version) และบันทึก data version (hash ของไฟล์ทั้งหมด)

รันเอง: python -m src.data.ingest [--force]
ต้องตั้ง ROBOFLOW_API_KEY ใน .env
"""

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from src.config import load_params


def compute_data_hash(dataset_dir: Path) -> tuple[str, int]:
    """hash ของ path + เนื้อหาทุกไฟล์ (เรียงลำดับคงที่) -> ข้อมูลเปลี่ยนแม้ไฟล์เดียว hash ก็เปลี่ยน"""
    h = hashlib.sha256()
    files = sorted(
        p for p in dataset_dir.rglob("*")
        if p.is_file() and p.name != "data_version.json" and p.suffix != ".cache"  # .cache = ไฟล์ที่ ultralytics สร้างเอง
    )
    for p in files:
        h.update(p.relative_to(dataset_dir).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16], len(files)


def raw_path(params: dict, source: dict) -> Path:
    return Path(params["data"]["raw_dir"]) / source["name"] / f"v{source['roboflow_version']}"


def download_source(params: dict, source: dict, force: bool = False) -> Path:
    target = raw_path(params, source)
    ref = f"{source['roboflow_workspace']}/{source['roboflow_project']}/{source['roboflow_version']}"
    if (target / "data.yaml").exists() and not force:
        print(f"ℹ️  {source['name']}: พบข้อมูลที่ {target} แล้ว ข้ามการดาวน์โหลด (ใช้ --force เพื่อโหลดใหม่)")
    else:
        load_dotenv()
        api_key = os.getenv("ROBOFLOW_API_KEY")
        if not api_key:
            raise ValueError("ROBOFLOW_API_KEY is missing. คัดลอก .env.example เป็น .env แล้วใส่ key")
        from roboflow import Roboflow  # import ตอนใช้ เพราะ serving ไม่ต้องใช้

        project = Roboflow(api_key=api_key).workspace(source["roboflow_workspace"]).project(source["roboflow_project"])
        project.version(source["roboflow_version"]).download(
            params["data"]["export_format"], location=str(target), overwrite=True
        )

    data_hash, n_files = compute_data_hash(target)
    version_info = {
        "source": f"roboflow:{ref}",
        "sha256_16": data_hash,
        "n_files": n_files,
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    (target / "data_version.json").write_text(json.dumps(version_info, indent=2), encoding="utf-8")
    print(f"✅ {source['name']} raw data @ {target} | data version {data_hash} ({n_files} files)")
    return target


def download(params: dict | None = None, force: bool = False) -> dict[str, Path]:
    """คืน {ชื่อ source: path ของ raw data}"""
    params = params or load_params()
    return {s["name"]: download_source(params, s, force) for s in params["data"]["sources"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    download(force=parser.parse_args().force)
