"""Pokemon PSA Grader API

Serving pattern: Cascade แบบ real-time (Stage 1 YOLO หาตำหนิ -> Stage 2 ประเมินช่วงเกรด)
+ /predict/batch สำหรับผู้ขายที่ส่งการ์ดหลายใบพร้อมกัน

Endpoints: /predict /predict/batch /feedback /health /ready /metrics /monitoring/drift /admin/reload
"""

import json
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel

from src.config import load_params
from src.data.validate import DataValidationError, validate_upload
from src.features.transform import image_stats, load_image
from src.models.grader import band_names

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("api")

PARAMS = load_params()
LOG_DIR = Path(os.getenv("LOG_DIR", PARAMS["monitoring"]["log_dir"]))
MAX_BATCH = 16

# ---------------------------------------------------------------- metrics
REQUEST_LATENCY = Histogram(
    "psa_request_latency_seconds", "End-to-end request latency", ["endpoint"],
    buckets=(0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0),
)
REQUESTS = Counter("psa_requests_total", "Requests", ["endpoint", "status"])
REJECTED = Counter("psa_rejected_inputs_total", "Inputs rejected by schema validation", ["reason"])
PREDICTED_BAND = Counter("psa_predicted_band_total", "Predicted PSA band", ["band"])
DEFECTS = Counter("psa_detected_defects_total", "Detected defects", ["type"])
MODEL_LOADED = Gauge("psa_model_loaded", "1 if a model is loaded")
MODEL_VERSION = Gauge("psa_model_version", "Registry version of the loaded model")
DATA_DRIFT = Gauge("psa_data_drift_detected", "1 if data drift detected")
DRIFT_PSI = Gauge("psa_drift_psi", "PSI per input feature", ["feature"])
CONCEPT_DRIFT = Gauge("psa_concept_drift_detected", "1 if concept drift detected")
AGREEMENT = Gauge("psa_feedback_agreement", "Agreement with real PSA grade (latest window)")


# ---------------------------------------------------------------- model state
class ModelState:
    def __init__(self):
        self.predictor = None
        self.version: str | None = None
        self.source: str | None = None
        self.loaded_at: str | None = None
        self.lock = threading.Lock()


state = ModelState()
STARTED = time.time()


def load_model() -> None:
    """โหลด champion จาก MLflow Registry; ถ้าไม่ได้ ใช้ MODEL_PATH (ไฟล์ weights) แทน"""
    reg = PARAMS["registry"]
    predictor, version, source = None, None, None
    try:
        import mlflow
        import yaml
        from mlflow import MlflowClient

        from src.models.predictor import Predictor

        mv = MlflowClient().get_model_version_by_alias(reg["model_name"], reg["champion_alias"])
        uri = f"models:/{reg['model_name']}/{mv.version}"
        # ดาวน์โหลดไฟล์ของโมเดลแล้วอ่านเอง แทน mlflow.pyfunc.load_model เพราะ
        # (1) โมเดลที่ log จาก Windows เก็บ path แบบ "\" ซึ่ง Linux หาไม่เจอ
        # (2) ใช้โค้ด src/ ของ image นี้ ไม่ใช่โค้ดเก่าที่ถูกแนบมากับโมเดล
        art = Path(mlflow.artifacts.download_artifacts(artifact_uri=uri)) / "artifacts"
        params = yaml.safe_load((art / "params_snapshot.yaml").read_text(encoding="utf-8"))
        params.setdefault("serving", PARAMS["serving"])  # โมเดลรุ่นเก่าที่ยังไม่มีค่านี้
        reference = json.loads((art / "reference_stats.json").read_text(encoding="utf-8"))
        predictor = Predictor(next(art.glob("*.pt")), params, reference)
        version, source = str(mv.version), uri
    except Exception as e:
        logger.warning("โหลดจาก MLflow ไม่ได้: %s", e)
        model_path = os.getenv("MODEL_PATH")
        if model_path and Path(model_path).exists():
            from src.models.predictor import Predictor

            predictor, version, source = Predictor(model_path, PARAMS), "local", model_path
    with state.lock:
        state.predictor, state.version, state.source = predictor, version, source
        state.loaded_at = datetime.now(UTC).isoformat() if predictor else None
    MODEL_LOADED.set(1 if predictor else 0)
    MODEL_VERSION.set(float(version) if version and version.isdigit() else 0)
    logger.info("model loaded: version=%s source=%s", version, source)


@asynccontextmanager
async def lifespan(app: FastAPI):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    load_model()
    yield


app = FastAPI(title="Pokemon PSA Grader API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    endpoint = request.url.path
    if endpoint not in {"/metrics", "/health"}:
        REQUEST_LATENCY.labels(endpoint).observe(time.perf_counter() - t0)
        REQUESTS.labels(endpoint, str(response.status_code)).inc()
    return response


def _append_jsonl(name: str, record: dict) -> None:
    with open(LOG_DIR / name, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _predict_bytes(data: bytes, filename: str | None) -> dict:
    """validate -> transform (ชุดเดียวกับตอนเทรน) -> Stage 1 + Stage 2 -> log"""
    if state.predictor is None:
        raise HTTPException(503, "model not loaded")
    request_id = uuid.uuid4().hex
    t0 = time.perf_counter()
    try:
        validate_upload(data, PARAMS["schema"])
    except DataValidationError as e:
        for err in e.errors:
            REJECTED.labels(err.split(":")[0]).inc()
        logger.warning(json.dumps({"event": "rejected_input", "request_id": request_id, "file": filename,
                                   "errors": e.errors}))
        raise HTTPException(422, {"message": str(e), "errors": e.errors, "request_id": request_id}) from e

    img = load_image(data, max_side=PARAMS["transform"]["max_side"])
    stats = image_stats(img)
    with state.lock:
        predictor, version = state.predictor, state.version
    result = predictor.predict(img)
    latency_ms = (time.perf_counter() - t0) * 1000

    PREDICTED_BAND.labels(result["band"]).inc()
    for defect, n in result["defect_counts"].items():
        if n:
            DEFECTS.labels(defect).inc(n)
    _append_jsonl("predictions.jsonl", {
        "request_id": request_id,
        "ts": datetime.now(UTC).isoformat(),
        "model_version": version,
        "input": stats,
        "prediction": {k: result[k] for k in ("score", "band", "n_defects", "defect_area_ratio")},
        "latency_ms": round(latency_ms, 2),
    })
    return {"request_id": request_id, "model_version": version, "latency_ms": round(latency_ms, 2), **result}


# ---------------------------------------------------------------- endpoints
@app.get("/health")
def health():
    """liveness: process ยังทำงาน (ตอบ 200 เสมอ พร้อมบอกสถานะโมเดล)"""
    return {
        "status": "ok" if state.predictor else "degraded",
        "model_loaded": state.predictor is not None,
        "model_version": state.version,
        "model_source": state.source,
        "loaded_at": state.loaded_at,
        "uptime_s": round(time.time() - STARTED, 1),
    }


@app.get("/ready")
def ready():
    """readiness: พร้อมรับ request หรือยัง (ไม่มีโมเดล -> 503)"""
    if state.predictor is None:
        raise HTTPException(503, "model not loaded")
    return {"ready": True, "model_version": state.version}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    return _predict_bytes(file.file.read(), file.filename)


@app.post("/predict/batch")
def predict_batch(files: list[UploadFile] = File(...)):
    if len(files) > MAX_BATCH:
        raise HTTPException(413, f"max {MAX_BATCH} files per batch")
    results = []
    for f in files:
        try:
            results.append({"file": f.filename, **_predict_bytes(f.file.read(), f.filename)})
        except HTTPException as e:
            results.append({"file": f.filename, "error": e.detail, "status_code": e.status_code})
    return {"n": len(results), "results": results}


class Feedback(BaseModel):
    request_id: str
    true_band: str  # ผลเกรดจริงจาก PSA เช่น "PSA 9-10"


@app.post("/feedback")
def feedback(fb: Feedback):
    if fb.true_band not in band_names(PARAMS["grader"]):
        raise HTTPException(422, f"true_band must be one of {band_names(PARAMS['grader'])}")
    _append_jsonl("feedback.jsonl", {**fb.model_dump(), "ts": datetime.now(UTC).isoformat()})
    return {"ok": True}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/monitoring/drift")
def monitoring_drift():
    from src.monitoring.drift import check_drift

    if state.predictor is None or not state.predictor.reference:
        raise HTTPException(503, "no reference stats (model not loaded from registry)")
    report = check_drift(state.predictor.reference, PARAMS, LOG_DIR, model_version=state.version)
    data, concept = report["data_drift"], report["concept_drift"]
    DATA_DRIFT.set(1 if data["detected"] else 0)
    for feature, v in data.get("features", {}).items():
        DRIFT_PSI.labels(feature).set(v["psi"])
    CONCEPT_DRIFT.set(1 if concept["detected"] else 0)
    if "current_agreement" in concept:
        AGREEMENT.set(concept["current_agreement"])
    return report


@app.post("/admin/reload")
def admin_reload():
    load_model()
    if state.predictor is None:
        raise HTTPException(503, "reload failed: no model available")
    return {"reloaded": True, "model_version": state.version}
