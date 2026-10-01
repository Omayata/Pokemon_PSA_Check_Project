"""Predictor = Stage 1 (YOLO detector) + Stage 2 (grader) ใช้ตัวเดียวกันทั้งตอนประเมินผลและตอน serving

PSAGraderPyfunc ห่อ Predictor เป็น MLflow pyfunc model: weights + config + reference stats
ถูกเก็บรวมกันใน model version เดียว -> serving ได้ config ชุดเดียวกับตอนเทรนเสมอ
"""

import json
from pathlib import Path

import mlflow.pyfunc
import yaml
from PIL import Image

from src.models.grader import Detection, grade


class Predictor:
    def __init__(self, weights_path: str | Path, params: dict, reference: dict | None = None):
        from ultralytics import YOLO  # import ช้า -> โหลดเมื่อใช้จริง

        self.detector = YOLO(str(weights_path))
        self.params = params
        self.reference = reference or {}

    def detect(self, img: Image.Image) -> list[Detection]:
        result = self.detector.predict(
            img,
            imgsz=self.params["train"]["imgsz"],
            conf=self.params["grader"]["min_confidence"],
            device=self.params["serving"]["device"],
            verbose=False,
        )[0]
        return [
            Detection(cls=result.names[int(c)], confidence=float(s), box=tuple(float(v) for v in xyxy))
            for c, s, xyxy in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy, strict=True)
        ]

    def predict(self, img: Image.Image) -> dict:
        return grade(self.detect(img), img.size, self.params["grader"])


class PSAGraderPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        params = yaml.safe_load(Path(context.artifacts["params"]).read_text(encoding="utf-8"))
        reference = json.loads(Path(context.artifacts["reference_stats"]).read_text(encoding="utf-8"))
        self.predictor = Predictor(context.artifacts["weights"], params, reference)

    def predict(self, context, model_input, params=None):
        """model_input: list ของ path ภาพ (หรือ DataFrame คอลัมน์ 'path')"""
        from src.features.transform import load_image

        paths = model_input["path"].tolist() if hasattr(model_input, "columns") else list(model_input)
        max_side = self.predictor.params["transform"]["max_side"]
        return [self.predictor.predict(load_image(p, max_side=max_side)) for p in paths]
