"""Predictor = Stage 1 (YOLO detector) + Stage 2 (classifier good/defective)
ใช้ตัวเดียวกันทั้งตอนประเมินผลและตอน serving

CardConditionPyfunc ห่อ Predictor เป็น MLflow pyfunc model: weights + config (รวม threshold) + reference stats
ถูกเก็บรวมกันใน model version เดียว -> serving ได้ config ชุดเดียวกับตอนเทรนเสมอ
"""

import json
from pathlib import Path

import mlflow.pyfunc
import yaml
from PIL import Image

from src.models.classifier import Detection, classify


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
            conf=self.params["classifier"]["detect_conf"],
            device=self.params["serving"]["device"],
            verbose=False,
        )[0]
        return [
            Detection(cls=result.names[int(c)], confidence=float(s), box=tuple(float(v) for v in xyxy))
            for c, s, xyxy in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy, strict=True)
        ]

    def predict(self, img: Image.Image) -> dict:
        return classify(self.detect(img), self.params["classifier"])


class CardConditionPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        def artifact(key: str) -> Path:  # โมเดลที่ log จาก Windows เก็บ path ด้วย "\"
            return Path(context.artifacts[key].replace("\\", "/"))

        params = yaml.safe_load(artifact("params").read_text(encoding="utf-8"))
        reference = json.loads(artifact("reference_stats").read_text(encoding="utf-8"))
        self.predictor = Predictor(artifact("weights"), params, reference)

    def predict(self, context, model_input, params=None):
        """model_input: list ของ path ภาพ (หรือ DataFrame คอลัมน์ 'path')"""
        from src.features.transform import load_image

        paths = model_input["path"].tolist() if hasattr(model_input, "columns") else list(model_input)
        max_side = self.predictor.params["transform"]["max_side"]
        return [self.predictor.predict(load_image(p, max_side=max_side)) for p in paths]
