"""การแปลงข้อมูลชุดเดียว ใช้ทั้งตอนเตรียมข้อมูลเทรนและตอนให้บริการ (กัน Training-Serving Skew)

- load_image(): ถูกเรียกใน src/data/prepare.py (ตอนสร้าง data/processed) และใน app/main.py (ทุก request)
- image_stats(): คุณลักษณะของภาพที่ใช้เฝ้าระวัง data drift (คำนวณแบบเดียวกันทั้ง reference และ production)
ขั้นตอนภายในโมเดล (letterbox/normalize) ทำโดย ultralytics เวอร์ชันเดียวกัน + imgsz เดียวกันจาก params.yaml
"""

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def load_image(source: bytes | str | Path, max_side: int = 1280) -> Image.Image:
    """อ่านภาพ -> แก้การหมุนจาก EXIF -> RGB -> ย่อให้ด้านยาวไม่เกิน max_side"""
    if isinstance(source, bytes):
        img = Image.open(io.BytesIO(source))
    else:
        img = Image.open(source)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.BILINEAR)
    return img


def image_stats(img: Image.Image) -> dict:
    """คุณลักษณะภาพสำหรับตรวจ data drift: ความสว่าง ความต่างสี ความคม สัดส่วนภาพ"""
    gray = np.asarray(img.convert("L").resize((256, 256)), dtype=np.float32)
    # ความคม = variance ของ Laplacian (ภาพเบลอ -> ค่าต่ำ)
    lap = (
        gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1]
    )
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness": float(lap.var()),
        "aspect_ratio": float(img.width / img.height),
        "width": int(img.width),
        "height": int(img.height),
    }
