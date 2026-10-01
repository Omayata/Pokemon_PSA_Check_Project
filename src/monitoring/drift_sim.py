"""การแปลงภาพที่จำลองสภาพแวดล้อมจริงเปลี่ยน (data drift)

ใช้ 2 ที่: scripts/simulate_drift.py (สร้าง drift เพื่อทดสอบระบบเฝ้าระวัง)
และ src/data/prepare.py (เพิ่มภาพแบบเดียวกันเข้า train set ตอน retrain)
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def dark(img: Image.Image) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(0.45)  # ถ่ายในห้องมืด


def bright(img: Image.Image) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(1.6)  # แสงแฟลชแรง


def blur(img: Image.Image) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=3))  # กล้องมือถือโฟกัสไม่ติด


def low_contrast(img: Image.Image) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(0.4)  # ถ่ายผ่านซองพลาสติกขุ่น


def noise(img: Image.Image, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.asarray(img, dtype=np.float32) + rng.normal(0, 25, size=(img.height, img.width, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))  # กล้องคุณภาพต่ำ


def rotate(img: Image.Image) -> Image.Image:
    return img.rotate(90, expand=True)  # ถ่ายแนวนอน (ใช้จำลองเท่านั้น เพราะเปลี่ยนตำแหน่ง bbox)


# drift ที่ไม่เปลี่ยนตำแหน่ง bbox -> ใช้เป็น augmentation ตอน retrain ได้
PHOTOMETRIC_DRIFTS = {"dark": dark, "bright": bright, "blur": blur, "low_contrast": low_contrast, "noise": noise}
ALL_DRIFTS = {**PHOTOMETRIC_DRIFTS, "rotate": rotate}

# feature ที่ drift -> augmentation ที่ควรเพิ่มตอน retrain
FEATURE_TO_AUGMENT = {
    ("brightness", "down"): "dark",
    ("brightness", "up"): "bright",
    ("sharpness", "down"): "blur",
    ("contrast", "down"): "low_contrast",
    ("sharpness", "up"): "noise",
}
