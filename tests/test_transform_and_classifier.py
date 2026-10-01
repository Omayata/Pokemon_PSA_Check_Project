import numpy as np
from PIL import Image

from src.features.transform import image_stats, load_image
from src.models.classifier import Detection, classify
from src.models.evaluate import binary_metrics, roc_auc, tune_threshold
from src.monitoring.drift import psi
from src.monitoring.drift_sim import blur, dark
from tests.conftest import card_image, jpeg_bytes


def test_load_image_same_result_from_bytes_and_path(tmp_path):
    """train อ่านจากไฟล์ / serving อ่านจาก bytes -> ต้องได้ภาพเดียวกัน (กัน training-serving skew)"""
    img = card_image(size=(1600, 2240))
    path = tmp_path / "c.jpg"
    img.save(path)
    a = load_image(path, max_side=1280)
    b = load_image(jpeg_bytes(Image.open(path)), max_side=1280)
    assert a.size == b.size == (914, 1280)
    assert a.mode == b.mode == "RGB"


def test_load_image_converts_rgba_and_grayscale():
    assert load_image(jpeg_bytes(card_image().convert("RGBA"), "PNG")).mode == "RGB"
    assert load_image(jpeg_bytes(card_image().convert("L"))).mode == "RGB"


def test_image_stats_detect_dark_and_blur():
    img = card_image()
    base = image_stats(img)
    assert image_stats(dark(img))["brightness"] < base["brightness"] * 0.6
    assert image_stats(blur(img))["sharpness"] < base["sharpness"] * 0.5


CARD = Detection("Card", 0.95, (0, 0, 300, 420))


def test_clean_card_is_good(params):
    result = classify([CARD], params["classifier"])
    assert result["verdict"] == "good"
    assert result["defect_probability"] == 0.0
    assert result["card_found"] and result["n_defects"] == 0


def test_defect_above_threshold_is_defective(params):
    cfg = params["classifier"]
    result = classify([CARD, Detection("Edge-Wear", cfg["threshold"] + 0.1, (0, 0, 300, 20)),
                       Detection("corner_wear", 0.9, (0, 0, 40, 40))], cfg)
    assert result["verdict"] == "defective"
    assert result["defect_probability"] == 0.9  # = confidence สูงสุดของตำหนิ
    assert result["defect_counts"] == {"corner wear": 1, "edge wear": 1, "scratch": 0}


def test_defect_below_threshold_is_good_but_scored(params):
    cfg = params["classifier"]
    result = classify([CARD, Detection("Scratch", cfg["threshold"] - 0.01, (0, 0, 50, 50))], cfg)
    assert result["verdict"] == "good"
    assert result["n_defects"] == 0  # ไม่แสดงกรอบที่ต่ำกว่า threshold
    assert 0 < result["defect_probability"] < cfg["threshold"]  # แต่ยังใช้คำนวณ ROC ได้


def test_card_box_is_not_a_defect(params):
    assert classify([Detection("Card", 0.99, (0, 0, 10, 10))], params["classifier"])["verdict"] == "good"


def test_binary_metrics_counts():
    y = np.array([1, 1, 1, 0, 0])
    s = np.array([0.9, 0.6, 0.1, 0.7, 0.05])
    m = binary_metrics(y, s, 0.5)
    assert (m["img_tp"], m["img_fn"], m["img_fp"], m["img_tn"]) == (2, 1, 1, 1)
    assert m["img_recall"] == 2 / 3 and m["img_precision"] == 2 / 3 and m["img_specificity"] == 0.5


def test_roc_auc_perfect_random_and_ties():
    y = np.array([1, 1, 0, 0])
    assert roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 1.0
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 0.0
    assert roc_auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5


def test_tune_threshold_separates_classes():
    y = np.array([1, 1, 1, 0, 0, 0])
    s = np.array([0.8, 0.7, 0.4, 0.3, 0.1, 0.0])
    t = tune_threshold(y, s)
    assert 0.3 < t <= 0.4
    assert binary_metrics(y, s, t)["img_f1"] == 1.0


def test_psi_same_vs_shifted():
    ref = list(range(1000))
    assert psi(ref, ref) < 0.01
    assert psi(ref, [v + 600 for v in ref]) > 0.2
