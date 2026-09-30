from PIL import Image

from src.features.transform import image_stats, load_image
from src.models.grader import Detection, band_names, grade
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


def test_grade_clean_card_is_top_band(params):
    cfg = params["grader"]
    result = grade([Detection("Card", 0.95, (0, 0, 300, 420))], (300, 420), cfg)
    assert result["score"] == 10.0
    assert result["band"] == band_names(cfg)[0]
    assert result["card_found"]


def test_grade_more_defects_lower_score(params):
    cfg = params["grader"]
    card = Detection("Card", 0.95, (0, 0, 300, 420))
    few = grade([card, Detection("Scratch", 0.8, (10, 10, 30, 30))], (300, 420), cfg)
    many = grade(
        [card, Detection("Scratch", 0.9, (10, 10, 120, 120)), Detection("Edge-Wear", 0.9, (0, 0, 300, 20)),
         Detection("corner_wear", 0.9, (0, 0, 40, 40))],
        (300, 420), cfg,
    )
    assert many["score"] < few["score"] < 10
    assert many["n_defects"] == 3


def test_grade_ignores_low_confidence(params):
    cfg = params["grader"]
    result = grade([Detection("Scratch", cfg["min_confidence"] - 0.01, (0, 0, 50, 50))], (300, 420), cfg)
    assert result["n_defects"] == 0


def test_psi_same_vs_shifted():
    ref = list(range(1000))
    assert psi(ref, ref) < 0.01
    assert psi(ref, [v + 600 for v in ref]) > 0.2
