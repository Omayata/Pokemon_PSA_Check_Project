"""ข้อมูลดีต้องผ่าน / ข้อมูลเสียทุกแบบต้องถูกหยุดพร้อมเหตุผล"""

import shutil

import pytest
from PIL import Image

from src.data.validate import DataValidationError, validate_dataset, validate_upload
from tests.conftest import card_image, jpeg_bytes


def test_valid_dataset_passes(yolo_dataset, params):
    report = validate_dataset(yolo_dataset, params["schema"], alert=False)
    assert report["passed"]
    assert report["splits"] == {"train": 4, "valid": 1, "test": 1}


@pytest.mark.parametrize(
    "corrupt, expected",
    [
        (lambda d: (d / "train/labels/train_0.txt").write_text("9 0.5 0.5 0.2 0.2"), "out of range"),
        (lambda d: (d / "train/labels/train_0.txt").write_text("1 1.4 0.5 0.2 0.2"), "bbox out of range"),
        (lambda d: (d / "train/labels/train_0.txt").write_text("1 0.5 0.5"), "expected 5 values"),
        (lambda d: (d / "train/labels/train_0.txt").unlink(), "missing label"),
        (lambda d: (d / "train/images/train_0.jpg").write_bytes(b"garbage"), "corrupt image"),
        (lambda d: shutil.copy(d / "train/images/train_1.jpg", d / "test/images/leak.jpg")
         or shutil.copy(d / "train/labels/train_1.txt", d / "test/labels/leak.txt"), "leakage"),
        (lambda d: (d / "data.yaml").write_text("nc: 2\nnames: [card, dent]"), "class names mismatch"),
        (lambda d: shutil.rmtree(d / "valid"), "valid: only 0 images"),
    ],
)
def test_bad_dataset_is_rejected(yolo_dataset, params, corrupt, expected):
    corrupt(yolo_dataset)
    with pytest.raises(DataValidationError) as exc:
        validate_dataset(yolo_dataset, params["schema"], alert=False)
    assert any(expected in e for e in exc.value.errors), exc.value.errors


def test_valid_upload_passes(params):
    validate_upload(jpeg_bytes(card_image()), params["schema"])


def test_rgba_png_is_accepted(params):
    validate_upload(jpeg_bytes(card_image().convert("RGBA"), "PNG"), params["schema"])


@pytest.mark.parametrize(
    "data, expected",
    [
        (b"", "empty_file"),
        (b"not an image", "cannot_decode"),
        (jpeg_bytes(Image.new("RGB", (50, 70), "red")), "image_too_small"),
        (jpeg_bytes(Image.new("RGB", (300, 420), "white")), "blank_image"),
        (jpeg_bytes(card_image(size=(2000, 300))), "aspect_ratio_out_of_range"),
        (jpeg_bytes(card_image(), "GIF"), "format_not_allowed"),
    ],
    ids=["empty", "not_image", "too_small", "blank", "panorama", "gif"],
)
def test_bad_upload_is_rejected(params, data, expected):
    with pytest.raises(DataValidationError) as exc:
        validate_upload(data, params["schema"])
    assert any(e.startswith(expected) for e in exc.value.errors), exc.value.errors
