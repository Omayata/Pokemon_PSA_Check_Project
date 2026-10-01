import copy
import io

import numpy as np
import pytest
import yaml
from PIL import Image

from src.config import load_params

CLASSES = ["Card", "Corner Wear", "Edge Wear", "Scratch"]


@pytest.fixture
def params():
    p = copy.deepcopy(load_params())
    p["schema"]["min_images_per_split"] = {"train": 3, "valid": 1, "test": 1}
    p["schema"]["min_instances_per_class"] = 1
    return p


def card_image(seed: int = 0, size=(300, 420)) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))


def jpeg_bytes(img: Image.Image, fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def yolo_dataset(tmp_path):
    """dataset YOLO ขนาดเล็กที่ถูกต้องตาม schema"""
    root = tmp_path / "ds"
    seed = 0
    for split, n in {"train": 4, "valid": 1, "test": 1}.items():
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for i in range(n):
            seed += 1
            card_image(seed).save(root / split / "images" / f"{split}_{i}.jpg")
            lines = ["0 0.5 0.5 0.9 0.9"] + [f"{c} 0.3 0.3 0.1 0.1" for c in (1, 2, 3)]
            (root / split / "labels" / f"{split}_{i}.txt").write_text("\n".join(lines))
    (root / "data.yaml").write_text(yaml.safe_dump({"nc": 4, "names": CLASSES}))
    return root
