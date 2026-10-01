"""สาธิตข้อมูลเสีย: ระบบต้องหยุดและแจ้งเตือน

1) dataset เสีย -> validate_dataset หยุด pipeline + alert
   python scripts/make_bad_data.py dataset
2) ภาพเสียส่งเข้า API -> ตอบ 422 พร้อมเหตุผล + นับใน psa_rejected_inputs_total
   python scripts/make_bad_data.py api --url http://localhost:8000
"""

import argparse
import io
import shutil
import sys
from pathlib import Path

import httpx
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_params  # noqa: E402
from src.data.validate import DataValidationError, validate_dataset  # noqa: E402


def corrupt_dataset(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    labels = sorted((dst / "train" / "labels").glob("*.txt"))
    images = sorted((dst / "train" / "images").glob("*.jpg"))
    labels[0].write_text("9 0.5 0.5 0.2 0.2\n")            # class id ที่ไม่มีอยู่จริง
    labels[1].write_text("1 1.4 0.5 0.2 0.2\n")            # พิกัดเกิน 1
    labels[2].write_text("1 0.5 0.5\n")                    # จำนวนค่าไม่ครบ
    images[3].write_bytes(images[3].read_bytes()[:500])    # ไฟล์ภาพถูกตัดกลางทาง
    shutil.copy(images[4], dst / "test" / "images" / images[4].name)  # leakage: ภาพ train อยู่ใน test
    shutil.copy(labels[4], dst / "test" / "labels" / labels[4].name)
    labels[5].unlink()                                     # label หาย


def demo_dataset() -> int:
    params = load_params()
    src, dst = Path(params["data"]["processed_dir"]), Path("data/bad_dataset")
    corrupt_dataset(src, dst)
    try:
        validate_dataset(dst, params["schema"])
        print("❌ ไม่ควรผ่าน!")
        return 1
    except DataValidationError as e:
        print(f"✅ ระบบหยุดตามที่คาด: {e}")
        for err in e.errors[:10]:
            print("   -", err)
        return 0


def bad_uploads() -> dict[str, tuple[str, bytes, str]]:
    def img_bytes(img: Image.Image, fmt: str = "JPEG") -> bytes:
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        return buf.getvalue()

    return {
        "not_an_image": ("card.jpg", b"this is not an image", "image/jpeg"),
        "too_small": ("tiny.jpg", img_bytes(Image.new("RGB", (50, 70), "red")), "image/jpeg"),
        "blank": ("blank.jpg", img_bytes(Image.new("RGB", (600, 840), "white")), "image/jpeg"),
        "wrong_format": ("card.gif", img_bytes(Image.new("RGB", (600, 840), "blue"), "GIF"), "image/gif"),
        "panorama": ("pano.jpg", img_bytes(Image.effect_noise((3000, 400), 60).convert("RGB")), "image/jpeg"),
        "empty": ("empty.jpg", b"", "image/jpeg"),
    }


def demo_api(url: str) -> int:
    ok = True
    for case, (name, data, mime) in bad_uploads().items():
        r = httpx.post(f"{url}/predict", files={"file": (name, data, mime)}, timeout=30)
        passed = r.status_code == 422
        ok &= passed
        print(f"{'✅' if passed else '❌'} {case}: HTTP {r.status_code} {r.text[:150]}")
    return 0 if ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=["dataset", "api"])
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()
    sys.exit(demo_dataset() if args.target == "dataset" else demo_api(args.url))
