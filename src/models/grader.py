"""Stage 2 ของ Cascade: ตำหนิที่ Stage 1 (YOLO) detect ได้ -> คะแนน 1-10 -> ช่วงเกรด PSA

ใช้ rubric ที่กำหนดใน params.yaml (grader:) เพราะ dataset ไม่มี label เกรด PSA จริง
คะแนน = 10 - Σ (น้ำหนักตามชนิดตำหนิ × confidence × (1 + area_weight × สัดส่วนพื้นที่ต่อการ์ด))
ถ้าเก็บผลเกรด PSA จริงได้ (ผ่าน /feedback) สามารถนำมาปรับน้ำหนัก/เทรนเป็นโมเดลแทน rubric ได้
"""

from dataclasses import asdict, dataclass

from src.config import normalize_name


@dataclass
class Detection:
    cls: str
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 (pixel)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def grade(detections: list[Detection], image_size: tuple[int, int], cfg: dict) -> dict:
    card_class = normalize_name(cfg["card_class"])
    weights = {normalize_name(k): v for k, v in cfg["defect_weights"].items()}
    dets = [d for d in detections if d.confidence >= cfg["min_confidence"]]

    cards = [d for d in dets if normalize_name(d.cls) == card_class]
    card = max(cards, key=lambda d: d.confidence) if cards else None
    card_area = card.area if card and card.area > 0 else float(image_size[0] * image_size[1])

    defects = [d for d in dets if normalize_name(d.cls) in weights]
    penalty = 0.0
    counts = {name: 0 for name in weights}
    total_area_ratio = 0.0
    for d in defects:
        name = normalize_name(d.cls)
        ratio = min(d.area / card_area, 1.0)
        counts[name] += 1
        total_area_ratio += ratio
        penalty += weights[name] * d.confidence * (1 + cfg["area_weight"] * ratio)

    score = round(min(10.0, max(1.0, 10.0 - penalty)), 2)
    band = next(label for threshold, label in cfg["bands"] if score >= threshold)
    return {
        "score": score,
        "band": band,
        "card_found": card is not None,
        "n_defects": len(defects),
        "defect_counts": counts,
        "defect_area_ratio": round(total_area_ratio, 4),
        "defects": [asdict(d) for d in defects],
    }


def band_names(cfg: dict) -> list[str]:
    """รายชื่อช่วงเกรดเรียงจากดีไปแย่"""
    return [label for _, label in cfg["bands"]]
