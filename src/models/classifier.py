"""Stage 2 ของ Cascade: ตำหนิที่ Stage 1 (YOLO) detect ได้ -> จำแนกการ์ดระดับภาพ good / defective

defect_probability = confidence สูงสุดของกรอบตำหนิในภาพ (ไม่มีตำหนิ = 0)
defect_probability >= threshold -> "defective" ไม่งั้น "good"
threshold ถูกเลือกบน valid set ตอนเทรน (F1 สูงสุด) แล้วเก็บไปกับโมเดล -> serving ใช้ค่าเดียวกัน
"""

from dataclasses import asdict, dataclass

from src.config import normalize_name


@dataclass
class Detection:
    cls: str
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 (pixel)


def classify(detections: list[Detection], cfg: dict) -> dict:
    defect_classes = {normalize_name(c) for c in cfg["defect_classes"]}
    card_class = normalize_name(cfg["card_class"]) if cfg.get("card_class") else None
    threshold = cfg["threshold"]

    defects = [d for d in detections if normalize_name(d.cls) in defect_classes]
    defect_probability = max((d.confidence for d in defects), default=0.0)
    shown = [d for d in defects if d.confidence >= threshold]  # กรอบที่ใช้อธิบายผลให้ผู้ใช้
    counts = {name: 0 for name in sorted(defect_classes)}
    for d in shown:
        counts[normalize_name(d.cls)] += 1

    return {
        "verdict": "defective" if defect_probability >= threshold else "good",
        "defect_probability": round(defect_probability, 4),
        "threshold": threshold,
        # None = โมเดลไม่มี class การ์ด (ตรวจไม่ได้ว่ามีการ์ดในภาพไหม)
        "card_found": any(normalize_name(d.cls) == card_class for d in detections) if card_class else None,
        "n_defects": len(shown),
        "defect_counts": counts,
        "defects": [asdict(d) for d in shown],
    }
