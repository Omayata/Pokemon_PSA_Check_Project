# AI Project Canvas — Pokemon Card Condition Check

> ⚠️ Draft: ตัวเลขผลลัพธ์ให้อัปเดตจาก `reports/experiment_comparison.json` และ `reports/gate_report.json` ล่าสุดก่อนส่ง

| Component | Details |
|---|---|
| **Problem** | ผู้ขาย/นักสะสมการ์ดโปเกม่อนต้องตรวจสภาพการ์ดก่อนส่งเกรด PSA (ค่าส่งประมาณ $25+ ต่อใบ รอหลายสัปดาห์) หรือก่อนตั้งราคาขาย การตรวจด้วยตาช้าและพลาดตำหนิเล็ก ๆ ได้ง่าย |
| **Users / stakeholders** | ผู้ขายการ์ดมือสอง, ร้านการ์ด (ตรวจทีละหลายใบ), ผู้ซื้อออนไลน์ (เช็คก่อนซื้อ) |
| **Value proposition** | อัปโหลดรูปการ์ด -> รู้ทันทีว่า **"สภาพดี" หรือ "มีตำหนิ"** พร้อมกรอบชี้ตำแหน่งตำหนิ (scratch / edge wear / corner wear) ช่วยคัดการ์ดที่คุ้มส่งเกรดและลดการตรวจด้วยตา |
| **Why ML, not rules** | ตำหนิมีรูปร่าง ขนาด ตำแหน่งไม่แน่นอนและเล็กมาก (6-25 px ในภาพ 416 px) การ์ด holo/full-art มีแสงสะท้อนและลายที่ดูคล้ายรอยขีด กฎแบบ threshold สี/ขอบแยกไม่ได้ ต้องใช้ object detection ที่เรียนจากตัวอย่าง |
| **ML task** | Cascade: (1) YOLOv8 object detection 4 classes (Card, Corner Wear, Edge Wear, Scratch) -> (2) จำแนกระดับภาพ: `defect_probability` = confidence สูงสุดของตำหนิ, ถ้า >= threshold -> **defective** ไม่งั้น **good** (threshold เลือกบน valid set แล้วเก็บไปกับโมเดล) |
| **Data** | Roboflow `group-6-major-project/card-grader` v5 (CC BY 4.0) 1,510 ภาพ (train 1,311 ซึ่ง Roboflow augment ไว้ 3 เท่าจากภาพต้นฉบับ ~437, valid 122, test 71) resize fit-within 416 px มี bbox label / label ระดับภาพ: มีกรอบตำหนิ >= 1 = defective (~70%), ไม่มี = good (~30%) |
| **Data risks** | ภาพต้นฉบับ ~630 < 5,000 -> overfit ง่าย, test set เล็ก (71 ภาพ, good 25) ตัวเลขแกว่ง ±1-4% / รับมือ: transfer learning จาก COCO, augmentation, เทียบ baseline ไม่มี aug, เลือก threshold บน valid เท่านั้น ใช้ test เฉพาะ gate, ตรวจ leakage ข้าม split, imgsz 832 ให้ตำหนิเล็กมองเห็นได้ |
| **Optimizing metric** | F1 ของคลาส defective ระดับภาพ (เลือกโมเดลจาก valid set) |
| **Gating metrics** | บน test set: recall(defective) >= 0.90, precision(defective) >= 0.90, ROC-AUC >= 0.90, p95 latency โมเดล <= 300 ms (CPU), ขนาด <= 50 MB, F1 แย่กว่า champion ไม่เกิน 0.02 |
| **Supporting metrics** | mAP50 ของ detector ราย class (คุณภาพของกรอบที่แสดงให้ผู้ใช้), accuracy, specificity |
| **Business metrics** | (1) อัตราที่ผลทายตรงกับผลตรวจจริงโดยคน (จาก `/feedback`) (2) จำนวนการ์ดมีตำหนิที่ถูกคัดออกก่อนส่งเกรด = ค่าเกรดที่ประหยัดได้ (3) เวลาตรวจต่อใบที่ลดลงเทียบกับตรวจด้วยตา |
| **Link model -> business** | recall สูง -> การ์ดมีตำหนิหลุดไปส่งเกรดน้อย = ประหยัดค่าเกรด; precision/specificity สูง -> ไม่ตีการ์ดดีว่าเสีย = ผู้ขายไม่ตั้งราคาต่ำเกินจริง |
| **Cost of errors** | False negative (การ์ดมีตำหนิแต่บอก good): ผู้ใช้เสียค่าเกรดฟรี/ผู้ซื้อได้ของไม่ตรงปก **(แย่กว่า -> recall เป็นเกณฑ์หลัก)** / False positive (การ์ดดีแต่บอก defective): ผู้ขายตั้งราคาต่ำหรือไม่ส่งเกรดการ์ดดี |
| **Serving** | Real-time REST API `/predict` (ผู้ใช้รอผล) + `/predict/batch` สูงสุด 16 ภาพสำหรับร้านค้า (หน้า+หลัง หรือหลายใบ) |
| **SLO** | p50 <= 200 ms, p95 <= 500 ms (end-to-end ที่ API), throughput >= 5 req/s บน CPU 1 เครื่อง, error rate <= 1%, availability 99% |
| **Data drift (expected)** | กล้อง/แสงเปลี่ยน (มืด เบลอ แฟลช), ถ่ายผ่านซองพลาสติก, ถ่ายเอียง/มีพื้นหลัง, ชุดการ์ดใหม่ที่ลายต่างไป |
| **Concept drift (expected)** | มาตรฐาน "สภาพดี" เข้มขึ้น (ตำหนิเล็กที่เคยยอมรับได้กลายเป็นตำหนิ), ตำหนิแบบใหม่ที่โมเดลไม่รู้จัก (เช่นรอยพับ) -> ภาพหน้าตาเดิมแต่ label จริงเปลี่ยน |
| **Monitoring** | ระบบ: latency, error rate, API up, model loaded / คุณภาพ: สัดส่วน good/defective ที่ทาย, distribution ของ defect_probability, อัตรา input ที่ถูกปฏิเสธ, PSI ของ input, agreement กับ feedback |
| **Retraining policy** | retrain เมื่อ (1) data drift: PSI > 0.2 และ KS p < 0.01 ใน 200 request ล่าสุด หรือ (2) concept drift: agreement ลด > 15% หรือ (3) โมเดลอายุเกิน 30 วัน / ตัวใหม่ต้องผ่าน gate ก่อนแทน champion |
| **Rollback** | gate ไม่ผ่าน -> ไม่ถูก promote เลย / พบปัญหาหลัง deploy -> `python -m src.registry.promote rollback` ย้าย alias champion กลับไป version ก่อนหน้าแล้ว reload API |
| **Ethics / limits** | เป็นการคัดกรองเบื้องต้น ไม่ใช่เกรดทางการ; ตรวจได้เฉพาะ 3 ชนิดตำหนิที่มีใน dataset (**ตรวจรอยพับ/รอยบุบไม่ได้**); ภาพถ่ายเองที่ต่างจากภาพเทรนมาก (พื้นหลัง แสง) อาจแม่นน้อยลง |
