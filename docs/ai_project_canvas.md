# AI Project Canvas — Pokemon Card Pre-Grader

> ⚠️ Draft: ตัวเลขในวงเล็บ [ ] ต้องใส่ค่าจริงหลังเทรน/วัดผล

| Component | Details |
|---|---|
| **Problem** | ผู้ขาย/นักสะสมการ์ดโปเกม่อนไม่รู้ว่าการ์ดควรส่งเกรด PSA หรือไม่ ค่าส่งเกรดประมาณ $25+ ต่อใบ และรอหลายสัปดาห์ ถ้าได้เกรดต่ำจะขาดทุน |
| **Users / stakeholders** | ผู้ขายการ์ดมือสอง, ร้านการ์ด, ผู้ซื้อออนไลน์ (ใช้ดูความเสี่ยงก่อนซื้อ) |
| **Value proposition** | อัปโหลดรูปการ์ด -> เห็นตำแหน่งตำหนิ (scratch / edge wear / corner wear) และช่วงเกรดโดยประมาณในไม่กี่วินาที ช่วยตัดสินใจก่อนจ่ายค่าเกรด |
| **Why ML, not rules** | ตำหนิมีรูปร่าง ขนาด ตำแหน่งไม่แน่นอน การ์ด holo/full-art มีแสงสะท้อนที่ดูคล้ายรอยขีด กฎแบบ threshold สี/ขอบแยกไม่ได้ ต้องใช้ object detection ที่เรียนจากตัวอย่าง |
| **ML task** | Cascade: (1) Object detection 4 classes (Card, Corner Wear, Edge Wear, Scratch) ด้วย YOLOv8 -> (2) grader แปลงตำหนิเป็นคะแนน 1-10 และช่วง PSA |
| **Data** | Roboflow `group-6-major-project/card-grader` v5 (CC BY 4.0) ~632 ภาพ มี bbox label; ไม่มี label เกรด PSA |
| **Data risks** | น้อยกว่า 5,000 ภาพ -> overfit ได้ง่าย / รับมือ: transfer learning จาก COCO, augmentation, เทียบ baseline ไม่มี aug, ใช้ test set แยกเฉพาะ gate, ตรวจ leakage ข้าม split |
| **Optimizing metric** | mAP@50 บน test set (ยิ่งสูงยิ่งดี) |
| **Gating metrics** | mAP50 >= 0.50, recall(scratch) >= 0.40, p95 latency โมเดล <= 300 ms (CPU), ขนาด <= 50 MB, แย่กว่า champion ไม่เกิน 0.01 |
| **Business metrics** | (1) อัตราที่ช่วงเกรดที่ทายตรงกับผล PSA จริง (จาก /feedback) (2) เงินค่าเกรดที่ประหยัดได้จากการ์ดที่ระบบเตือนว่าไม่ถึง PSA 9 (3) จำนวนการใช้งานซ้ำต่อผู้ใช้ |
| **Link model -> business** | recall ของ scratch สูง -> พลาดตำหนิน้อย -> ผู้ใช้ไม่ส่งการ์ดที่มีตำหนิไปเสียค่าเกรด; precision สูง -> ไม่กดราคาการ์ดดีเกินจริง |
| **Cost of errors** | False negative (ไม่เห็นตำหนิ): ผู้ใช้เสียค่าเกรดฟรีและเสียเวลา (แย่กว่า) / False positive (เห็นตำหนิที่ไม่มี): ผู้ใช้ขายถูกเกินไปหรือไม่ส่งเกรดการ์ดดี |
| **Serving** | Real-time REST API `/predict` (ผู้ใช้รอผล) + `/predict/batch` สูงสุด 16 ภาพสำหรับร้านค้า |
| **SLO** | p50 <= 200 ms, p95 <= 500 ms (end-to-end ที่ API), throughput >= 5 req/s บน CPU 1 เครื่อง, error rate <= 1%, availability 99% |
| **Data drift (expected)** | กล้อง/แสงเปลี่ยน (มืด เบลอ แฟลช), ถ่ายผ่านซองพลาสติก, ถ่ายเอียง/แนวนอน, ชุดการ์ดใหม่ที่ลายต่างไป |
| **Concept drift (expected)** | PSA เปลี่ยนความเข้มงวดของการให้เกรด (ตำหนิเท่าเดิมแต่เกรดต่ำลง), ตลาดให้คุณค่ากับตำหนิบางแบบต่างไป |
| **Monitoring** | ระบบ: latency, error rate, API up, model loaded / คุณภาพ: สัดส่วนช่วงเกรดที่ทาย, จำนวนตำหนิ, อัตรา input ที่ถูกปฏิเสธ, PSI ของ input, agreement กับ feedback |
| **Retraining policy** | retrain เมื่อ (1) data drift: PSI > 0.2 และ KS p < 0.01 ใน 200 request ล่าสุด หรือ (2) concept drift: agreement ลด > 15% หรือ (3) โมเดลอายุเกิน 30 วัน / ตัวใหม่ต้องผ่าน gate ก่อนแทน champion |
| **Rollback** | ถ้าตัวใหม่แย่กว่าเดิม (gate ไม่ผ่าน -> ไม่ถูก promote เลย) หรือพบปัญหาหลัง deploy -> `python -m src.registry.promote rollback` ย้าย alias champion กลับไป version ก่อนหน้าแล้ว reload API |
| **Ethics / limits** | ไม่ใช่เกรดทางการ แสดงเป็น "ช่วงโดยประมาณ" เสมอ; Stage 2 เป็น rubric จนกว่าจะเก็บผล PSA จริงได้พอ |
