# Pokemon Card Condition Check — MLOps Project

ระบบประเมินสภาพการ์ดโปเกม่อนจากรูปถ่าย: หาตำหนิ (scratch / edge wear / corner wear) ด้วย YOLOv8
แล้วจำแนกการ์ดเป็น **good** (สภาพดี) หรือ **defective** (มีตำหนิ) ครบวงจร MLOps ตั้งแต่ข้อมูลดิบจนถึงการให้บริการและเฝ้าระวัง

- สถาปัตยกรรม: [docs/architecture.md](docs/architecture.md)
- AI Project Canvas, metrics, SLO, นโยบาย retrain: [docs/ai_project_canvas.md](docs/ai_project_canvas.md)
- ค่าตั้งทั้งหมด (schema, gates, SLO, drift thresholds): [configs/params.yaml](configs/params.yaml)

## โครงสร้าง

```
├── .github/workflows/ci.yml     CI: lint, tests, data validation, model gate, docker build
├── configs/params.yaml          ค่าตั้งเดียวของทั้งระบบ (รวมรายการ dataset + class map)
├── src/
│   ├── data/ingest.py           ดาวน์โหลดทุก dataset จาก Roboflow (pin version) + data version hash
│   ├── data/validate.py         schema ของ dataset และของภาพขาเข้า
│   ├── data/prepare.py          รวม dataset + map class + แบ่ง split ใหม่ -> processed (transform เดียวกับ serving)
│   ├── features/transform.py    load_image() + image_stats() ใช้ร่วม train/serve
│   ├── models/train.py          เทรน + บันทึก MLflow ครบ 6 อย่าง
│   ├── models/evaluate.py       mAP ราย class, latency, reference stats
│   ├── models/classifier.py     Stage 2: ตำหนิ -> good / defective
│   ├── models/predictor.py      Stage 1 + 2 และ MLflow pyfunc wrapper
│   ├── registry/                gate, promote, rollback
│   ├── monitoring/drift.py      data drift (PSI+KS) / concept drift (feedback)
│   └── alerts.py                log + webhook
├── pipelines/flow.py            Prefect DAG: train / monitor
├── app/main.py                  FastAPI
├── scripts/                     load test, จำลอง drift, ข้อมูลเสีย, gate สำหรับ CI
├── tests/                       pytest
├── monitoring/                  Prometheus (alert rules) + Grafana dashboard
└── docs/
```

## เริ่มจากเครื่องเปล่า

ต้องมี: Docker Desktop, Git และ Roboflow API key (ฟรี: roboflow.com -> Settings -> API Keys)

```bash
git clone https://github.com/Omayata/Pokemon_PSA_Check_Project.git
cd Pokemon_PSA_Check_Project
cp .env.example .env              # แล้วใส่ ROBOFLOW_API_KEY

docker compose up -d --build      # API :8000, MLflow :5000, Prefect :4200, Prometheus :9090, Grafana :3000
docker compose run --rm trainer   # ⭐ คำสั่งเดียว: ingest -> validate -> prepare -> train x2 -> gate -> register -> deploy
```

ทดสอบ pipeline เร็ว ๆ (เทรน 3 epochs): `docker compose run --rm trainer python -m pipelines.flow train --quick`

### เทรนด้วย GPU (เร็วกว่ามาก)

Docker บน Windows มองไม่เห็น GPU (ถ้าไม่ตั้งค่าเพิ่ม) จึงเทรนบนเครื่องโดยตรง แต่ยังใช้ MLflow/Prefect ใน Docker
(`device: auto` ใน params.yaml จะใช้ GPU เองถ้ามี ส่วน API ให้บริการบน CPU เสมอ)

```bash
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python -m pip install -r requirements-dev.txt -c constraints.txt

docker compose up -d --build                         # เปิด MLflow, Prefect, API ฯลฯ
.venv\Scripts\python -m pipelines.flow train --quick # ทดสอบก่อน
.venv\Scripts\python -m pipelines.flow train         # เทรนเต็ม -> ผ่าน gate แล้ว API reload เอง
.venv\Scripts\python -m pipelines.flow regate        # เปลี่ยนเกณฑ์/ตัวชี้วัด: ประเมิน weights เดิมใหม่ ไม่ต้องเทรน
```

| URL | ใช้ทำอะไร |
|---|---|
| http://localhost:8000/docs | ทดลองเรียก API |
| http://localhost:5000 | MLflow: เทียบ experiment / registry |
| http://localhost:4200 | Prefect: ดู DAG และประวัติการรัน |
| http://localhost:3000 | Grafana dashboard "Pokemon Card Condition" |
| http://localhost:9090/alerts | Prometheus alert rules |

## API

```bash
curl -F "file=@card.jpg" http://localhost:8000/predict
curl -F "files=@a.jpg" -F "files=@b.jpg" http://localhost:8000/predict/batch
curl -X POST http://localhost:8000/feedback -H "Content-Type: application/json" \
     -d '{"request_id": "<id จาก /predict>", "true_label": "defective"}'
curl http://localhost:8000/health           # liveness + model version
curl http://localhost:8000/ready            # 503 ถ้ายังไม่มีโมเดล
curl http://localhost:8000/metrics          # Prometheus
curl http://localhost:8000/monitoring/drift # ตรวจ drift ตอนนี้
```

## การสาธิต (demo)

รันบนเครื่อง (นอก Docker) ต้องติดตั้ง dependency ก่อน:
```bash
python -m venv .venv && .venv\Scripts\activate          # Windows (macOS/Linux: source .venv/bin/activate)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt -c constraints.txt
```

| สาธิต | คำสั่ง |
|---|---|
| ข้อมูลเสีย -> pipeline หยุด + alert | `python scripts/make_bad_data.py dataset` |
| ภาพเสีย -> API ตอบ 422 พร้อมเหตุผล | `python scripts/make_bad_data.py api` |
| Latency p50/p95 + throughput เทียบ SLO | `python scripts/load_test.py --n 200` |
| Data drift (ภาพมืด) | `python scripts/simulate_drift.py data --kind dark --n 100` |
| Concept drift (มาตรฐานสภาพดีเข้มขึ้น) | `python scripts/simulate_drift.py concept --n 120` |
| ตรวจ drift -> retrain อัตโนมัติ | `docker compose run --rm trainer python -m pipelines.flow monitor` |
| ดู registry | `docker compose run --rm trainer python -m src.registry.promote list` |
| Rollback | `docker compose run --rm trainer python -m src.registry.promote rollback` |
| Tests / lint | `pytest -v` / `ruff check .` |

## Reproducibility

- ทุก dependency ถูก pin: `requirements.txt` (ตรง) + `constraints.txt` (lock ทั้งหมด)
- ข้อมูล pin ที่ Roboflow version 5 + hash ของไฟล์ทุกไฟล์ (`data_version.json`) บันทึกลง MLflow
- seed คงที่ (`seed: 42`) + `deterministic=True` ของ ultralytics
- split train/valid/test ตายตัวตาม Roboflow version และตรวจ leakage ข้าม split ทุกครั้ง

## Git workflow

`main` ถูกป้องกัน -> ทำงานบน branch `feature/<ชื่องาน>` -> เปิด Pull Request -> CI ต้องเขียวก่อน merge

## การใช้ AI ช่วยเขียนโค้ด

โครงสร้างโปรเจกต์และโค้ดเริ่มต้นสร้างโดยใช้ Claude (Anthropic) ช่วยเขียน สมาชิกทุกคนต้องอ่าน
ทำความเข้าใจ และอธิบายโค้ดทุกบรรทัดได้ — ระบุรายละเอียดส่วนที่ใช้ AI ในรายงาน
