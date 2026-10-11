# โครงสร้างโค้ด: อธิบายทีละไฟล์

เอกสารนี้ตอบ 3 คำถาม: (1) ภาพหนึ่งใบเดินผ่านโค้ดไฟล์ไหนบ้างตอนผู้ใช้อัปโหลด (2) pipeline เทรนเรียกอะไรตามลำดับ
(3) แต่ละไฟล์ทำอะไรและมีฟังก์ชันหลักอะไร ภาพรวมระดับระบบดูที่ [architecture.md](architecture.md)

## 1. ภาพหนึ่งใบตอน serving (`POST /predict`)

```
app/main.py  predict()
  └─ _predict_bytes()
       ├─ src/data/validate.py      validate_upload()   ภาพเสีย -> 422 + เหตุผล
       ├─ src/features/transform.py load_image()        EXIF, RGB, ย่อด้านยาวสุด 640 (โค้ดเดียวกับตอนเตรียมข้อมูลเทรน)
       ├─ src/models/predictor.py   Predictor.detect()  Stage 1: YOLOv8 หากรอบตำหนิ (conf >= 0.05)
       ├─ src/models/classifier.py  classify()          Stage 2: max confidence >= threshold -> defective
       ├─ src/features/transform.py image_stats()       brightness / contrast / sharpness / aspect ratio
       └─ logs/predictions.jsonl                        ใช้ตรวจ drift ภายหลัง + Prometheus counters
```

## 2. Pipeline เทรน (`python -m pipelines.flow train`)

```
pipelines/flow.py  training_pipeline()
  1. src/data/ingest.py      download()          ดึง 2 dataset จาก Roboflow (pin version) + hash ทุกไฟล์
  2. src/data/validate.py    validate_dataset()  ตรวจ raw ของแต่ละ source  (ไม่ผ่าน -> หยุด + alert)
  3. src/data/prepare.py     prepare_dataset()   map class, แบ่ง split ตามภาพต้นฉบับ, ใช้ load_image() ชุดเดียวกับ serving
  4. src/data/validate.py    validate_dataset()  ตรวจ data/processed อีกรอบ
  5. src/models/train.py     train_experiment()  เทรนทุก experiment ใน params.yaml (baseline / aug) + log MLflow
       └─ src/models/evaluate.py                 tune threshold บน valid, วัดระดับภาพบน valid+test, mAP, latency
  6. src/registry/gates.py   select_best()       เลือกจาก valid F1 (test ไม่ถูกใช้เลือก)
  7. src/registry/promote.py gate_and_register() check_gates() บน test -> register ทุกตัว, ผ่าน = alias champion
  8. src/registry/promote.py reload_api()        สั่ง API โหลด champion ใหม่ (POST /admin/reload)
```

pipeline อื่นในไฟล์เดียวกัน:
- `regate`: ประเมิน weights เดิมด้วยตัวชี้วัด/เกณฑ์ใหม่ ไม่ต้องเทรนซ้ำ (`reevaluate_experiment()`)
- `monitor`: `check_drift()` -> ถ้าเจอ drift ตามนโยบาย -> `training_pipeline(drift_augment=[...])`

## 3. ทีละไฟล์

### ค่าตั้งและของใช้ร่วม

| ไฟล์ | หน้าที่ | ฟังก์ชันหลัก |
|---|---|---|
| `configs/params.yaml` | ค่าตั้งเดียวของทั้งระบบ: dataset + class_map, schema, transform, experiments, gates, SLO, threshold, drift | - |
| `src/config.py` | โหลด params.yaml (cache), เลือก device, normalize ชื่อ class | `load_params`, `resolve_device`, `normalize_name` |
| `src/alerts.py` | แจ้งเตือน: log + `logs/alerts.jsonl` + webhook (Discord/Slack) ถ้าตั้งไว้ | `send_alert` |

### Data (`src/data/`, `src/features/`)

| ไฟล์ | หน้าที่ | ฟังก์ชันหลัก |
|---|---|---|
| `ingest.py` | ดาวน์โหลดทุก source จาก Roboflow ตาม version ที่ pin แล้วคำนวณ hash ของทุกไฟล์ = data version | `download`, `compute_data_hash` |
| `validate.py` | ตรวจ dataset ตอนเทรน (class, จำนวนภาพ/instance, label format, bbox, ภาพเสีย, leakage) และตรวจภาพที่ผู้ใช้อัปโหลด (format, ขนาด, สัดส่วน, ภาพสีเดียว) | `validate_dataset`, `validate_upload`, `raw_schema` |
| `prepare.py` | รวม 2 source เป็น data/processed: แปลงชื่อ class ผ่าน `class_map` (null = ตัดทิ้ง), resplit dataset ที่สองตามภาพต้นฉบับ (`<ชื่อ>.rf.<hash>` -> hash + seed), เพิ่มภาพ drift ตอน retrain | `class_id_map`, `remap_label`, `group_key`, `assign_split`, `prepare_dataset` |
| `transform.py` | การแปลงภาพชุดเดียวที่ใช้ทั้ง train และ serve (กัน training-serving skew) + คุณลักษณะภาพสำหรับตรวจ drift | `load_image`, `image_stats` |

### Model (`src/models/`)

| ไฟล์ | หน้าที่ | ฟังก์ชันหลัก |
|---|---|---|
| `classifier.py` | Stage 2: `defect_probability` = confidence สูงสุดของกรอบตำหนิ, `>= threshold` -> defective, คืนกรอบที่ใช้อธิบายผล | `Detection`, `classify` |
| `predictor.py` | Stage 1 + 2 รวมกัน ใช้ตัวเดียวทั้งประเมินผลและ serving; `CardConditionPyfunc` ห่อเป็น MLflow model ที่เก็บ weights + params (รวม threshold) + reference stats ไว้ด้วยกัน | `Predictor.detect`, `Predictor.predict`, `CardConditionPyfunc` |
| `evaluate.py` | ตัวชี้วัดระดับภาพ (recall/precision/F1/specificity/ROC-AUC), แยกตาม source, mAP ราย class, latency, หา threshold ที่ F1 สูงสุดบน valid, reference stats สำหรับ drift | `binary_metrics`, `roc_auc`, `per_source_metrics`, `tune_threshold`, `evaluate_detector`, `measure_latency`, `build_reference_stats` |
| `train.py` | เทรน 1 experiment ด้วย ultralytics แล้ว log MLflow ครบ 6 อย่าง (code, data version, params, metrics, artifacts, environment) | `train_experiment`, `reevaluate_experiment`, `_log_context`, `_evaluate_and_log` |

### Registry (`src/registry/`)

| ไฟล์ | หน้าที่ | ฟังก์ชันหลัก |
|---|---|---|
| `gates.py` | เกณฑ์ที่ต้องผ่านทุกข้อ (ไม่มี dependency หนัก ใช้ใน CI ได้) + เลือกโมเดลจาก valid (เสมอกันภายใน 0.01 -> เลือกตัวเล็กกว่า แล้วดู val AUC) | `check_gates`, `select_best` |
| `promote.py` | register ทุก candidate พร้อม tag `status` approved/rejected; ผ่าน gate -> ย้าย alias `champion` (ตัวเดิมเป็น `previous`); rollback คำสั่งเดียว; เทียบ F1 กับ champion เฉพาะเมื่อ test set ชุดเดียวกัน | `gate_and_register`, `rollback`, `reload_api`, `list_versions` |

### Monitoring (`src/monitoring/`)

| ไฟล์ | หน้าที่ | ฟังก์ชันหลัก |
|---|---|---|
| `drift.py` | Data drift: PSI + KS ของ image stats และ prediction ใน 200 request ล่าสุดเทียบ reference / Concept drift: agreement กับ `/feedback` ช่วงแรกเทียบช่วงล่าสุด | `psi`, `detect_data_drift`, `detect_concept_drift`, `check_drift` |
| `drift_sim.py` | การแปลงภาพจำลองสภาพแวดล้อมเปลี่ยน (มืด, สว่าง, เบลอ, contrast ต่ำ, noise, หมุน) ใช้ทั้งสาธิตและเพิ่มข้อมูลตอน retrain | `dark`, `bright`, `blur`, `low_contrast`, `noise`, `rotate` |

### Orchestration และ Serving

| ไฟล์ | หน้าที่ | ฟังก์ชันหลัก |
|---|---|---|
| `pipelines/flow.py` | Prefect DAG 3 แบบ: train / regate / monitor ดูข้อ 2 | `training_pipeline`, `regate_pipeline`, `drift_monitor` |
| `app/main.py` | FastAPI: โหลด champion จาก MLflow (ไม่ได้ก็ใช้ไฟล์ weights), ทับ threshold ด้วย `serving.threshold`, เลือก version อื่นที่ approved ได้ (`?version=N`, cache แบบ LRU), log ทุก request, Prometheus metrics | endpoints: `/`, `/models`, `/predict`, `/predict/batch` (สูงสุด 16), `/feedback`, `/health`, `/ready`, `/metrics`, `/monitoring/drift`, `/admin/reload` |
| `app/web/index.html` | หน้าเว็บอัปโหลดรูป แสดงผล good/defective + กรอบตำหนิ เลือก model version ได้ | - |

### Scripts (สาธิตและวัดผล)

| ไฟล์ | ใช้ทำอะไร |
|---|---|
| `check_model_gate.py` | CI อ่าน `reports/candidate_metrics.json` แล้วเช็ค gate, ไม่ผ่าน = exit 1 |
| `compare_versions.py` | ประเมินหลาย version ใหม่บน test ชุดเดียวกัน + การ์ดดีจากภายนอก (เทียบอย่างยุติธรรม) |
| `eval_clean_cards.py` | วัด false positive บนการ์ดดี 110 ใบจาก dataset อื่น (ใช้ทดสอบอย่างเดียว) |
| `load_test.py` | ยิง API จริงด้วย concurrency ตาม SLO วัด p50/p95/p99 + throughput |
| `make_bad_data.py` | สาธิตข้อมูลเสีย: dataset เสีย -> pipeline หยุด, ภาพเสีย -> 422 |
| `simulate_drift.py` | ส่งภาพที่ drift (data) หรือ feedback ที่เปลี่ยน (concept) เข้า API จริง |

### Tests (`tests/`)

| ไฟล์ | ทดสอบอะไร |
|---|---|
| `test_data_validation.py` | dataset/ภาพเสียทุกแบบต้องถูกปฏิเสธพร้อมเหตุผล (label ผิด, bbox เกิน, ภาพเสีย, leakage, class ไม่ตรง ฯลฯ) |
| `test_prepare.py` | map class, ตัด class ที่เป็น null, สำเนา augment อยู่ split เดียวกัน |
| `test_transform_and_classifier.py` | อ่านจากไฟล์กับจาก bytes ได้ภาพเดียวกัน, ตรรกะ threshold, metrics, ROC-AUC, PSI |
| `test_model_gate.py` | ทุก gate ทำให้ตกได้, regression vs champion, การเลือกตัวที่เสมอกัน, exit code ของสคริปต์ CI |
| `test_api.py` | API ด้วย predictor ปลอม: predict, batch, 422, 503, feedback, metrics, เลือก version |

### Infra

| ไฟล์ | หน้าที่ |
|---|---|
| `Dockerfile`, `docker-compose.yml` | image เดียวทั้ง API และ trainer + MLflow, Prefect, Prometheus, Grafana |
| `.github/workflows/ci.yml` | 5 job: ruff, pytest, validate ข้อมูลจริง, model gate, docker build + `/health` |
| `monitoring/` | Prometheus scrape config + alert rules, Grafana dashboard |
| `requirements.txt`, `constraints.txt` | dependency ที่ pin ทั้งหมด |

## 4. ถ้าโดนถามว่า "ตรงนี้อยู่ไฟล์ไหน"

| คำถาม | เปิดไฟล์ |
|---|---|
| ตัดสิน good/defective ยังไง | `src/models/classifier.py` |
| เลือก threshold ยังไง | `tune_threshold` ใน `src/models/evaluate.py` + `serving.threshold` ใน `configs/params.yaml` |
| กัน data leakage ยังไง | `group_key` / `assign_split` ใน `src/data/prepare.py` + เช็ค leakage ใน `src/data/validate.py` |
| กัน training-serving skew ยังไง | `src/features/transform.py` (ถูกเรียกจาก `prepare.py` และ `app/main.py`) |
| เกณฑ์ขึ้น production | `gates:` ใน `configs/params.yaml` + `src/registry/gates.py` |
| rollback | `rollback` ใน `src/registry/promote.py` |
| คำนวณ drift | `src/monitoring/drift.py` |
