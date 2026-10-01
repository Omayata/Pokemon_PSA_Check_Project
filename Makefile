# ทางลัดคำสั่ง (Linux/macOS/Git Bash). บน Windows PowerShell ใช้คำสั่งใน README ได้ตรง ๆ
.PHONY: up down pipeline pipeline-quick monitor test lint load-test drift-data drift-concept bad-data rollback models

up:              ## เปิด API + MLflow + Prefect + Prometheus + Grafana
	docker compose up -d --build

down:
	docker compose down

pipeline:        ## ข้อมูลดิบ -> validate -> train x3 -> gate -> register -> deploy (คำสั่งเดียว)
	docker compose run --rm trainer

pipeline-quick:  ## เหมือน pipeline แต่เทรนแค่ quick_epochs (ทดสอบระบบ)
	docker compose run --rm trainer python -m pipelines.flow train --quick

monitor:         ## ตรวจ drift -> retrain อัตโนมัติถ้าเข้าเงื่อนไข
	docker compose run --rm trainer python -m pipelines.flow monitor

test:
	pytest -v

lint:
	ruff check .

load-test:
	python scripts/load_test.py --n 200 --concurrency 8

drift-data:
	python scripts/simulate_drift.py data --kind dark --n 100

drift-concept:
	python scripts/simulate_drift.py concept --n 120

bad-data:
	python scripts/make_bad_data.py dataset && python scripts/make_bad_data.py api

rollback:
	docker compose run --rm trainer python -m src.registry.promote rollback

models:
	docker compose run --rm trainer python -m src.registry.promote list
