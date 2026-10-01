# System Architecture

```mermaid
flowchart LR
    subgraph Data["Data"]
        RF[(Roboflow x2<br/>card-grader v5<br/>card-defect v6)] -->|ingest + hash| RAW[data/raw/source]
        RAW -->|validate schema| V1{OK?}
        V1 -->|no| STOP1[[stop + alert]]
        V1 -->|yes| PREP[prepare<br/>src/features/transform.py]
        PREP --> PROC[data/processed] -->|validate| V2{OK?}
        V2 -->|no| STOP1
    end

    subgraph Train["Training (Prefect DAG)"]
        V2 -->|yes| T1[baseline yolov8n] & T2[yolov8n + aug] & T3[yolov8s + aug]
        T1 & T2 & T3 --> SEL[select best<br/>by val F1, image level]
        SEL --> GATE{gates on test set<br/>recall, precision, AUC,<br/>latency, size}
    end

    subgraph Registry["MLflow"]
        T1 & T2 & T3 -.->|code, data version, params,<br/>metrics, artifacts, env| TRACK[(Tracking)]
        GATE -->|pass| CH[registry<br/>alias: champion]
        GATE -->|fail| REJ[registry<br/>status: rejected]
        CH -.->|rollback| PREV[alias: previous]
    end

    subgraph Serving["Serving (Docker)"]
        USER((User / seller)) -->|POST /predict| API[FastAPI]
        API --> VAL{validate_upload}
        VAL -->|bad| E422[422 + reason]
        VAL -->|ok| TR[transform<br/>same code as training]
        TR --> S1[Stage 1: YOLO<br/>detect defects] --> S2[Stage 2: classifier<br/>max defect conf >= threshold<br/>-> good / defective]
        S2 --> USER
        CH -->|load champion| API
        API -->|predictions.jsonl<br/>feedback.jsonl| LOGS[(logs)]
    end

    subgraph Monitor["Monitoring"]
        API -->|/metrics| PROM[Prometheus<br/>+ alert rules] --> GRAF[Grafana]
        LOGS --> DRIFT[drift check<br/>PSI + KS / feedback agreement]
        DRIFT -->|drift found| RETRAIN[retrain policy]
        RETRAIN -->|drift_augment| PREP
    end

    subgraph CI["GitHub Actions"]
        PR[Pull Request] --> L[ruff] & PT[pytest + bad data] & DV[real data validation] & MG[model gate] --> DK[docker build + /health]
    end
```

## Components and tools

| Role | Tool | Why |
|---|---|---|
| Version control | Git + GitHub (branch + PR) | Required; commit history is used for individual grading |
| Containerization | Docker + docker compose | One image for API and trainer, so transform code is shared |
| Data validation | Custom (`src/data/validate.py`) | Images + YOLO labels don't fit table-oriented tools like GE/Pandera |
| Experiment tracking | MLflow Tracking | Records all 6 items and compares runs in the UI |
| Model registry | MLflow Registry (aliases + tags) | `champion`/`previous` aliases make rollback one command |
| Orchestration | Prefect 3 | Runs on Windows, lighter than Airflow, has a UI that shows the DAG |
| Serving | FastAPI + Uvicorn | Simple, automatic docs at `/docs`, easy to add endpoints |
| Monitoring | Prometheus + Grafana + custom drift checks | System metrics + PSI/KS for data drift + feedback for concept drift |
| CI/CD | GitHub Actions | 5 jobs: lint, tests, data validation, model gate, docker |
