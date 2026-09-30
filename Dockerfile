# image เดียวใช้ทั้ง API และ trainer (โค้ด transform/grader ชุดเดียวกัน -> ไม่มี training-serving skew)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# opencv (ใช้โดย ultralytics) ต้องการ libGL / glib
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ติดตั้ง torch แบบ CPU ก่อน (เล็กกว่า CUDA หลาย GB) แล้วค่อยติดตั้งที่เหลือตาม lock file
COPY requirements.txt constraints.txt ./
RUN pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt -c constraints.txt

COPY configs ./configs
COPY src ./src
COPY app ./app
COPY pipelines ./pipelines
COPY scripts ./scripts

ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=${GIT_COMMIT}

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
