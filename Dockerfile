# ใช้ Python 3.9 เป็นฐาน
FROM python:3.9-slim

WORKDIR /app

# คัดลอกไฟล์ requirements.txt และติดตั้ง (ระบุเวอร์ชันไลบรารี[cite: 3])
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกโค้ดทั้งหมดลงใน Container
COPY . .

# เปิดพอร์ต 8000
EXPOSE 8000

# รัน FastAPI Server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]