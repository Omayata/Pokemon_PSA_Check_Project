from fastapi import FastAPI, UploadFile, File, HTTPException
import time
import shutil
import os
from src.preprocessing import preprocess_image
from src.data_validation import validate_image_schema

app = FastAPI(title="Pokemon PSA Prediction API")

# สมมติฟังก์ชันโหลดโมเดลจาก MLflow Model Registry
# model = load_model_from_registry("Pokemon_PSA_Model", version="Production")

@app.get("/health")
def health_check():
    """Endpoint สำหรับตรวจสุขภาพระบบ[cite: 4, 6]"""
    return {"status": "healthy", "model_version": "v1.0"}

@app.post("/predict")
async def predict_psa(file: UploadFile = File(...)):
    start_time = time.time()
    
    # บันทึกไฟล์ชั่วคราว
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # 1. ตรวจสอบ Schema ของข้อมูล[cite: 3]
        validate_image_schema(temp_file_path)
        
        # 2. แปลงข้อมูลด้วยโค้ดชุดเดียวกับตอนเทรน[cite: 4, 5]
        processed_img = preprocess_image(temp_file_path)
        
        # 3. ทำนายผล (จำลอง)
        # prediction = model.predict(processed_img)
        predicted_grade = 9 # จำลองผลลัพธ์ PSA 9
        
    except ValueError as e:
        os.remove(temp_file_path)
        raise HTTPException(status_code=400, detail=str(e))
        
    os.remove(temp_file_path)
    
    # วัด Latency (ความหน่วง) ของระบบ[cite: 4, 6]
    latency_ms = (time.time() - start_time) * 1000
    
    return {
        "predicted_psa_grade": predicted_grade,
        "latency_ms": round(latency_ms, 2)
    }