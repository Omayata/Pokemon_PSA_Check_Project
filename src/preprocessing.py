import numpy as np
from PIL import Image

def preprocess_image(image_path, target_size=(224, 224)):
    """แปลงข้อมูล (Resize, Normalize) ใช้ร่วมกันทั้ง Train และ Serving[cite: 4, 5]"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize ค่าพิกเซล 0-1
    # ขยายมิติเป็น (1, 224, 224, 3) สำหรับป้อนเข้าโมเดล
    img_array = np.expand_dims(img_array, axis=0)
    return img_array