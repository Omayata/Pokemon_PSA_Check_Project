import os
from PIL import Image

def validate_image_schema(image_path):
    """ตรวจสอบว่ารูปภาพใช้งานได้จริง ตรงตาม Schema ที่กำหนดหรือไม่"""
    try:
        with Image.open(image_path) as img:
            img.verify() # ตรวจสอบไฟล์เสีย
            width, height = img.size
            
            # สมมติ Schema: รูปภาพการ์ดต้องมีขนาดขั้นต่ำ 200x200
            if width < 200 or height < 200:
                raise ValueError(f"Image too small: {width}x{height}. Expected at least 200x200.")
                
            # สมมติโหมดภาพต้องเป็น RGB
            if img.mode != 'RGB':
                raise ValueError(f"Invalid image mode: {img.mode}. Expected RGB.")
                
    except Exception as e:
        # ระบบหยุดทำงานและแจ้งเตือนเมื่อพบข้อมูลเสีย
        print(f"🚨 [ALERT] Data Validation Failed for {image_path}: {e}")
        raise
    return True