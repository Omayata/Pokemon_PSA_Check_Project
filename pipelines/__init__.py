"""โหลด .env ก่อน import prefect เพื่อให้ PREFECT_API_URL ชี้ไป server ใน Docker (ไม่ใช่ server ชั่วคราว)"""

from dotenv import load_dotenv

load_dotenv()
