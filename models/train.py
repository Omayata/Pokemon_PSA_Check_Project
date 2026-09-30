import mlflow
import mlflow.keras
from src.preprocessing import preprocess_image
# สมมติว่ามีการเตรียมชุดข้อมูลและสร้างโมเดล (เช่น CNN) ไว้แล้ว

def train_model():
    # ตั้งค่า Experiment
    mlflow.set_experiment("Pokemon_PSA_Grading")
    
    with mlflow.start_run() as run:
        # 1. บันทึก Hyperparameters
        learning_rate = 0.001
        epochs = 10
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("epochs", epochs)
        
        # [ส่วนจำลองการเทรนโมเดลด้วย TensorFlow/Keras]
        # model.fit(x_train, y_train, epochs=epochs)
        
        # 2. จำลองผลลัพธ์การวัดผล (Metrics)
        # ระบุ Optimizing metric (accuracy) และ Gating metric (latency)[cite: 3]
        accuracy = 0.85 
        inference_time_ms = 45 
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("inference_latency_ms", inference_time_ms)
        
        # 3. ตรวจสอบด่าน (Gating Metric) ก่อนบันทึกลง Registry[cite: 4, 6]
        if accuracy >= 0.80 and inference_time_ms < 50:
            print("Model passed the gating metrics. Registering model...")
            # สมมติบันทึกโมเดลจำลอง
            # mlflow.keras.log_model(model, "psa_model", registered_model_name="Pokemon_PSA_Model")
        else:
            print("Model did not pass gating metrics. Rejected.")

if __name__ == "__main__":
    train_model()