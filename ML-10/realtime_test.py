import cv2
from ultralytics import YOLO

def run_realtime_detection(model_path="runs/detect/epoch_50/weights/best.pt", source=0, conf_threshold=0.25):
    # โหลดโมเดล
    model = YOLO(model_path)

    # เปิดกล้อง Webcam (source=0) หรือไฟล์วิดีโอ (ใส่ path ไฟล์วิดีโอ)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("ไม่สามารถเปิดกล้องหรือไฟล์วิดีโอได้")
        return

    print("กด 'q' เพื่อออกจากโปรแกรม")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("ไม่พบเฟรม หรือวิดีโอสิ้นสุดแล้ว")
            break

        # ตรวจจับวัตถุในแต่ละเฟรม
        results = model(frame, conf=conf_threshold)

        # วาด Bounding Box, Class Labels และ Confidence
        annotated_frame = results[0].plot()

        # แสดงผล Real-time
        cv2.imshow("YOLO Real-time Detection", annotated_frame)

        # กด 'q' เพื่อเลิกทำงาน
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # source=0 สำหรับ Webcam
    run_realtime_detection(source=0)