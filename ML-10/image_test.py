import cv2
from ultralytics import YOLO

def test_single_image(image_path, model_path="runs/detect/epoch_50/weights/best.pt", conf_threshold=0.25):
    # โหลดโมเดลที่เทรนแล้ว
    model = YOLO(model_path)

    # ทำการตรวจจับวัตถุ
    results = model.predict(source=image_path, conf=conf_threshold)

    # แสดงผลและบันทึกภาพ
    for result in results:
        # พล็อต Bounding Box, Label, Confidence Score ลงบนรูป
        annotated_frame = result.plot()

        # แสดงผลภาพบนหน้าต่าง
        cv2.imshow("YOLO Detection Test", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # บันทึกภาพผลลัพธ์
        output_path = "output_result.jpg"
        cv2.imwrite(output_path, annotated_frame)
        print(f"บันทึกภาพผลลัพธ์ไว้ที่ {output_path}")

if __name__ == "__main__":
    # ใส่ path ภาพที่ต้องการทดสอบ
    test_image = "dataset/test/images/sample.jpg"
    test_single_image(test_image)