from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model("violation_20250624-171156.jpg")

for result in results:
    boxes = result.boxes
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"Class: {model.names[cls_id]}, Confidence: {conf:.2f}")
