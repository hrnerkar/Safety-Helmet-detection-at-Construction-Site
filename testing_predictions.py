from ultralytics import YOLO
import cv2

model = YOLO("model/yolov8n.pt")
# img = cv2.imread("violation_20250624-165331.jpg") 
img = cv2.imread("violation_20250624-171156.jpg")   # Replace with actual test image
results = model(img)

results[0].show()  # ✅ Fix: use results[0]

for box in results[0].boxes:
    cls = int(box.cls[0])
    conf = float(box.conf[0])
    print("Detected:", results[0].names[cls], "Confidence:", conf)
