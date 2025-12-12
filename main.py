# main.py
import cv2
import os
import csv
from datetime import datetime
from ultralytics import YOLO
from ocr_utils import extract_chest_number
from email_utils import send_email
from cleanup import delete_old_images

# Paths
VIOLATION_FOLDER = "images/violations"
CSV_LOG_FILE = "logs/violations.csv"
MODEL_PATH = "model/best.pt"
MANAGER_EMAIL = "harshrameshnerkar@gmail.com"

# Setup
os.makedirs(VIOLATION_FOLDER, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Load model and video
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0)
print("[INFO] System started...")

def is_helmet_on_head(person_box, helmet_box):
    px1, py1, px2, py2 = person_box
    hx1, hy1, hx2, hy2 = helmet_box
    head_y_limit = py1 + int((py2 - py1) * 0.4)
    return (
        hx1 >= px1 and hx2 <= px2 and
        hy1 >= py1 and hy2 <= head_y_limit
    )

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    helmet_boxes = []
    person_boxes = []

    for det in results.boxes:
        if det.conf[0] < 0.5:
            continue
        cls = int(det.cls[0])
        label = model.names[cls].lower()
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        color = (0, 255, 0) if label in ["helmet", "hardhat"] else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if label in ["helmet", "hardhat"]:
            helmet_boxes.append((x1, y1, x2, y2))
        elif label in ["person", "worker"]:
            person_boxes.append((x1, y1, x2, y2))

    for (x1, y1, x2, y2) in person_boxes:
        person_crop = frame[y1:y2, x1:x2]
        person_has_helmet = any(is_helmet_on_head((x1, y1, x2, y2), h) for h in helmet_boxes)

        if person_has_helmet:
            cv2.putText(frame, "Helmet Detected", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            print(f"[INFO] Helmet Detected for person at ({x1},{y1})")
        else:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"violation_{timestamp}.jpg"
            filepath = os.path.join(VIOLATION_FOLDER, filename)
            cv2.imwrite(filepath, frame)

            try:
                suit_id = extract_chest_number(person_crop)
            except Exception as e:
                print(f"[ERROR] OCR failed: {e}")
                suit_id = "UNKNOWN"

            with open(CSV_LOG_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, suit_id, filename])
            print(f"[ALERT] No Helmet - Suit ID: {suit_id}")

            try:
                send_email(MANAGER_EMAIL, suit_id, filepath)
            except Exception as e:
                print(f"[ERROR] Email send failed: {e}")

    cv2.imshow("Helmet Detection Feed", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
delete_old_images(VIOLATION_FOLDER, days=30)
