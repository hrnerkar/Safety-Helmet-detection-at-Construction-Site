import cv2

cap = cv2.VideoCapture(0)  # Try 1 or 2 if 0 fails

if not cap.isOpened():
    print("[ERROR] Webcam not accessible")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Frame not read")
        break

    cv2.imshow("Test Webcam", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
