import cv2
import easyocr

reader = easyocr.Reader(['en'])

def extract_chest_number(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        result = reader.readtext(thresh)
        for (bbox, text, prob) in result:
            if prob > 0.5 and text.strip().isalnum():
                return text.strip()
    except Exception as e:
        print(f"[OCR ERROR] {e}")
    return "UNKNOWN"
