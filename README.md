# 🛡️ Helmet Detection System for Construction Sites 🏗️

## 📌 Overview

This project ensures safety compliance at construction sites by monitoring live camera feeds to detect whether workers are wearing helmets. If a person is **not wearing a helmet**, the system will:

- Capture approximately **10 violation images**.
- Send those images as email alerts to **site managers**.

If the person is wearing a helmet, no action is taken.

---

## 🚀 Features

- ✅ Real-time video monitoring using webcam or CCTV.
- 🎯 Helmet detection using deep learning.
- 📸 Violation snapshot capturing (approx. 10 per incident).
- 📧 Automatic email notifications to managers with attached images.
- 🔒 Privacy-respecting: no full video recording, only violation frames saved.

---

## 🧠 Technologies Used

- **Language**: Python
- **Libraries**: OpenCV, TensorFlow or YOLOv5/YOLOv8, smtplib
- **Camera Input**: USB Webcam / IP Camera
- **Optional**: Flask (for web interface)

---

## 📁 Project Structure

helmet-detection/
│
├── snapshots/ # Captured violation images
├── model/ # Pre-trained or custom helmet detection model
├── utils/
│ ├── email_sender.py # Sends emails with violation images
│ └── violation_logger.py # Logs and manages captured violations
├── detector.py # Helmet detection logic
├── app.py # Main application entry point
├── requirements.txt # Python dependencies
└── README.md # Project documentation

pip install -r requirements.txt
