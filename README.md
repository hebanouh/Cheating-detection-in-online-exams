
# **Cheating Detection in Online Exams – AI-Powered Monitoring System**

A complete AI-based proctoring system that detects multiple forms of cheating during online exams using computer vision, audio analysis, pose estimation, and object detection.
The system runs in real time using the student’s webcam and microphone and automatically generates detailed reports of violations.

---

## ⭐ **Features**

### 🎯 **1. Face Monitoring**

* Detects if the student leaves the frame
* Detects multiple faces
* Tracks head position using pose estimation

### 👀 **2. Eye & Gaze Tracking**

* Gaze direction (Left/Right/Center/Up/Down)
* Blink detection using Eye Aspect Ratio (EAR)
* Detects suspicious repeated gaze deviations

### 📘 **3. Object Detection**

Detects **FORBIDDEN** objects such as:

* Mobile phones
* Books
* Smart watches
* Earbuds / Headphones
* Sunglasses
* Laptops
* Face masks
* Any custom objects (via YOLOv8 model)

### 🗣 **4. Audio Monitoring**

* Detects speech, whispering, or loud noise
* Can be integrated with Whisper for speech-to-text
* Flags when student is talking

### 🕺 **5. Pose Estimation**

* Detects abnormal posture
* Detects leaving the seat

### 🖥 **6. Real-Time Dashboard**

* Live view of webcam
* Live detection alerts
* Timeline of violations
* System health & running modules

### 📄 **7. Automatic Report Generation**

Each violation is logged into:

* **CSV report**
* **Saved screenshot evidence**
* Timestamped event list
* Summary at the end of session

### 🔔 **8. Real-Time Alerting**

* On-screen alerts
* Optional voice alerts (“Stop looking away from the screen”)

---

## 🚀 **Technologies Used**

### **Computer Vision**

* OpenCV
* MediaPipe (Face Mesh, Pose)
* YOLOv8 (Ultralytics)

### **Audio Analysis**

* Pyaudio / SpeechRecognition
* Whisper (optional)

### **Backend / Dashboard**

* Flask
* HTML + CSS + JS (Realtime UI)

### **Scripting**

* Python 3.8+

---

## 📦 **Installation**

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/Cheating-detection-in-online-exams.git
cd Cheating-detection-in-online-exams
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Download Whisper models

```bash
pip install whisper
```

---

## ▶ **Run the System**

### Start monitoring:

```bash
python src/main.py
```

### Start the dashboard:

```bash
python src/dashboard/app.py
```

Then open:
👉 [http://localhost:5000](http://localhost:5000)

---

## 🏗 **System Architecture (ASCII Diagram)**

```
Cheating-Detection-System/
│
├── Camera Input + Microphone
│
├── src/
│   ├── detection/
│   │   ├── face_detection.py
│   │   ├── gaze_tracking.py
│   │   ├── object_detection.py
│   │   ├── pose_estimation.py
│   │   └── audio_detection.py
│   │
│   ├── reporting/
│   │   ├── report_generator.py
│   │   └── logger.py
│   │
│   ├── dashboard/
│   │   ├── app.py
│   │   └── static/
│   │
│   └── main.py
│
├── models/ (YOLO + face models)
├── evidence/ (screenshots)
├── logs/
└── exam_report.csv
```

---

## ⚙ **Configuration (config.yaml)**

Example:

```yaml
video:
  source: 0
  fps: 30

detection:
  objects:
    confidence: 0.60
  eyes:
    blink_threshold: 0.25
    gaze_limit_seconds: 2
  audio:
    energy_threshold: 0.001
    whisper_enabled: false

reporting:
  save_evidence: true
  evidence_path: "./evidence"
```

---

## 🧪 **Troubleshooting**

### ❌ *YOLO not detecting correctly*

* Retrain with more samples
* Increase confidence threshold
* Confirm class names are correct

### ❌ *Gaze detection inaccurate*

* Improve lighting
* Camera should be at eye level

### ❌ *Audio sensitivity too high*

* Reduce energy threshold

---

## 🤝 **Contributing Guide**

1. Fork the repository
2. Create a new branch

```bash
git checkout -b feature-new
```

3. Commit your changes

```bash
git commit -m "Add new feature"
```

4. Push your branch

```bash
git push origin feature-new
```

5. Open a Pull Request

## ☕ **Support**

If this project helped you, star ⭐ the repo!


