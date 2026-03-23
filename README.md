# 🎥 Object Detection & Tracking in Video Pipeline

## 📌 Overview
This repository contains an end-to-end machine learning pipeline designed to ingest raw video feeds, detect and track specific objects, and export the tracking data into structured CSV reports. Built using the state-of-the-art YOLO11 architecture and ByteTrack algorithms, this project demonstrates how to transform unstructured computer vision data into actionable, quantifiable metrics suitable for enterprise reporting and analytics.

---
## 🎯 Key Objectives
- Detect objects from video frames
- Track objects across frames using unique IDs
- Count objects without duplication
- Export results into structured data formats

---
## 🧠 Core Features

### 🔹 Object Detection (YOLO11)
- Uses Ultralytics YOLO11m
- Trained on COCO (80 classes)
- Configuration:
  - Epochs: 100
  - Batch Size: 16
  - Image Size: 640
---
### 🔹 Multi-Object Tracking (ByteTrack)
- Maintains persistent tracking IDs
- Prevents:
  - Double counting
  - Frame-to-frame inconsistency
---
### 🔹 ROI-Based Counting System
- Defines a dynamic counting zone
- Counts objects only when entering the region
- Ensures accurate event-based counting

```python
if zone_x1 < cx < zone_x2 and zone_y1 < cy < zone_y2:
    if track_id not in counted_ids:
        counted_ids.add(track_id)
```
---
### 🔹 Automated Data Export
- Generates structured output:
```bash
total_counts.csv
```
Example:
```bash
| Object Type | Total Count |
|------------|------------|
| Person     | 32         |
| Car        | 18         |
| Dog        | 5          |
```
---
### 🔹 Video Annotation & Export
- Annotates:
  - Bounding boxes
  - Class labels
  - Tracking IDs
  - Counting zone
- Converts output to web-ready MP4 using FFmpeg
---
## 📊 Model Performance

### 🔹 Training Results
<p align="center"> <img src="Results/Training_result.png" width="500"/> </p>

- Shows loss convergence, precision, recall, and mAP trends
- Indicates stable training and good generalization

### 🔹 Confusion Matrix
<p align="center"> <img src="Results/Confusion_matrix.png" width="500"/> </p>

- Strong diagonal → correct classifications
- Minimal class confusion across dataset

---
## 📁 Project Structure
``` bash
📦 object-detection-in-video
│
├── Models/
│   └── best.pt
│
├── NoteBook/
│   └── object_detection_in_video.ipynb
│
├── outputs/
│   ├── web_ready_output.mp4
│   └── total_counts.csv
│
├── Results/
│   ├── Training_result.png
│   └── Confusion_matrix.png
│
└── README.md
```
---

## ⚙️ Tech Stack

| Category            | Technology              |
|--------------------|------------------------|
| Language           | Python                 |
| Detection          | YOLO11 (Ultralytics)   |
| Tracking           | ByteTrack              |
| Video Processing   | OpenCV                 |
| Dataset Platform   | Roboflow               |
| Compression        | FFmpeg                 |

---
## 🚀 How to Run

### 1️⃣ Clone Repository
```bash
git clone https://github.com/TamimHq/traffic-analysis-pipeline.git
cd traffic-analysis-pipeline
```
### 2️⃣ Install Dependencies
```bash
pip install ultralytics roboflow opencv-python
```
### 3️⃣ Run Notebook
```bash
NoteBook/object_detection_in_video.ipynb
```
Run all cells to:
- Train model (optional)
- Evaluate performance
- Process video
- Generate outputs
---

## 📦 Outputs

| Output Types | File |
|--------------|------|
|🎥 Annotated Video|web_ready_output.mp4|
|📊 CSV Report|total_counts.csv|
---

## 💡 Real-World Applications
- 🚗 Traffic flow monitoring
- 🏙️ Smart city infrastructure
- 🏭 Industrial object tracking
- 🛒 Retail analytics
- 🎥 Surveillance systems
---

## 🔒 Notes
- API keys removed for security
- Model weights included for inference
- Compatible with Google Colab + local setup
---

## 📬 Contact
Tamim Haque
Feel free to connect for collaboration or questions!