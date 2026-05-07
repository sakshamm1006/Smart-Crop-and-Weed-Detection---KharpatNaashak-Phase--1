# 🌱 Smart Crop & Weed Detection System

### *KharpatNaashak – Phase 1*

---

## 📌 Overview

Agriculture is one of the most important sectors for sustaining human life, but one of its major challenges is the uncontrolled growth of weeds. Weeds compete with crops for essential resources such as nutrients, water, and sunlight, ultimately reducing agricultural productivity.

This project presents an **AI-powered crop and weed detection system** built using deep learning. The system leverages a **fine-tuned YOLOv3 (You Only Look Once) model** to detect and classify crops and weeds in real-time.

The solution is deployed through an **interactive Streamlit web application**, supporting multiple input modes such as images, videos, and webcam streams. This makes it a practical tool for real-world agricultural monitoring and analysis.

---

## 🎯 Problem Statement

Weeds negatively impact crop yield by competing for resources and forcing farmers to rely heavily on chemical pesticides. These chemicals can harm both the environment and human health.

The goal of this project is to build an **automated detection system** that:

* Identifies crops and weeds accurately
* Reduces reliance on harmful chemicals
* Assists in smarter decision-making for farmers

---

## 🚀 Features

* 🌱 Crop vs Weed Detection using YOLOv3
* 🖼 Multi-image upload detection
* 🎥 Video-based detection
* 📷 Webcam-based real-time detection
* 🌐 Interactive web interface using Streamlit
* 📊 Field analysis:

  * Crop count
  * Weed count
  * Weed density calculation
* 📈 Visualization using graphs
* 📄 Downloadable detection report
* ⚡ Adjustable confidence threshold

---

## 🧠 Methodology / Approach

The system follows a structured deep learning pipeline:

### 1. Model Selection

YOLOv3 was chosen for:

* Real-time detection capability
* High accuracy
* Efficiency in multi-object detection

### 2. Fine-Tuning

* Used **pre-trained YOLOv3 weights**
* Fine-tuned on a custom agricultural dataset
* Classes:

  * Crop 🌾
  * Weed 🌿

### 3. Detection Pipeline

* Image preprocessing using OpenCV
* Forward pass through YOLO network
* Bounding box extraction
* **Non-Max Suppression (NMS)** to remove duplicates

### 4. UI Integration

* Built using **Streamlit**
* Allows real-time interaction and visualization

---

## 📊 Dataset

* Format: YOLO annotation format
* Image size: 512 × 512
* Classes:

  * Crop
  * Weed

---

## 📈 Results

* **mAP@50:** ~91%
* **Precision:** ~87%
* **Recall:** ~86%

The model performs reliably across different field conditions and lighting variations.

---

## 📊 Model Performance & Analysis

### 📉 Training Metrics

![Training Graph](assets/results.png)

The training curves show consistent improvement, with decreasing loss and increasing mAP, indicating effective fine-tuning.

---

### 🔍 Confusion Matrix

![Confusion Matrix](assets/confusion_matrix.png)

The confusion matrix demonstrates high classification accuracy, with most predictions correctly classified along the diagonal.

---

### 📊 Detection Statistics

![Graph](assets/graph.png)

This visualization shows the distribution of detected crops and weeds.

---

### 📄 Generated Report

![Report](assets/report.png)

The system generates a detailed report including crop count, weed count, and actionable recommendations.

---

## 📸 Application Preview

### 🖥️ User Interface

![UI](assets/ui.png)

### 🖼 Image Detection

![Image Detection](assets/image.png)

### 🎥 Video Detection

![Video Detection](assets/video.png)

### 📷 Webcam Detection

![Webcam](assets/webcam.png)

---

## ⚙️ Tech Stack

* Python
* OpenCV
* YOLOv3
* Streamlit
* NumPy / Pandas

---

## ▶️ Installation & Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the application

```bash
streamlit run app.py
```

### 3. Use the system

* Upload image/video OR use webcam
* View detection results
* Analyze crop vs weed distribution
* Download report

---

## 📂 Project Structure

```
Smart-Crop-Weed-Detection/
│
├── app.py
├── models/
│   └── best.pt
├── performing_detection/
├── training/
├── assets/
├── requirements.txt
├── README.md
```

---

## 💡 Key Learnings

* Fine-tuning pre-trained deep learning models
* Understanding YOLO architecture
* Handling dataset annotation formats
* Building end-to-end AI applications
* Integrating ML models with web interfaces

---

## 🔮 Future Scope (Phase 2 – KharpatNaashak)

This project is part of a larger system named **KharpatNaashak**.

Future improvements include:

* Upgrading to **YOLOv11**
* Improving detection accuracy and speed
* Real-world deployment in agricultural fields
* Integration with automated weed removal systems

> 🔧 **Note:** The YOLOv11-based Phase 2 (KharpatNaashak) is being developed by my teammate **Mahi Singh**.

## 👨‍💻 Authors & Contributors

* **Saksham Trivedi** — Model fine-tuning, system integration, Streamlit application development
* **Mahi Singh** — Future development (KharpatNaashak Phase 2 – YOLOv11 based system)
* **Nandani Tripathi** — Project support and contributions

---

## 🤝 Collaboration

This project is part of an evolving AI-based agricultural system under **KharpatNaashak**, developed collaboratively with contributions from team members working on different phases and enhancements.


## ⭐ Conclusion

This project demonstrates how deep learning and computer vision can be applied to solve real-world agricultural challenges. By combining YOLOv3 with an interactive UI, the system provides a practical and scalable solution for weed detection and smart farming.
