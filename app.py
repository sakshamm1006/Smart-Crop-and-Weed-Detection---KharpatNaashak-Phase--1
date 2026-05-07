import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
import time

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Smart Weed Detection", layout="wide")

st.title("🌱 Smart Crop & Weed Detection System")
st.markdown("AI-powered precision agriculture system")

# =========================
# SIDEBAR
# =========================
mode = st.sidebar.radio("Mode", ["Image", "Video", "Webcam"])

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.1, 1.0, 0.5, step=0.05
)

show_boxes = st.sidebar.checkbox("Show Bounding Boxes", True)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    net = cv2.dnn.readNet(
        "performing_detection/data/weights/crop_weed_detection.weights",
        "performing_detection/data/cfg/crop_weed.cfg"
    )

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return net, output_layers

net, output_layers = load_model()
classes = ["crop", "weed"]

# =========================
# DETECTION FUNCTION
# =========================
def detect(frame):
    height, width, _ = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (512, 512), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold:
                cx, cy = int(detection[0]*width), int(detection[1]*height)
                w, h = int(detection[2]*width), int(detection[3]*height)
                x, y = int(cx - w/2), int(cy - h/2)

                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)

    crop, weed = 0, 0

    if len(indexes) > 0:
        for i in indexes.flatten():
            x,y,w,h = boxes[i]
            label = classes[class_ids[i]]

            if label == "crop":
                crop += 1
                color = (0,255,0)
            else:
                weed += 1
                color = (0,0,255)

            if show_boxes:
                cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
                cv2.putText(frame,label,(x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

    return frame, crop, weed

# =========================
# REPORT FUNCTION
# =========================
def generate_report(crop, weed):
    total = crop + weed
    density = (weed/total)*100 if total else 0

    report = f"""
SMART CROP & WEED DETECTION REPORT

Total Crops: {crop}
Total Weeds: {weed}
Weed Density: {density:.2f}%

Recommendation:
"""
    if density > 50:
        report += "High weed presence. Immediate action required."
    elif density > 20:
        report += "Moderate weeds. Monitor closely."
    else:
        report += "Low weeds. No major action needed."

    return report, density

# =========================
# IMAGE MODE
# =========================
if mode == "Image":

    files = st.file_uploader("Upload Images", accept_multiple_files=True)

    if files:
        total_crop, total_weed = 0, 0

        for file in files:
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
            out, c, w = detect(img)

            total_crop += c
            total_weed += w

            st.image(out, channels="BGR")

        report, density = generate_report(total_crop, total_weed)

        st.subheader("📊 Summary")
        st.write(report)

        # GRAPH
        df = pd.DataFrame({
            "Type": ["Crop", "Weed"],
            "Count": [total_crop, total_weed]
        })
        st.bar_chart(df.set_index("Type"))

        # DOWNLOAD
        st.download_button("📥 Download Report", report, file_name="report.txt")

# =========================
# VIDEO MODE
# =========================
if mode == "Video":

    file = st.file_uploader("Upload Video")

    if file:
        temp = "temp.mp4"
        with open(temp, "wb") as f:
            f.write(file.read())

        cap = cv2.VideoCapture(temp)
        frame_window = st.image([])

        total_crop, total_weed = 0, 0
        prev_time = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            curr_time = time.time()
            fps = 1/(curr_time - prev_time) if prev_time else 0
            prev_time = curr_time

            frame, c, w = detect(frame)
            total_crop += c
            total_weed += w

            cv2.putText(frame, f"FPS: {int(fps)}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)

            frame_window.image(frame, channels="BGR")

        cap.release()

        report, density = generate_report(total_crop, total_weed)

        st.subheader("📊 Video Summary")
        st.write(report)

        df = pd.DataFrame({
            "Type": ["Crop", "Weed"],
            "Count": [total_crop, total_weed]
        })
        st.bar_chart(df.set_index("Type"))

        st.download_button("📥 Download Report", report, file_name="video_report.txt")

# =========================
# WEBCAM MODE
# =========================
if mode == "Webcam":

    img = st.camera_input("Capture Image")

    if img:
        frame = cv2.imdecode(np.frombuffer(img.read(), np.uint8), 1)

        start = time.time()
        frame, c, w = detect(frame)
        fps = 1/(time.time()-start)

        cv2.putText(frame, f"FPS: {int(fps)}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)

        st.image(frame, channels="BGR")

        report, density = generate_report(c, w)

        st.write(report)

        df = pd.DataFrame({
            "Type": ["Crop", "Weed"],
            "Count": [c, w]
        })
        st.bar_chart(df.set_index("Type"))

        st.download_button("📥 Download Report", report, file_name="webcam_report.txt")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("🚀 KharpatNaashak Phase 1 | EPICS Project")