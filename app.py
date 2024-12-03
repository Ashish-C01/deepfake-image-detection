import streamlit as st
from transformers import AutoImageProcessor, AutoModelForImageClassification
import cv2
import torch
import numpy as np
import tempfile

image_processor = AutoImageProcessor.from_pretrained(
    'ashish-001/deepfake-detection-using-ViT')
model = AutoModelForImageClassification.from_pretrained(
    'ashish-001/deepfake-detection-using-ViT')


def classify_frame(frame):
    inputs = image_processor(images=frame, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.sigmoid(logits)
    pred = torch.argmax(logits, dim=1).item()
    lab = 'Real' if pred == 1 else 'Fake'
    confidence, _ = torch.max(probs, dim=1)
    return f"{lab}::{format(confidence.item(), '.2f')}"


st.title("Deepfake detector")
uploaded_file = st.file_uploader(
    "Upload an image or video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv"]
)
placeholder = st.empty()
if st.button('Detect'):
    if uploaded_file is not None:
        clf = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        mime_type = uploaded_file.type
        if mime_type.startswith("image"):
            file_bytes = uploaded_file.read()
            np_arr = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = clf.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in faces:
                cv2.rectangle(image_rgb, (x, y), (x+w, y+h), (0, 0, 255), 2)
                face = image_rgb[y:y + h, x:x + w]
                img = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                label = classify_frame(img)
                new_frame = cv2.putText(
                    image_rgb, label, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                st.image(new_frame)

        elif mime_type.startswith('video'):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_video_path = temp_file.name
                cap = cv2.VideoCapture(temp_video_path)
                if not cap.isOpened():
                    st.error("Error: Cannot open video file.")
                else:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = clf.detectMultiScale(
                            gray, scaleFactor=1.3, minNeighbors=5)
                        for (x, y, w, h) in faces:
                            cv2.rectangle(
                                frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                            face = frame[y:y + h, x:x + w]
                            img = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                            label = classify_frame(img)
                            frame = cv2.putText(
                                frame, label, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                        placeholder.image(frame)
                    cap.release()

if st.button('Use Example Video'):
    clf = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture("Sample.mp4")
    if not cap.isOpened():
        st.error("Error: Cannot open video file.")
    else:
        st.write(f"Video credits: 'Deep Fakes' Are Becoming More Realistic Thanks To New Technology. Link:https://www.youtube.com/watch?v=CDMVaQOvtxU")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = clf.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in faces:
                cv2.rectangle(
                    frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                face = frame[y:y + h, x:x + w]
                img = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                label = classify_frame(img)
                frame = cv2.putText(
                    frame, label, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            placeholder.image(frame)
        cap.release()
