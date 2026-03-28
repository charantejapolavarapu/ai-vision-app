import streamlit as st
import torch
from torchvision import models
from PIL import Image
import pandas as pd
import os
from gtts import gTTS
from datetime import datetime
from ultralytics import YOLO
import numpy as np
import cv2

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="Image Recognition Pro+", layout="wide")

# -------------------------
# SESSION
# -------------------------
if "user" not in st.session_state:
    st.session_state.user = None
if "users" not in st.session_state:
    st.session_state.users = {}
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------
# LOGIN
# -------------------------
if st.session_state.user is None:
    st.title("🔐 Login")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            if u in st.session_state.users and st.session_state.users[u] == p:
                st.session_state.user = u
                st.rerun()
            else:
                st.error("Invalid")

    with tab2:
        u = st.text_input("New Username")
        p = st.text_input("New Password", type="password")
        if st.button("Register"):
            st.session_state.users[u] = p
            st.success("Registered")

# -------------------------
# MAIN APP
# -------------------------
else:

    st.title("🚀 Image Recognition Pro+")
    st.write(f"Welcome {st.session_state.user}")

    if st.button("Logout"):
        st.session_state.user = None
        st.rerun()

    # -------------------------
    # LOAD MODELS
    # -------------------------
    @st.cache_resource
    def load_models():
        weights = models.MobileNet_V2_Weights.DEFAULT
        clf = models.mobilenet_v2(weights=weights)
        clf.eval()

        yolo = YOLO("yolov8n.pt")  # lightweight & fast

        return clf, weights, yolo

    clf_model, weights, yolo_model = load_models()
    transform = weights.transforms()
    labels = weights.meta["categories"]

    # -------------------------
    # SETTINGS
    # -------------------------
    st.sidebar.title("⚙️ Settings")
    language = st.sidebar.selectbox("Language", ["English", "Telugu"])
    use_voice = st.sidebar.toggle("Voice Output")

    # -------------------------
    # FUNCTIONS
    # -------------------------
    def classify(image):
        image = image.convert("RGB")
        img = transform(image).unsqueeze(0)

        with torch.inference_mode():
            out = clf_model(img)
            probs = torch.nn.functional.softmax(out[0], dim=0)

        top5 = torch.topk(probs, 5)

        return pd.DataFrame([
            {"Label": labels[top5.indices[i]],
             "Confidence (%)": round(float(top5.values[i])*100,2)}
            for i in range(5)
        ])

    # 🔥 IMPROVED DETECTION
    def detect(image):
        img = np.array(image)

        results = yolo_model(img, conf=0.45, iou=0.5)[0]

        objects = []
        important = ["person","dog","cat","cow","horse","sheep","bird"]

        for box in results.boxes:
            cls = int(box.cls[0])
            label = yolo_model.names[cls]

            if label in important:
                objects.append(label)

        annotated = results.plot()
        return Image.fromarray(annotated), objects

    # 🧠 FACE DETECTION
    def detect_faces(image):
        img = np.array(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

        return Image.fromarray(img), len(faces)

    def explain(text):
        if language == "Telugu":
            return f"ఈ చిత్రంలో {text} గుర్తించబడింది."
        return f"In this image, {text} is detected."

    def speak(text):
        tts = gTTS(text, lang='te' if language=="Telugu" else 'en')
        tts.save("voice.mp3")
        return "voice.mp3"

    # -------------------------
    # TABS
    # -------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Classification",
        "🎯 Object Detection",
        "🧠 Face Detection",
        "📊 Analytics",
        "🎥 Live"
    ])

    # -------------------------
    # CLASSIFICATION
    # -------------------------
    with tab1:
        file = st.file_uploader("Upload Image", type=["jpg","png"])
        if file:
            image = Image.open(file)

            st.image(image, use_container_width=True)

            df = classify(image)
            pred = df.iloc[0]["Label"]

            st.success(f"Predicted: {pred}")
            st.dataframe(df)

    # -------------------------
    # OBJECT DETECTION
    # -------------------------
    with tab2:
        file = st.file_uploader("Upload for Detection", type=["jpg","png"], key="det")

        if file:
            image = Image.open(file)

            detected_img, objects = detect(image)

            st.image(detected_img, use_container_width=True)

            if objects:
                st.success(f"Detected: {set(objects)}")
                st.write(explain(", ".join(set(objects))))

                if use_voice:
                    st.audio(speak(", ".join(set(objects))))
            else:
                st.warning("No humans/animals detected")

    # -------------------------
    # FACE DETECTION
    # -------------------------
    with tab3:
        file = st.file_uploader("Upload for Face Detection", type=["jpg","png"], key="face")

        if file:
            image = Image.open(file)

            face_img, count = detect_faces(image)

            st.image(face_img)
            st.success(f"Faces Detected: {count}")

    # -------------------------
    # ANALYTICS
    # -------------------------
    with tab4:
        st.metric("App Status", "Running")

    # -------------------------
    # LIVE DETECTION (LOCAL ONLY)
    # -------------------------
    with tab5:
        run = st.checkbox("Start Camera")

        cap = cv2.VideoCapture(0)
        frame_window = st.image([])

        while run:
            ret, frame = cap.read()
            if not ret:
                break

            results = yolo_model(frame, conf=0.45)[0]
            frame = results.plot()

            frame_window.image(frame)

        cap.release()
