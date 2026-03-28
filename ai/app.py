import streamlit as st
import torch
from torchvision import models
from PIL import Image
import pandas as pd
import numpy as np
from datetime import datetime
from gtts import gTTS

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
                st.error("Invalid credentials")

    with tab2:
        u = st.text_input("New Username")
        p = st.text_input("New Password", type="password")
        if st.button("Register"):
            if u and p:
                st.session_state.users[u] = p
                st.success("Registered")
            else:
                st.error("Fill all fields")

# -------------------------
# MAIN APP
# -------------------------
else:

    st.title("🚀 Image Recognition Pro+")
    st.write(f"Welcome **{st.session_state.user}** 👋")

    if st.button("Logout"):
        st.session_state.user = None
        st.rerun()

    # -------------------------
    # MODEL
    # -------------------------
    @st.cache_resource
    def load_model():
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
        model.eval()
        return model, weights

    model, weights = load_model()
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
            outputs = model(img)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)

        top5 = torch.topk(probs, 5)

        return pd.DataFrame([
            {"Label": labels[top5.indices[i]],
             "Confidence (%)": round(float(top5.values[i])*100,2)}
            for i in range(5)
        ])

    # 🔥 HUMAN + ANIMAL DETECTION (SMART FILTER)
    def detect_animals_humans(df):
        keywords = [
            "person","man","woman","boy","girl",
            "dog","cat","cow","horse","sheep","animal"
        ]

        detected = []
        for _, row in df.iterrows():
            label = row["Label"].lower()

            if any(k in label for k in keywords):
                detected.append(label)

        return list(set(detected))

    def explain(text):
        if language == "Telugu":
            return f"ఈ చిత్రంలో {text} గుర్తించబడింది."
        return f"In this image, {text} is detected."

    def speak(text):
        try:
            tts = gTTS(text, lang='te' if language=="Telugu" else 'en')
            tts.save("voice.mp3")
            return "voice.mp3"
        except:
            return None

    # -------------------------
    # TABS
    # -------------------------
    tab1, tab2 = st.tabs(["🔍 Analyze", "📊 Analytics"])

    # -------------------------
    # TAB 1: ANALYSIS
    # -------------------------
    with tab1:

        file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

        if file:
            image = Image.open(file)

            col1, col2 = st.columns(2)

            with col1:
                st.image(image, use_container_width=True)

            with col2:
                df = classify(image)
                pred = df.iloc[0]["Label"]

                st.success(f"🎯 Predicted: {pred}")

                # 📊 Table
                st.dataframe(df)

                # 📈 Chart
                st.bar_chart(df.set_index("Label"))

                # 👨🐶 Detection
                objects = detect_animals_humans(df)

                if objects:
                    st.success(f"👀 Detected: {objects}")
                    explanation = explain(", ".join(objects))
                else:
                    st.warning("No humans/animals detected")
                    explanation = explain(pred)

                st.subheader("🧠 Explanation")
                st.write(explanation)

                # 🔊 Voice
                if use_voice:
                    audio = speak(explanation)
                    if audio:
                        st.audio(audio)

                # 📊 Save history
                st.session_state.history.append({
                    "time": datetime.now(),
                    "prediction": pred
                })

    # -------------------------
    # TAB 2: ANALYTICS
    # -------------------------
    with tab2:

        st.metric("Total Predictions", len(st.session_state.history))

        if st.session_state.history:
            df = pd.DataFrame(st.session_state.history)

            st.bar_chart(df["prediction"].value_counts())

            st.dataframe(df)

            csv = df.to_csv(index=False)
            st.download_button("Download CSV", csv)

    st.markdown("---")
    st.markdown("🔥 Cloud Ready AI Project")
