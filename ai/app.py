import streamlit as st
import torch
from torchvision import models
from PIL import Image
import pandas as pd
import os
from gtts import gTTS

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="Image Recognition Pro", layout="wide")

# -------------------------
# SESSION
# -------------------------
if "user" not in st.session_state:
    st.session_state.user = None
if "users" not in st.session_state:
    st.session_state.users = {}

# -------------------------
# UI STYLE
# -------------------------
st.markdown("""
<style>
body {background-color: #0e1117;}
.card {
    background: rgba(255,255,255,0.06);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(12px);
    margin-bottom: 20px;
}
.title {
    text-align: center;
    font-size: 36px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# LOGIN
# -------------------------
if st.session_state.user is None:

    st.markdown("<div class='title'>🔐 Image Recognition</div>", unsafe_allow_html=True)

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

    st.markdown("<div class='title'>🧠 Image Recognition Pro</div>", unsafe_allow_html=True)
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
    # SIDEBAR
    # -------------------------
    st.sidebar.title("⚙️ Settings")
    use_ai = st.sidebar.toggle("AI Explanation")
    use_voice = st.sidebar.toggle("Voice")

    language = st.sidebar.selectbox("Language", ["English", "Telugu"])

    # -------------------------
    # OPENAI SAFE
    # -------------------------
    client = None
    if use_ai:
        try:
            from openai import OpenAI
            key = os.getenv("OPENAI_API_KEY")
            if key:
                client = OpenAI(api_key=key)
        except:
            pass

    # -------------------------
    # FUNCTIONS
    # -------------------------
    def analyze(image):
        image = image.convert("RGB")
        img = transform(image).unsqueeze(0)

        with torch.inference_mode():
            outputs = model(img)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)

        top3 = torch.topk(probs, 3)

        return pd.DataFrame([
            {"Label": labels[top3.indices[i]],
             "Confidence (%)": round(float(top3.values[i]) * 100, 2)}
            for i in range(3)
        ])

    # ✅ REAL TELUGU EXPLANATION
    def get_explanation(label, lang):

        fallback_en = f"This image is predicted as {label} based on visual features like shape and texture."
        fallback_te = f"ఈ చిత్రం {label} గా గుర్తించబడింది. ఆకారం మరియు రంగు వంటి లక్షణాల ఆధారంగా ఇది నిర్ణయించబడింది."

        if client is None:
            return fallback_te if lang == "Telugu" else fallback_en

        try:
            if lang == "Telugu":
                prompt = f"{label} అనే వస్తువు ఏమిటి మరియు ఈ చిత్రం ఎందుకు {label}గా గుర్తించబడిందో సులభంగా తెలుగులో వివరించండి."
            else:
                prompt = f"Explain why this image is {label} in simple terms."

            res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )

            return res.choices[0].message.content

        except:
            return fallback_te if lang == "Telugu" else fallback_en

    def text_to_speech(text):
        tts = gTTS(text, lang='te' if language=="Telugu" else 'en')
        file = "voice.mp3"
        tts.save(file)
        return file

    # -------------------------
    # UI
    # -------------------------
    st.markdown("### 📷 Upload Image")
    file = st.file_uploader("", type=["jpg", "png", "jpeg"])

    if file:
        image = Image.open(file)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            df = analyze(image)

            predicted_label = df.iloc[0]["Label"]

            st.markdown("<div class='card'>", unsafe_allow_html=True)

            # 🎯 PREDICTED VALUE
            st.subheader("🎯 Predicted Value")
            st.success(predicted_label)

            # 📊 FULL TABLE
            st.subheader("📊 Predictions")
            st.dataframe(df)

            st.subheader("📈 Chart")
            st.bar_chart(df.set_index("Label"))

            # 👤 USER INPUT ACTUAL VALUE
            actual = st.text_input("Enter Actual Value (Optional)")

            if actual:
                st.subheader("🔁 Comparison")
                if actual.lower() == predicted_label.lower():
                    st.success("✅ Prediction Correct")
                else:
                    st.error("❌ Prediction Incorrect")

            # 🧠 EXPLANATION
            explanation = get_explanation(predicted_label, language)

            st.subheader("🧠 Explanation")
            st.write(explanation)

            # 🔊 VOICE
            if use_voice:
                audio = text_to_speech(explanation)
                st.audio(audio)

            st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------
    # DASHBOARD
    # -------------------------
    st.markdown("### 📊 Dashboard")
    st.metric("User", st.session_state.user)
    st.metric("Model", "MobileNetV2")
    st.metric("Status", "Active")

    st.markdown("---")
    st.markdown("🚀 AI Vision App | Telugu Enabled")
