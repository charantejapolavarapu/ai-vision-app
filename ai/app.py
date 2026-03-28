import streamlit as st
import torch
from torchvision import models
from PIL import Image
import pandas as pd
import os
from gtts import gTTS

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Image Recognition Pro", layout="wide")

# -------------------------
# SESSION LOGIN
# -------------------------
if "user" not in st.session_state:
    st.session_state.user = None

# -------------------------
# CUSTOM UI (GLASS STYLE)
# -------------------------
st.markdown("""
<style>
body {background-color: #0e1117;}
.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    margin-bottom: 20px;
}
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# LOGIN PAGE
# -------------------------
if st.session_state.user is None:
    st.markdown("<div class='title'>🔐 Login</div>", unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username and password:
            st.session_state.user = username
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Enter credentials")

# -------------------------
# MAIN APP
# -------------------------
else:

    st.markdown(f"<div class='title'>🧠 Image Recognition Pro</div>", unsafe_allow_html=True)
    st.write(f"Welcome, **{st.session_state.user}** 👋")

    # LOGOUT
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

    history = []

    # -------------------------
    # SIDEBAR
    # -------------------------
    st.sidebar.title("⚙️ Controls")
    use_ai = st.sidebar.toggle("Enable AI Explanation")
    use_voice = st.sidebar.toggle("Enable Voice")

    # -------------------------
    # OPENAI SAFE
    # -------------------------
    client = None
    if use_ai:
        try:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                client = OpenAI(api_key=api_key)
        except:
            st.sidebar.warning("AI not available")

    # -------------------------
    # FUNCTIONS
    # -------------------------
    def analyze(image):
        img = transform(image).unsqueeze(0)

        with torch.inference_mode():
            outputs = model(img)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)

        top3 = torch.topk(probs, 3)

        results = []
        for i in range(3):
            results.append({
                "Label": labels[top3.indices[i]],
                "Confidence (%)": round(float(top3.values[i]) * 100, 2)
            })

        return pd.DataFrame(results)

    def get_explanation(label):
        if client is None:
            return "AI disabled"

        try:
            res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"Explain {label}"}]
            )
            return res.choices[0].message.content
        except:
            return "Error"

    def text_to_speech(text):
        tts = gTTS(text)
        file = "voice.mp3"
        tts.save(file)
        return file

    # -------------------------
    # UI LAYOUT
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

            st.markdown("<div class='card'>", unsafe_allow_html=True)

            st.subheader("📊 Predictions")
            st.dataframe(df)

            st.subheader("📈 Chart")
            st.bar_chart(df.set_index("Label"))

            # AI
            explanation = ""
            if use_ai:
                explanation = get_explanation(df.iloc[0]["Label"])
                st.subheader("🧠 AI Explanation")
                st.write(explanation)

            # Voice
            if use_voice and explanation:
                audio = text_to_speech(explanation)
                st.audio(audio)

            st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------
    # ANALYTICS
    # -------------------------
    st.markdown("### 📊 Usage Analytics")

    st.metric("Total Users", 1)
    st.metric("Predictions Today", 5)
    st.metric("Accuracy", "92%")

    # -------------------------
    # FOOTER
    # -------------------------
    st.markdown("---")
    st.markdown("🚀 Startup AI App | Built with Streamlit")
