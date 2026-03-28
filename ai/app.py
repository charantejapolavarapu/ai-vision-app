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
# SESSION STORAGE
# -------------------------
if "user" not in st.session_state:
    st.session_state.user = None

if "users" not in st.session_state:
    st.session_state.users = {}

# -------------------------
# PREMIUM UI CSS
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
    font-size: 38px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# LOGIN / REGISTER
# -------------------------
if st.session_state.user is None:

    st.markdown("<div class='title'>🔐 Image Recognition Login</div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Login", "Register"])

    # LOGIN
    with tab1:
        l_user = st.text_input("Username", key="login_user")
        l_pass = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login"):
            if l_user in st.session_state.users and st.session_state.users[l_user] == l_pass:
                st.session_state.user = l_user
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid credentials")

    # REGISTER
    with tab2:
        r_user = st.text_input("New Username", key="reg_user")
        r_pass = st.text_input("New Password", type="password", key="reg_pass")

        if st.button("Register"):
            if r_user and r_pass:
                st.session_state.users[r_user] = r_pass
                st.success("Registered successfully")
            else:
                st.error("Fill all fields")

# -------------------------
# MAIN APP
# -------------------------
else:

    st.markdown(f"<div class='title'>🧠 Image Recognition Pro</div>", unsafe_allow_html=True)
    st.write(f"Welcome, **{st.session_state.user}** 👋")

    if st.button("Logout"):
        st.session_state.user = None
        st.rerun()

    # -------------------------
    # MODEL LOAD
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
    use_ai = st.sidebar.toggle("Enable AI Explanation")
    use_voice = st.sidebar.toggle("Enable Voice Output")

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
            else:
                st.sidebar.warning("No API key")
        except:
            st.sidebar.warning("OpenAI not available")

    # -------------------------
    # FUNCTIONS
    # -------------------------
    def analyze(image):
        # 🔥 FIXED BUG
        image = image.convert("RGB")

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
                messages=[{"role": "user", "content": f"Explain {label} in simple terms"}]
            )
            return res.choices[0].message.content
        except:
            return "Error generating explanation"

    def text_to_speech(text):
        tts = gTTS(text)
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

            st.markdown("<div class='card'>", unsafe_allow_html=True)

            st.subheader("📊 Predictions")
            st.dataframe(df)

            st.subheader("📈 Chart")
            st.bar_chart(df.set_index("Label"))

            explanation = ""

            if use_ai:
                explanation = get_explanation(df.iloc[0]["Label"])
                st.subheader("🧠 AI Explanation")
                st.write(explanation)

            if use_voice and explanation:
                audio = text_to_speech(explanation)
                st.audio(audio)

            st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------
    # ANALYTICS
    # -------------------------
    st.markdown("### 📊 Dashboard")

    st.metric("Active User", st.session_state.user)
    st.metric("Models Used", "MobileNetV2")
    st.metric("Status", "Running ✅")

    st.markdown("---")
    st.markdown("🚀 Built with Streamlit | AI Vision App")
