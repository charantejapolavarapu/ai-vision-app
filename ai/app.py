import streamlit as st
import torch
from torchvision import models
from PIL import Image
import pandas as pd
import os
from gtts import gTTS
from datetime import datetime

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
# UI STYLE
# -------------------------
st.markdown("""
<style>
body {background-color:#0e1117;}
.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
}
.title {
    text-align:center;
    font-size:40px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# LOGIN
# -------------------------
if st.session_state.user is None:
    st.markdown("<div class='title'>🔐 Login</div>", unsafe_allow_html=True)

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
# MAIN
# -------------------------
else:

    st.markdown("<div class='title'>🚀 Image Recognition Pro+</div>", unsafe_allow_html=True)
    st.write(f"Welcome **{st.session_state.user}**")

    if st.button("Logout"):
        st.session_state.user = None
        st.rerun()

    # MODEL
    @st.cache_resource
    def load_model():
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
        model.eval()
        return model, weights

    model, weights = load_model()
    transform = weights.transforms()
    labels = weights.meta["categories"]

    # SIDEBAR
    st.sidebar.title("⚙️ Options")
    language = st.sidebar.selectbox("Language", ["English", "Telugu"])
    use_voice = st.sidebar.toggle("Voice")

    # FUNCTIONS
    def analyze(image):
        image = image.convert("RGB")
        img = transform(image).unsqueeze(0)

        with torch.inference_mode():
            outputs = model(img)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)

        top5 = torch.topk(probs, 5)

        results = []
        for i in range(5):
            results.append({
                "Label": labels[top5.indices[i]],
                "Confidence": round(float(top5.values[i])*100,2)
            })

        return pd.DataFrame(results)

    def explanation(label):
        if language == "Telugu":
            return f"ఈ చిత్రం {label} గా గుర్తించబడింది. ఇది ఆకారం మరియు రంగుల ఆధారంగా గుర్తించబడింది."
        return f"This image is classified as {label} based on visual patterns."

    def speak(text):
        tts = gTTS(text, lang='te' if language=="Telugu" else 'en')
        tts.save("voice.mp3")
        return "voice.mp3"

    # TABS
    tab1, tab2, tab3 = st.tabs(["🔍 Analyze", "📊 Analytics", "🧾 History"])

    # -------------------------
    # TAB 1: ANALYZE
    # -------------------------
    with tab1:

        st.markdown("### Upload or Capture Image")

        file = st.file_uploader("Upload", type=["jpg","png"])
        cam = st.camera_input("Or Take Photo")

        if file or cam:

            image = Image.open(file if file else cam)

            col1, col2 = st.columns(2)

            with col1:
                st.image(image, use_container_width=True)

            with col2:
                df = analyze(image)
                pred = df.iloc[0]["Label"]

                st.success(f"🎯 Predicted: {pred}")

                # PROGRESS BARS
                for i,row in df.iterrows():
                    st.write(row["Label"])
                    st.progress(int(row["Confidence"]))

                st.dataframe(df)

                exp = explanation(pred)
                st.write("🧠", exp)

                if use_voice:
                    st.audio(speak(exp))

                # SAVE HISTORY
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

    # -------------------------
    # TAB 3: HISTORY
    # -------------------------
    with tab3:

        if st.session_state.history:
            df = pd.DataFrame(st.session_state.history)

            search = st.text_input("Search")
            if search:
                df = df[df["prediction"].str.contains(search)]

            st.dataframe(df)

            # DOWNLOAD
            csv = df.to_csv(index=False)
            st.download_button("Download CSV", csv)

        else:
            st.write("No history yet")

    st.markdown("---")
    st.markdown("🔥 Advanced AI Project")
