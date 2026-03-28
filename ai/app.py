import streamlit as st
import torch
from torchvision import models
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
import os

# -------------------------
# OPENAI
# -------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------
# MODEL
# -------------------------
weights = models.MobileNet_V2_Weights.DEFAULT
model = models.mobilenet_v2(weights=weights)
model.eval()

transform = weights.transforms()
labels = weights.meta["categories"]

# -------------------------
# APP UI
# -------------------------
st.set_page_config(page_title="AI Vision App", layout="wide")

st.title("🚀 AI Vision App")
st.subheader("Image Recognition + ChatGPT Explanation")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

# -------------------------
# FUNCTION
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
            "label": labels[top3.indices[i]],
            "confidence": float(top3.values[i]) * 100
        })

    return pd.DataFrame(results)

# -------------------------
# CHATGPT
# -------------------------
def get_explanation(label):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": f"Explain why this image is {label}"}
            ]
        )
        return response.choices[0].message.content
    except:
        return "AI explanation unavailable"

# -------------------------
# MAIN
# -------------------------
if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    df = analyze(image)

    st.subheader("📊 Predictions")
    st.dataframe(df)

    # Chart
    st.bar_chart(df.set_index("label"))

    # Explanation
    explanation = get_explanation(df.iloc[0]["label"])
    st.subheader("🧠 AI Explanation")
    st.write(explanation)