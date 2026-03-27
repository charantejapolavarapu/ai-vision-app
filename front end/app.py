import gradio as gr
import requests
import pandas as pd
import matplotlib.pyplot as plt

API_URL = "https://YOUR-BACKEND-URL"

history = []

def analyze(image):
    files = {"file": ("image.png", image, "image/png")}
    res = requests.post(f"{API_URL}/predict/", files=files).json()

    preds = res["predictions"]
    explanation = res["explanation"]

    df = pd.DataFrame(preds)

    # Chart
    labels = df["label"]
    conf = df["confidence"]

    plt.figure()
    plt.barh(labels, conf)
    plt.xlabel("Confidence")

    chart_path = "chart.png"
    plt.savefig(chart_path)
    plt.close()

    history.extend(preds)
    hist_df = pd.DataFrame(history)

    return df, explanation, chart_path, hist_df

def load_dashboard():
    res = requests.get(f"{API_URL}/dashboard/").json()

    total = res["total_predictions"]
    avg = res["avg_confidence"]

    return total, avg

with gr.Blocks() as demo:
    gr.Markdown("# 🧠 AI Vision SaaS")

    with gr.Row():
        img = gr.Image(source="webcam", type="pil")
        btn = gr.Button("Analyze")

    output = gr.Dataframe(label="Predictions")
    explanation = gr.Textbox(label="AI Explanation")
    chart = gr.Image(label="Chart")
    history_table = gr.Dataframe(label="History")

    btn.click(analyze, inputs=img, outputs=[output, explanation, chart, history_table])

    gr.Markdown("## 📊 Dashboard")

    total = gr.Number(label="Total Predictions")
    avg = gr.Number(label="Average Confidence")

    dash_btn = gr.Button("Refresh Dashboard")
    dash_btn.click(load_dashboard, outputs=[total, avg])

demo.launch(server_name="0.0.0.0", server_port=7860)