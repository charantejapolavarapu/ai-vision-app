from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from model_utils import classify, clip_analysis, explain
from database import insert_prediction, get_history
from auth import register, login

app = FastAPI()

# CORS FIX
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# AUTH
# -------------------------
@app.post("/register/")
def register_user(username: str, password: str):
    return register(username, password)

@app.post("/login/")
def login_user(username: str, password: str):
    return login(username, password)

# -------------------------
# PREDICTION
# -------------------------
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    preds = classify(image)
    explanation = explain(preds)

    for p in preds:
        insert_prediction(p["label"], p["confidence"])

    return {
        "predictions": preds,
        "explanation": explanation
    }

# -------------------------
# DASHBOARD
# -------------------------
@app.get("/dashboard/")
def dashboard():
    data = get_history()

    total = len(data)
    avg = sum([d[1] for d in data]) / total if total else 0

    return {
        "total_predictions": total,
        "avg_confidence": avg,
        "history": data
    }