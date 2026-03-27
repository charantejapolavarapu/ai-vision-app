import torch
from torchvision import models
from transformers import CLIPProcessor, CLIPModel
from openai import OpenAI

device = "cuda" if torch.cuda.is_available() else "cpu"

# EfficientNet
weights = models.EfficientNet_B0_Weights.DEFAULT
model = models.efficientnet_b0(weights=weights)
model.eval().to(device)
transform = weights.transforms()
labels = weights.meta["categories"]

# CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# OpenAI
client = OpenAI(api_key="YOUR_OPENAI_API_KEY")

def classify(image):
    img = transform(image).unsqueeze(0).to(device)

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

    return results

def clip_analysis(image):
    texts = ["a dog", "a cat", "a car", "a person", "a tree"]

    inputs = clip_processor(text=texts, images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)

    probs = outputs.logits_per_image.softmax(dim=1)
    idx = probs.argmax().item()

    return texts[idx], float(probs[0][idx]) * 100

def explain(predictions):
    top = predictions[0]["label"]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Explain why this image is {top}"}]
    )

    return response.choices[0].message.content