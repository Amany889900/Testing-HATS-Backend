import os
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Path to your converted model
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "onnx_model")

# Load model and tokenizer globally
# This prevents Vercel from reloading them on every single request
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
session = ort.InferenceSession(os.path.join(MODEL_DIR, "model.onnx"))

class TextRequest(BaseModel):
    text: str

@app.get("/")
def health_check():
    return {"status": "AI Detector is Online 🚀"}

@app.post("/predict")
def predict(request: TextRequest):
    # 1. Tokenize
    inputs = tokenizer(request.text, return_tensors="np", truncation=True, max_length=512)
    
    # 2. Run Inference
    onnx_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64)
    }
    logits = session.run(None, onnx_inputs)[0]
    
    # 3. Process Results (Softmax)
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
    prediction = np.argmax(probs, axis=-1)[0]

    return {
        "prediction": "AI-Generated" if prediction == 1 else "Human-Written",
        "confidence": round(float(probs[0][prediction]), 4)
    }