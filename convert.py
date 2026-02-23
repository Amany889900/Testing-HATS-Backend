from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

model_id = "abhi099k/ai-text-detector-L0"
save_dir = "onnx_model"

# This one line handles: 
# 1. Downloading the PyTorch model
# 2. Converting it to ONNX (handling LayerNorm/Opset automatically)
# 3. Validating the conversion
model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Save the ONNX model and tokenizer to your folder
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"Done! Your Vercel-ready model is in the '{save_dir}' folder.")