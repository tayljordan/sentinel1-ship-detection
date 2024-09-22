from transformers import AutoModelForImageClassification, AutoProcessor
from PIL import Image
import torch
import os

# Load the processor and model
model_name = "dima806/vessel_ship_types_image_detection"
model = AutoModelForImageClassification.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)

# Load an image
image_path = os.path.join(os.getcwd(), "objects", "military.png")
image = Image.open(image_path).convert("RGB")  # Convert to RGB

# Resize the image
image = image.resize((224, 224))  # Example resize to 224x224

# Preprocess the image
inputs = processor(images=image, return_tensors="pt")

# Make a prediction
with torch.no_grad():
    outputs = model(**inputs)

# Get the predicted class
predicted_class = outputs.logits.argmax(-1).item()

'''
    "0": "Cargo",
    "1": "Carrier",
    "2": "Cruise",
    "3": "Military",
    "4": "Tankers"
'''

print(f"Predicted class ID: {predicted_class}")
