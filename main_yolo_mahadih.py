from ultralytics import YOLO
import cv2
import os

model_path = os.path.join(os.getcwd(), "models", "ship.pt")
image_path = os.path.join(os.getcwd(), "images", "singapore_jurong.png")
output_path = os.path.join(os.getcwd(), "output", "output_yolo_mahadih.jpg")

# Load the YOLO model
model = YOLO(model_path)

# Load an image for ship detection (replace with your image path)
image = cv2.imread(image_path)

# Run inference on the image with a lower confidence threshold (e.g., 0.3)
results = model(image, conf=0.05)  # Lowering confidence threshold

# Print detected boxes and their corresponding information
for result in results:
    print(f"Detected {len(result.boxes)} objects:")
    for box in result.boxes:
        print(f"Class ID: {box.cls}, Confidence: {box.conf}, Bounding Box: {box.xyxy}")

# Display the results
for result in results:
    result.show()

# Save the result image with bounding boxes
results[0].save(output_path)
