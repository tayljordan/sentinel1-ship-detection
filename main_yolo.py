from ultralytics import YOLO
import cv2

# Load the YOLOv8 model (ship detection model)
model = YOLO("/Users/jordantaylor/computer_imaging/models/ship.pt")

# Load an image for ship detection (replace with your image path)
image_path = "/Users/jordantaylor/computer_imaging/images/kotor_large.png"
image = cv2.imread(image_path)

# Run inference on the image
results = model(image)

# Print detected boxes and their corresponding information
for result in results:
    print(f"Detected {len(result.boxes)} objects:")
    for box in result.boxes:
        print(f"Class ID: {box.cls}, Confidence: {box.conf}, Bounding Box: {box.xyxy}")

# Display the results
for result in results:
    result.show()

# Save the result image with bounding boxes
results[0].save("output_image.jpg")
