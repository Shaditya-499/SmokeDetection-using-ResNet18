import torch
from torchvision import models, transforms
from PIL import Image
import cv2

# Define the model architecture
model = models.resnet18(pretrained=False)  # Assuming you're using ResNet-18 for smoke detection

# Load the model state_dict
model.load_state_dict(torch.load('C:/Users/shadi/TFIC/models/smoke_detection.pth', map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, change to 1, 2, etc., for other cameras

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Load and preprocess the unseen image
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if the frame was captured successfully
    if not ret:
        print("Error: Could not read frame.")
        break

    # Preprocess the frame
    input_tensor = preprocess(frame)
    input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(input_batch)

    # Get the predicted class
    _, predicted_class = output.max(1)

    # Map the predicted class to the class name
    class_names = ['NO_smoke', 'Smoke']  # Make sure these class names match your training data
    predicted_class_name = class_names[predicted_class.item()]

    # Display the predicted class on the frame
    cv2.putText(frame, predicted_class_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
