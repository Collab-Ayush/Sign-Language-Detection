import cv2
import torch
import torch.nn as nn
import numpy as np

# 1. Redefine the exact same CNN architecture
class SignLanguageCNN(nn.Module):
    def __init__(self):
        super(SignLanguageCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 26)

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# 2. Load the trained weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SignLanguageCNN()
model.load_state_dict(torch.load('sign_mnist_cnn.pth', map_location=device))
model.to(device)
model.eval()

# 3. Define the alphabet mapping 
# (Sign MNIST skips 'J' and 'Z' due to motion, but we map the indices 0-25 directly)
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
letter_map = {i: letter for i, letter in enumerate(alphabet)}

# 4. Start the Webcam
cap = cv2.VideoCapture(0)

print("Starting live detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Flip the frame so it acts like a mirror
    frame = cv2.flip(frame, 1)
    
    # Define the Region of Interest (ROI) coordinates (X, Y, Width, Height)
    roi_x, roi_y, roi_w, roi_h = 300, 100, 250, 250
    
    # Draw a rectangle on the main frame to guide the user
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
    cv2.putText(frame, "Place Hand Here", (roi_x, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Extract the ROI from the frame
    roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    
    # Preprocess the ROI for the model
    # 1. Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 2. Add Gaussian Blur to smooth out camera noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Resize to 28x28 exactly like the training data
    resized = cv2.resize(blurred, (28, 28))
    
    # 4. Normalize pixel values (0 to 1) and convert to PyTorch tensor
    normalized = resized / 255.0
    tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    # Make the prediction
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_idx = predicted.item()
        
        # Get the corresponding letter
        predicted_letter = letter_map.get(predicted_idx, "?")
        
    # Display the prediction on the screen
    cv2.putText(frame, f"Prediction: {predicted_letter}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
    
    # Show the main frame
    cv2.imshow("Sign Language Translator", frame)
    
    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()