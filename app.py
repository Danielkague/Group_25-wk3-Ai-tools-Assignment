import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import os

# Load the trained model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout(0.25)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(128)
        self.drop3 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(128 * 2 * 2, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.drop4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.drop5 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.drop1(x)

        x = F.relu(self.bn2(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.drop2(x)

        x = F.relu(self.bn3(self.conv5(x)))
        x = self.drop3(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.drop4(x)
        x = F.relu(self.fc2(x))
        x = self.drop5(x)
        x = self.fc3(x)
        return x

# Initialize model and load weights with error handling
@st.cache_resource
def load_model():
    model = CNNModel()
    try:
        state_dict = torch.load('mnist_cnn_model.pth', map_location=torch.device('cpu'))
        print("Available keys in state dict:", state_dict.keys())
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

# Define image transformation with improved preprocessing
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),  # Ensure single channel
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def preprocess_image(image):
    # Convert to grayscale if not already
    if image.mode != 'L':
        image = image.convert('L')
    
    # Invert colors if the digit is white on black background
    image_array = np.array(image)
    if np.mean(image_array) > 127:
        image = Image.fromarray(255 - image_array)
    
    # Apply transformations
    img_tensor = transform(image)
    return img_tensor

# Streamlit app
st.title('MNIST Digit Prediction')
st.write('Upload an image of a handwritten digit (0-9) to predict its value.')

uploaded_file = st.file_uploader('Choose an image...', type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    try:
        # Load and preprocess image
        image = Image.open(uploaded_file)
        
        # Display original image
        st.image(image, caption='Uploaded Image', width=200)
        
        # Preprocess the image
        img_tensor = preprocess_image(image).unsqueeze(0)
        
        if model is not None:
            # Make prediction
            with torch.no_grad():
                output = model(img_tensor)
                pred = torch.argmax(output, 1).item()
                confidence = torch.softmax(output, 1).max().item()
            
            st.write(f'Predicted Digit: {pred}')
            st.write(f'Confidence: {confidence:.4f}')
            
            # Display confidence scores for all digits
            probs = torch.softmax(output, 1)[0].numpy()
            st.write('Confidence scores for all digits:')
            for i, prob in enumerate(probs):
                st.write(f'Digit {i}: {prob:.4f}')
            
            # Add warning if confidence is low
            if confidence < 0.95:
                st.warning('Warning: The model is not very confident in its prediction. Please ensure the image is clear and the digit is centered.')
        else:
            st.error("Model not loaded properly. Please check the model file.")
            
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.write("Please make sure you've uploaded a valid image file.") 