import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# Original CNN Model - keeping exact architecture from your working code
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

def create_prediction_chart(probabilities):
    """Create a bar chart showing prediction probabilities"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    digits = list(range(10))
    colors = ['red' if i == np.argmax(probabilities) else 'skyblue' for i in digits]
    
    bars = ax.bar(digits, probabilities, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax.set_xlabel('Digit', fontsize=14, fontweight='bold')
    ax.set_ylabel('Confidence Score', fontsize=14, fontweight='bold')
    ax.set_title('MNIST Digit Classification - Prediction Confidence for Each Digit (0-9)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(digits)
    ax.set_ylim(0, max(probabilities) * 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Highlight the predicted digit
    predicted_digit = np.argmax(probabilities)
    ax.text(predicted_digit, probabilities[predicted_digit] + max(probabilities) * 0.03, 
            '← PREDICTED', ha='left', va='bottom', 
            fontweight='bold', color='red', fontsize=14, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Add horizontal line at 10% for reference
    ax.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='10% threshold')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    return fig

# Streamlit app
st.title('🔢 MNIST Digit Classifier')
st.markdown('Upload an image of a handwritten digit (0-9) to get accurate predictions!')

if model is None:
    st.error("⚠️ Model not loaded. Please ensure 'mnist_cnn_model.pth' is in the current directory.")
    st.stop()

st.markdown("---")

# File upload section
st.markdown("### 📁 Upload Handwritten Digit Image")
st.info("💡 For best results, upload clear images with dark digits on light backgrounds")

uploaded_file = st.file_uploader(
    'Choose an image file...', 
    type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
    help="Supported formats: PNG, JPG, JPEG, BMP, TIFF"
)

if uploaded_file is not None:
    try:
        # Load and preprocess image
        image = Image.open(uploaded_file)
        
        # Create three columns for better layout
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            st.markdown("#### 📸 Original Image")
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Show image info
            st.markdown("**Image Details:**")
            st.write(f"- **Size:** {image.size[0]} × {image.size[1]} pixels")
            st.write(f"- **Mode:** {image.mode}")
            st.write(f"- **Format:** {image.format}")
        
        with col2:
            st.markdown("#### 🔄 Preprocessed Image")
            
            # Preprocess the image
            img_tensor = preprocess_image(image).unsqueeze(0)
            
            # Show preprocessed image
            processed_img = img_tensor.squeeze().numpy()
            # Denormalize for display
            processed_img = (processed_img * 0.3081) + 0.1307
            processed_img = np.clip(processed_img, 0, 1)
            
            fig_img, ax_img = plt.subplots(figsize=(4, 4))
            ax_img.imshow(processed_img, cmap='gray')
            ax_img.set_title('28×28 Processed Image', fontweight='bold')
            ax_img.axis('off')
            st.pyplot(fig_img)
            plt.close(fig_img)
        
        with col3:
            st.markdown("#### 🎯 Prediction Results")
            
            # Make prediction
            with torch.no_grad():
                output = model(img_tensor)
                pred = torch.argmax(output, 1).item()
                confidence = torch.softmax(output, 1).max().item()
                probabilities = torch.softmax(output, 1)[0].numpy()
            
            # Create prediction display with styling
            prediction_html = f"""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 10px 0;">
                <h2 style="color: #1f77b4; margin: 0;">Predicted Digit: <span style="color: #d62728; font-size: 2em;">{pred}</span></h2>
                <h3 style="color: #2ca02c; margin: 10px 0 0 0;">Confidence: {confidence:.4f} ({confidence:.2%})</h3>
            </div>
            """
            st.markdown(prediction_html, unsafe_allow_html=True)
            
            # Confidence indicator
            if confidence >= 0.95:
                st.success('✅ **Excellent prediction!** Very high confidence.')
            elif confidence >= 0.80:
                st.info('ℹ️ **Good prediction.** High confidence.')
            elif confidence >= 0.60:
                st.warning('⚠️ **Moderate prediction.** Consider using a clearer image.')
            else:
                st.error('❌ **Low confidence prediction.** Please try a different image.')
        
        # Full-width prediction chart
        st.markdown("---")
        st.markdown("### 📊 Detailed Prediction Analysis")
        
        # Create and display prediction chart
        fig_chart = create_prediction_chart(probabilities)
        st.pyplot(fig_chart)
        plt.close(fig_chart)
        
        # Detailed probabilities table
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📋 All Digit Probabilities")
            for i, prob in enumerate(probabilities):
                emoji = "🎯" if i == pred else "  "
                st.write(f'{emoji} **Digit {i}:** {prob:.6f} ({prob:.2%})')
        
        with col2:
            st.markdown("#### 📈 Top 3 Predictions")
            # Get top 3 predictions
            top_3_indices = np.argsort(probabilities)[-3:][::-1]
            for rank, idx in enumerate(top_3_indices, 1):
                medal = ["🥇", "🥈", "🥉"][rank-1]
                st.write(f'{medal} **Digit {idx}:** {probabilities[idx]:.6f} ({probabilities[idx]:.2%})')
        
        # Model certainty analysis
        st.markdown("---")
        st.markdown("#### 🧠 Model Analysis")
        
        # Calculate entropy for uncertainty measure
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        max_entropy = np.log(10)  # Maximum entropy for 10 classes
        uncertainty = entropy / max_entropy
        
        analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
        
        with analysis_col1:
            st.metric("Prediction Confidence", f"{confidence:.2%}", 
                     delta=f"{(confidence-0.1)*100:.1f}%" if confidence > 0.1 else None)
        
        with analysis_col2:
            st.metric("Model Uncertainty", f"{uncertainty:.2%}",
                     delta=f"{(0.5-uncertainty)*100:.1f}%" if uncertainty < 0.5 else None)
        
        with analysis_col3:
            second_best = np.sort(probabilities)[-2]
            margin = confidence - second_best
            st.metric("Prediction Margin", f"{margin:.4f}",
                     delta="High" if margin > 0.5 else "Low")
        
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        st.write("Please make sure you've uploaded a valid image file.")
        st.write("**Troubleshooting tips:**")
        st.write("- Ensure the image file is not corrupted")
        st.write("- Try a different image format (PNG recommended)")
        st.write("- Make sure the image contains a clear digit")

else:
    # Show sample images when no file is uploaded
    st.markdown("### 📝 Sample Usage")
    st.info("Upload an image above to see the prediction results!")
    
    # Instructions
    st.markdown("#### 🎯 Tips for Best Results:")
    tips_html = """
    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #17a2b8;">
        <ul style="margin: 0; padding-left: 20px;">
            <li><strong>Image Quality:</strong> Use clear, high-resolution images</li>
            <li><strong>Contrast:</strong> Dark digits on light backgrounds work best</li>
            <li><strong>Centering:</strong> Ensure the digit is centered in the image</li>
            <li><strong>Size:</strong> The digit should fill most of the image space</li>
            <li><strong>Clarity:</strong> Avoid blurry or distorted images</li>
            <li><strong>Background:</strong> Clean background without noise</li>
        </ul>
    </div>
    """
    st.markdown(tips_html, unsafe_allow_html=True)

# Model information
st.markdown("---")
with st.expander("ℹ️ Model Architecture & Information"):
    st.markdown("""
    ### 🏗️ CNN Model Architecture
    
    **Convolutional Layers:**
    - **Block 1:** Conv2d(1→32, 3×3) → BatchNorm → ReLU → Conv2d(32→32, 3×3) → ReLU → MaxPool(2×2) → Dropout(0.25)
    - **Block 2:** Conv2d(32→64, 3×3) → BatchNorm → ReLU → Conv2d(64→64, 3×3) → ReLU → MaxPool(2×2) → Dropout(0.25)  
    - **Block 3:** Conv2d(64→128, 3×3) → BatchNorm → ReLU → Dropout(0.25)
    
    **Fully Connected Layers:**
    - **FC1:** Linear(512) → BatchNorm → ReLU → Dropout(0.5)
    - **FC2:** Linear(256) → ReLU → Dropout(0.5)
    - **Output:** Linear(10) → Softmax
    
    ### 📊 Model Details
    - **Input Size:** 28×28 grayscale images
    - **Normalization:** Mean=0.1307, Std=0.3081 (MNIST statistics)
    - **Output Classes:** 10 digits (0-9)
    - **Parameters:** ~500K trainable parameters
    - **Training Dataset:** MNIST (60,000 training images)
    
    ### 🔧 Preprocessing Pipeline
    1. Convert to grayscale (if needed)
    2. Invert colors (white digits → black digits)  
    3. Resize to 28×28 pixels
    4. Convert to tensor
    5. Normalize with MNIST statistics
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; font-size: 14px; margin-top: 20px;">
        🤖 MNIST Digit Classifier | Built with Streamlit & PyTorch
    </div>
    """, 
    unsafe_allow_html=True
)