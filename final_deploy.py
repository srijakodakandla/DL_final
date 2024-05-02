import streamlit as st
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# Function to preprocess the uploaded image
def transform_image(uploaded_file):
    with Image.open(uploaded_file) as img:
        # Resizing the image to 64x64
        img = img.resize((64, 64))
        # Normalize pixel values to be between 0 and 1
        img_array = np.array(img) / 255.0

    return img_array

# Defining VGG-13 CNN architecture
class VGG(nn.Module):
    def __init__(self, num_classes=43):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Creating a VGG model instance
vgg13_opt_model = VGG(num_classes=43)
vgg13_opt_model.load_state_dict(torch.load('best_vgg13_opt_model.pth', map_location=torch.device('cpu')))
vgg13_opt_model.eval()

# label mapping
num_classes = ['Speed limit (20km/h)', 'Speed limit (30km/h)', 'No passing for vehicles over 3.5 metric tons', 'Right-of-way at the next intersection',
               'Priority road', 'Yield', 'Stop', 'No vehicles', 'Vehicles over 3.5 metric tons prohibited', 'No entry', 'General caution',
               'Dangerous curve to the left', 'Speed limit (50km/h)', 'Dangerous curve to the right', 'Double curve', 'Bumpy road', 'Slippery road',
               'Road narrows on the right', 'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing',
               'Speed limit (60km/h)', 'Beware of ice/snow', 'Wild animals crossing', 'End of all speed and passing limits', 'Turn right ahead',
               'Turn left ahead', 'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right', 'Keep left', 'Speed limit (70km/h)',
               'Roundabout mandatory', 'End of no passing', 'End of no passing by vehicles over 3.5 metric tons', 'Speed limit (80km/h)',
               'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)','No passing']




# Streamlit web interface
st.markdown("<h1 style='text-align: center; color: white;'>Interactive CNN Image Classifier</h1>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose an image:", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', width=80)

    if st.button('Predict'):
        # Preprocess the uploaded image
        image = transform_image(uploaded_file)
        image_tensor = torch.tensor(image.transpose((2, 0, 1)), dtype=torch.float32).unsqueeze(0)

        # Perform prediction
        with torch.no_grad():
            output = vgg13_opt_model(image_tensor)
            probs = F.softmax(output, dim=1)
            pred_class_idx = torch.argmax(probs, dim=1).item()
            pred_class = num_classes[pred_class_idx]

        st.write(f'Predicted Class: {pred_class}')

