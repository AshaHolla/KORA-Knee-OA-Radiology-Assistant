import torch
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st
import custom_densenets

import torch.nn as nn
import torch.nn.functional as F

# Load the model
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

model = custom_densenets.se_densenet121_model(5)
model.load_state_dict(torch.load('models/SE_DenseNet121_42.ckpt', map_location=torch.device(device)))


# Set the model to evaluation mode
model.eval()

# Define the preprocessing steps for the input image
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ColorJitter(brightness=0),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

grad_model = nn.Sequential(*list(model.children())[:-1], nn.Identity())

# Set the final layer's activation function to None
grad_model.eval()
for param in grad_model.parameters():
    param.requires_grad = False
    
# Define the function to make predictions
def predict(image):
    # Preprocess the input image
    input_tensor = transform(image)
    #gray_tensor = F.rgb_to_grayscale(input_tensor)
    # Add a batch dimension to the input tensor
    input_tensor = input_tensor.unsqueeze(0)

    # Make predictions
    with torch.no_grad():
        output = model(input_tensor)

    # Convert output to probabilities using softmax function
    probabilities = torch.nn.functional.softmax(output, dim=1)

    # Get the predicted class
    _, predicted_class = torch.max(probabilities, dim=1)

    # Return the predicted class
    return predicted_class.item()


# Define the Streamlit app
st.title("Intelligent Radiology Assistant")
st.write("Severity Analysis of Knee Osteoarthritis")
icon = Image.open("C:/Users/asha/Desktop/FYP/monka.png")
# Add a file uploader widget
uploaded_file = st.file_uploader("Choose X-Ray Image", type="png")

with st.sidebar:
    st.image(icon)
    st.subheader("Get A KL Grade Classification of MRI Scans Instantly")
    st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e7bcf,#2e7bcf);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )

    
    
    
# Make predictions when the user uploads an image
if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    
    
    # Display the image
    st.image(image, caption="Uploaded Image", use_column_width=True, width=400)

    # Make predictions
    prediction = predict(image)

    # Display the predicted class
    st.write(f"Prediction is {prediction}",style={"font-size": "40px"})
    #heatmap = make_gradcam_heatmap(grad_model, img_tensor)
    #save_and_display_gradcam(image, heatmap)