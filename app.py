import torch
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st
import custom_densenets
import torchvision.transforms.functional as F
import base64
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

icon = Image.open("images/logo.png")
st.set_page_config(layout = "wide",
    page_title="KORA",
    page_icon=icon
)
with st.sidebar:
    st.image(icon)
    ori = '<p></p>'
    st.markdown(ori, unsafe_allow_html=True)
    st.markdown(ori, unsafe_allow_html=True)
    st.markdown(ori, unsafe_allow_html=True)
    st.markdown(ori, unsafe_allow_html=True)
    st.markdown(ori, unsafe_allow_html=True)
    st.markdown(ori, unsafe_allow_html=True)
    
    st.subheader("Upload image")
    uploaded_file = st.file_uploader("Choose x-ray image")
    st.sidebar.write("")



# Add a file uploader widget
#uploaded_file = st.file_uploader("Choose an image...", type="png")
break1 = '<br>'

# Make predictions when the user uploads an image
if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    original_title2 = '<p style="font-family: Franklin Gothic Medium; color:White; font-size: 50px; text-align: center">KORA Diagnosis - Severity Analysis </p>'
    st.markdown(original_title2, unsafe_allow_html=True)
    st.markdown(ori, unsafe_allow_html=True)
    st.markdown(ori, unsafe_allow_html=True)
    # Display the image
    #col1, col2, col3 = st.columns(3)
    #with col1:
    #    st.write(' ')
    #with col2:
    #    st.image(image, caption="Uploaded MRI scan", width=400)
    #with col3:
    #    st.write(' ')

#    Classification = ["Normal","Doubtful","Mild","Moderate","Severe"]
#    Description = ["No features of OA","Minute Osteophyte : Doubtful Significance","Definite Osteophyte : Normal Joint Space","Moderate Joint Space reduction","Joint space greatly reduced : Subchondral Sclerosis"]

    # Make predictions
    col11,col12 = st.columns(2,gap="medium")
    prediction = predict(image)
    st.markdown(ori, unsafe_allow_html=True)
    # Display the predicted class
    if(prediction == 0):
        with col11:
            st.write('<p style="font-family: Georgia; color:White; font-size: 30px; text-align: justify">Uploaded Image</p>', unsafe_allow_html=True)
            st.markdown(ori, unsafe_allow_html=True)
            st.image(image, use_column_width=True)
            st.markdown(ori, unsafe_allow_html=True)
            st.write('<div align="center"; border: 10px white;><table><tr style=font-size:20px;font-family:Georgia;text-align: center><td style="text-align: center" colspan="2" >RESULTS</td></tr><tr style=font-size:20px;font-family:Georgia;text-align: justify><td>KL Grade</td><td>0</td></tr><tr style=font-size:20px;font-family:Georgia; text-align: justify><td>Classification</td><td>Normal</td></tr><tr style=font-size:20px;font-family:Georgia; text-align: justify><td>Description</td><td>No features of OA : Healthy Knee Joint</td></tr></table></div>', unsafe_allow_html=True)
        with col12:    
            original5 = '<p style="font-family: Georgia; color:White; font-size: 30px; text-align: justify">Download your report</p>'
            st.markdown(original5, unsafe_allow_html=True)
            st.markdown(ori, unsafe_allow_html=True)
            with open("reports/grade_0 .pdf","rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="500" height="690" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)

    if(prediction == 1):
        with col11:
            st.write('<p style="font-family: Georgia; color:White; font-size: 30px; text-align: justify">Uploaded Image</p>', unsafe_allow_html=True)
            st.markdown(ori, unsafe_allow_html=True)
            st.image(image, use_column_width=True)
            st.markdown(ori, unsafe_allow_html=True)
            st.write('<div align="center"; border: 10px white;><table><tr style=font-size:20px;font-family:Georgia;text-align: justify><td>KL Grade</td><td>1</td></tr><tr style=font-size:20px;font-family:Georgia;text-align: justify><td>Classification</td><td>Doubtful</td></tr><tr style=font-size:20px;font-family:Georgia;text-align: justify><td>Description</td><td>Minute Osteophyte : Doubtful Significance</td></tr></table></div>', unsafe_allow_html=True)
        with col12: 
            original4 = '<p style="font-family: Georgia; color:White; font-size: 30px; text-align: left">Download your report</p>'
            st.markdown(original4, unsafe_allow_html=True)
            st.markdown(ori, unsafe_allow_html=True)
            with open("reports/grade_1.pdf","rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="500" height="690" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)

    if(prediction == 2):
        with col11:
            st.write('<p style="font-family: Georgia; color:White; font-size: 30px; text-align: justify">Uploaded Image</p>', unsafe_allow_html=True)
            st.markdown(ori, unsafe_allow_html=True)
            st.image(image, use_column_width=True)
            st.markdown(ori, unsafe_allow_html=True)
            st.write('<div align="center"; border: 10px white;><table><tr style=font-size:20px;font-family:Georgia;text-align: justify><td>KL Grade</td><td>2</td></tr><tr style=font-size:20px;font-family:Georgia;text-align: justify><td>Classification</td><td>Mild</td></tr><tr style=font-size:20px;font-family:Georgia;text-align: justify><td>Description</td><td>Definite Osteophyte : Normal Joint Space</td></tr></table></div>', unsafe_allow_html=True)
        with col12: 
            original3 = '<p style="font-family: Georgia; color:White; font-size: 30px; text-align: left">Download your report</p>'
            st.markdown(original3, unsafe_allow_html=True)
            st.markdown(ori, unsafe_allow_html=True)
            with open("reports/grade_2.pdf","rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="500" height="690" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)

    if(prediction == 3):
        with col11:
            st.write('<p style="font-family: Georgia; color:White; font-size: 30px; text-align: justify">Uploaded Image</p>', unsafe_allow_html=True)
            st.markdown(ori, unsafe_allow_html=True)
            st.image(image, use_column_width=True)
            st.markdown(ori, unsafe_allow_html=True)
            st.write('<div align="center"; border: 10px white;><table><tr style=font-size:20px;font-family:Georgia;text-align: justify><td>KL Grade</td><td>3</td></tr><tr style=font-size:20px;font-family:Georgia;text-align: justify><td>Classification</td><td>Moderate</td></tr><tr style=font-size:20px;font-family:Georgia;text-align: justify><td>Description</td><td>Moderate Joint Space reduction</td></tr></table></div>', unsafe_allow_html=True)
        with col12: 
            original2 = '<p style="font-family: Georgia; color:White; font-size: 30px; text-align: left">Download your report</p>'
            st.markdown(original2, unsafe_allow_html=True)
            st.markdown(ori, unsafe_allow_html=True)
            with open("reports/grade_3.pdf","rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="500" height="690" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)

    if(prediction == 4):
        with col11:
            st.write('<p style="font-family: Georgia; color:White; font-size: 30px; text-align: justify">Uploaded Image</p>', unsafe_allow_html=True)
            st.markdown(ori, unsafe_allow_html=True)
            st.image(image, use_column_width=True)
            st.markdown(ori, unsafe_allow_html=True)
            st.write('<div align="center"; border: 10px white;><table><tr style=font-size:20px;font-family:Georgia;text-align: justify><td>KL Grade</td><td>4</td></tr><tr style=font-size:20px;font-family:Georgia;text-align: justify><td>Classification</td><td>Severe</td></tr><tr style=font-size:20px;font-family:Georgia;text-align: justify><td>Description</td><td>Joint space greatly reduced : Subchondral Sclerosis</td></tr></table></div>', unsafe_allow_html=True)
        with col12:
            original1 = '<p style="font-family: Georgia; color:White; font-size: 30px; text-align: left">Download your report</p>'
            st.markdown(original1, unsafe_allow_html=True)
            st.markdown(ori, unsafe_allow_html=True)
            with open("reports/grade_4.pdf","rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="500" height="690" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)

else:
    
    original_title2 = '<p style="font-family: Franklin Gothic Medium; color:White; font-size: 60px; text-align: center">Knee Osteoarthritis Radiology Assistant</p>'
    st.markdown(original_title2, unsafe_allow_html=True)
    
    st.markdown(ori, unsafe_allow_html=True)
  
    original_title = '<p style="font-family:Georgia; color:White; font-size: 20px; text-align: justify">Knee osteoarthritis (OA), also known as degenerative joint disease of the knee, is typically the result of wear and tear and progressive loss of articular cartilage. </p>'
    st.markdown(original_title, unsafe_allow_html=True)
    st.markdown(ori, unsafe_allow_html=True)
    original_title1 = '<p style="font-family:Georgia; color:White; font-size: 20px; text-align: justify">Based on radiological data, the Kellgren-Lawrence (KL) scoring system is used to evaluate the severity of the condition. The KL system is the most widely used approach for grading Osteoarthritis (OA) in the knee joint into five separate classes, primarily to determine the severity of the illness.</p>'
    st.markdown(original_title1, unsafe_allow_html=True)
    st.markdown(ori, unsafe_allow_html=True)
    conmat1 = Image.open("images/KLgrade.png")
    conmat = Image.open("images/conmat.png")
    rocauc = Image.open("images/rocauc.jpg")
    st.image(conmat1, caption="KL grading system for OA", use_column_width=True)
    st.markdown(ori, unsafe_allow_html=True)
    original_title3 = '<p style="font-family: Franklin Gothic Medium; color:White; font-size: 35px; "><b>How does KORA work?</b></p>'
    st.markdown(original_title3, unsafe_allow_html=True)
    original_title3 = '<p style="font-family: Georgia; color:White; font-size: 20px; text-align: justify">KORA learns the OA progressions from nearly 2000 knee MRI images deploying a cutting-edge deep learning model that incorporates Densenet and Squueze Excitation modules. After training, the model has the ability to recognise and grade patient scans.</p>'
    st.markdown(original_title3, unsafe_allow_html=True)
    st.markdown(ori, unsafe_allow_html=True)
    original_title4 = '<p style="font-family: Georgia; color:White; font-size: 20px; text-align: justify">The algorithm implemented in KORA exhibits an accuracy range of <b>75%-80%</b> and consistently produces precise results.</p>'
    st.markdown(original_title4, unsafe_allow_html=True)
    st.markdown(ori, unsafe_allow_html=True)
    col11, col12 = st.columns(2)
    with col11:
        st.image(conmat, caption="Confusion matrix showing the percentage of correct predictions", width=360)
    with col12:
        st.image(rocauc, caption="ROC AUC curve showing how well model can distinguish between classes", use_column_width=True)
    
    
    
    
    
#st.markdown(
#    """
#    <style>
#    .stApp {
#        background: linear-gradient(to bottom right, #0000FF, #ffffff);
#    }
#    .stButton>button {
#        background-color: #2f3e98;
#        color: white;
#        border-color: #2f3e98;
#        border-radius: 20px;
#        font-size: 20px;
#        padding: 10px 20px;
#        margin-top: 10px;
#    }
#    .stTextInput>div>div>input {
#        border-radius: 20px;
#        padding: 10px 20px;
#    }
#    </style>
#    """,
#    unsafe_allow_html=True,
#)
