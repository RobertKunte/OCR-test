import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import streamlit as st
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Constants
MODEL_ID = "microsoft/trocr-base-printed"
DEVICE = 'cpu'

# Initialize Streamlit
st.set_page_config(
    page_title="OCR App",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource(show_spinner="Loading OCR model...")
def load_model():
    """Load the OCR model and processor."""
    try:
        processor = TrOCRProcessor.from_pretrained(MODEL_ID)
        model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID).to(DEVICE)
        model.eval()
        return model, processor
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None, None

# Load model
model, processor = load_model()

# UI
st.title("📝 OCR Model Tester")
st.markdown("---")

# (This section was moved to the top of the file)

# Main app
st.header("Upload an Image")
st.write("Upload an image containing text to extract the text using OCR.")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    try:
        # Display the uploaded image
        with st.spinner("Processing image..."):
            image = Image.open(uploaded_file).convert("RGB")
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            
            with col2:
                if model is not None and processor is not None:
                    try:
                        # Process the image
                        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
                        generated_ids = model.generate(**inputs)
                        result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                        
                        # Display results
                        st.subheader("Extracted Text")
                        st.text_area("Extracted Text", 
                                    value=result, 
                                    height=200, 
                                    label_visibility="collapsed")
                        
                        # Add a download button
                        st.download_button(
                            label="💾 Download Text",
                            data=result,
                            file_name="extracted_text.txt",
                            mime="text/plain"
                        )
                    except Exception as e:
                        st.error(f"Error processing image: {str(e)}")
                else:
                    st.error("Model failed to load. Please check the console for errors.")
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
else:
    st.info("Please upload an image file to get started.")

# Add some styling
st.markdown("""
<style>
    .stApp {
        max-width: 1000px;
        margin: 0 auto;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)
