import requests
import streamlit as st
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

MODEL_ID = "reducto/RolmOCR"

st.title("OCR Model Tester")

@st.cache_resource
def load_model():
    """Load the OCR model and processor."""
    try:
        model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)
        processor = TrOCRProcessor.from_pretrained(MODEL_ID)
        return model, processor
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None

@st.cache_data
def check_model_availability():
    """Verify that the model is accessible on Hugging Face Hub."""
    try:
        resp = requests.get(f"https://huggingface.co/api/models/{MODEL_ID}")
        return resp.status_code == 200
    except Exception:
        return False

if check_model_availability():
    st.success(f"Model {MODEL_ID} is available.")
else:
    st.warning(f"Could not confirm availability of {MODEL_ID}.")

model, processor = load_model()

uploaded_file = st.file_uploader("Import image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None and model and processor:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image")
    inputs = processor(images=image, return_tensors="pt")
    generated_ids = model.generate(**inputs)
    result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    st.text_area("OCR result", result, height=150)
