# OCR Model Test App

This repository provides a small application for testing OCR models using [Hugging Face Transformers](https://huggingface.co/docs/transformers/index).

The application loads `reducto/RolmOCR` and lets you upload an image to extract text.

## Requirements

- Python 3.8+
- [streamlit](https://streamlit.io)
- [transformers](https://pypi.org/project/transformers/)
- [torch](https://pytorch.org/) (required by transformers)
- [Pillow](https://pypi.org/project/Pillow/)
- [requests](https://pypi.org/project/requests/)

Internet access is required on the first run so that the model can be downloaded from Hugging Face Hub.

Install dependencies with:

```bash
pip install streamlit transformers torch Pillow requests
```

## Usage

To launch the app:

```bash
streamlit run app.py
```

1. Use the **Import image** widget to upload a picture containing text.
2. The application will process the image and display the detected text under **OCR result**.

The app also checks whether the model `reducto/RolmOCR` is accessible from Hugging Face. If the model is unavailable or there is a connection issue, an error message will appear.

