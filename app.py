import os
import time
from threading import Thread

from flask import Flask, render_template, request
import torch
import numpy as np
from PIL import Image
import cv2

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    AutoModelForImageTextToText,
    AutoProcessor,
    TextIteratorStreamer,
)

# Constants for text generation
MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

# Select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32

# Load RolmOCR
MODEL_ID_M = "reducto/RolmOCR"
processor_m = AutoProcessor.from_pretrained(MODEL_ID_M, trust_remote_code=True)
model_m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_M,
    trust_remote_code=True,
    torch_dtype=dtype
)
if device.type == "cuda":
    model_m = model_m.to(device)
model_m = model_m.eval()

# Load Qwen2-VL-OCR-2B-Instruct
#MODEL_ID_X = "prithivMLmods/Qwen2-VL-OCR-2B-Instruct"
#processor_x = AutoProcessor.from_pretrained(MODEL_ID_X, trust_remote_code=True)
#model_x = Qwen2VLForConditionalGeneration.from_pretrained(
#    MODEL_ID_X,
#    trust_remote_code=True,
#    torch_dtype=torch.float16
#).to(device).eval()

# Load Nanonets-OCR-s
#MODEL_ID_V = "nanonets/Nanonets-OCR-s"
#processor_v = AutoProcessor.from_pretrained(MODEL_ID_V, trust_remote_code=True)
#model_v = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#    MODEL_ID_V,
#    trust_remote_code=True,
#    torch_dtype=torch.float16
#).to(device).eval()

# Load aya-vision-8b
#MODEL_ID_A = "CohereForAI/aya-vision-8b"
#processor_a = AutoProcessor.from_pretrained(MODEL_ID_A, trust_remote_code=True)
#model_a = AutoModelForImageTextToText.from_pretrained(
#    MODEL_ID_A,
#    trust_remote_code=True,
#    torch_dtype=torch.float16
#).to(device).eval()


def downsample_video(video_path):
    """Downsamples the video to evenly spaced frames."""
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frames = []
    frame_indices = np.linspace(0, total_frames - 1, 10, dtype=int)
    for i in frame_indices:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, image = vidcap.read()
        if success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            timestamp = round(i / fps, 2)
            frames.append((pil_image, timestamp))
    vidcap.release()
    return frames


def generate_image(
    model_name: str,
    text: str,
    image: Image.Image,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
): 
    """Generates responses using the selected model for image input."""
    start = time.time()
    if model_name == "RolmOCR":
        processor = processor_m
        model = model_m
    elif model_name == "Qwen2-VL-OCR-2B-Instruct":
        processor = processor_x
        model = model_x
    elif model_name == "Nanonets-OCR-s":
        processor = processor_v
        model = model_v
    elif model_name == "Aya-Vision":
        processor = processor_a
        model = model_a
    else:
        yield "Invalid model selected."
        return

    if image is None:
        yield "Please upload an image."
        return

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": text},
        ]
    }]
    prompt_full = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[prompt_full],
        images=[image],
        return_tensors="pt",
        padding=True,
        truncation=False,
        max_length=MAX_INPUT_TOKEN_LENGTH,
    )
    if device.type == "cuda":
        inputs = inputs.to(device)
    streamer = TextIteratorStreamer(
        processor, skip_prompt=True, skip_special_tokens=True
    )
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": repetition_penalty,
    }
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    buffer = ""
    for new_text in streamer:
        buffer += new_text
        buffer = buffer.replace("<|im_end|>", "")
        time.sleep(0.01)
        yield buffer
    end = time.time()
    print(f"⏱️ Verarbeitung dauerte: {end - start:.2f} Sekunden")


def generate_video(
    model_name: str,
    text: str,
    video_path: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
):
    """Generates responses using the selected model for video input."""
    start = time.time()
    if model_name == "RolmOCR":
        processor = processor_m
        model = model_m
    elif model_name == "Qwen2-VL-OCR-2B-Instruct":
        processor = processor_x
        model = model_x
    elif model_name == "Nanonets-OCR-s":
        processor = processor_v
        model = model_v
    elif model_name == "Aya-Vision":
        processor = processor_a
        model = model_a
    else:
        yield "Invalid model selected."
        return

    if video_path is None:
        yield "Please upload a video."
        return

    frames = downsample_video(video_path)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [{"type": "text", "text": text}]},
    ]
    for frame in frames:
        image, timestamp = frame
        messages[1]["content"].append({"type": "text", "text": f"Frame {timestamp}:"})
        messages[1]["content"].append({"type": "image", "image": image})
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        truncation=False,
        max_length=MAX_INPUT_TOKEN_LENGTH,
    )
    if device.type == "cuda":
        inputs = inputs.to(device)
    streamer = TextIteratorStreamer(
        processor, skip_prompt=True, skip_special_tokens=True
    )
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": repetition_penalty,
    }
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    buffer = ""
    for new_text in streamer:
        buffer += new_text
        buffer = buffer.replace("<|im_end|>", "")
        time.sleep(0.01)
        yield buffer
    end = time.time()
    print(f"⏱️ Verarbeitung dauerte: {end - start:.2f} Sekunden")


app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None)


def run_generator(gen):
    result = ""
    for chunk in gen:
        result = chunk
    return result


@app.route("/image", methods=["POST"])
def image_route():
    text = request.form.get("text", "")
    model = request.form.get("model", "Nanonets-OCR-s")
    file = request.files.get("image")
    if not file or file.filename == "":
        return render_template("index.html", result="Please upload an image.")
    path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(path)
    image = Image.open(path).convert("RGB")
    result = run_generator(
        generate_image(model, text, image)
    )
    os.remove(path)
    return render_template("index.html", result=result)


@app.route("/video", methods=["POST"])
def video_route():
    text = request.form.get("text", "")
    model = request.form.get("model", "Nanonets-OCR-s")
    file = request.files.get("video")
    if not file or file.filename == "":
        return render_template("index.html", result="Please upload a video.")
    path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(path)
    result = run_generator(
        generate_video(model, text, path)
    )
    os.remove(path)
    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
