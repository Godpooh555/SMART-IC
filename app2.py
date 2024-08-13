import gradio as gr
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
import os
import zipfile

DRIVE_URL = "https://drive.google.com/uc?id=1cc_KscsoMtedAqDAbDX0h4FkEXUZPFrv"
ZIP_PATH = "model_05.zip"
KERAS_PATH = "model_05.keras"

def download_file_from_drive(url, destination):
    response = requests.get(url, stream=True)
    with open(destination, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"File downloaded successfully: {destination}")

def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"File extracted successfully: {zip_path}")

if not os.path.isfile(ZIP_PATH):
    print(f"{ZIP_PATH} not found. Downloading...")
    download_file_from_drive(DRIVE_URL, ZIP_PATH)

if not os.path.isfile(KERAS_PATH):
    print(f"{KERAS_PATH} not found. Extracting...")
    extract_zip(ZIP_PATH, '.')

try:
    model = tf.keras.models.load_model(KERAS_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

def predict_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    try:
        prediction = model.predict(image_array)
        return "Stroke Detected" if prediction[0][0] > 0.5 else "Stroke Not Detected"
    except Exception as e:
        return f"Error during prediction: {e}"

interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Ischemic Stroke Detection",
    description="Upload an MRI image to detect if ischemic stroke is present."
)

interface.launch()