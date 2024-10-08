import gradio as gr
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
import os

def download_model_from_drive(url, output_path):
    response = requests.get(url)
    with open(output_path, 'wb') as f:
        f.write(response.content)
    print(f"Model downloaded to {output_path}")

drive_url = "https://drive.google.com/uc?id=1cc_KscsoMtedAqDAbDX0h4FkEXUZPFrv"
model_path = "model_05.keras"

if not os.path.exists(model_path):
    print("Downloading model...")
    download_model_from_drive(drive_url, model_path)

print("Loading model...")
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully.")

def predict_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    prediction = model.predict(image_array)
    print(f"Prediction: {prediction}")
    return "Stroke Detected" if prediction[0][0] > 0.5 else "Stroke Not Detected"

interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Ischemic Stroke Detection",
    description="Upload an MRI image to detect if ischemic stroke is present."
)

print("Launching interface...")
interface.launch(share=True)
