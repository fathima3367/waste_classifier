import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("waste_classifier_model.h5")
class_names = ['organic', 'recyclable']  # Update based on your dataset order

def predict(image):
    image = image.resize((150, 150))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = class_names[0] if prediction < 0.5 else class_names[1]
    confidence = prediction if prediction > 0.5 else 1 - prediction

    return f"{label} ({confidence*100:.2f}% confidence)"

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Waste Classifier",
    description="Upload an image of waste to classify as Organic or Recyclable."
)

if __name__ == "__main__":
    interface.launch()
