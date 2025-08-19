import gradio as gr
import tensorflow as tf
import cv2
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("model.keras")

def recognize_alpha(image):
    if image is not None:
        
        image = cv2.resize(255 - image, (28, 28))
        image = image.reshape((1, 28, 28, 1)).astype("float32") / 255

        prediction = model.predict(image)[0]
        labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        return {labels[i]: float(prediction[i]) for i in range(26)}
    else:
        return ''

iface = gr.Interface(
    fn=recognize_alpha,
    inputs=gr.Sketchpad(type="numpy"),  
    outputs=gr.Label(num_top_classes=26),
    live=True
)

iface.launch()
