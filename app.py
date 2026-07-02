import gradio as gr
from PIL import Image
from app.inference import predict_image

def predict(img):
    result = predict_image(img)
    return result

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="CropMind AI",
    description="Plant disease detection from crop leaf images"
)

interface.launch()
