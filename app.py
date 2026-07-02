import gradio as gr
from PIL import Image
from app.inference import predict_image

def predict(img):
    result = predict_image(img)
    return result

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="CropMind AI",
    description="Upload a crop leaf image to detect disease"
)

demo.launch()