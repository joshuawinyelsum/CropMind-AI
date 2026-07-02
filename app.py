import gradio as gr

def predict(img):
    return "SPACE IS NOW RUNNING ✔"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text"
)

# IMPORTANT: MUST BE THIS EXACT VARIABLE NAME
demo.launch()