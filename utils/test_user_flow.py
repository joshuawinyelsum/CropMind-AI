import requests

img_path = "data/train/Tomato___Bacterial_spot/001.jpg"

files = {'file': open(img_path, 'rb')}

response = requests.post(
    "http://127.0.0.1:8000/predict",
    files=files
)

print(response.json())
