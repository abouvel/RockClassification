import base64
import requests
from PIL import Image
import io

def resize_to_128x128(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((128, 128))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()

def image_to_base64(image_bytes):
    return base64.b64encode(image_bytes).decode("utf-8")

def call_classify_shape(api_url, base64_str):
    payload = {"image_base64": base64_str}
    response = requests.post(f"{api_url}/classify-shape", json=payload)
    print("🟢 /classify-shape response:", response.status_code)
    print(response.json())

def call_detect(api_url, base64_str):
    payload = {"image_base64": base64_str}
    response = requests.post(f"{api_url}/detect", json=payload)
    print("🟢 /detect response:", response.status_code)
    print(response.json())

# === MAIN USAGE ===
if __name__ == "__main__":
    image_path = "/home/abouvel/claws/AI-NLP-24-25/IMGDetector/shapes_dataset/cylinderImg/image_5.png"  # 🔁 change this path
    api_url = "http://127.0.0.1:5001"           # 🔁 change if running on another port

    # Resize to 128x128, convert to base64
    resized_bytes = resize_to_128x128(image_path)
    b64 = image_to_base64(resized_bytes)

    print("\n=== 🔍 Classify Shape ===")
    call_classify_shape(api_url, b64)

    print("\n=== 🎯 Detect with YOLO ===")
    #call_detect(api_url, b64)
