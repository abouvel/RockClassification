from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import io
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
import cv2
import logging
from sklearn.cluster import KMeans
from ultralytics import YOLO
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === CONFIG ===
IMG_SIZE = (128, 128)
MODEL_PATH = "shape_model.keras"
YOLO_MODEL_PATH = "best (2).pt"
class_names = ['cube', 'cylinder', 'ellipsoid', 'sphere']  # adjust as needed

# === LOAD MODELS ===
model = tf.keras.models.load_model(MODEL_PATH)
yolo_model = YOLO(YOLO_MODEL_PATH)

# === FASTAPI SETUP ===
app = FastAPI()

class ImageRequest(BaseModel):
    image_base64: str

def decode_exact_128x128_image(base64_str):
    try:
        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data)).convert("RGB")

        if img.size != IMG_SIZE:
            raise ValueError(f"Image must be exactly {IMG_SIZE}, got {img.size}")

        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

        if img_array.shape != (1, 128, 128, 3):
            raise ValueError(f"Expected shape (1,128,128,3), got {img_array.shape}")
        return img_array
    except Exception as e:
        print("❌ Error:", e)
        raise HTTPException(status_code=400, detail=f"Image decoding error: {e}")

@app.post("/classify-shape")
def classify_shape(req: ImageRequest):
    try:
        img_tensor = decode_exact_128x128_image(req.image_base64)
        predictions = model.predict(img_tensor)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = float(np.max(predictions))
        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4)
        }
    except Exception as e:
        print("❌ Prediction error:", e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

def get_dominant_color(image_crop, k=1):
    pixels = image_crop.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(pixels)
    return kmeans.cluster_centers_[0].astype(int).tolist()  # [R, G, B]

@app.post("/detect")
def detect(req: ImageRequest):
    try:
        # Decode base64 image
        image_data = base64.b64decode(req.image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Image could not be decoded.")

        # Run YOLO detection
        results = yolo_model(image)

        # Prepare detections with color
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = image[y1:y2, x1:x2]

                # Avoid crash on bad crops
                if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
                    dominant_color = [0, 0, 0]
                else:
                    dominant_color = get_dominant_color(crop)

                detections.append({
                    "class": int(box.cls),
                    "confidence": float(box.conf),
                    "bbox": [x1, y1, x2, y2],
                    "dominant_color": dominant_color  # RGB
                })

        return {"detections": detections}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
