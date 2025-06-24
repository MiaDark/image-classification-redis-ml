from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
import redis
from PIL import Image
import io
import os
import tensorflow as tf
import numpy as np
from dotenv import load_dotenv

load_dotenv()
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

app = FastAPI()

# Redis connection
r = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    # ssl=True,
    ssl_cert_reqs=None,  # Rhe free version of Redis does not support SSL
    decode_responses=False,
)
# Test connection
try:
    r.ping()
    print("✅ Redis connection successful!")
except Exception as e:
    print(f"❌ Redis connection failed: {e}")

model = tf.keras.applications.MobileNetV2(weights="imagenet")


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    r.set(f"image:{file.filename}", image_bytes)

    # Preprocess the image
    processed_image = preprocess_image(image_bytes)

    # Make prediction
    predictions = model.predict(processed_image)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(
        predictions, top=1
    )[0]
    predicted_class = decoded_predictions[0][1]
    confidence = float(decoded_predictions[0][2])

    return JSONResponse(
        content={
            "prediction": predicted_class,
            "confidence": confidence,
            "filename": file.filename,
            "cached": False,
        }
    )


@app.get("/", response_class=HTMLResponse)
async def upload_interface():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)
