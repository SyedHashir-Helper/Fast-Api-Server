from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from PIL import Image
import tensorflow as tf
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

Model = tf.keras.models.load_model('2.keras')
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get('/hello/{name}')
async def hello(name):
    return f"Welcome to FastAPI Tutorial, {name}"

def readFile(file)->np.ndarray:
    return np.array(Image.open(BytesIO(file)))

@app.post('/predict')
async def predict(file: UploadFile):
    image =  readFile(await file.read())
    image_batch = np.expand_dims(image,0)
    prediction = Model.predict(image_batch)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)*100
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)