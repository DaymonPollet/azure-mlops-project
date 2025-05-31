from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
from tensorflow import keras
import tensorflow as tf
import os
import io # Make sure io is imported

# 1. Create an app
app = FastAPI()

# 2. Configure CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

# Define animal labels (based on your training)
ANIMALS = ['Cat', 'Dog', 'Panda']

# 3. Load in the AI Model using an environment variable for the base path
# The environment variable 'MODEL_BASE_PATH' will point to the directory
# containing 'best_model.h5' inside the container.
# We provide a default path for local development if the env var isn't set,
# assuming the model is placed relative to 'inference/main.py' as before.
MODEL_BASE_PATH_DEFAULT_RELATIVE = os.path.join("animal-classification", "INPUT_model_path", "animal-cnn")
model_full_path = os.path.join(
    os.environ.get('MODEL_BASE_PATH', MODEL_BASE_PATH_DEFAULT_RELATIVE),
    "best_model.h5"
)

# Load the TensorFlow Keras model directly from .h5
model = keras.models.load_model(model_full_path)

# 4. Define the inference endpoint
@app.post('/upload/image')
async def uploadImage(img: UploadFile = File(...)):
    # Read the bytes and process as an image
    original_image = Image.open(io.BytesIO(await img.read()))

    # Resize image to (64, 64) for your model
    resized_image = original_image.resize((64, 64))

    # Our AI Model wanted a list of images, but we only have one, so we expand its dimension
    images_to_predict = np.expand_dims(np.array(resized_image), axis=0)

    # The result will be a list with predictions in the one-hot encoded format: [ [0 1 0] ]
    predictions = model.predict(images_to_predict)

    # Extract prediction probabilities (adjust 'activation_7' if your model's output layer has a different name)
    # Assuming 'activation_7' is the name of the output layer in your trained model
    if isinstance(predictions, dict) and 'activation_7' in predictions:
        prediction_probabilities = predictions['activation_7']
    else:
        # Fallback if predictions is a numpy array directly (common for load_model from .h5)
        prediction_probabilities = predictions

    # Fetch the index of the highest value in this list [ [1] ]
    classifications = prediction_probabilities.argmax(axis=1)

    # Fetch the first item in our classifications array, format it as a list first, result will be e.g.: "Dog"
    return ANIMALS[classifications.tolist()[0]]