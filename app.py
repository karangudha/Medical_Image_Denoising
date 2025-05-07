from flask import Flask, request, jsonify, send_from_directory
import os
import numpy as np
import tensorflow as tf
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('output/autoencoder_combined.keras')

# Function to preprocess the image (resize and convert to grayscale)
def preprocess_image(image):
    # Convert to grayscale
    image = image.convert('L')  # 'L' mode = grayscale

    # Resize image
    image = image.resize((128, 128))

    # Convert to numpy and normalize
    image_array = np.array(image).astype('float32') / 255.0

    # Add channel and batch dimensions
    image_array = image_array[..., np.newaxis]  # (128, 128, 1)
    return np.expand_dims(image_array, axis=0)  # (1, 128, 128, 1)


# Function to save denoised image
def save_denoised_image(image_array):
    # Convert the image from [0, 1] range to [0, 255] range
    denoised_image = (image_array.squeeze() * 255).astype(np.uint8)  # Convert back to 0-255 scale
    image = Image.fromarray(denoised_image)
    output_path = 'static/denoised_image.jpg'
    image.save(output_path)
    return output_path

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if an image is provided
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files['image']
        image = Image.open(file.stream)
        
        # Preprocess the image (resize, grayscale, normalization)
        image_array = preprocess_image(image)

        # Get the denoised output from the model
        denoised_image_array = model.predict(image_array)
        
        # Save the denoised image and get the path
        denoised_image_path = save_denoised_image(denoised_image_array)

        # Return response with path to denoised image
        return jsonify({"message": "Prediction successful", "denoised_image_path": denoised_image_path}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

if __name__ == "__main__":
    app.run(debug=True)
