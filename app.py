from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2

app = Flask(__name__)

# Load the trained model
model = load_model('emotion_model.h5')

# Define the emotion labels
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']

@app.route('/predict', methods=['POST'])
def predict():
    # Read the image from the request
    file = request.files['image']
    img_array = np.fromstring(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

    # Resize and preprocess the image
    img_resized = cv2.resize(img, (48, 48))
    img_reshaped = np.reshape(img_resized, (1, 48, 48, 1))
    img_normalized = img_reshaped / 255.0

    # Predict the emotion
    prediction = model.predict(img_normalized)
    emotion_index = np.argmax(prediction)
    
    # Return the predicted emotion as JSON
    return jsonify({'emotion': emotion_labels[emotion_index]})

if __name__ == '__main__':
    app.run(debug=True)
