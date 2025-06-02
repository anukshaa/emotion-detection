import cv2
import numpy as np
from tensorflow.keras.models import load_mode
# Load the trained model
model = load_model('emotion_model.h5')
# Define the emotion labels
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']

# Start the webcam capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize and preprocess the image
    gray_resized = cv2.resize(gray, (48, 48))
    image = np.reshape(gray_resized, (1, 48, 48, 1)) / 255.0

    # Predict the emotion
    prediction = model.predict(image)

    # Get the emotion with the highest probability
    emotion_index = np.argmax(prediction)

    # Display the emotion label on the frame
    cv2.putText(frame, emotion_labels[emotion_index], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Emotion Recognition', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
