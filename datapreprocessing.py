
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def load_data():
    data = pd.read_csv('fer2013.csv')  # Replace with the path to your dataset

    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = [np.fromstring(pixel_sequence, sep=' ').reshape((width, height, 1)) for pixel_sequence in pixels]
    faces = np.array(faces, dtype='float32')
    faces /= 255.0

    emotions = to_categorical(data['emotion'], num_classes=7)  # Assuming 7 classes of emotions

    return train_test_split(faces, emotions, test_size=0.2, random_state=42)

if _name_ == "_main_":
    X_train, X_val, y_train, y_val = load_data()
    np.save('X_train.npy', X_train)
    np.save('X_val.npy', X_val)
    np.save('y_train.npy', y_train)
    np.save('y_val.npy', y_val)
