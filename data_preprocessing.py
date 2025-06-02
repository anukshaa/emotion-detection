def load_data():
    # Define paths to the train and test directories
    train_dir = r'C:\Users\Anshika\OneDrive\Desktop\mood_music_recommendation\fer2013\train'
    test_dir = r'C:\Users\Anshika\OneDrive\Desktop\mood_music_recommendation\fer2013\test'

    # Define data generators
    datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

    # Load training data
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=64,
        subset='training'
    )

    # Load validation data
    validation_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=64,
        subset='validation'
    )

    # Load test data
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(48, 48),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=64
    )

    return train_generator, validation_generator, test_generator

def generator_to_array(generator):
    X, y = [], []
    for i in range(len(generator)):
        batch_X, batch_y = generator[i]
        X.append(batch_X)
        y.append(batch_y)
    return np.vstack(X), np.vstack(y)

if __name__ == "__main__":
    train_gen, val_gen, test_gen = load_data()

    # Convert data generators to arrays
    X_train, y_train = generator_to_array(train_gen)
    X_val, y_val = generator_to_array(val_gen)
    X_test, y_test = generator_to_array(test_gen)

    # **Ensure labels are one-hot encoded** (This is the added step)
    print("Original y_train shape:", y_train.shape)
    y_train = to_categorical(np.argmax(y_train, axis=-1), num_classes=7)
    y_val = to_categorical(np.argmax(y_val, axis=-1), num_classes=7)
    y_test = to_categorical(np.argmax(y_test, axis=-1), num_classes=7)
    print("Updated y_train shape:", y_train.shape)  # Should now be (num_samples, 7)

    # Save the arrays
    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)
    np.save('X_val.npy', X_val)
    np.save('y_val.npy', y_val)
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)
