import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical

def getModel():
    model_file = 'mnist_model.h5'
    if os.path.exists(model_file):
        loaded_model = load_model(model_file)
        print("Model was loaded")
        return loaded_model
    else:
        print("Training new model")

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train / 255.0
        x_test = x_test / 255.0

        y_train = to_categorical(y_train, num_classes=10)
        y_test = to_categorical(y_test, num_classes=10)

        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        # Fully connected layers
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))
        train_loss, train_accuracy = model.evaluate(x_train, y_train, verbose=2)
        print(f"Train data accuracy: {train_accuracy * 100:.2f}%")

        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
        print(f"Test data accuracy: {test_accuracy * 100:.2f}%")

        # Save the model in a specific directory
        model.save(model_file)
        return model

def classify_image(model, image_path):
    img = image.load_img(image_path, target_size=(28, 28), color_mode="grayscale")

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)

    predicted_class = np.argmax(predictions)

    return predictions, predicted_class

workModel = getModel()

predictions, predicted_class = classify_image(workModel, 'images/image.png')

print(f"Predicted Class: {predicted_class}")
print(f"Predictions: {predictions}")
