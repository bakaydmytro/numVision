import os
import numpy as np
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image

def get_model(model_path='mnist_model.h5'):
    if os.path.exists(model_path):
        print("✅ Model loaded")
        return load_model(model_path)

    print("🚀 Training new model")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train, x_test = x_train[..., np.newaxis], x_test[..., np.newaxis]  # Add channel dim

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        Flatten(),
        Dense(10, activation='softmax')
    ])


    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test), verbose=2)

    model.save(model_path)
    print("💾 Model saved")
    return model

def classify_image(model, image_path):
    img = image.load_img(image_path, target_size=(28, 28), color_mode="grayscale")
    img_array = np.expand_dims(image.img_to_array(img), axis=0)  # shape: (1, 28, 28, 1)
    predictions = model.predict(img_array)
    return predictions, np.argmax(predictions)

# --- Run ---
model = get_model()
predictions, predicted_class = classify_image(model, 'images/image.png')
print(f"📌 Predicted Class: {predicted_class}")
print(f"🔢 Predictions: {predictions}")
