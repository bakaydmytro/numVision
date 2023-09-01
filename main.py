import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models

(x_train, y_train), (x_test, y_test) = mnist.load_data()


print("Size x_train:", x_train.shape)
print("Size y_train:", y_train.shape)
print("Size x_test:", x_test.shape)
print("Size y_test:", y_test.shape)


model = models.Sequential()

model.add(layers.Flatten(input_shape=(28, 28)))

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

train_loss, train_accuracy = model.evaluate(x_train, y_train, verbose=2)
print(f"Train data accuracy: {train_accuracy*100:.2f}%")

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"Test data accuracy: {test_accuracy*100:.2f}%")


