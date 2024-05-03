import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten


class ModelTrainer:

    def __init__(self, input_shape, num_classes):
        self.model = self.build_model(input_shape, num_classes)

    def build_model(self, input_shape, num_classes):
        model = Sequential([
            Conv2D(32, 3, activation='relu', input_shape=input_shape, padding='same'),
            MaxPool2D(),
            Conv2D(64, 3, activation='relu', padding='same'),
            MaxPool2D(),
            Conv2D(128, 3, activation='relu', padding='same'),
            MaxPool2D(),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='SGD', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def save_model(self, file_path='model.h5'):
        self.model.save(file_path)

    def load_model(self, file_path='model.h5'):
        self.model = load_model(file_path)

    def train(self, X_train, y_train, batch_size, epochs):
        self.log = self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
        self.model.summary()

    def predict(self, image):
        prediction = self.model.predict(image[None,:,:,:])[0]
        return prediction

    def evaluate_model(self, X_test, y_test, batch_size=16):
        loss, accuracy = self.model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
        print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")