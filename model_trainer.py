import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten

class ModelTrainer:

    _instance = None
    model_initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ModelTrainer, cls).__new__(cls)
        return cls._instance

    def initialize_model(self, input_shape, num_classes):
        if not self.model_initialized:
            self.model = self.build_model(input_shape, num_classes)
            self.model_initialized = True

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
        self.model_initialized = True

    def train(self, X_train, y_train, X_test, y_test, batch_size, epochs):
        self.log = self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))
        self.model.summary()

    def predict(self, image):
        return self.model.predict(image[None, :, :, :])[0]

    def evaluate_model(self, X_test, y_test, batch_size=512):
        loss, accuracy = self.model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
        print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")