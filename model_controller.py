from flask import jsonify, request
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import glob
import os
from skimage import io
import base64
import uuid 
import numpy as np
from model_trainer import ModelTrainer 

# --------------------------------------------
# Función para subir un archivo
# --------------------------------------------
def upload_data(request):
    try:
        # Verificar si se envió la imagen en base64
        img_data = request.json.get('image').replace("data:image/png;base64,","")
        subfolder = request.json.get('subfolder')
        
        # Verificar si se proporcionó la subcarpeta y si es válida
        if not subfolder or subfolder not in ['A', 'E', 'I', 'O', 'U']:
            return jsonify({'status': 'error', 'message': 'Invalid or missing subfolder'})
        
        # Decodificar la imagen base64 y guardarla
        img_bytes = base64.b64decode(img_data)
        
        # Definir el path donde guardarás los archivos incluyendo la subcarpeta
        upload_folder = os.path.join('data/train/', subfolder)

        # Asegurarte de que el directorio existe
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        
        # Generar un nombre único para el archivo
        filename = str(uuid.uuid4()) + ".png"

        # Guardar el archivo con el nombre único
        filepath = os.path.join(upload_folder, filename)
        with open(filepath, "wb") as fh:
            fh.write(img_bytes)

        # Aquí puedes agregar cualquier procesamiento adicional necesario
        return jsonify({'status': 'success', 'message': 'File uploaded successfully', 'path': filepath})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# --------------------------------------------
# Función para preparar el dataset
# --------------------------------------------
def prepare_dataset():
    try:
        images = []
        d = ['A', 'E', 'I', 'O', 'U']
        digits = []
        for digit in d:
            filelist = glob.glob('data/train/{}/*.png'.format(digit))
            images_read = io.concatenate_images(io.imread_collection(filelist))
            images_read = images_read[:, :, :, 3]
            digits_read = np.array([digit] * images_read.shape[0])
            images.append(images_read)
            digits.append(digits_read)

        images = np.vstack(images)
        digits = np.concatenate(digits)
        np.save('data/test/X.npy', images, allow_pickle=True)
        np.save('data/test/y.npy', digits, allow_pickle=True)

        return jsonify({'status': 'success', 'message': 'Dataset prepared successfully'})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# --------------------------------------------
# Función para entrenar el modelo
# --------------------------------------------
def train_model():
    try:
        # Preparamos los datos
        X_raw = np.load('data/test/X.npy')
        X_raw = X_raw / 255.0
        y = np.load('data/test/y.npy')
        X = []
        size = (28, 28)
        for x in X_raw:
            X.append(resize(x, size))
        X = np.array(X)

        # Dividimos los datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

        # Preparar las dimensiones de las imágenes si es necesario
        if X_train.ndim == 3:
            X_train = X_train[..., None]  # Añadir una dimensión adicional al final de las imágenes de entrenamiento
            X_test = X_test[..., None]    # Añadir una dimensión adicional al final de las imágenes de prueba
            print(X_train.shape, X_test.shape)  # Imprimir las nuevas formas de las imágenes de entrenamiento y prueba

        # Creamos un objeto LabelEncoder
        label_encoder = LabelEncoder()

        # Convertimos las etiquetas de letras en números enteros
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)

        # Crear e instanciar el modelo
        num_classes = len(np.unique(y))
        input_shape = X_train.shape[1:]  # Asume que todas las imágenes tienen el mismo shape
        trainer = ModelTrainer(input_shape, num_classes)
        # Entrenar el modelo bs = 16, epochs = 200
        trainer.train(X_train, y_train_encoded, X_test, y_test_encoded, batch_size=16, epochs=200)
        trainer.save_model('data/test/trained_model.h5')  # Guardar modelo
        return jsonify({'status': 'success', 'message': 'Model trained and saved successfully'})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# --------------------------------------------
# Función para predecir una imagen
# --------------------------------------------
def predict_data(request):
    trainer = ModelTrainer(input_shape=(28, 28, 1), num_classes=5)
    trainer.load_model('data/test/trained_model.h5')
    try: 
        img_data = request.json.get('image').replace("data:image/png;base64,","")

        # Decodificar la imagen base64 y guardarla
        img_bytes = base64.b64decode(img_data)

        # Guardar el archivo con el nombre único
        filepath = os.path.join('data/test/', "predict.png")
        with open(filepath, "wb") as fh:
            fh.write(img_bytes)

        # Cargar la imagen
        image = Image.open(filepath)

        # Redimensionar la imagen
        image = image.resize((28, 28))

        # Convertir la imagen a un array numpy
        image_array = np.array(image)

        # Seleccionar el canal de color que deseas mantener
        image_array = image_array[:, :, 3]

        image_array = np.expand_dims(image_array, axis=-1)

        # Realizar la predicción
        prediction = trainer.predict(image_array)
        # Obtener el índice de la predicción con mayor probabilidad
        maxIndex = np.argmax(prediction)
        # Crear una respuesta JSON
        response = jsonify({'status': 'success', 'message': 'Prediction successful', 'prediction': prediction.tolist(), 'letter': ['A', 'E', 'I', 'O', 'U'][maxIndex]})

        return response

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})