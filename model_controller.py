from flask import jsonify, request
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from PIL import Image
import glob
import os
import uuid 
import numpy as np
from model_trainer import ModelTrainer 

# --------------------------------------------
# Función para subir un archivo
# --------------------------------------------
def upload_data(request):
    # imprimir cuando llega el request
    print(request)
    # Verificar si el request contiene el archivo en el cuerpo del formulario
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'})

    file = request.files['image']
    
    # Verificar si se envió un archivo
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'})

    # Obtener el nombre de la subcarpeta del cuerpo de la solicitud, si existe
    subfolder = request.form.get('subfolder')
    if not subfolder or subfolder not in ['A', 'E', 'I', 'O', 'U']:
        return jsonify({'status': 'error', 'message': 'Invalid or missing subfolder'})

    try:
        # Definir el path donde guardarás los archivos incluyendo la subcarpeta
        upload_folder = os.path.join('data/train/', subfolder)

        # Asegurarte de que el directorio existe
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        
        # Generar un nombre único para el archivo
        filename = str(uuid.uuid4()) + ".png"

        # Guardar el archivo con el nombre único
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)

        # Aquí puedes agregar cualquier procesamiento adicional necesario
        return jsonify({'status': 'success', 'message': 'File uploaded successfully', 'path': filepath})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# --------------------------------------------
# Función para preparar el dataset
# --------------------------------------------
def prepare_dataset():
    try:
        X = []
        y = []
        categories = ['A', 'E', 'I', 'O', 'U']
        for i, subfolder in enumerate(categories):
            files = glob.glob(os.path.join('data/train/', subfolder, '*.png'))
            for file in files:
                with Image.open(file) as image:
                    image = image.convert('L')  # Convertir a escala de grises
                    image = image.resize((28, 28))  # Redimensionar a 28x28 que es el input_shape esperado
                    image_array = np.array(image)
                    image_array = image_array / 255.0  # Normalización
                    X.append(image_array)
                    y.append(i)

        # Convertir listas a arrays de numpy y guardar
        X = np.array(X)
        y = np.array(y)
        np.save('data/test/X.npy', X, allow_pickle=True)
        np.save('data/test/y.npy', y, allow_pickle=True)

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
            X_train = X_train[..., None]
            X_test = X_test[..., None]

        # Crear e instanciar el modelo
        num_classes = len(np.unique(y))
        input_shape = X_train.shape[1:]  # Asume que todas las imágenes tienen el mismo shape

        # Entrenar el modelo
        trainer = ModelTrainer()
        trainer.initialize_model(input_shape, num_classes)
        trainer.train(X_train, y_train, X_test, y_test, batch_size=30, epochs=50)
        trainer.save_model('data/test/trained_model.h5')  # Guardar modelo

        return jsonify({'status': 'success', 'message': 'Model trained and saved successfully'})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# --------------------------------------------
# Función para predecir una imagen
# --------------------------------------------
def predict_data(request):
    trainer = ModelTrainer()
    # Verificar si el modelo está cargado, si no, cargarlo
    if not trainer.model_initialized:
        try:
            trainer.load_model('data/test/trained_model.h5')  # Cargar modelo
        except Exception as e:
            return jsonify({'status': 'error', 'message': 'Model could not be loaded: ' + str(e)})

    # Recibir la imagen
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image provided'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No image selected'})

    try:
        # Abrir la imagen con PIL y convertirla en formato adecuado
        image = Image.open(file.stream)
        image = image.convert('L')  # Convertir a escala de grises
        image = image.resize((28, 28))  # Redimensionar a 28x28 que es el input_shape esperado
        image_array = np.array(image)
        image_array = image_array / 255.0  # Normalización
        image_array = image_array.reshape((1, 28, 28, 1))  # Reshape para que coincida con el input_shape del modelo

        # Realizar la predicción
        prediction = trainer.predict(image_array)

        return jsonify({'status': 'success', 'message': 'Prediction successful', 'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})