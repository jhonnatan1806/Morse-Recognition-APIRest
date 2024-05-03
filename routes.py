from flask import request, send_from_directory
import model_controller

def init_app(app):

    @app.route('/', methods=['GET'])
    def index():
        return 'Hello, World!'
    
    @app.route('/upload', methods=['POST'])
    def upload_data():
        return model_controller.upload_data(request)

    @app.route('/train', methods=['GET'])
    def train_model():
        return model_controller.train_model()
    
    @app.route('/prepare', methods=['GET'])
    def prepare_dataset():
        return model_controller.prepare_dataset()
    
    @app.route('/download/<path:filename>', methods=['GET'])
    def download_data(filename):
        return send_from_directory('data/test', filename)

    @app.route('/predict', methods=['POST'])
    def predict_data():
        return model_controller.predict_data(request)