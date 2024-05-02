from flask import Flask
from flask_cors import CORS
import routes

app = Flask(__name__)
CORS(app)
routes.init_app(app)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
