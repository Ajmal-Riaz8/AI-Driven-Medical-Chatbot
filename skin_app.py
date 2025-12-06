from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input
from werkzeug.utils import secure_filename
import requests

# Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"),
            static_folder=os.path.join(BASE_DIR, "static"))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = load_model(
    os.path.join(BASE_DIR, "DenseNet201_best.keras"),
    custom_objects={'preprocess_input': preprocess_input}
)

class_names = [
    'actinic keratosis',
    'basal cell carcinoma',
    'dermatofibroma',
    'melanoma',
    'nevus',
    'pigmented benign keratosis',
    'seborrheic keratosis',
    'squamous cell carcinoma',
    'vascular lesion'
]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat")
def chat():
    return render_template("chat.html")

@app.route("/detection")
def detection_page():
    return render_template("detection.html")

@app.route("/predict-skin", methods=["POST"])
def predict_skin():
    if "image" not in request.files:
        return jsonify({"result": "No file uploaded."})
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"result": "No selected file."})

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        img = image.load_img(filepath, target_size=(180, 180))
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)[0]
        predicted_label = class_names[np.argmax(prediction)]
        return jsonify({"result": predicted_label})
    except Exception as e:
        return jsonify({"result": f"❌ Error: {str(e)}"})

@app.route("/get_answer", methods=["POST"])
def get_answer():
    question = request.json.get("question", "").strip()
    if not question:
        return jsonify({"answer": "Please enter a valid question."})

    try:
        response = requests.post("http://127.0.0.1:5001/chatbot", json={"question": question})
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"answer": f"❌ Error talking to chatbot: {str(e)}"})

if __name__ == "__main__":
    app.run(port=5000, debug=True)
