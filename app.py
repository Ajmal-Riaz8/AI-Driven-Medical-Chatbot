from flask import Flask, render_template, request, jsonify
import os
import json
import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.densenet import preprocess_input
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from langchain_groq import ChatGroq

# ==== Setup ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"),
            static_folder=os.path.join(BASE_DIR, "static"))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==== Load Skin Detection Model (DISABLED) ====
# model = load_model(
#     os.path.join(BASE_DIR, "DenseNet201_best.keras"),
#     custom_objects={'preprocess_input': preprocess_input}
# )

# class_names = [
#     'actinic keratosis',
#     'basal cell carcinoma',
#     'dermatofibroma',
#     'melanoma',
#     'nevus',
#     'pigmented benign keratosis',
#     'seborrheic keratosis',
#     'squamous cell carcinoma',
#     'vascular lesion'
# ]

# ==== RAG Setup ====
with open(os.path.join(BASE_DIR, "documents.pkl"), "rb") as f:
    documents = pickle.load(f)

faiss_index = faiss.read_index(os.path.join(BASE_DIR, "rag_index.faiss"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")

os.environ["GROQ_API_KEY"] = "Use your API key"
llm = ChatGroq(model="llama3-8b-8192", temperature=0.7)

# ==== Safety Filter ====
UNSAFE_KEYWORDS = [
    "dosage", "overdose", "kill", "suicide", "self-harm", "medicine",
    "tablet", "remedy", "cure", "prescription", "drug", "treatment"
]

def is_safe(text):
    return not any(word in text.lower() for word in UNSAFE_KEYWORDS)

def retrieve_context(query, k=3):
    query_vector = embedder.encode([query]).astype("float32")
    _, I = faiss_index.search(query_vector, k)
    return [documents[i] for i in I[0]]

def build_prompt(query, contexts):
    context_block = "\n\n---\n\n".join(contexts)
    return f"""You are a Medical Diagnostic Assistant AI. Your role is to help users identify possible medical conditions based solely on the symptoms they provide.

Follow these guidelines:

- Only respond to medical symptom-related questions. If a user asks about anything unrelated to health or symptoms, politely inform them that you are only designed to assist with medical symptom analysis.
- Always respond in a clear and paragraph format.
- Use simple and easy-to-understand language. Avoid medical jargon unless necessary, and explain any medical terms used.
- Keep responses concise and focused. Do not be verbose or overly technical and it should be in paragraph format.
- Never recommend or suggest any medications (prescription or over-the-counter), home remedies, treatments, dosages, or medical procedures. Never advise on actions that could be harmful or risky.
- If a user's input is unclear, incomplete, or vague, ask relevant follow-up questions to gather more information.
- If you're unsure based on the provided symptoms, say so. Do not make guesses or present uncertain information as fact.
- If the symptoms described are urgent or life-threatening (e.g., chest pain, difficulty breathing, confusion, unconsciousness), advise the user to seek immediate medical help or call emergency services.
- Always include this disclaimer at the end of every response:  
  “This is not a medical diagnosis. Please consult a licensed healthcare professional for a confirmed diagnosis or treatment.”

Context:
{context_block}

Question: {query}
Answer:"""

# ==== Routes ====
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat")
def chat():
    return render_template("chat.html")

@app.route("/detection")
def detection_page():
    return render_template("detection.html")

# ==== Predict Skin Route (DISABLED) ====
# @app.route("/predict-skin", methods=["POST"])
# def predict_skin():
#     if "image" not in request.files:
#         return jsonify({"result": "No file uploaded."})
#     file = request.files["image"]
#     if file.filename == "":
#         return jsonify({"result": "No selected file."})
#
#     filename = secure_filename(file.filename)
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     file.save(filepath)
#
#     try:
#         img = image.load_img(filepath, target_size=(180, 180))
#         img_array = image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#         prediction = model.predict(img_array)[0]
#         predicted_label = class_names[np.argmax(prediction)]
#         print("Probabilities:", prediction)
#         return jsonify({"result": predicted_label})
#     except Exception as e:
#         return jsonify({"result": f"❌ Error: {str(e)}"})

@app.route("/get_answer", methods=["POST"])
def get_answer():
    question = request.json.get("question", "").strip()
    if not question:
        return jsonify({"answer": "Please enter a valid question."})

    if not is_safe(question):
        return jsonify({"answer": "⚠️ Sorry, I cannot answer unsafe medical questions. Please consult a doctor."})

    try:
        context = retrieve_context(question)
        prompt = build_prompt(question, context)
        response = llm.invoke(prompt).content
    except Exception as e:
        response = f"❌ Error: {str(e)}"

    return jsonify({"answer": response})

if __name__ == "__main__":
    app.run(debug=True)