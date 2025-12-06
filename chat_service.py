from flask import Flask, request, jsonify
import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "documents.pkl"), "rb") as f:
    documents = pickle.load(f)

faiss_index = faiss.read_index(os.path.join(BASE_DIR, "rag_index.faiss"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")

os.environ["GROQ_API_KEY"] = "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"
llm = ChatGroq(model="llama3-8b-8192", temperature=0.7)

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

app = Flask(__name__)

@app.route("/chatbot", methods=["POST"])
def chatbot():
    question = request.json.get("question", "").strip()
    if not question:
        return jsonify({"answer": "Please enter a valid question."})
    if not is_safe(question):
        return jsonify({"answer": "⚠️ Sorry, I cannot answer unsafe medical questions. Please consult a doctor."})

    try:
        context = retrieve_context(question)
        prompt = build_prompt(question, context)
        answer = llm.invoke(prompt).content
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"answer": f"❌ Chatbot Error: {str(e)}"})

@app.route("/", methods=["GET"])
def home():
    return "Chat service running. Use POST /chatbot."

if __name__ == "__main__":
    app.run(port=5001, debug=True)
