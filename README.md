 **HealthBot – AI-Driven Medical Assistant Chatbot**

HealthBot is an AI-powered medical assistant designed to provide **instant healthcare support** through intelligent symptom analysis, first-aid guidance, emergency awareness, and basic skin-disease classification. It was developed to address healthcare accessibility challenges in regions like **Pakistan**, where medical resources are often limited and people struggle to receive timely, reliable medical advice.


 **Why HealthBot?**

In many areas, especially underserved communities, people face:

* Difficulty accessing doctors quickly
* Lack of reliable medical information
* Delays in receiving guidance during emergencies
* Dependence on unverified online sources

**HealthBot bridges this gap** by offering fast, AI-generated medical responses that help users make safe, informed decisions—without needing to physically visit a hospital.


 **How HealthBot Works**

HealthBot supports **two types of inputs**:


 **1. Text-Based Medical Queries (RAG + LLaMA)**

When users describe symptoms or ask health-related questions:

1. The text is processed using **SentenceTransformer** to generate vector embeddings.
2. A **FAISS index** retrieves medically relevant documents.
3. The **LLaMA 3 language model**, powered through the **Groq API**, generates a clear, medically informed response.
4. The system includes **safety filters** to handle sensitive topics such as overdosing, self-harm, or emergencies.

This ensures responses are fast, reliable, and grounded in real medical content.


 **2. Image-Based Skin Disease Detection (CNN)**

Users can upload images of visible skin conditions.
HealthBot uses a **TensorFlow-based Convolutional Neural Network (CNN)** to classify the condition and provide appropriate guidance.

This dual-input design allows the chatbot to support a wide range of healthcare concerns.


 **Technology Stack**

 **Backend**

* Python
* Flask
* TensorFlow (CNN Model)
* SentenceTransformer
* FAISS
* LLaMA 3 via Groq API

 **Frontend**

* HTML
* CSS
* JavaScript


 **Performance & Safety Features**

* Responds in **under 5 seconds**
* Detects harmful or unsafe user inputs
* Offers emergency instructions when necessary
* Minimizes misinformation by relying on trusted medical content
* Reduces non-urgent hospital visits by providing instant digital support


 **Future Enhancements**

Planned features include:

* Multilingual support (Urdu & regional languages)
* Voice-based interaction
* Android mobile application
* Real-time telemedicine integration
* Connecting users with local hospitals or pharmacies


 **Full Project Download (Google Drive)**

 **Google Drive Link:**
  https://drive.google.com/drive/folders/17Uq_jlajg88a8cpL20JJBDC1LSwhVgbK?usp=sharing


  **Repository Structure (GitHub Version)**

This GitHub repository includes only the **main source code**, excluding heavy model files.

```
HealthBot/
│── app.py
│── templates/
│── static/
│── models/ (placeholder)
│── faiss_index/ (placeholder)
│── utils/
│── README.md
```

---
