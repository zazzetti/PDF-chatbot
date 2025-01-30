# 📚 FastAPI PDF Chatbot with LangChain & Docker

This project is a FastAPI-based chatbot that processes PDF documents, extracts text, and enables conversational retrieval using LangChain, FAISS, and Mistral AI.

---

## 🚀 Features
- 📂 Upload and process PDFs from a folder
- 🔎 Vectorize text with FAISS and Sentence Transformers
- 🗣️ Conversational memory with LangChain
- 🤖 AI-powered responses using Mistral AI
- 🐳 Dockerized for easy deployment



## 🐳 Docker Setup

### 1️⃣ Set up environment variables
Create a `.env` file in the root directory and add:
```env
MISTRAL_API_KEY=your_mistral_api_key
```

### 2️⃣ Build the Docker image
```bash
docker build -t fastapi-pdf-chatbot .
```

### 3️⃣ Run the container
```bash
docker run -p 8000:8000 fastapi-pdf-chatbot
```

---

API will be accessible at: http://localhost:8000/docs

---

## 📡 API Endpoints

### 📂 Process PDFs from a folder
```
POST /process_folder/
```
- Extracts and processes text from PDFs inside the `data/` folder

### 💬 Chat with processed PDFs
```
POST /chat/
```
- **Request Body:** `question: str`
- **Response:** AI-generated answer based on document content



