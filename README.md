# YouTube RAG Chatbot
> Chat with any YouTube video and get exact, timestamped answers instantly.

## 📖 Overview
The YouTube RAG (Retrieval-Augmented Generation) Chatbot is an intelligent web application that allows you to paste a YouTube video URL and converse directly with its content. It automatically downloads the video's transcript, processes it completely locally on your machine, and provides accurate answers to your questions. This saves you from having to scrub through hours of video just to find a single piece of specific information.

## ✨ Features
- 📺 **Instant Video Chat**: Paste any YouTube URL and start chatting with the transcript immediately.
- 🌍 **Auto-Language Detection**: Automatically detects and downloads the best available subtitle language (prioritizing English and Hindi).
- 🔒 **100% Local & Private**: All data processing and AI inference happen locally on your hardware. No data is sent to external AI providers.
- 🎯 **Pinpoint Accuracy**: Uses hybrid search (BM25 keyword search + semantic embeddings) and advanced cross-encoder reranking to find the exact information you need.
- ⏱️ **Timestamped Citations**: Every answer includes clickable timestamps, taking you exactly to the moment in the video where the fact was mentioned.
- 💬 **Sleek Web UI**: A clean, responsive chat interface built with HTML/JS that streams answers in real-time.

## 🛠 Tech Stack
- **Python**: The core programming language powering the backend logic and AI pipeline.
- **FastAPI**: A high-performance web framework used to serve the API and the web interface.
- **Ollama**: A tool for running large language models (LLMs) locally. It generates text answers and creates embeddings without relying on paid APIs.
- **LangChain**: A framework that simplifies building AI applications, used here for managing the in-memory vector store and document chunks.
- **yt-dlp**: A robust command-line tool used behind the scenes to download video transcripts and metadata directly from YouTube.
- **Sentence Transformers & Rank-BM25**: Core machine learning libraries used to rank and score the most relevant transcript segments for your query.
- **HTML/CSS/JS (Vanilla)**: Powers the frontend UI seamlessly using Server-Sent Events (SSE) for real-time text streaming.

## 📋 Prerequisites
Before you start, make sure you have the following installed on your machine:
- **Python 3.9+**: Ensure Python and `pip` are installed on your system.
- **Git**: Required to clone the repository.
- **Ollama**: Must be installed and running in the background (`http://localhost:11434`). You can download it from [ollama.com](https://ollama.com/).

## 🚀 Local Development (Step-by-Step)

### 1. Clone the repository
Open your terminal and run the following commands to download the code and navigate into the project folder:
```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Install dependencies
It is highly recommended to use a Python virtual environment to keep your dependencies isolated. Run these commands sequentially:

```bash
# Create a virtual environment named "venv"
python -m venv venv

# Activate the virtual environment
# On macOS and Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install the web framework dependencies
pip install -r requirements_web.txt

# Install the core RAG, AI, and processing libraries required by the engine
pip install numpy pysrt yt-dlp langchain-core langchain-ollama rank-bm25 sentence-transformers torch
```

### 3. Environment setup
Because this project processes data 100% locally using Ollama, **you do not need any third-party API keys (like OpenAI) and do not need to configure a `.env` file.**

However, you *must* download the required local AI models before running the app. Ensure the Ollama application is running on your machine, then execute these commands in your terminal to pull the models:

```bash
# Pull the generation model
ollama pull gemma4:31b-cloud

# Pull the embedding model
ollama pull dengcao/Qwen3-Embedding-8B:Q4_K_M

# Pull the reranking model
ollama pull Krakekai/qwen3-reranker-8b
```
*(Note: If you have limited hardware, you can open `rag_core.py` and modify `ANSWER_MODEL` to a smaller model you have downloaded, like `llama3` or `mistral`.)*

### 4. Run the development server
Once the models are downloaded and your Python dependencies are installed, start the FastAPI server:
```bash
python api.py
```
*(Alternatively, you can start it using Uvicorn directly: `uvicorn api:app --host 0.0.0.0 --port 8000 --reload`)*

Open your browser and navigate to [http://localhost:8000](http://localhost:8000) to use the application!

### 5. Build for production
Since the frontend is built entirely with vanilla HTML, CSS, and JavaScript, there is no separate build step for the UI (like you would have with React or Vue).

To run the server in a production environment, simply use a production-ready web server command without the `--reload` flag:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

## 🧠 How It Works (Architecture)
1. **Ingestion**: When you paste a URL into the UI, the backend uses `yt-dlp` to grab the video's transcript.
2. **Chunking**: The transcript is sliced into small, overlapping text blocks (chunks) so the AI context window isn't overwhelmed.
3. **Indexing**: Each chunk is converted into numbers (embeddings) using a local Ollama model and stored in an in-memory vector database. A keyword index (BM25) is simultaneously built.
4. **Hybrid Retrieval & Reranking**: When you ask a question, the system searches the chunks using both exact keywords and overall semantic meaning. It then reranks the best results using a Cross-Encoder to ensure maximum relevance.
5. **Answer Generation**: The top transcript chunks are handed over to the local generation LLM, which formulates a clear answer and attaches the exact video timestamps as citations!

## 📁 Folder Structure
```text
.
├── api.py                  # The main FastAPI web server and API routes
├── main.py                 # A standalone, command-line version of the application
├── rag_core.py             # Core logic: downloading, chunking, retrieving, and AI setup
├── requirements_web.txt    # Python dependencies required to run the web server
└── static/                 # Folder containing all frontend assets
    ├── index.html          # The interactive chat web interface
    ├── yt-logo.png         # Graphical assets for the UI
    └── yt-logo.jpeg        # Alternative graphical asset
```