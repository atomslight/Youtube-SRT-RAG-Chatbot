#!/bin/bash

# 1. Install system dependencies (zstd is required for Ollama installation)
echo "Installing system dependencies..."
apt-get update && apt-get install -y zstd

# 2. Install Python dependencies
echo "Installing Python dependencies..."
# Install web requirements if the file exists
if [ -f "requirements_web.txt" ]; then
    pip install -r requirements_web.txt
fi
# Install core AI dependencies directly to ensure they are present
pip install numpy pysrt yt-dlp langchain-core langchain-ollama rank-bm25 sentence-transformers torch pyngrok

# 3. Install Ollama (The AI Engine)
echo "Installing Ollama..."
curl -L https://ollama.com/install.sh | sh

# 4. Start Ollama server in the background
echo "Starting Ollama server..."
ollama serve > ollama.log 2>&1 &

# 5. Wait for Ollama to start and pull the models
echo "Waiting for Ollama to initialize..."
sleep 10

# Verify ollama is installed and pull models
if command -v ollama &> /dev/null
then
    echo "Pulling AI models (this may take a few minutes)..."
    ollama pull gemma4:31b-cloud
    ollama pull dengcao/Qwen3-Embedding-8B:Q4_K_M
    ollama pull Krakekai/qwen3-reranker-8b
else
    echo "ERROR: Ollama installation failed. Please check ollama.log"
    exit 1
fi

# 6. Start the Python API
echo "Launching Chatbot Server..."
python api.py &