#!/bin/bash

# 1. Install system dependencies and pyngrok
echo "Installing dependencies..."
pip install -r requirements.txt
pip install pyngrok

# 2. Install Ollama (The AI Engine)
echo "Installing Ollama..."
curl -L https://ollama.com/install.sh | sh

# 3. Start Ollama server in the background
echo "Starting Ollama server..."
ollama serve > ollama.log 2>&1 &

# 4. Wait for Ollama to start and pull the models
echo "Pulling AI models (this may take a few minutes)..."
sleep 10
ollama pull gemma4:31b-cloud
ollama pull dengcao/Qwen3-Embedding-8B:Q4_K_M
ollama pull Krakekai/qwen3-reranker-8b

# 5. Start the Python API
echo "Launching Chatbot Server..."
python api.py &
