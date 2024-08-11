#!/bin/bash

echo "Installing dependencies..."

pip install -r requirements.txt
pip install groq==0.9.0

if [ -z "$GROQ_API_KEY" ]; then
  echo "GROQ_API_KEY environment variable is not set. Please set it to your project's GROQ API key. to use the groq provider."
fi

pip install httpx==1.0.0.beta0 --force-reinstall

echo "Installation completed successfully!"