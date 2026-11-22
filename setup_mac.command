#!/bin/bash
set -e
cd "$(dirname "$0")"
if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is required. Install Python 3.10 or newer first."
  exit 1
fi
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env. Add your Gemini and Groq API keys before starting the app."
fi
echo "Setup complete. Edit .env, then double-click run_mac.command."
