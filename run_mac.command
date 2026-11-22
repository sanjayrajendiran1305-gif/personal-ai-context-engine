#!/bin/bash
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "Run setup_mac.command first."
  exit 1
fi
if [ ! -f .env ]; then
  echo "Missing .env. Copy .env.example to .env and add your API keys."
  exit 1
fi
source .venv/bin/activate
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
