#!/bin/bash
set -e

echo "=== Global Economic Risk Monitor ==="

# Install dependencies
echo "[1/4] Installing dependencies..."
pip install -r requirements.txt --quiet

# Ingest data
echo "[2/4] Fetching macroeconomic data..."
python -m data.ingest

# Run models
echo "[3/4] Running risk models..."
python -m models.recession
python -m models.inflation
python -m models.composite

# Launch app
echo "[4/4] Starting dashboard at http://localhost:8501"
streamlit run app.py
