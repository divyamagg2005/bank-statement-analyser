#!/bin/bash
set -e

# Install system dependencies
echo "Installing system dependencies..."
apt-get update
apt-get install -y --no-install-recommends \
    ghostscript \
    python3-dev \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
echo "Installing Python dependencies..."
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo "Setup completed successfully!"
