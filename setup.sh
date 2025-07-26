#!/bin/bash
set -e

echo "=== Starting setup ==="

# Install system dependencies
echo "Installing system dependencies..."
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ghostscript \
    python3.10-dev \
    python3-pip \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
update-alternatives --set python3 /usr/bin/python3.10

# Ensure pip is using the correct Python version
python3 -m pip install --upgrade pip setuptools wheel

# Install Python dependencies
echo "Installing Python dependencies..."
python3 -m pip install -r requirements.txt

# Install ghostscript Python bindings
echo "Installing ghostscript Python bindings..."
python3 -m pip install --no-deps ghostscript==0.7

echo "=== Setup completed successfully ==="
