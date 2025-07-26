#!/bin/bash
set -e

echo "=== Starting setup ==="

# Install system dependencies
echo "Installing system dependencies..."
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ghostscript \
    python3-dev \
    python3-pip \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Ensure we have a recent version of pip and setuptools
echo "Upgrading pip and setuptools..."
python3 -m pip install --upgrade pip setuptools wheel

# Install numpy first with specific build options to avoid compilation
echo "Installing numpy with specific build options..."
NUMPY_CFLAGS="-O3 -march=native -mtune=native" python3 -m pip install --no-binary :all: --only-binary=numpy,scipy numpy==1.26.0

# Install Python dependencies
echo "Installing Python dependencies..."
python3 -m pip install -r requirements.txt

# Install ghostscript Python bindings
echo "Installing ghostscript Python bindings..."
python3 -m pip install --no-deps ghostscript==0.7

# Verify installations
echo "Verifying installations..."
python3 -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python3 -c "import pandas; print(f'Pandas version: {pandas.__version__}')"
python3 -c "import streamlit; print(f'Streamlit version: {streamlit.__version__}')"

echo "=== Setup completed successfully ==="
