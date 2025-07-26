#!/bin/bash

# Install Ghostscript for PDF processing
apt-get update
apt-get install -y ghostscript

# Install Python dependencies
pip install -r requirements.txt
