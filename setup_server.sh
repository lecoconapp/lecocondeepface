#!/usr/bin/env bash
# One-time setup for the Le Cocon DeepFace server on a fresh Ubuntu 22.04/24.04 VPS.
# Run as a normal user with sudo rights (e.g. the root user or the default user).
#
#   curl -fsSL https://raw.githubusercontent.com/lecoconapp/lecocondeepface/main/setup_server.sh | bash
#
set -e

echo "==> Updating packages"
sudo apt-get update -y
sudo apt-get upgrade -y

echo "==> Installing Python 3.11 + pip"
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update -y
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
python3.11 -m ensurepip --upgrade || true

echo "==> Installing system libs (OpenCV needs these)"
sudo apt-get install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 wget

echo "==> Creating project folder"
sudo mkdir -p /opt/lecocon-deepface
sudo chown $USER /opt/lecocon-deepface
cd /opt/lecocon-deepface

echo "==> Downloading server code from GitHub"
wget -q https://raw.githubusercontent.com/lecoconapp/lecocondeepface/main/app.py
wget -q https://raw.githubusercontent.com/lecoconapp/lecocondeepface/main/requirements.txt
# Procfile not needed on a VPS (we run with gunicorn).

echo "==> Creating venv"
python3.11 -m venv venv
./venv/bin/pip install --upgrade pip

echo "==> Installing DeepFace + gunicorn (this downloads ~500MB of models, takes a few minutes)"
./venv/bin/pip install -r requirements.txt
./venv/bin/pip install gunicorn

echo "==> All installed."
echo "NEXT: start it with:"
echo "  cd /opt/lecocon-deepface && ./venv/bin/gunicorn --workers 1 --threads 1 --timeout 600 -b 0.0.0.0:8000 app:app"
echo "For first test:  curl http://127.0.0.1:8000/health"
