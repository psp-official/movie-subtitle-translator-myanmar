#!/usr/bin/env bash
set -o errexit

apt-get update
apt-get install -y ffmpeg libgl1-mesa-glx libglib2.0-0

pip install -r requirements.txt
