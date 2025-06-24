# Image Classification with Redis and FastAPI

This project is a web-based image classification system that allows users to upload images and receive predictions using a pre-trained MobileNetV2 model. The application leverages FastAPI for the API, Redis for storing image data, and TensorFlow for machine learning inference. 

## Features
- Upload images via a browser interface.
- Real-time classification using MobileNetV2 (ImageNet weights).
- Store uploaded images in a Redis database.
- Responsive HTML interface with JavaScript for user interaction.

## Prerequisites
- Python 3.9+
- Redis Cloud account (free tier available at [redis.com/try-free](https://redis.com/try-free))
- Required Python packages (listed in `requirements.txt`)

