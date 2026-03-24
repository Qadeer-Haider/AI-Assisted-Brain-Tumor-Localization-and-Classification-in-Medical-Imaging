# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies (needed for opencv)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies inside building stage
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# We'll copy the application code
COPY app/ ./app/

# In Docker Compose we'll mount the whole project root or weights dir to /workspace or 
# copy them if we are standalone. But for generic docker, we can mount it.
# To make it self contained, we could copy weights, but they are large.
# Let's assume mounting is used, but we'll set working dir inside /app.
WORKDIR /app/app
# Expose port
EXPOSE 8501

# Command to run the application
CMD [\"streamlit\", \"run\", \"app.py\", \"--server.port=8501\", \"--server.address=0.0.0.0\"]
