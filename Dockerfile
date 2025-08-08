# Use a small Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for some packages like unstructured
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    tesseract-ocr \
    libmagic-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Expose port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "backend_new:app", "--host", "0.0.0.0", "--port", "8000"]
