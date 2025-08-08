# Dockerfile
FROM python:3.11-slim

# Install system dependencies (libgl1 for OpenCV + others for unstructured)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app
COPY . .

# Run the FastAPI app
CMD ["uvicorn", "backend_new:app", "--host", "0.0.0.0", "--port", "8000"]
