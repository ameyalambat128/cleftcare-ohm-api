# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the entire project into the container
COPY . /app

# Install system-level dependencies (for librosa and ffmpeg)
RUN apt-get update && apt-get install -y ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the FastAPI port
EXPOSE 8080

# Set the default command to run the FastAPI app
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}
