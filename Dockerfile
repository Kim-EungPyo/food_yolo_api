FROM python:3.10-slim

# Avoid interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# Install required system packages
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy the app
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Expose FastAPI port
EXPOSE 5000

# Run FastAPI app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
