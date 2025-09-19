# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# System dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project first (so setup.py is available for -e .)
COPY . .

# Install Python dependencies (including editable install of your project)
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask API port
EXPOSE 5000

# Start Flask API
CMD ["python", "app.py"]
