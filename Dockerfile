FROM docker.arvancloud.ir/python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for sentence-transformers and LanceDB
RUN apt-get update && apt-get install -y \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Run embedding generation during build
# RUN python run_embedding.py

# Expose the Streamlit port
EXPOSE 7777

# Command to run the application with persistence
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=7777", "--server.address=0.0.0.0"]