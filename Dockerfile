# Lightweight Python 3.11 base image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install curl for healthchecks and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and pyproject first for layer caching
COPY requirements.txt pyproject.toml ./

# Install CPU PyTorch first (drastically reduces image size from ~6GB to ~1GB)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and configuration
COPY src/ ./src/
COPY app/ ./app/
COPY configs/ ./configs/
COPY runs/ ./runs/

# Install the plantdx package in editable mode
RUN pip install --no-cache-dir -e .

# Expose Streamlit default port
EXPOSE 8501

# Streamlit health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch the Streamlit application
ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]

