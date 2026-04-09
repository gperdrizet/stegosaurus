FROM python:3.12-slim

# DEVICE=cpu (default) or gpu — controls which PyTorch wheel index is used
ARG DEVICE=cpu
# TORCH_DTYPE is baked in as an ENV so it's available at runtime
ARG TORCH_DTYPE=float32

# Create directory for application code
WORKDIR /stegosaurus

# Copy requirements file into app directory
COPY requirements-deploy.txt .

# Install Python dependencies, selecting the correct wheel index for the device
RUN if [ "$DEVICE" = "gpu" ]; then \
      pip install --no-cache-dir -r requirements-deploy.txt \
        --extra-index-url https://download.pytorch.org/whl/cu126; \
    else \
      pip install --no-cache-dir -r requirements-deploy.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu; \
    fi

# Move source code
COPY src/ src/
COPY demo/ demo/

# Set directory for model downloads and cache
ENV HF_HOME=/tmp/huggingface
ENV TORCH_DTYPE=${TORCH_DTYPE}
ENV MODEL=Qwen/Qwen2.5-1.5B

# Set and expose port for Gradio
ENV PORT=8080
EXPOSE 8080

# Launch the app
CMD ["python", "demo/app.py"]