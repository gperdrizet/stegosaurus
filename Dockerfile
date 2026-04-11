FROM python:3.12-slim

# Create directory for application code
WORKDIR /stegosaurus

# Copy requirements file into app directory
COPY requirements.txt .

# Install Python dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Move source code
COPY src/ src/
COPY demo/ demo/

# Move model(s) to cache directory
RUN mkdir -p /tmp/huggingface
COPY models/ /tmp/huggingface/

# Set directory for model downloads and cache
ENV HF_HOME=/tmp/huggingface
ENV MODEL=Qwen/Qwen3-0.6B

# Set and expose port for Gradio
ENV PORT=8080
EXPOSE 8080

# Run Hugging Face in offline mode to use local models
ENV HF_HUB_OFFLINE=1
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV TRANSFORMERS_OFFLINE=1

# Launch the app
CMD ["python", "demo/app.py"]