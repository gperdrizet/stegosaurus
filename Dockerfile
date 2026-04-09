FROM python:3.12-slim

# Create directory for application code
WORKDIR /stegosaurus

# Copy requirements file into app directory
COPY requirements-deploy.txt .

# Install Python dependencies.
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Move source code
COPY src/ src/
COPY demo/ demo/

# Set directory for model downloads and cache
ENV HF_HOME=/tmp/huggingface
ENV TORCH_DTYPE=float32
ENV MODEL=Qwen/Qwen2.5-1.5B

# Set and expose port for Gradio
ENV PORT=8080
EXPOSE 8080

# Launch the app
CMD ["python", "demo/app.py"]