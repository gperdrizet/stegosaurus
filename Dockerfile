FROM python:3.12-slim

# Create directory for application code
WORKDIR /stegosaurus

# Copy requirements.txt file into app directory
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r ./requirements.txt

# Move source code
COPY src/ src/
COPY demo/ demo/

# Set some environment vars
ENV PORT=8080
ENV HF_HOME=/tmp/huggingface

# Expose port for Gradio
EXPOSE 8080

# Launch the app
CMD ["python", "demo/app.py"]