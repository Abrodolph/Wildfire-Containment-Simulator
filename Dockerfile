FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Expose port for HF Spaces
EXPOSE 7860

# Start the OpenEnv HTTP server on port 7860
CMD ["python", "server/app.py"]
