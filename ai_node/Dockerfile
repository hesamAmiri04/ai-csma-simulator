FROM python:3.11-slim

WORKDIR /app

# Install PyTorch (CPU-only, lightweight)
RUN pip install --no-cache-dir \
    torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu \
    numpy==1.26.4

COPY replay_buffer.py .
COPY agent.py .
COPY csma_mac.py .
COPY main.py .

CMD ["python", "-u", "main.py"]
