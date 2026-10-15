FROM python:3.12-slim AS base
WORKDIR /app
# PyTorch's build for the CPU: the server has no GPU, and the CUDA build is gigabytes bigger.
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
ENV HOST=0.0.0.0

# The completions API: docker build --target server -t mvgpt-server .
FROM base AS server
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt
COPY server.py tokenizer.py ./
EXPOSE 8000
CMD ["python", "server.py"]

# The Gradio app. It's the last stage, so it's what docker build makes without a --target.
FROM base AS app
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt
COPY app.py tokenizer.py ./
EXPOSE 7860
CMD ["python", "app.py"]
