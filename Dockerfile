FROM python:3.12-slim
WORKDIR /app
# PyTorch's build for the CPU: the server has no GPU, and the CUDA build is gigabytes bigger.
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt
COPY app.py tokenizer.py ./
ENV HOST=0.0.0.0
EXPOSE 7860
CMD ["python", "app.py"]
