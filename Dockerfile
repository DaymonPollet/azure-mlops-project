# Stage 1: Build/Install Dependencies
FROM python:3.9-slim-buster AS builder

WORKDIR /app

# copy only requirements.txt first to leverage Docker cache
# if requirements.txt doesn't change
COPY requirements.txt .

# install dependancies: use --no-cache-dir for smaller image, and
# --compile for pre-compiling Python files, which can speed up startup.
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime Image
FROM python:3.9-slim-buster

WORKDIR /app

# copy pathing installer / packages
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

# copy application code AND the model (since the model is inside inference/)
COPY inference/ ./inference/

ENV MODEL_BASE_PATH=/app/inference/animal-classification/INPUT_model_path/animal-cnn

EXPOSE 8000

CMD ["uvicorn", "inference.main:app", "--host", "0.0.0.0", "--port", "8000"]