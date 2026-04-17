FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_SERVER_PORT=7871

WORKDIR /app

# Runtime libs needed by opencv/ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps from requirements
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# App + models (same directory as script)
COPY gradio_yolo_presence_app_V2.py /app/gradio_yolo_presence_app_V2.py
COPY modele_1.pt /app/modele_1.pt
COPY modele_2.pt /app/modele_2.pt

EXPOSE 7871

CMD ["python", "gradio_yolo_presence_app_V2.py"]
