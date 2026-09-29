# syntax=docker/dockerfile:1
#
# apps/web, containerised. Torch-free like the app itself: the model runs on
# onnxruntime, so no CUDA base image is needed. The ONNX export is mounted at
# /model rather than baked in -- docker-compose.yml wires that up.
#
#   cd docker && SA4BIRDS_MODEL=../../../ckpts/onnx/ssa_LT_seed1 docker compose up --build
#
# The build context is the PARENT directory, so the COPY paths below are relative
# to the app directory, not to this file.
#
FROM python:3.12-slim

# Gradio decodes every uploaded file that is not WAV (mp3, ogg, m4a from a
# phone...) with ffmpeg.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 app
ENV HOME=/home/app
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./
COPY reference/ ./reference/

USER app
ENV GRADIO_TEMP_DIR=/tmp/gradio
EXPOSE 7860

ENTRYPOINT ["python", "-u", "app.py"]

# --host 0.0.0.0 is not optional: app.py defaults to 127.0.0.1, which inside a
# container binds the container's own loopback, so a published port still refuses
# connections.
CMD ["--ckpt", "/model", "--host", "0.0.0.0", "--port", "7860"]
