FROM python:3.13-slim

# ffmpeg: shelled out to by app/api/routes/voice_samples.py and
# voice_enrollment.py to convert whatever format the browser's
# MediaRecorder produced (webm/opus) to 16kHz mono WAV.
# libgomp1: required at runtime by torch/onnxruntime (OpenMP), not
# pulled in automatically by the slim base image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# CPU-only torch build - this backend's own CUDA probe
# (StreamingTranscriber.__init__) falls back to CPU here anyway (see
# scripts/benchmark_command_latency.py's real measurement this
# session), so the ~2GB CUDA wheel would be dead weight. Swap this back
# to a plain `pip install -r requirements.txt` if you deploy somewhere
# with working GPU passthrough and want CUDA torch instead.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
