# TOmni-Care Agent — Dockerfile
# 多階段建構：builder 安裝依賴 → runtime 只含執行時必要檔案

# ── Stage 1: builder ────────────────────────────────────────
FROM python:3.11-slim AS builder

# 系統依賴（librosa 需要 libsndfile，faster-whisper 需要 ffmpeg）
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# 先只複製 requirements 讓 Docker layer cache 生效
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn[standard]

# ── Stage 2: runtime ────────────────────────────────────────
FROM python:3.11-slim AS runtime

# 只安裝執行時系統依賴
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 從 builder 複製已安裝的 Python 套件
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

# 複製專案原始碼
COPY src/ ./src/
COPY data/knowledge/ ./data/knowledge/
COPY .env.example .env.example

# 環境變數預設值（實際值由 docker-compose / -e 覆蓋）
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    DEVICE=cpu \
    ASR_MODEL_SIZE=small

# 預設執行 FastAPI API 服務
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
