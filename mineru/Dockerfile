# Официальный образ MinerU (docker/global/Dockerfile) собран на базе
# vllm/vllm-openai — это оправдано для vlm/hybrid-бэкендов, но лишнее для
# нашего случая: подтверждён "pipeline"-бэкенд (кириллица через явную
# recognition-модель). Он тянет torch + torchvision + onnxruntime и НЕ тянет
# paddle, поэтому GPU здесь — это обычный CUDA-torch, а не отдельная
# paddlepaddle-gpu сборка. Берём обычный python-образ: CUDA-рантайм приезжает
# вместе с колёсами torch (nvidia-*), драйвер пробрасывается с хоста через
# nvidia-container-toolkit. Свой CUDA-базовый образ не нужен.

FROM python:3.12-slim-bookworm

# libgl1/libglib2.0 — стандартная нужда OpenCV/Paddle-зависимостей внутри
# mineru[pipeline], без них падает на "libGL.so.1: cannot open shared
# object file" при первом вызове разбора документа.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Вариант torch задаётся явно, а не отдаётся на откуп дефолту PyPI:
#   CUDA 12.4 : https://download.pytorch.org/whl/cu124   (по умолчанию)
#   CUDA 12.8 : https://download.pytorch.org/whl/cu128
#   только CPU: https://download.pytorch.org/whl/cpu     (образ ~3 ГБ легче)
# Версия должна быть не новее CUDA-драйвера на хосте (`nvidia-smi`).
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision --index-url ${TORCH_INDEX_URL}

# torch уже стоит нужного варианта — mineru его не переустановит.
RUN pip install --no-cache-dir "mineru[core,pipeline]"

# huggingface по умолчанию; если сервер не видит HF (та же история, что
# была с Baidu-registry для другого агента) — пересобрать с
# --build-arg MINERU_MODEL_SOURCE=modelscope
ARG MINERU_MODEL_SOURCE=huggingface

# Модели тянутся один раз при сборке образа, а не при каждом старте
# контейнера — тот же принцип, что и у ppocr_v5-бэкенда OCR-агента.
# -m pipeline: только модели pipeline-бэкенда, без vlm (мы vlm не используем).
RUN mineru-models-download -s ${MINERU_MODEL_SOURCE} -m pipeline

# После сборки образ самодостаточен — mineru.json уже указывает на
# локальные пути моделей, сеть при старте контейнера не нужна.
ENV MINERU_MODEL_SOURCE=local

EXPOSE 8010

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=5 \
    CMD curl -f http://localhost:8010/health || exit 1

CMD ["mineru-api", "--host", "0.0.0.0", "--port", "8010"]