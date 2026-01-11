# Stage 1: Builder - Install dependencies
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel AS builder

RUN pip install --no-cache-dir uv

# Install build-time dependencies
RUN apt update && \
    apt install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app/

# Install Python dependencies
RUN uv pip install --system --no-cache -e . && \
    uv pip install --system --no-cache -e .[compile]

# Compile Python files and remove source
RUN python -m compileall -b -q --invalidation-mode unchecked-hash \
    /app/entry_points.py \
    /app/tts_processor.py \
    /app/CT_generic_server_client && \
    find /app/CT_generic_server_client -type d -name '__pycache__' -prune -exec rm -rf '{}' + && \
    find /app -maxdepth 1 -type d -name '__pycache__' -prune -exec rm -rf '{}' + && \
    rm -f /app/entry_points.py /app/tts_processor.py && \
    find /app/CT_generic_server_client -type f -name '*.py' -delete

# Stage 2: Runtime - Clean image without build tools
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime AS zonos_server

ENV XDG_CACHE_HOME=/tmp/.cache \
    TRITON_CACHE_DIR=/tmp/triton_cache \
    PYTHONUNBUFFERED=1

# Install only runtime dependencies
RUN apt update && \
    apt install -y --no-install-recommends espeak-ng curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy entire conda environment from builder (includes all site-packages)
COPY --from=builder /opt/conda /opt/conda

# Copy compiled application from builder
COPY --from=builder /app /app

# Create user and set permissions
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -m appuser && \
    chown -R appuser:appgroup /app && \
    chmod +x /app/health-check.sh && \
    mkdir -p "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" /.config/pulse && \
    chmod 777 "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" /.config /.config/pulse

USER appuser

EXPOSE 8189

CMD ["python3", "-m", "CT_generic_server_client.server", "--port", "8189"]