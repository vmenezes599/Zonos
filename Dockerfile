FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel AS zonos_server

ENV XDG_CACHE_HOME=/tmp/.cache \
    TRITON_CACHE_DIR=/tmp/triton_cache \
    PYTHONUNBUFFERED=1 \
    TZ=Europe/Amsterdam

RUN pip install --no-cache-dir uv

# Install runtime dependencies
RUN apt update && \
    apt install -y --no-install-recommends espeak-ng curl tzdata && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app/

# Install Python dependencies
RUN uv pip install --system --no-cache -e .

# Compile Python files and remove source
RUN python -m compileall -b -q --invalidation-mode unchecked-hash \
    /app/entry_points.py \
    /app/tts_processor.py \
    /app/CT_generic_server_client && \
    find /app/CT_generic_server_client -type d -name '__pycache__' -prune -exec rm -rf '{}' + && \
    find /app -maxdepth 1 -type d -name '__pycache__' -prune -exec rm -rf '{}' + && \
    rm -f /app/entry_points.py /app/tts_processor.py && \
    find /app/CT_generic_server_client -type f -name '*.py' -delete

# Create user and set permissions
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -m appuser && \
    chown -R appuser:appgroup /app && \
    chmod +x /app/health-check.sh /app/entrypoint.sh && \
    mkdir -p "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" /.config/pulse && \
    chmod 777 "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" /.config /.config/pulse

USER appuser

EXPOSE 8189

ENTRYPOINT ["/app/entrypoint.sh"]
