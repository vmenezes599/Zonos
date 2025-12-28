FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel AS builder

ARG APP_USER=appuser
ARG APP_GROUP=appgroup
ARG APP_UID=1000
ARG APP_GID=1000

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

# Prime dependency layer for caching
COPY pyproject.toml uv.lock ./

# Install build tooling
RUN uv pip install --no-cache-dir build

# Bring in package source and build wheels for the app (including compile extras) plus deps
COPY zonos ./zonos
RUN uv pip wheel --wheel-dir /tmp/wheels .[compile]

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime AS zonos_server

ARG APP_USER=appuser
ARG APP_GROUP=appgroup
ARG APP_UID=1000
ARG APP_GID=1000

ENV XDG_CACHE_HOME=/tmp/.cache \
    TRITON_CACHE_DIR=/tmp/triton_cache \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends espeak-ng && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd -g ${APP_GID} ${APP_GROUP} && \
    useradd -u ${APP_UID} -g ${APP_GROUP} -m ${APP_USER}

WORKDIR /app

# Install from the prebuilt wheelhouse
COPY --from=builder /tmp/wheels /tmp/wheels
RUN pip install --no-cache-dir --no-index --find-links /tmp/wheels zonos[compile]

# Copy runtime scripts not captured by the wheel
COPY CT_generic_server_client ./CT_generic_server_client
COPY entry_points.py tts_processor.py health-check.sh ./

RUN chmod +x /app/health-check.sh && \
    mkdir -p "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" && \
    chown -R ${APP_USER}:${APP_GROUP} "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR"

USER ${APP_USER}

EXPOSE 8189

CMD [ "python3", "-m", "CT_generic_server_client.server", "--port", "8189" ]
