FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel AS builder

RUN pip install uv

RUN apt update && \
    apt install -y espeak-ng curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app/

RUN groupadd -g 1000 appgroup && useradd -u 1000 -g appgroup -m appuser
RUN chown -R appuser:appgroup /app

# Create triton cache directory with proper permissions
RUN mkdir -p /tmp/triton_cache && chmod 777 /tmp/triton_cache
RUN mkdir -p /.cache && chmod 777 /.cache
RUN mkdir -p /.config/pulse && chmod 777 /.config && chmod 777 /.config/pulse

RUN chmod +x /app/health-check.sh

RUN uv pip install --system -e . && uv pip install --system -e .[compile]

# Compile Python files and remove source
RUN python -m compileall -b -q --invalidation-mode unchecked-hash /app/entry_points.py /app/tts_processor.py /app/CT_generic_server_client && \
    find /app/CT_generic_server_client -type d -name '__pycache__' -prune -exec rm -rf '{}' + && \
    find /app -maxdepth 1 -type d -name '__pycache__' -prune -exec rm -rf '{}' + && \
    rm -f /app/entry_points.py /app/tts_processor.py && \
    find /app/CT_generic_server_client -type f -name '*.py' -delete

FROM builder AS zonos_server

EXPOSE 8189

CMD [ "python3", "-m", "CT_generic_server_client.server", "--port", "8189" ]