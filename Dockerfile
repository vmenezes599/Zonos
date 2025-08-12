FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
RUN pip install uv

RUN apt update && \
    apt install -y espeak-ng curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . ./

RUN groupadd -g 1000 appgroup && useradd -u 1000 -g appgroup -m appuser
RUN chown -R appuser:appgroup /app

# Create triton cache directory with proper permissions
RUN mkdir -p /tmp/triton_cache && chmod 777 /tmp/triton_cache
RUN mkdir -p /.cache && chmod 777 /.cache
RUN mkdir -p /.config/pulse && chmod 777 /.config && chmod 777 /.config/pulse

RUN chmod +x /app/health-check.sh

RUN uv pip install --system -e . && uv pip install --system -e .[compile]
