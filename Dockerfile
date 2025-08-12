FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
RUN pip install uv

RUN apt update && \
    apt install -y espeak-ng && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . ./

# Create triton cache directory with proper permissions
RUN mkdir -p /tmp/triton_cache && chmod 777 /tmp/triton_cache

RUN uv pip install --system -e . && uv pip install --system -e .[compile]
