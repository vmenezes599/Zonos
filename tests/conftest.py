"""Shared fixtures for Zonos integration tests."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Generator

import pytest
import requests


def _is_server_running(health_url: str) -> bool:
    """Check if the Zonos server is already running."""
    try:
        response = requests.get(health_url, timeout=1)
        return response.status_code == 200
    except requests.RequestException:
        return False


def _start_uvicorn_server(server_dir: Path, port: int, health_url: str) -> subprocess.Popen:
    """Start a uvicorn server for Zonos and wait until healthy."""
    python_dir = Path(sys.executable).parent
    uvicorn_path = python_dir / "uvicorn"

    if uvicorn_path.exists():
        cmd = [str(uvicorn_path), "generic_ai_server_client.server:app", "--host", "0.0.0.0", "--port", str(port)]
    else:
        cmd = [sys.executable, "-m", "uvicorn", "generic_ai_server_client.server:app", "--host", "0.0.0.0", "--port", str(port)]

    process = subprocess.Popen(
        cmd,
        cwd=server_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    for _ in range(30):
        if _is_server_running(health_url):
            return process
        time.sleep(0.5)

    process.kill()
    pytest.fail(f"Zonos server failed to start on port {port}")


@pytest.fixture(name="server_process", scope="module")
def _server_process() -> Generator[subprocess.Popen | None, None, None]:
    """Use a running server if present, otherwise start one for the tests."""
    server_dir = Path(__file__).parent.parent
    port = 8189
    health_url = f"http://127.0.0.1:{port}/health"

    if _is_server_running(health_url):
        yield None
        return

    process = _start_uvicorn_server(server_dir, port, health_url)
    yield process

    if process:
        process.terminate()
        process.wait(timeout=5)


@pytest.fixture(name="server_url")
def _server_url() -> str:
    """Base URL for Zonos server endpoints."""
    return "http://127.0.0.1:8189"


@pytest.fixture(name="assets_dir")
def _assets_dir() -> Path:
    """Path to bundled test assets."""
    return Path(__file__).parent.parent / "assets"
