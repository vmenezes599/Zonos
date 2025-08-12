#!/bin/bash
# Health check script for command server

set -e

# Check if the server is responding
if curl -f -k -s http://127.0.0.1:8189/health > /dev/null 2>&1; then
    echo "TTS Server is healthy"
    exit 0
else
    echo "TTS Server is not responding"
    exit 1
fi