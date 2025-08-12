#!/bin/bash
# Health check script for command server

set -e

# Check if the server is responding
if curl -f -k -s http://127.0.0.1:8188/ > /dev/null 2>&1; then
    echo "Command server is healthy"
    exit 0
else
    echo "Command server is not responding"
    exit 1
fi