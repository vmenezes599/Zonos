#!/bin/bash

VERSION=$(cat "$(dirname "$0")/docker_freeze_version")

# Build zonos_server prod image
DOCKER_BUILDKIT=1 docker build \
  --target zonos_server \
  -t ghcr.io/vmenezes599/ct_zonos_server:${VERSION} \
  .
