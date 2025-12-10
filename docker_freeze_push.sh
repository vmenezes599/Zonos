#!/bin/bash

VERSION=$(cat "$(dirname "$0")/docker_freeze_version")

docker push ghcr.io/vmenezes599/ct_zonos_server:${VERSION}
