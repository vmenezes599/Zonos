#!/bin/bash

DOCKER_COMPOSE_FILE="docker-compose.prod.yml"
DEV_DOCKER_COMPOSE_FILE="docker-compose.dev.yml"

case "$1" in

build | b)
    echo "🔨 Building Zonos..."
    docker compose -f $DOCKER_COMPOSE_FILE down
    docker compose -f $DOCKER_COMPOSE_FILE build
    docker compose -f $DOCKER_COMPOSE_FILE up -d
    echo "✅ Build completed!"
    ;;

dev_build | db)
    echo "🔨 Building dev_Zonos..."
    docker compose -f $DEV_DOCKER_COMPOSE_FILE down
    docker compose -f $DEV_DOCKER_COMPOSE_FILE build --no-cache
    docker compose -f $DEV_DOCKER_COMPOSE_FILE up -d
    echo "✅ Build completed!"
    ;;

*)
    echo "Zonos Server - Docker Management"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "🔨 BUILD:"
    echo "  b/build        - Build Zonos"
    echo "  db/dev_build    - Build Zonos in development mode"
    echo ""
    echo ""
    ;;
esac
