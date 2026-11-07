#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Script para compilar Fritchii en Linux nativamente

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_DIR="${BUILD_DIR:-build}"
JOBS="${JOBS:-$(nproc)}"

echo "=== Configurando Fritchii para Linux ($BUILD_TYPE) ==="

CMAKE_FLAGS=(
    -S .
    -B "$BUILD_DIR"
    -G Ninja
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DBUILD_TESTING=OFF
    -DYUZU_TESTS=OFF
    -DYUZU_CMD=ON
    -DYUZU_ROOM=OFF
    -DYUZU_ROOM_STANDALONE=OFF
    -DYUZU_USE_BUNDLED_QT=OFF
    -DYUZU_USE_BUNDLED_SDL3=ON
    -DVulkanHeaders_FORCE_BUNDLED=ON
    -DVulkanUtilityLibraries_FORCE_BUNDLED=ON
    -DENABLE_UPDATE_CHECKER=ON
    -DUSE_DISCORD_PRESENCE=OFF
)

# Ejecutar CMake
cmake "${CMAKE_FLAGS[@]}"

echo "=== Compilando con $JOBS hilos ==="
cmake --build "$BUILD_DIR" --target yuzu yuzu-cmd --parallel "$JOBS"

echo "=== Compilación completada con éxito ==="
echo "Ejecutables listos en:"
echo "  - GUI: $BUILD_DIR/bin/lichii"
echo "  - CLI: $BUILD_DIR/bin/eden-cli"
