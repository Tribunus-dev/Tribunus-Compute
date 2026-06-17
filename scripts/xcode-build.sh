#!/bin/bash
# Xcode build script — invokes Cargo for the Tribunus server.
#
# Environment variables set by Xcode:
#   SRCROOT       = source root
#   CONFIGURATION = Debug | Release
#   ARCHS         = arm64 (Apple Silicon)
#   BUILT_PRODUCTS_DIR = output directory for built products

set -euo pipefail
 
 # macOS workaround: unset MallocStackLogging inherited from Xcode/LLDB/shell
 # before any subprocess (cmake, ninja, omp threads) inherits it.
 # This suppresses "can't turn off malloc stack logging because it was not enabled"
 # on stderr during process shutdown.
 export MallocStackLogging=
 export MallocStackLoggingNoCompact=

cd "$SRCROOT"

# Map Xcode configuration to Cargo profile
CARGO_FLAGS="--features server"
PROFILE_DIR="debug"
if [ "$CONFIGURATION" = "Release" ]; then
    CARGO_FLAGS="$CARGO_FLAGS --release"
    PROFILE_DIR="release"
fi

echo "Building tribunus-server (${CONFIGURATION})..."
cargo build $CARGO_FLAGS --package tribunus-compute-core --bin tribunus-server 2>&1

# Copy built binary to Xcode's built products dir
BINARY_PATH="target/${PROFILE_DIR}/tribunus-server"
if [ -f "$BINARY_PATH" ]; then
    cp "$BINARY_PATH" "$BUILT_PRODUCTS_DIR/"
    echo "Copied tribunus-server to ${BUILT_PRODUCTS_DIR}/"
else
    echo "Warning: Built binary not found at ${BINARY_PATH}"
fi
