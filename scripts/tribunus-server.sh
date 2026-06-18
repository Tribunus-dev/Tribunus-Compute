#!/bin/bash
# Launcher for tribunus-server — unsets MallocStackLogging before any
# macOS process startup so libsystem_malloc doesn't print the spurious
# "can't turn off malloc stack logging because it was not enabled" message
# on stderr during process or thread shutdown.
#
# Usage:
#   ./scripts/tribunus-server.sh [--port 11434] [--model-path ...]

set -euo pipefail

# Unset before this process starts — macOS libsystem_malloc reads this
# during process init (before main()), so remove_var in Rust main() is
# too late for subprocesses like omp threads spawned during Metal init.
export MallocStackLogging=0
export MallocStackLoggingNoCompact=0

cd "$(dirname "$0")/.."

BINARY="compute-native/target/release/tribunus-server"
if [ ! -f "$BINARY" ]; then
    echo "Building tribunus-server..."
    cargo build --release --package tribunus-compute-core --bin tribunus-server \
        --features "server,mlx-backend,ane,accelerate,exo" 2>&1
fi

exec "$BINARY" "$@"
