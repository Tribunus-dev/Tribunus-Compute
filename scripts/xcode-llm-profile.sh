#!/bin/bash
# scripts/xcode-llm-profile.sh
# Installs the Tribunus Xcode AI provider profile.
#
# Usage: ./scripts/xcode-llm-profile.sh [install|uninstall]

set -euo pipefail

cd "$(dirname "$0")/.."

PROFILE_SRC="Tribunus/Resources/tribunus-xcode-provider.plist"
PROFILE_DST="$HOME/Library/Preferences/com.apple.dt.Xcode.CustomLLMProvider.plist"

case "${1:-install}" in
    install)
        if [ ! -f "$PROFILE_SRC" ]; then
            echo "Error: $PROFILE_SRC not found"
            exit 1
        fi
        cp "$PROFILE_SRC" "$PROFILE_DST"
        echo "Xcode profile installed. Restart Xcode to pick up changes."
        ;;
    uninstall)
        rm -f "$PROFILE_DST"
        echo "Xcode profile removed."
        ;;
    *)
        echo "Usage: $0 [install|uninstall]"
        exit 1
        ;;
esac
