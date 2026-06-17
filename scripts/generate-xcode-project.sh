#!/bin/bash
# Regenerates Tribunus.xcodeproj from xcodegen.yml
#
# Usage: scripts/generate-xcode-project.sh
# Prerequisites: brew install xcodegen

set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v xcodegen &> /dev/null; then
    echo "xcodegen not found. Install via: brew install xcodegen"
    exit 1
fi

xcodegen generate --spec xcodegen.yml

echo "Xcode project generated at Tribunus.xcodeproj"
