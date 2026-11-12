#!/bin/sh
# Builds internal/backend/coreai's Swift shim (libcaishim.dylib, embedded with go:embed). Needs a Mac
# with Xcode 27 (its SDK has Core AI's subframeworks); the Command Line Tools' swiftc is enough.
set -e
cd "$(dirname "$0")/../.."
SDK=${SDK:-$(xcrun --sdk macosx --show-sdk-path 2>/dev/null)}
[ -d "$SDK/System/Library/SubFrameworks/CoreAIRuntime.framework" ] || SDK=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
SWIFTC=${SWIFTC:-swiftc}
"$SWIFTC" -sdk "$SDK" -target arm64-apple-macos27.0 -O -parse-as-library -emit-library \
  -Xlinker -install_name -Xlinker @rpath/libcaishim.dylib \
  internal/backend/coreai/shim/shim.swift -o internal/backend/coreai/libcaishim.dylib
ls -l internal/backend/coreai/libcaishim.dylib
