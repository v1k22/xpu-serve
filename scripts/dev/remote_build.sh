#!/usr/bin/env bash
# Dev helper: copy this working tree to a remote test machine and build it there.
#
#   scripts/dev/remote_build.sh <system> <tag> [cmake args...]
#
# <system> is a name known to the `remote` CLI (e.g. ptl-86, b60-dgpu). <tag> keeps parallel work apart: the tree
# lands in ~/xpu-serve/src-<tag> and builds in ~/xpu-serve/src-<tag>/build. Run binaries there with
#   remote exec <system> "cd ~/xpu-serve/src-<tag> && source scripts/dev/remote_env.sh && build/src/xpu-serve ..."
set -euo pipefail
system=${1:?system}
tag=${2:?tag}
shift 2
root=$(cd "$(dirname "$0")/../.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
tar -C "$root" --exclude='./build*' --exclude='./.git' --exclude='__pycache__' --exclude='./tests/fixtures/generated' \
    -czf "$tmp/src.tgz" .
remote copy "$system" "$tmp/src.tgz:/tmp/xs-src-$tag.tgz" >/dev/null
remote exec "$system" "mkdir -p ~/xpu-serve/src-$tag && tar -C ~/xpu-serve/src-$tag -xzf /tmp/xs-src-$tag.tgz && rm /tmp/xs-src-$tag.tgz" 120 | head -1
remote exec "$system" "cd ~/xpu-serve/src-$tag && source scripts/dev/remote_env.sh && cmake -S . -B build -G Ninja -DOpenVINO_DIR=\$XS_OV_ROOT/runtime/cmake $* >/dev/null && cmake --build build 2>&1 | tail -15" 900
