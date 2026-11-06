#!/usr/bin/env bash
# Install a Node.js LTS (GEAK's headless runtime needs Node >= 20) under ~/.local/opt without root.
# Usage: install_node.sh [vXX.YY.Z]   (default: newest LTS from nodejs.org/dist/index.json)
# Verifies the tarball against nodejs.org's SHASUMS256.txt and points ~/.local/opt/node at it.
set -euo pipefail

ARCH_SUFFIX="linux-x64"
case "$(uname -m)" in
  x86_64) ARCH_SUFFIX="linux-x64" ;;
  aarch64) ARCH_SUFFIX="linux-arm64" ;;
  *) echo "unsupported machine $(uname -m)" >&2; exit 2 ;;
esac

VERSION="${1:-}"
if [[ -z "$VERSION" ]]; then
  VERSION="$(curl -sSfL https://nodejs.org/dist/index.json |
    python3 -c 'import json,sys; print(next(x["version"] for x in json.load(sys.stdin) if x["lts"]))')"
fi

PREFIX="$HOME/.local/opt"
TARBALL="node-$VERSION-$ARCH_SUFFIX.tar.xz"
WORK="$(mktemp -d /tmp/node_dl.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

curl -sSfL -o "$WORK/$TARBALL" "https://nodejs.org/dist/$VERSION/$TARBALL"
curl -sSfL -o "$WORK/SHASUMS256.txt" "https://nodejs.org/dist/$VERSION/SHASUMS256.txt"
(cd "$WORK" && grep " $TARBALL\$" SHASUMS256.txt | sha256sum -c -)

mkdir -p "$PREFIX"
tar -xJf "$WORK/$TARBALL" -C "$PREFIX"
ln -sfn "$PREFIX/node-$VERSION-$ARCH_SUFFIX" "$PREFIX/node"
"$PREFIX/node/bin/node" --version
echo "add to PATH: export PATH=$PREFIX/node/bin:\$PATH"
