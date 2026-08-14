#!/usr/bin/env bash
set -e

echo "📦 Setting up project..."

# 1. Install Git LFS if needed
if ! command -v git-lfs &> /dev/null; then
    echo "⚠️ Git LFS not found. Installing..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get update && sudo apt-get install -y git-lfs
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install git-lfs
    fi
    git lfs install
else
    git lfs install
fi

# 2. Pull LFS files
echo "⬇️  Downloading large assets via Git LFS..."
git lfs pull

# 3. Sync Python deps
echo "🐍 Syncing Python environment with uv..."
uv sync

# 4. Drop stray legacy *.egg-info metadata left in site-packages.
# The lerobot wheel ships a top-level `lerobot-<ver>.egg-info` PKG-INFO file alongside
# its .dist-info. uv reads that as a second, distutils-installed copy of the package and
# then refuses the next sync with "distutils-installed distributions do not include the
# metadata required to uninstall safely". The .dist-info is the authoritative record, so
# the .egg-info is safe to remove.
echo "🧹 Cleaning stray legacy egg-info metadata..."
find .venv/lib/python*/site-packages -maxdepth 1 -name '*.egg-info' -type f -delete 2>/dev/null || true

echo "✅ Project setup complete!"
