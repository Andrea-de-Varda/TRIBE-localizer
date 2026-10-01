#!/bin/bash
# Clone LLM_Modularity at the pinned commit into data/external (git-ignored).
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=https://github.com/Pengrui-Han/LLM_Modularity.git
COMMIT=e3ac7fbb3a6caea05c88343a8de6ec04a4035db8
DEST=data/external/LLM_Modularity
if [ ! -d "$DEST/.git" ]; then
    git clone --filter=blob:limit=5m "$REPO" "$DEST"
fi
git -C "$DEST" fetch --depth 1 origin "$COMMIT" 2>/dev/null || true
git -C "$DEST" checkout -q "$COMMIT"
echo "LLM_Modularity at $(git -C "$DEST" rev-parse HEAD)"
