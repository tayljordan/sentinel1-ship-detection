#!/usr/bin/env bash
# Fetch the Allen Institute for AI vessel detection code and the Sentinel-1 model weights.
# The weights are stored with Git LFS, so they are downloaded directly.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AI2_DIR="${AI2_VESSEL_DETECTION_DIR:-$REPO_ROOT/third_party/vessel-detection-sentinels}"
AI2_URL="https://github.com/allenai/vessel-detection-sentinels"
MEDIA_URL="https://media.githubusercontent.com/media/allenai/vessel-detection-sentinels/main"

if [ ! -d "$AI2_DIR/.git" ]; then
  echo "Cloning $AI2_URL"
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 -q "$AI2_URL" "$AI2_DIR"
fi

for path in \
  src/model_artifacts/frcnn_cmp2/3dff445/best.pth \
  src/model_artifacts/attr/c34aa37/best.pth; do
  target="$AI2_DIR/$path"
  if [ -f "$target" ] && [ "$(wc -c < "$target")" -gt 1000000 ]; then
    echo "Already downloaded: $path"
    continue
  fi
  echo "Downloading $path"
  curl -fsSL -o "$target.tmp" "$MEDIA_URL/$path"
  mv "$target.tmp" "$target"
done

echo "Models ready in $AI2_DIR"
