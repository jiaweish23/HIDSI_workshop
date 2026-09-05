#!/usr/bin/env bash
# =============================================================================
# Fetch the large diffusion checkpoint (LDM_ema.pt, ~1.5 GB) that is NOT stored
# in the git repo.  It lives as an asset on a GitHub *Release* of this repo
# (GitHub allows up to 2 GB per release asset, vs a 100 MB limit inside the repo).
#
# Usage (after cloning, from the repo root):
#     ./download_checkpoint.sh
#
# Override the source if you forked / renamed:
#     REPO=youruser/yourrepo TAG=checkpoints ./download_checkpoint.sh
# =============================================================================
set -euo pipefail

# ---- EDIT THESE TWO to match where you upload the release asset -------------
REPO="${REPO:-jiaweish23/HIDSI_workshop}"         # override with REPO=... if you fork
TAG="${TAG:-checkpoints}"                          # the Release tag holding the asset
# -----------------------------------------------------------------------------
ASSET="LDM_ema.pt"
DEST="checkpoints/${ASSET}"
EXPECTED_MD5="51dad29c3d697584b0b68dede5f1ecd3"

cd "$(dirname "$0")"
mkdir -p checkpoints

verify() {
  if command -v md5sum >/dev/null 2>&1; then
    local got; got="$(md5sum "$DEST" | awk '{print $1}')"
    if [ "$got" != "$EXPECTED_MD5" ]; then
      echo "ERROR: md5 mismatch for $DEST"
      echo "  expected $EXPECTED_MD5"
      echo "  got      $got"
      exit 1
    fi
    echo "md5 OK ($got)"
  fi
}

if [ -f "$DEST" ]; then
  echo "$DEST already present — verifying ..."
  verify
  echo "Nothing to do."
  exit 0
fi

echo "Downloading $ASSET from release '$TAG' of $REPO ..."
if command -v gh >/dev/null 2>&1; then
  gh release download "$TAG" --repo "$REPO" --pattern "$ASSET" --dir checkpoints
else
  URL="https://github.com/${REPO}/releases/download/${TAG}/${ASSET}"
  echo "gh CLI not found; using wget: $URL"
  wget -O "$DEST" "$URL"
fi

verify
echo "Done: $DEST"
