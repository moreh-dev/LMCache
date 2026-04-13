#!/usr/bin/env bash
# LMCache bench-build 브랜치 빌드 & 설치 스크립트
#
# 사용법:
#   ./scripts/rebuild_and_install.sh           # editable install (기본)
#   ./scripts/rebuild_and_install.sh --clean   # 캐시 정리 후 설치
#   ./scripts/rebuild_and_install.sh --no-edit # non-editable install
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CLEAN=false
EDITABLE=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean)    CLEAN=true; shift ;;
    --no-edit)  EDITABLE=false; shift ;;
    *)          echo "Unknown option: $1"; exit 1 ;;
  esac
done

cd "$REPO_DIR"

echo "============================================"
echo " LMCache Build & Install"
echo "============================================"
echo "Branch:    $(git branch --show-current)"
echo "Commit:    $(git log --oneline -1)"
echo "Editable:  $EDITABLE"
echo "Clean:     $CLEAN"
echo "============================================"

# Clean
if [[ "$CLEAN" == "true" ]]; then
  echo ">>> Cleaning build artifacts..."
  rm -rf build/ dist/ *.egg-info src/*.egg-info
  # 구버전 dist-packages 잔재 제거
  DIST_PATH="/usr/local/lib/python3.12/dist-packages/lmcache"
  if [[ -d "$DIST_PATH" ]]; then
    echo "    Removing stale install at $DIST_PATH"
    rm -rf "$DIST_PATH"
  fi
  pip uninstall -y lmcache 2>/dev/null || true
fi

# Build & Install
echo ">>> Installing..."
if [[ "$EDITABLE" == "true" ]]; then
  TORCH_DONT_CHECK_COMPILER_ABI=1 \
  CXX=hipcc \
  BUILD_WITH_HIP=1 \
  python3 -m pip install --no-build-isolation -e .
else
  TORCH_DONT_CHECK_COMPILER_ABI=1 \
  CXX=hipcc \
  BUILD_WITH_HIP=1 \
  python3 -m pip install --no-build-isolation .
fi

# Verify
echo ""
echo ">>> Verifying..."
python3 -c "
import lmcache
print(f'  lmcache path:    {lmcache.__file__}')
from lmcache.v1.config import LMCacheEngineConfig
print(f'  LMCacheEngineConfig: OK')
from lmcache.v1.lookup_client.mooncake_lookup_client import MooncakeLookupClient
print(f'  MooncakeLookupClient: OK')
"

echo ""
echo "============================================"
echo " Done!"
echo "============================================"
