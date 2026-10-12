#!/usr/bin/env bash
set -Eeuo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
# Help, analysis and planning require only the standard library, not a GPU/env install.
case "${1:-}" in
  --help|-h|--plan-only|--analyze) exec python3 "$HERE/run.py" "$@" ;;
esac
CONDA_BIN="${CONDA_EXE:-}"
if [[ ! -x "$CONDA_BIN" ]]; then CONDA_BIN="$(command -v conda || true)"; fi
if [[ ! -x "$CONDA_BIN" ]]; then
  for candidate in "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" /opt/conda/bin/conda; do
    if [[ -x "$candidate" ]]; then CONDA_BIN="$candidate"; break; fi
  done
fi
if [[ ! -x "$CONDA_BIN" ]]; then
  echo 'Conda chưa có trên máy này. Cài Miniforge/Conda tại login node rồi chạy lại; không thay đổi Python hệ thống.' >&2
  exit 1
fi
ENV_NAME="${BITMARK_ENV:-bitmark-radioactivity}"
if [[ ! "$ENV_NAME" =~ ^[a-zA-Z0-9_-]+$ || "$ENV_NAME" == base ]]; then
  echo 'BITMARK_ENV phải là tên một môi trường riêng, không phải base.' >&2; exit 1
fi
mkdir -p "$HERE/cache"
# Serialize setup across concurrent jobs; release before starting experiments.
exec 9>"$HERE/cache/environment.lock"
flock 9
if ! "$CONDA_BIN" run -n "$ENV_NAME" python -c 'import sys; assert sys.version_info[:2] == (3, 11)' >/dev/null 2>&1; then
  "$CONDA_BIN" create -y -n "$ENV_NAME" python=3.11 pip
fi
"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python "$HERE/setup_env.py"
flock -u 9
exec 9>&-
if [[ "${1:-}" == --setup-only ]]; then exit 0; fi
exec "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python "$HERE/run.py" "$@"
