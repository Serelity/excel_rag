#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ENV_FILE=${RAG_ENV_FILE:-$PROJECT_ROOT/deploy/.env}

if [[ ! -r $ENV_FILE ]]; then
  echo "deployment environment is not readable: $ENV_FILE" >&2
  echo "create it with: cp deploy/.env.example deploy/.env" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

: "${QDRANT_HOST:=127.0.0.1}"
: "${QDRANT_HTTP_PORT:=6333}"
: "${QDRANT_GRPC_PORT:=6334}"
: "${QDRANT_STORAGE_PATH:=/srv/rag/qdrant}"

if [[ $QDRANT_HOST != 127.0.0.1 ]]; then
  echo "QDRANT_HOST must remain 127.0.0.1; configure authenticated TLS proxying separately" >&2
  exit 1
fi
if [[ -z ${QDRANT_BIN:-} || ! -x $QDRANT_BIN ]]; then
  echo "QDRANT_BIN must point to an executable Qdrant server binary" >&2
  exit 1
fi
if [[ $QDRANT_BIN != /* ]]; then
  echo "QDRANT_BIN must be an absolute path" >&2
  exit 1
fi
if [[ $QDRANT_STORAGE_PATH != /* ]]; then
  echo "QDRANT_STORAGE_PATH must be an absolute path" >&2
  exit 1
fi
qdrant_version=$("$QDRANT_BIN" --version 2>&1) || {
  echo "unable to execute Qdrant version check" >&2
  exit 1
}
if [[ ! $qdrant_version =~ (^|[[:space:]])1\.14\.1($|[[:space:]]) ]]; then
  echo "QDRANT_BIN must be version 1.14.1; found: $qdrant_version" >&2
  exit 1
fi

mkdir -p "$QDRANT_STORAGE_PATH"
if ! ulimit -n 65535 2>/dev/null; then
  echo "warning: could not raise the open-file limit to 65535" >&2
fi

export QDRANT__SERVICE__HOST="$QDRANT_HOST"
export QDRANT__SERVICE__HTTP_PORT="$QDRANT_HTTP_PORT"
export QDRANT__SERVICE__GRPC_PORT="$QDRANT_GRPC_PORT"
export QDRANT__STORAGE__STORAGE_PATH="$QDRANT_STORAGE_PATH"
export QDRANT__TELEMETRY_DISABLED=true
if [[ -n ${QDRANT_API_KEY:-} ]]; then
  export QDRANT__SERVICE__API_KEY="$QDRANT_API_KEY"
else
  unset QDRANT__SERVICE__API_KEY || true
fi

cd "$PROJECT_ROOT"
exec "$QDRANT_BIN"
