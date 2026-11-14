#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${MLX_ROOT:?Set MLX_ROOT to an installed MLX 0.32.4 C++ SDK prefix}"
MLX_ROOT="$(cd "$MLX_ROOT" && pwd)"
ASR_PREFIX="${ASR_PREFIX:-$PWD/.native}"
if [[ -z "${ASR_BACKEND:-}" ]]; then
  case "$(uname -s)" in Darwin) ASR_BACKEND=metal ;; Linux) ASR_BACKEND=cuda ;; *) echo "Unsupported OS" >&2; exit 1 ;; esac
fi
cmake -S native -B "build/$ASR_BACKEND" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$MLX_ROOT;${CMAKE_PREFIX_PATH:-}" \
  -DCMAKE_INSTALL_PREFIX="$ASR_PREFIX" -DCMAKE_INSTALL_LIBDIR=lib \
  -DQWEN_ASR_BACKEND="$ASR_BACKEND"
cmake --build "build/$ASR_BACKEND" --parallel "${BUILD_JOBS:-4}"
cmake --install "build/$ASR_BACKEND"
printf '\nGo build environment:\nexport PKG_CONFIG_PATH="%s/lib/pkgconfig:${PKG_CONFIG_PATH:-}"\n' "$ASR_PREFIX"
