# qwen-asr-go

Native Go bindings for **Qwen3-ASR-0.6B**, powered by the MLX C++ runtime.

```text
Go → cgo → C++ / MLX → Metal or CUDA
```

The model stays loaded between calls. Inference runs inside the Go process,
without a Python interpreter, an HTTP API, or a separate worker process.
Asynchronous decoding with one-token lookahead is enabled by default.

## Status

| Backend | Status |
| --- | --- |
| macOS / Apple Silicon / Metal | Built and smoke-tested locally |
| Linux / NVIDIA / CUDA | Build configuration prepared; not built or tested on NVIDIA |
| Linux / CUDA Docker image | Dockerfile provided; not built or tested |
| CPU | Explicit build/runtime option; performance is not validated |

Only MLX-format **Qwen3-ASR-0.6B, affine 4-bit, group size 64** checkpoints
are supported. Model weights are not included.

CUDA operator compatibility, transcription quality, performance, and resource
handling still need validation on a Linux/NVIDIA host.

## Requirements

- Go 1.24.1 or later with cgo enabled.
- A C++20 compiler, CMake 3.25 or later, and `pkg-config`.
- MLX C++ SDK **0.32.4**.
- nlohmann-json 3.11 or later.
- For Metal: Apple Silicon and Xcode Command Line Tools.
- For CUDA: Linux, a supported NVIDIA driver, CUDA Toolkit, cuDNN, and BLAS/LAPACK.

The MLX headers and libraries can come from an installed `mlx` package.
The native inference runtime does not use its Python bindings.

## Installation

Install the Go module in your application:

```sh
go get github.com/nkschmidt/qwen-asr-go@v0.1.2
```

`go get` downloads Go source. It does **not** install MLX or build the native
library. Build and install that library first, then expose its `.pc` file when
building your application:

```sh
git clone https://github.com/nkschmidt/qwen-asr-go.git
cd qwen-asr-go
MLX_ROOT=/absolute/path/to/mlx-sdk make native
export PKG_CONFIG_PATH="$PWD/.native/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
make build
```

The SDK directory must contain `include/` and `lib/`. On macOS, a path such as
`/path/to/site-packages/mlx` can be used as `MLX_ROOT`.

The default native install prefix is `.native`. Set `ASR_PREFIX` to use a
shared SDK location. On macOS, installation bundles `libmlx.dylib` and `mlx.metallib` next to
`libqwen_asr_mlx.dylib` by default and uses `@loader_path`, so inference does not
load its MLX runtime from the original SDK or Python environment. Keep the native
installation available at runtime. Set `QWEN_ASR_BUNDLE_MLX_RUNTIME=OFF` in CMake
only when managing MLX shared libraries separately. Linux uses the configured
MLX installation and the Docker runtime copies both library directories.

Build separately for each OS and architecture. A macOS binary or native
installation cannot be reused in Linux.

## Go API

Create a recognizer when the application starts and reuse it across requests:

```go
import asr "github.com/nkschmidt/qwen-asr-go"

recognizer, err := asr.New(ctx, asr.Config{
    ModelPath:  "/models/qwen3-asr-0.6b-4bit",
    Device:     asr.Auto,
    DecodeMode: asr.Async,
})
if err != nil {
    return err
}
defer recognizer.Close()

result, err := recognizer.Recognize(ctx, pcm16, "de")
// result.Text, result.Language, result.Device, result.ProcessingTime
```

`Recognize` accepts **PCM16LE, mono, 16 kHz**, from 25 milliseconds to
28 seconds. Inputs shorter than one second are zero-padded. Supply an ISO
language code such as `de`, `ru`, or `en`; use `auto` or an empty string for
automatic language detection. Supported codes are listed in `languages.go`.

`Result` includes the text, language code, model identifier, selected device,
total processing time, and decoding time after prefill.

### Device and decoding options

- `asr.Auto`: selects the backend configured in the native build.
- `asr.Metal`: requires a Metal-enabled build and an available Metal GPU.
- `asr.CUDA`: requires a CUDA-enabled build and an available CUDA GPU.
- `asr.CPU`: explicitly selects CPU execution with a CPU-enabled MLX runtime.
- `asr.Async`: asynchronous decoding with one-token lookahead; the default.
- `asr.Sync`: synchronous decoding for comparison.

An unavailable GPU does not silently fall back to CPU. In a CPU-only native
build, `Auto` selects that explicitly configured CPU backend.

### Concurrency and lifecycle

A recognizer is safe to call from multiple goroutines. Requests execute
sequentially on a dedicated OS thread. All recognizers in this module also
share a native execution lock protecting MLX defaults and tokenizer state.
Multiple recognizers do not provide parallel inference in this version.

Context cancellation returns control to the caller. Native code checks a
cancellation flag between stages and tokens; a running GPU kernel is not
preempted. Pending GPU work, including extra lookahead after EOS, is drained
before request resources are released.

`Close` is idempotent, cancels active work, waits for cleanup, and releases the
model. It can wait for a running load or GPU kernel. Canceling `New` can return
before native loading finishes; cleanup then completes in the background.

This library recognizes complete audio utterances. Transport, voice activity
detection, and application-specific orchestration belong to the caller.

## Command-line example

```sh
./bin/transcribe \
  -model /path/to/model \
  -wav /path/to/speech.wav \
  -language de \
  -device auto \
  -decode async
```

The CLI expects a PCM16 mono 16 kHz WAV and prints a JSON result. Applications
can import the root package directly without using the CLI.

## Linux / CUDA build — unverified

Build MLX C++ 0.32.4 with CUDA support, then build the native library:

```sh
git clone --depth 1 --branch v0.32.4 https://github.com/ml-explore/mlx.git
cmake -S mlx -B mlx-build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/opt/mlx -DCMAKE_INSTALL_LIBDIR=lib \
  -DBUILD_SHARED_LIBS=ON -DMLX_BUILD_CUDA=ON -DMLX_BUILD_METAL=OFF \
  -DMLX_BUILD_CPU=ON -DMLX_BUILD_PYTHON_BINDINGS=OFF \
  -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build mlx-build --parallel 4
cmake --install mlx-build

# Run from the qwen-asr-go checkout.
MLX_ROOT=/opt/mlx ASR_BACKEND=cuda ASR_PREFIX=/opt/qwen-asr ./scripts/build-native.sh
export PKG_CONFIG_PATH=/opt/qwen-asr/lib/pkgconfig
make build ASR_PREFIX=/opt/qwen-asr
```

`80` is an example CUDA architecture; choose the value for your GPU. Installation
prefixes must be writable or replaced with suitable paths.

For an explicit CPU-only native build, set `ASR_BACKEND=cpu` and use an MLX SDK
built with CPU support.

## Docker / CUDA — unverified

`Dockerfile.cuda` builds MLX without Python bindings, the native shared
library, and the example CLI. Python may be installed as a build tool in the
build stage; native inference does not invoke it.

The host needs the NVIDIA driver and NVIDIA Container Toolkit:

```sh
docker build --platform linux/amd64 -f Dockerfile.cuda -t qwen-asr-go:cuda .
docker run --rm --gpus all \
  -v /absolute/path/model:/models/qwen:ro \
  -v /absolute/path/audio:/audio:ro \
  qwen-asr-go:cuda \
  -model /models/qwen -wav /audio/speech.wav -language de -device cuda
```

Build arguments: `BUILD_JOBS`, `CUDA_ARCHITECTURES` (default: `80;86;89;90`),
and `MLX_VERSION`. The image runs a CLI, not a network service.

When embedding the library in another Go application, use the same MLX/native
installation in its build stage and copy `/opt/mlx/lib` and `/opt/qwen-asr/lib`
into its runtime image alongside the application binary. Model weights are
mounted separately.

## Development and tests

```sh
make test-go  # Pure Go lifecycle contracts; no native runtime required.
make test     # Race tests; requires the installed native library.
make vet
```

`CGO_ENABLED=0` does not enable inference: `New` returns
`ErrNativeUnavailable`. This mode supports pure Go contract tests.

Contract tests cover serialization, request cancellation, PCM buffer ownership,
idempotent close, and cleanup after canceled loading. Native model smoke tests
are opt-in and do not download weights or recordings:

```sh
QWEN_ASR_TEST_MODEL=/path/to/model \
QWEN_ASR_TEST_PCM=/path/to/audio.pcm \
QWEN_ASR_TEST_LANGUAGE=de \
PKG_CONFIG_PATH="$PWD/.native/lib/pkgconfig:${PKG_CONFIG_PATH:-}" \
  go test -race -run TestNativeModelLifecycle -v .
```

For local development in a consuming application:

```sh
go mod edit -replace github.com/nkschmidt/qwen-asr-go=../qwen-asr-go
```

## License and attribution

MIT. See `LICENSE` and `NOTICE`. Third-party notices are included under
`licenses/` and `native/vendor/qwen-asr/`. Model weights have their own license
and are not distributed with this repository.

## Decoder diagnostics

`Result.Diagnostics` contains decoder mode, input/audio/generated token counts,
forced language, final token/EOS IDs, repeated-token metrics and the stop reason.
Native decoding errors are wrapped in `*DecodeError`; use `errors.As` to inspect
its `Diagnostics`. The SDK does not log audio or partial text automatically.

Stop reasons are `eos`, `generation_limit`, `canceled`, or `error`. The 512-token
limit is unchanged. No automatic retry or promotion of incomplete text is added.
Repetition metrics are diagnostic signals, not proof that a transcript is invalid.
The KV cache remains request-local and asynchronous lookahead is drained before
request resources are released. Per-request timings and diagnostics are reset.

Rebuild/reinstall the native library before using this SDK revision: the C ABI
adds `asr_diagnostics`. Older native installations do not export this symbol.

`TestNativeSyncAsyncDiagnosticsAndRequestIsolation` compares repeated sync and
async recognition of the same PCM fixture. Configure `QWEN_ASR_TEST_MODEL` and
`QWEN_ASR_TEST_PCM` as for the existing native lifecycle test. It checks EOS and
output consistency. A passing fixture does not rule out failures on other audio.
