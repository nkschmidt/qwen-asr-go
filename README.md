# qwen-asr-go

Go-библиотека Qwen3-ASR-0.6B: Go → cgo → C++ MLX → Metal или CUDA.
Без Python, HTTP и отдельного inference worker во время распознавания.
Модель загружается один раз. По умолчанию используется async lookahead на один токен.

**Статус:** macOS/Metal проверяется локально. Linux/CUDA и Dockerfile подготовлены,
но не собирались и не запускались на NVIDIA. Совместимость всех CUDA-операций,
точность и скорость на Linux пока не подтверждены. CPU — явный дополнительный режим,
его скорость не является целью этой версии. Поддержан один checkpoint-формат:
MLX Qwen3-ASR-0.6B, affine 4 bit, group size 64. Веса не включены.

## Как подключать в Go

Это отдельный Go module, а не микросервис. У него нет зависимостей от Mixcall.
Нужны и Go-код, и установленная нативная библиотека с MLX:

```sh
# Для приватного репозитория: доступ GitHub должен быть настроен в Git.
export GOPRIVATE="${GOPRIVATE:+$GOPRIVATE,}github.com/nkschmidt/*"
go get github.com/nkschmidt/qwen-asr-go@v0.1.0
```

Если GOPRIVATE уже настроен, добавьте этот шаблон к существующим значениям.
Локальная разработка в потребляющем приложении:

```sh
go mod edit -replace github.com/nkschmidt/qwen-asr-go=../qwen-asr-go
```

`go get` загружает исходники Go; он не устанавливает MLX/CUDA и не собирает CMake.
Перед сборкой приложения выполните установку native ниже и задайте PKG_CONFIG_PATH.
Это cgo-приложение: `CGO_ENABLED=0` не включает распознавание. В таком режиме
`New` возвращает `ErrNativeUnavailable`; чистые Go-тесты доступны без MLX.

```go
import (
    "context"
    asr "github.com/nkschmidt/qwen-asr-go"
)

recognizer, err := asr.New(ctx, asr.Config{
    ModelPath: "/models/qwen3-asr-0.6b-4bit",
    Device: asr.Auto, // Metal на Mac; CUDA в CUDA-сборке. Без скрытого CPU fallback.
    DecodeMode: asr.Async,
})
if err != nil { return err }
defer recognizer.Close()

result, err := recognizer.Recognize(context.Background(), pcm16, "de")
// result.Text, result.Language, result.Device, result.ProcessingTime
```

PCM16LE mono 16000 Hz, 25 мс–28 с; короткие фразы дополняются до одной секунды.
Язык — ISO code (de, ru, en, …), `auto` или пустая строка. Языки — languages.go.
Неподдерживаемый язык и неверный формат отклоняются до вызова модели.

Вызовы безопасны из нескольких goroutine и выполняются последовательно на закреплённом
OS thread. Все экземпляры внутри Go-модуля также разделяют блокировку MLX/tokenizer.
Для MVP обычно нужен один экземпляр на ASR-процесс, не экземпляр на звонок.
Отмена context возвращает управление вызывающему коду; native кооперативно проверяет
флаг между стадиями/токенами. Запущенное GPU-ядро не прерывается. Перед освобождением
ресурсов завершается вся работа GPU, включая дополнительный шаг после EOS.
`Close` идемпотентен, отменяет активный запрос и ждёт очистки. Отмена загрузки модели
может вернуть ошибку раньше, чем завершится фоновая загрузка и её очистка.

Это библиотека распознавания готовых фраз. SIP, VAD, аудиопотоки и брокер остаются в ASR/Gateway.

## macOS / Metal

Нужны Apple Silicon, Xcode Command Line Tools, Go 1.24.1+, CMake 3.25+,
pkg-config, nlohmann-json 3.11+ и MLX C++ SDK **0.32.4**.
Можно использовать include/lib из установленного пакета mlx; Python runtime не нужен.

```sh
MLX_ROOT=/absolute/path/to/site-packages/mlx make native
export PKG_CONFIG_PATH="$PWD/.native/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
make build
./bin/transcribe -model /path/to/model -wav /path/to/speech.wav -language de
```

Нативная библиотека устанавливается в `.native` или `ASR_PREFIX`. Этот каталог и
каталог библиотек MLX должны сохраняться при запуске приложения (rpath записывается
при сборке). Для общей установки задайте `ASR_PREFIX=/your/sdk/prefix`; root не требуется.
Не используйте macOS `.native` или собранный бинарник в Linux: там нужна отдельная сборка.

## Linux / CUDA — подготовлено, не проверено

Нужен Linux/NVIDIA с совместимым драйвером, CUDA Toolkit, cuDNN, C++20,
BLAS/LAPACK, CMake, pkg-config и nlohmann-json. Сначала MLX C++ 0.32.4:

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
MLX_ROOT=/opt/mlx ASR_BACKEND=cuda ASR_PREFIX=/opt/qwen-asr ./scripts/build-native.sh
export PKG_CONFIG_PATH=/opt/qwen-asr/lib/pkgconfig
make build ASR_PREFIX=/opt/qwen-asr
```

`80` — пример архитектуры, задайте нужную для своей NVIDIA.
Указанные `/opt` prefixes должны быть доступны на запись либо заменены своими.
`Device=CUDA` проверяет наличие CUDA backend; неподдерживаемый backend не подменяется.
`ASR_BACKEND=cpu` собирает явный CPU-вариант с MLX, собранным с CPU поддержкой.

## Docker — подготовлено, не проверено

Dockerfile.cuda собирает MLX без Python bindings, native shared library и smoke CLI.
Python может быть установлен в build-stage для инструментов сборки; runtime его не запускает.
Нужен NVIDIA Container Toolkit на хосте. Контейнер — CLI для проверки, не HTTP-сервис.

```sh
docker build --platform linux/amd64 -f Dockerfile.cuda -t qwen-asr-go:cuda .
docker run --rm --gpus all \
  -v /absolute/path/model:/models/qwen:ro \
  -v /absolute/path/audio:/audio:ro \
  qwen-asr-go:cuda -model /models/qwen -wav /audio/speech.wav -language de -device cuda
```

Параметры сборки: `BUILD_JOBS`, `CUDA_ARCHITECTURES` (по умолчанию 80;86;89;90),
`MLX_VERSION`. Для потребляющего Go-сервиса используйте ту же установку MLX/native
в build-stage и перенесите `/opt/mlx/lib`, `/opt/qwen-asr/lib` в runtime вместе с бинарником.
Веса передаются volume; ни веса, ни записи в образ/репозиторий не включаются.

## Подключение к Mixcall

Интерфейс принадлежит application Mixcall. В infrastructure пишется маленький адаптер:

```go
type MLX struct { recognizer *asr.Recognizer }

func (m *MLX) Recognize(ctx context.Context, pcm []byte, language string) (ports.Recognition, error) {
    result, err := m.recognizer.Recognize(ctx, pcm, language)
    if err != nil { return ports.Recognition{}, err }
    return ports.Recognition{Text: result.Text, Language: result.Language, Engine: result.Engine}, nil
}
```

`asr.Result` и `ports.Recognition` — разные Go-типы, поэтому нужен этот адаптер.
В `cmd/asr/main.go` создаются `asr.Recognizer` и адаптер; в application передаётся
только `ports.Recognizer`. Оригинальный брокер и Gateway менять не требуется.
Рабочий Mixcall этим репозиторием автоматически не переключается.

## Проверки

```sh
make test-go       # Go contracts without native dependencies
make test          # race tests, requires installed native library
make vet
```

Проверяются сериализация, отмена активного/ожидающего запроса, владение PCM,
повторное закрытие и очистка отменённой загрузки. Отдельные model smoke-проверки
запускаются CLI на локальных данных. Linux/CUDA сборка и runtime в этой версии
намеренно не проверялись. Требуются проверка операций CUDA, сравнение текстов,
повторные вызовы, отмена и измерение задержек/памяти на реальной NVIDIA.

Источники, версии и лицензии — NOTICE, LICENSE и licenses/.
