.PHONY: native build test test-go vet fmt
ASR_PREFIX ?= $(CURDIR)/.native
native:
	MLX_ROOT="$(MLX_ROOT)" ASR_PREFIX="$(ASR_PREFIX)" ./scripts/build-native.sh
build:
	mkdir -p bin
	PKG_CONFIG_PATH="$(ASR_PREFIX)/lib/pkgconfig:$$PKG_CONFIG_PATH" go build -o bin/transcribe ./cmd/transcribe
test:
	PKG_CONFIG_PATH="$(ASR_PREFIX)/lib/pkgconfig:$$PKG_CONFIG_PATH" go test -race ./...
test-go:
	CGO_ENABLED=0 go test ./...
vet:
	PKG_CONFIG_PATH="$(ASR_PREFIX)/lib/pkgconfig:$$PKG_CONFIG_PATH" go vet ./...
fmt:
	gofmt -w .
