//go:build cgo && (darwin || linux)

package asr

/*
#cgo pkg-config: qwen-asr-mlx
#include <stdlib.h>
#include "qwen_asr_mlx.h"
*/
import "C"
import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"
	"unsafe"
)

type mlxModel struct{ state unsafe.Pointer }

func cstring(s string) (*C.char, func()) {
	p := C.CString(s)
	return p, func() { C.free(unsafe.Pointer(p)) }
}
func nativeError(p *C.char) error {
	if p == nil {
		return errors.New("native ASR failed")
	}
	defer C.asr_string_free(p)
	return errors.New(C.GoString(p))
}
func loadNative(cfg Config) (nativeModel, error) {
	path, fp := cstring(cfg.ModelPath)
	defer fp()
	device, fd := cstring(string(cfg.Device))
	defer fd()
	mode, fm := cstring(string(cfg.DecodeMode))
	defer fm()
	var err *C.char
	state := C.asr_load(path, device, mode, &err)
	if state == nil {
		return nil, nativeError(err)
	}
	return &mlxModel{state: state}, nil
}
func (m *mlxModel) recognize(ctx context.Context, pcm []byte, code, name, prompt string) (Result, error) {
	if err := ctx.Err(); err != nil {
		return Result{}, err
	}
	size := len(pcm) / 2
	if size < 16000 {
		size = 16000
	}
	audio := make([]float32, size)
	for i := 0; i < len(pcm)/2; i++ {
		audio[i] = float32(int16(binary.LittleEndian.Uint16(pcm[2*i:]))) / 32768
	}
	// The caller supplies request-local context; keep it empty for warmup/silence.
	lang, fl := cstring(name)
	defer fl()
	p, fp := cstring(prompt)
	defer fp()
	flag := C.asr_cancel_new()
	if flag == nil {
		return Result{}, errors.New("cannot allocate cancellation flag")
	}
	defer C.asr_cancel_free(flag)
	callbackDone := make(chan struct{})
	stop := context.AfterFunc(ctx, func() { C.asr_cancel_set(flag); close(callbackDone) })
	var err *C.char
	started := time.Now()
	text := C.asr_transcribe(m.state, (*C.float)(unsafe.Pointer(&audio[0])), C.int(len(audio)), lang, p, flag, &err)
	elapsed := time.Since(started)
	if !stop() {
		<-callbackDone
	}
	if ctx.Err() != nil {
		if text != nil {
			C.asr_string_free(text)
		}
		if err != nil {
			C.asr_string_free(err)
		}
		return Result{}, ctx.Err()
	}
	var diagnostics DecodeDiagnostics
	if parseErr := json.Unmarshal([]byte(C.GoString(C.asr_diagnostics(m.state))), &diagnostics); parseErr != nil {
		if text != nil {
			C.asr_string_free(text)
		}
		if err != nil {
			C.asr_string_free(err)
		}
		return Result{}, fmt.Errorf("decode diagnostics: %w", parseErr)
	}
	if text == nil {
		return Result{}, &DecodeError{Cause: nativeError(err), Diagnostics: diagnostics}
	}
	defer C.asr_string_free(text)
	if code == "" {
		detected := strings.ToLower(strings.TrimSpace(C.GoString(C.asr_language(m.state))))
		code = "und"
		for k, v := range languages {
			if strings.ToLower(v) == detected {
				code = k
				break
			}
		}
	}
	return Result{Diagnostics: diagnostics, Text: strings.TrimSpace(C.GoString(text)), Language: code, Engine: "qwen3-asr-0.6b-mlx-4bit", Device: Device(C.GoString(C.asr_device(m.state))), ProcessingTime: elapsed, DecodeTime: time.Duration(float64(C.asr_decode_seconds(m.state)) * float64(time.Second))}, nil
}
func (m *mlxModel) close() { C.asr_free(m.state); m.state = nil }
