// Package asr runs Qwen3-ASR through the native MLX C++ runtime.
// Build the native library before building with cgo; see README.md.
package asr

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
)

type Device string

const (
	Auto  Device = "auto"
	Metal Device = "metal"
	CUDA  Device = "cuda"
	CPU   Device = "cpu"
)

type DecodeMode string

const (
	Async DecodeMode = "async"
	Sync  DecodeMode = "sync"
)

var (
	ErrClosed            = errors.New("recognizer closed")
	ErrNativeUnavailable = errors.New("native ASR requires cgo and a supported OS")
)

type Config struct {
	ModelPath  string
	Device     Device     // Auto selects the backend enabled by the native build.
	DecodeMode DecodeMode // Empty selects Async.
}
type DecodeDiagnostics struct {
	DecodeMode           string  `json:"decode_mode"`
	AudioSamples         int     `json:"audio_samples"`
	ForcedLanguage       string  `json:"forced_language"`
	InputTokens          int     `json:"input_tokens"`
	AudioTokens          int     `json:"audio_tokens"`
	GeneratedTokens      int     `json:"generated_tokens"`
	LastTokenID          *int    `json:"last_token_id"`
	EOSTokenID           *int    `json:"eos_token_id"`
	MaxSameTokenRun      int     `json:"max_same_token_run"`
	RepeatedSuffixLength int     `json:"repeated_suffix_length"`
	StopReason           string  `json:"stop_reason"`
	DecodeSeconds        float64 `json:"decode_seconds"`
}
type DecodeError struct {
	Cause       error
	Diagnostics DecodeDiagnostics
}

func (e *DecodeError) Error() string { return e.Cause.Error() }
func (e *DecodeError) Unwrap() error { return e.Cause }

type Result struct {
	Diagnostics    DecodeDiagnostics `json:"diagnostics"`
	Text           string            `json:"text"`
	Language       string            `json:"language"`
	Engine         string            `json:"engine"`
	Device         Device            `json:"device"`
	ProcessingTime time.Duration     `json:"processing_time_ns"`
	DecodeTime     time.Duration     `json:"decode_time_ns"`
}
type nativeModel interface {
	recognize(context.Context, []byte, string, string, string) (Result, error)
	close()
}
type response struct {
	result Result
	err    error
}
type request struct {
	ctx            context.Context
	pcm            []byte
	language, name string
	prompt         string
	reply          chan response
}

// Recognizer serializes requests on an OS thread. It is safe for concurrent callers.
// Additional model instances in this module share the same native execution lock.
type Recognizer struct {
	jobs   chan request
	life   context.Context
	cancel context.CancelFunc
	done   chan struct{}
	once   sync.Once
}

// New loads one local model. Canceling ctx during loading returns promptly;
// native loading may finish in the background before its resources can be freed.
func New(ctx context.Context, cfg Config) (*Recognizer, error) {
	return newRecognizer(ctx, cfg, loadNative)
}
func newRecognizer(ctx context.Context, cfg Config, load func(Config) (nativeModel, error)) (*Recognizer, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if cfg.ModelPath == "" {
		return nil, errors.New("model path is required")
	}
	if cfg.Device == "" {
		cfg.Device = Auto
	}
	if cfg.DecodeMode == "" {
		cfg.DecodeMode = Async
	}
	if cfg.Device != Auto && cfg.Device != Metal && cfg.Device != CUDA && cfg.Device != CPU {
		return nil, fmt.Errorf("unsupported device %q", cfg.Device)
	}
	if cfg.DecodeMode != Async && cfg.DecodeMode != Sync {
		return nil, fmt.Errorf("unsupported decode mode %q", cfg.DecodeMode)
	}
	life, cancel := context.WithCancel(context.Background())
	r := &Recognizer{jobs: make(chan request), life: life, cancel: cancel, done: make(chan struct{})}
	ready := make(chan error, 1)
	go r.run(cfg, load, ready)
	select {
	case err := <-ready:
		if err != nil {
			cancel()
			return nil, err
		}
		if err = ctx.Err(); err != nil {
			cancel()
			return nil, err
		}
		return r, nil
	case <-ctx.Done():
		cancel()
		return nil, ctx.Err()
	}
}

// nativeMu also protects MLX global defaults and the reused tokenizer tables.
var nativeMu sync.Mutex

func (r *Recognizer) run(cfg Config, load func(Config) (nativeModel, error), ready chan<- error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	defer close(r.done)
	nativeMu.Lock()
	model, err := load(cfg)
	nativeMu.Unlock()
	ready <- err
	if err != nil {
		return
	}
	defer func() { nativeMu.Lock(); defer nativeMu.Unlock(); model.close() }()
	for {
		if r.life.Err() != nil {
			return
		}
		select {
		case <-r.life.Done():
			return
		case req := <-r.jobs:
			callCtx, cancel := context.WithCancel(req.ctx)
			stop := context.AfterFunc(r.life, cancel)
			nativeMu.Lock()
			var result Result
			if err = callCtx.Err(); err == nil {
				result, err = model.recognize(callCtx, req.pcm, req.language, req.name, req.prompt)
			}
			nativeMu.Unlock()
			stop()
			cancel()
			if r.life.Err() != nil {
				err = ErrClosed
			}
			req.reply <- response{result, err}
		}
	}
}

// Recognize accepts PCM16LE, mono, 16 kHz, from 25 ms to 28 seconds.
// Audio shorter than one second is zero-padded for Qwen. Language is an ISO code
// (de, ru, en, ...) or auto/empty. Context cancellation is cooperative between
// native stages/tokens; a running GPU kernel is not preempted.
func (r *Recognizer) Recognize(ctx context.Context, pcm []byte, language string) (Result, error) {
	return r.RecognizeWithContext(ctx, pcm, language, "")
}

// RecognizeWithContext supplies request-local vocabulary or conversational context.
// It biases recognition, not a guaranteed answer. Empty context preserves Recognize.
// Context is limited to 4096 UTF-8 bytes; native special-token delimiters and NUL
// are rejected. It is never retained for another request or model warmup.
func (r *Recognizer) RecognizeWithContext(ctx context.Context, pcm []byte, language, prompt string) (Result, error) {
	if len(prompt) > 4096 || !utf8.ValidString(prompt) || strings.ContainsAny(prompt, "\x00") || strings.Contains(prompt, "<|") || strings.Contains(prompt, "|>") {
		return Result{}, errors.New("context must be valid UTF-8, at most 4096 bytes, without NUL or special-token delimiters")
	}
	if err := ctx.Err(); err != nil {
		return Result{}, err
	}
	if r.life.Err() != nil {
		return Result{}, ErrClosed
	}
	if len(pcm) < 800 || len(pcm)%2 != 0 || len(pcm) > 16000*2*28 {
		return Result{}, errors.New("expected 25 ms to 28 s of PCM16LE mono 16 kHz")
	}
	code := strings.ToLower(strings.TrimSpace(language))
	name := ""
	if code == "auto" {
		code = ""
	}
	if code != "" {
		var ok bool
		name, ok = languages[code]
		if !ok {
			return Result{}, fmt.Errorf("unsupported language %q", language)
		}
	}
	req := request{ctx: ctx, pcm: append([]byte(nil), pcm...), language: code, name: name, prompt: prompt, reply: make(chan response, 1)}
	select {
	case <-ctx.Done():
		return Result{}, ctx.Err()
	case <-r.life.Done():
		return Result{}, ErrClosed
	case r.jobs <- req:
	}
	select {
	case <-ctx.Done():
		return Result{}, ctx.Err()
	case <-r.life.Done():
		return Result{}, ErrClosed
	case reply := <-req.reply:
		return reply.result, reply.err
	}
}

// Close cancels active work, drains native execution and releases the model.
// It is idempotent. It may wait for a running load/GPU kernel to finish.
func (r *Recognizer) Close() error { r.once.Do(r.cancel); <-r.done; return nil }
