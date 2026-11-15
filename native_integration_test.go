//go:build cgo && (darwin || linux)

package asr

import (
	"context"
	"errors"
	"os"
	"testing"
	"time"
)

// Explicit opt-in: no downloads, bundled recordings or automatic GPU assumption.
func TestNativeModelLifecycle(t *testing.T) {
	model, pcmPath := os.Getenv("QWEN_ASR_TEST_MODEL"), os.Getenv("QWEN_ASR_TEST_PCM")
	if model == "" || pcmPath == "" {
		t.Skip("set QWEN_ASR_TEST_MODEL and QWEN_ASR_TEST_PCM for native model smoke")
	}
	pcm, err := os.ReadFile(pcmPath)
	if err != nil {
		t.Fatal(err)
	}
	lang := os.Getenv("QWEN_ASR_TEST_LANGUAGE")
	if lang == "" {
		lang = "de"
	}
	r, err := New(context.Background(), Config{ModelPath: model, Device: Auto, DecodeMode: Async})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	result, err := r.Recognize(context.Background(), pcm, lang)
	if err != nil {
		t.Fatal(err)
	}
	if result.Text == "" || result.Language != lang {
		t.Fatal(result)
	}
	if expected := os.Getenv("QWEN_ASR_TEST_EXPECTED"); expected != "" && result.Text != expected {
		t.Fatalf("got %q", result.Text)
	}
	auto, err := r.Recognize(context.Background(), pcm, "auto")
	if err != nil || auto.Language != lang {
		t.Fatal(auto, err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()
	if _, err = r.Recognize(ctx, pcm, lang); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatal("expected cancellation", err)
	}
	// Cancellation must not poison the next request or free a still-running GPU cache.
	result, err = r.Recognize(context.Background(), pcm, lang)
	if err != nil || result.Text == "" {
		t.Fatal(result, err)
	}
	if err = r.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err = r.Recognize(context.Background(), pcm, lang); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
}

func TestNativeSyncAsyncDiagnosticsAndRequestIsolation(t *testing.T) {
	model, path := os.Getenv("QWEN_ASR_TEST_MODEL"), os.Getenv("QWEN_ASR_TEST_PCM")
	if model == "" || path == "" {
		t.Skip("set QWEN_ASR_TEST_MODEL and QWEN_ASR_TEST_PCM")
	}
	pcm, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var expected string
	for _, mode := range []DecodeMode{Sync, Async} {
		r, err := New(context.Background(), Config{ModelPath: model, Device: Auto, DecodeMode: mode})
		if err != nil {
			t.Fatal(err)
		}
		for run := 0; run < 4; run++ {
			result, err := r.Recognize(context.Background(), pcm, "de")
			if err != nil {
				r.Close()
				t.Fatal(mode, err)
			}
			d := result.Diagnostics
			if d.StopReason != "eos" || d.EOSTokenID == nil || (*d.EOSTokenID != 151643 && *d.EOSTokenID != 151645) || d.GeneratedTokens < 1 || d.GeneratedTokens > 512 || d.DecodeMode != string(mode) {
				r.Close()
				t.Fatal("invalid decode termination", d)
			}
			if expected == "" {
				expected = result.Text
			}
			if result.Text != expected {
				r.Close()
				t.Fatalf("mode=%s run=%d got=%q expected=%q", mode, run, result.Text, expected)
			}
			t.Logf("mode=%s run=%d elapsed_ms=%d decode_ms=%d tokens=%d eos=%d text=%q", mode, run, result.ProcessingTime.Milliseconds(), result.DecodeTime.Milliseconds(), d.GeneratedTokens, *d.EOSTokenID, result.Text)
		}
		if err := r.Close(); err != nil {
			t.Fatal(err)
		}
	}
}
