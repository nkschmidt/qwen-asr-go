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
