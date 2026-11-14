package asr

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"
)

type fakeModel struct {
	entered chan []byte
	release chan struct{}
	closed  atomic.Bool
	calls   atomic.Int32
	max     atomic.Int32
}

func (m *fakeModel) recognize(ctx context.Context, pcm []byte, code, name string) (Result, error) {
	n := m.calls.Add(1)
	defer m.calls.Add(-1)
	for old := m.max.Load(); n > old; old = m.max.Load() {
		if m.max.CompareAndSwap(old, n) {
			break
		}
	}
	if m.entered != nil {
		m.entered <- pcm
	}
	if m.release != nil {
		select {
		case <-ctx.Done():
			return Result{}, ctx.Err()
		case <-m.release:
		}
	}
	return Result{Text: "text", Language: code}, nil
}
func (m *fakeModel) close() { m.closed.Store(true) }
func openFake(t *testing.T, m *fakeModel) *Recognizer {
	t.Helper()
	r, err := newRecognizer(context.Background(), Config{ModelPath: "unused"}, func(Config) (nativeModel, error) { return m, nil })
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { r.Close() })
	return r
}
func receive[T any](t *testing.T, ch <-chan T) T {
	t.Helper()
	select {
	case value := <-ch:
		return value
	case <-time.After(3 * time.Second):
		t.Fatal("timed out")
		var zero T
		return zero
	}
}
func TestCanceledRequestOwnsPCMAndDoesNotBlockNext(t *testing.T) {
	m := &fakeModel{entered: make(chan []byte, 2), release: make(chan struct{})}
	r := openFake(t, m)
	pcm := make([]byte, 800)
	pcm[0] = 7
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { _, err := r.Recognize(ctx, pcm, "de"); done <- err }()
	snapshot := receive(t, m.entered)
	cancel()
	if err := receive(t, done); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	pcm[0] = 9
	if snapshot[0] != 7 {
		t.Fatal("native request retained caller-owned PCM")
	}
	close(m.release)
	result, err := r.Recognize(context.Background(), pcm, "DE")
	if err != nil || result.Language != "de" {
		t.Fatal(result, err)
	}
}
func TestCloseCancelsActiveRequestAndIsIdempotent(t *testing.T) {
	m := &fakeModel{entered: make(chan []byte, 1), release: make(chan struct{})}
	r := openFake(t, m)
	done := make(chan error, 1)
	go func() { _, err := r.Recognize(context.Background(), make([]byte, 800), "auto"); done <- err }()
	receive(t, m.entered)
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	r.Close()
	if !m.closed.Load() {
		t.Fatal("native resources not closed")
	}
	if err := receive(t, done); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
	if _, err := r.Recognize(context.Background(), make([]byte, 800), "de"); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
}
func TestConcurrentCallsAreSerialized(t *testing.T) {
	m := &fakeModel{entered: make(chan []byte, 8), release: make(chan struct{})}
	r := openFake(t, m)
	done := make(chan error, 8)
	for i := 0; i < 8; i++ {
		go func() { _, err := r.Recognize(context.Background(), make([]byte, 800), "en"); done <- err }()
	}
	receive(t, m.entered)
	close(m.release)
	for i := 0; i < 8; i++ {
		if err := receive(t, done); err != nil {
			t.Fatal(err)
		}
	}
	if m.max.Load() != 1 {
		t.Fatal("native calls overlapped", m.max.Load())
	}
}
func TestQueueCancellationAndValidation(t *testing.T) {
	m := &fakeModel{entered: make(chan []byte, 1), release: make(chan struct{})}
	r := openFake(t, m)
	first := make(chan error, 1)
	go func() { _, err := r.Recognize(context.Background(), make([]byte, 800), "en"); first <- err }()
	receive(t, m.entered)
	ctx, cancel := context.WithCancel(context.Background())
	queued := make(chan error, 1)
	go func() { _, err := r.Recognize(ctx, make([]byte, 800), "de"); queued <- err }()
	cancel()
	if err := receive(t, queued); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if m.calls.Load() != 1 {
		t.Fatal("queued canceled call reached native execution")
	}
	if _, err := r.Recognize(context.Background(), make([]byte, 801), "de"); err == nil {
		t.Fatal("accepted odd PCM")
	}
	if _, err := r.Recognize(context.Background(), make([]byte, 800), "bogus"); err == nil {
		t.Fatal("accepted unknown language")
	}
	close(m.release)
	if err := receive(t, first); err != nil {
		t.Fatal(err)
	}
}
func TestCanceledLoadEventuallyReleasesModel(t *testing.T) {
	entered := make(chan struct{})
	release := make(chan struct{})
	m := &fakeModel{}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := newRecognizer(ctx, Config{ModelPath: "unused"}, func(Config) (nativeModel, error) { close(entered); <-release; return m, nil })
		done <- err
	}()
	receive(t, entered)
	cancel()
	if err := receive(t, done); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	close(release)
	deadline := time.After(3 * time.Second)
	for !m.closed.Load() {
		select {
		case <-deadline:
			t.Fatal("canceled load leaked model")
		default:
			time.Sleep(time.Millisecond)
		}
	}
}
