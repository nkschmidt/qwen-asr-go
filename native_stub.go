//go:build !cgo || (!darwin && !linux)

package asr

func loadNative(Config) (nativeModel, error) { return nil, ErrNativeUnavailable }
