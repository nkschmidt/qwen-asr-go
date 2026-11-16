// Command transcribe is a file-based smoke CLI. Applications should import the root package.
package main

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	asr "github.com/nkschmidt/qwen-asr-go"
	"os"
	"os/signal"
	"syscall"
)

func readPCM(path string) ([]byte, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(b) < 12 || string(b[:4]) != "RIFF" || string(b[8:12]) != "WAVE" {
		return nil, errors.New("expected WAV")
	}
	valid := false
	var pcm []byte
	for pos := 12; pos+8 <= len(b); {
		size := int(binary.LittleEndian.Uint32(b[pos+4:]))
		if size > len(b)-pos-8 {
			return nil, errors.New("truncated WAV")
		}
		chunk := b[pos+8 : pos+8+size]
		switch string(b[pos : pos+4]) {
		case "fmt ":
			valid = len(chunk) >= 16 && binary.LittleEndian.Uint16(chunk) == 1 && binary.LittleEndian.Uint16(chunk[2:]) == 1 && binary.LittleEndian.Uint32(chunk[4:]) == 16000 && binary.LittleEndian.Uint16(chunk[14:]) == 16
		case "data":
			pcm = chunk
		}
		pos += 8 + size + size%2
	}
	if !valid || len(pcm) == 0 || len(pcm)%2 != 0 {
		return nil, errors.New("expected PCM16 mono 16kHz WAV")
	}
	return pcm, nil
}
func run() error {
	model := flag.String("model", "", "local Qwen3-ASR-0.6B affine 4-bit MLX checkpoint")
	wav := flag.String("wav", "", "PCM16 mono 16kHz WAV")
	lang := flag.String("language", "auto", "language code or auto")
	device := flag.String("device", "auto", "auto, metal, cuda or cpu")
	decode := flag.String("decode", "async", "async or sync")
	prompt := flag.String("context", "", "optional vocabulary or conversational context (4096 bytes max)")
	flag.Parse()
	if *model == "" || *wav == "" {
		flag.Usage()
		return errors.New("model and wav are required")
	}
	pcm, err := readPCM(*wav)
	if err != nil {
		return err
	}
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	recognizer, err := asr.New(ctx, asr.Config{ModelPath: *model, Device: asr.Device(*device), DecodeMode: asr.DecodeMode(*decode)})
	if err != nil {
		return err
	}
	defer recognizer.Close()
	result, err := recognizer.RecognizeWithContext(ctx, pcm, *lang, *prompt)
	if err != nil {
		return err
	}
	return json.NewEncoder(os.Stdout).Encode(result)
}
func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
