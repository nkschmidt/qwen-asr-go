#ifndef QWEN_ASR_MLX_H
#define QWEN_ASR_MLX_H
#ifdef __cplusplus
extern "C" {
#endif
typedef struct asr_cancel_flag asr_cancel_flag;
void *asr_load(const char *path, const char *device, const char *mode, char **error);
char *asr_transcribe(void *state, const float *audio, int samples, const char *language,
                     const char *prompt, const asr_cancel_flag *cancel, char **error);
void asr_free(void *state);
void asr_string_free(char *text);
double asr_decode_seconds(void *state);
const char *asr_diagnostics(void *state);
const char *asr_device(void *state);
const char *asr_language(void *state);
asr_cancel_flag *asr_cancel_new(void);
void asr_cancel_set(asr_cancel_flag *flag);
void asr_cancel_free(asr_cancel_flag *flag);
#ifdef __cplusplus
}
#endif
#endif
