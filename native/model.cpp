#include "qwen_asr_mlx.h"
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <new>
#if defined(QWEN_ASR_METAL)
#include <mlx/backend/metal/metal.h>
#elif defined(QWEN_ASR_CUDA)
#include <mlx/backend/cuda/cuda.h>
#endif
#include <mlx/fast.h>
#include <mlx/mlx.h>
#include <nlohmann/json.hpp>
#include <stdexcept>
extern "C" {
#include "qwen_asr_tokenizer.h"
int qwen_verbose = 0;
}
namespace mx = mlx::core;
using mx::array;
using json = nlohmann::json;
static array gelu(const array &x) {
    static auto fn = mx::compile(
        [](const std::vector<array> &a) {
            return std::vector<array>{a[0] * (1 + mx::erf(a[0] / float(std::sqrt(2.)))) / 2};
        },
        true);
    return fn({x})[0];
}
static array silu(const array &x) {
    static auto fn = mx::compile(
        [](const std::vector<array> &a) { return std::vector<array>{a[0] * mx::sigmoid(a[0])}; },
        true);
    return fn({x})[0];
}
struct asr_cancel_flag {
    std::atomic<bool> canceled{false};
};
static void check_cancel(const asr_cancel_flag *flag) {
    if (flag && flag->canceled.load(std::memory_order_relaxed))
        throw std::runtime_error("recognition canceled");
}
struct Model {
    std::unordered_map<std::string, array> w;
    std::unique_ptr<qwen_tokenizer_t, decltype(&qwen_tokenizer_free)> tok{nullptr,
                                                                          qwen_tokenizer_free};
    json ac, tc;
    int bits = 4, group = 64;
    bool async_decode = false;
    double decode_seconds = 0;
 json diagnostics;
 std::string diagnostics_json;
    mx::Device device{mx::Device::gpu};
    std::string backend, last_language;
    explicit Model(const std::string &path, const std::string &requested, const std::string &mode) {
        async_decode = mode == "async";
        if (mode != "sync" && mode != "async")
            throw std::runtime_error("decode mode must be sync or async");
        if (requested == "cpu") {
            device = mx::Device(mx::Device::cpu);
            backend = "cpu";
        }
#if defined(QWEN_ASR_METAL)
        else if (requested == "auto" || requested == "metal") {
            if (!mx::metal::is_available())
                throw std::runtime_error("Metal GPU unavailable");
            backend = "metal";
        }
#elif defined(QWEN_ASR_CUDA)
        else if (requested == "auto" || requested == "cuda") {
            if (!mx::cu::is_available())
                throw std::runtime_error("CUDA GPU unavailable");
            backend = "cuda";
        }
#else
        else if (requested == "auto") {
            device = mx::Device(mx::Device::cpu);
            backend = "cpu";
        }
#endif
        else
            throw std::runtime_error("requested device not enabled in this native build");
        if (!mx::is_available(device))
            throw std::runtime_error("requested MLX device unavailable");
        mx::set_default_device(device);
        std::ifstream cf(path + "/config.json");
        json cfg;
        cf >> cfg;
        ac = cfg.at("thinker_config").at("audio_config");
        tc = cfg.at("thinker_config").at("text_config");
        if (int(tc.at("hidden_size")) != 1024 || int(ac.at("d_model")) != 896 ||
            cfg.at("quantization").value("mode", std::string("affine")) != "affine")
            throw std::runtime_error(
                "prototype supports Qwen3-ASR-0.6B affine MLX checkpoint only");
        bits = cfg.at("quantization").at("bits");
        group = cfg.at("quantization").at("group_size");
        if (bits != 4 || group != 64)
            throw std::runtime_error("expected affine 4-bit group-64 checkpoint");
        w = mx::load_safetensors(path + "/model.safetensors").first;
        std::vector<array> all;
        for (auto &p : w)
            all.push_back(p.second);
        mx::eval(all);
        tok.reset(qwen_tokenizer_load((path + "/vocab.json").c_str()));
        if (!tok)
            throw std::runtime_error("tokenizer load failed");
    }
    array get(const std::string &s) { return w.at(s); }
    array linear(const array &x, const std::string &s) {
        array y = w.count(s + ".scales")
                      ? mx::quantized_matmul(x, get(s + ".weight"), get(s + ".scales"),
                                             get(s + ".biases"), true, group, bits)
                      : mx::matmul(x, mx::swapaxes(get(s + ".weight"), -1, -2));
        if (w.count(s + ".bias"))
            y = y + get(s + ".bias");
        return y;
    }
    array embed(const array &ids) {
        auto s = "model.embed_tokens";
        auto packed = mx::take(get(std::string(s) + ".weight"), ids, 0);
        auto scale = mx::take(get(std::string(s) + ".scales"), ids, 0);
        auto bias = mx::take(get(std::string(s) + ".biases"), ids, 0);
        return mx::dequantize(packed, scale, bias, group, bits);
    }
    array ln(const array &x, const std::string &s) {
        return mx::fast::layer_norm(x, get(s + ".weight"), get(s + ".bias"), 1e-5);
    }
    array rms(const array &x, const std::string &s) {
        return mx::fast::rms_norm(x, get(s + ".weight"), float(tc.at("rms_norm_eps")));
    }
    array features(const float *audio, int n) {
        // Whisper frontend: periodic Hann, centered reflect padding, Slaney mel.
        int frames = n / 160;
        std::vector<float> framed(frames * 400), filters(128 * 201);
        double top = 15. + std::log(8.) * 27. / std::log(6.4);
        std::vector<double> edges(130);
        for (int i = 0; i < 130; i++) {
            double m = top * i / 129.;
            edges[i] = m < 15. ? 200. * m / 3. : 1000. * std::exp((m - 15.) * std::log(6.4) / 27.);
        }
        for (int m = 0; m < 128; m++)
            for (int k = 0; k < 201; k++) {
                double hz = k * 40.;
                double a = (hz - edges[m]) / (edges[m + 1] - edges[m]);
                double b = (edges[m + 2] - hz) / (edges[m + 2] - edges[m + 1]);
                filters[m * 201 + k] =
                    float(std::max(0., std::min(a, b)) * 2. / (edges[m + 2] - edges[m]));
            }
        for (int t = 0; t < frames; t++)
            for (int k = 0; k < 400; k++) {
                int i = t * 160 + k - 200;
                if (i < 0)
                    i = -i;
                if (i >= n)
                    i = 2 * n - 2 - i;
                framed[t * 400 + k] = audio[i] * float(.5 - .5 * std::cos(2. * M_PI * k / 400.));
            }
        auto spectrum = mx::fft::rfft(array(framed.data(), {frames, 400}), 400, -1);
        auto power = mx::square(mx::abs(spectrum));
        auto mel = mx::matmul(array(filters.data(), {128, 201}), mx::transpose(power));
        mel = mx::log10(mx::maximum(mel, array(1e-10f)));
        return (mx::maximum(mel, mx::max(mel) - 8) + 4) / 4;
    }
    array encode(const array &features, int feature_len, const asr_cancel_flag *cancel) {
        int frames = features.shape(1), chunk_size = 2 * int(ac.at("n_window")),
            chunks = (feature_len + chunk_size - 1) / chunk_size;
        int maxlen = std::min(chunk_size, feature_len), d = ac.at("d_model"),
            h = ac.at("encoder_attention_heads"), hd = d / h;
        std::vector<array> parts;
        std::vector<int> lens;
        for (int i = 0; i < chunks; i++) {
            int len = std::min(chunk_size, feature_len - i * chunk_size);
            lens.push_back((len + 7) / 8);
            auto part = mx::slice(features, {0, i * chunk_size},
                                  {128, std::min(frames, i * chunk_size + len)});
            part = mx::pad(part, {{0, 0}, {0, maxlen - part.shape(1)}});
            parts.push_back(part);
        }
        array x = mx::expand_dims(mx::stack(parts), -1);
        for (int i = 1; i <= 3; i++) {
            std::string p = "audio_tower.conv2d" + std::to_string(i);
            x = gelu(mx::conv2d(x, get(p + ".weight"), {2, 2}, {1, 1}) + get(p + ".bias"));
        }
        int t = x.shape(2);
        x = mx::reshape(mx::transpose(x, {0, 2, 3, 1}), {chunks, t, -1});
        x = linear(x, "audio_tower.conv_out");
        auto inv = mx::exp(mx::arange(d / 2, mx::float32) * float(-std::log(10000.) / (d / 2 - 1)));
        auto pos = mx::expand_dims(mx::arange(t, mx::float32), 1) * mx::expand_dims(inv, 0);
        x = x + mx::expand_dims(mx::concatenate({mx::sin(pos), mx::cos(pos)}, 1), 0);
        parts.clear();
        for (int i = 0; i < chunks; i++)
            parts.push_back(
                mx::reshape(mx::slice(x, {i, 0, 0}, {i + 1, lens[i], d}), {lens[i], d}));
        x = mx::expand_dims(mx::concatenate(parts, 0), 0);
        int seq = x.shape(1), window = t * (int(ac.at("n_window_infer")) / chunk_size);
        std::vector<float> mask(seq * seq, -1e9f);
        for (int i = 0; i < seq; i++)
            for (int j = 0; j < seq; j++)
                if (i / window == j / window)
                    mask[i * seq + j] = 0;
        array attmask(mask.data(), {1, 1, seq, seq});
        for (int i = 0; i < int(ac.at("encoder_layers")); i++) {
            check_cancel(cancel);
            auto p = "audio_tower.layers." + std::to_string(i);
            auto a = p + ".self_attn";
            auto norm = ln(x, p + ".self_attn_layer_norm");
            auto q =
                mx::transpose(mx::reshape(linear(norm, a + ".q_proj") * float(1 / std::sqrt(hd)),
                                          {1, seq, h, hd}),
                              {0, 2, 1, 3});
            auto k = mx::transpose(mx::reshape(linear(norm, a + ".k_proj"), {1, seq, h, hd}),
                                   {0, 2, 1, 3});
            auto v = mx::transpose(mx::reshape(linear(norm, a + ".v_proj"), {1, seq, h, hd}),
                                   {0, 2, 1, 3});
            auto aout = mx::fast::scaled_dot_product_attention(q, k, v, 1., "", attmask);
            x = x + linear(mx::reshape(mx::transpose(aout, {0, 2, 1, 3}), {1, seq, d}),
                           a + ".out_proj");
            x = x + linear(gelu(linear(ln(x, p + ".final_layer_norm"), p + ".fc1")), p + ".fc2");
        }
        x = ln(mx::reshape(x, {seq, d}), "audio_tower.ln_post");
        x = linear(gelu(linear(x, "audio_tower.proj1")), "audio_tower.proj2");
        mx::eval(x);
        return x;
    }
    struct KV {
        std::optional<array> k, v;
    };
    array step(array x, std::vector<KV> &cache, int offset) {
        int seq = x.shape(1), heads = tc.at("num_attention_heads"),
            kvheads = tc.at("num_key_value_heads"), hd = tc.at("head_dim");
        for (int i = 0; i < int(tc.at("num_hidden_layers")); i++) {
            auto p = "model.layers." + std::to_string(i), a = p + ".self_attn";
            auto norm = rms(x, p + ".input_layernorm");
            auto q =
                rms(mx::reshape(linear(norm, a + ".q_proj"), {1, seq, heads, hd}), a + ".q_norm");
            auto k =
                rms(mx::reshape(linear(norm, a + ".k_proj"), {1, seq, kvheads, hd}), a + ".k_norm");
            auto v = mx::transpose(mx::reshape(linear(norm, a + ".v_proj"), {1, seq, kvheads, hd}),
                                   {0, 2, 1, 3});
            q = mx::fast::rope(mx::transpose(q, {0, 2, 1, 3}), hd, false,
                               float(tc.at("rope_theta")), 1., offset);
            k = mx::fast::rope(mx::transpose(k, {0, 2, 1, 3}), hd, false,
                               float(tc.at("rope_theta")), 1., offset);
            int needed = offset + seq;
            if (!cache[i].k || needed > cache[i].k->shape(2)) {
                int capacity = ((needed + 255) / 256) * 256;
                if (cache[i].k) {
                    int extra = capacity - cache[i].k->shape(2);
                    cache[i].k = mx::concatenate(
                        {*cache[i].k, mx::zeros({1, kvheads, extra, hd}, k.dtype())}, 2);
                    cache[i].v = mx::concatenate(
                        {*cache[i].v, mx::zeros({1, kvheads, extra, hd}, v.dtype())}, 2);
                } else {
                    cache[i].k = mx::zeros({1, kvheads, capacity, hd}, k.dtype());
                    cache[i].v = mx::zeros({1, kvheads, capacity, hd}, v.dtype());
                }
            }
            cache[i].k =
                mx::slice_update(*cache[i].k, k, {0, 0, offset, 0}, {1, kvheads, needed, hd});
            cache[i].v =
                mx::slice_update(*cache[i].v, v, {0, 0, offset, 0}, {1, kvheads, needed, hd});
            k = mx::slice(*cache[i].k, {0, 0, 0, 0}, {1, kvheads, needed, hd});
            v = mx::slice(*cache[i].v, {0, 0, 0, 0}, {1, kvheads, needed, hd});
            auto out = mx::fast::scaled_dot_product_attention(q, k, v, float(1 / std::sqrt(hd)),
                                                              seq > 1 ? "causal" : "");
            x = x +
                linear(mx::reshape(mx::transpose(out, {0, 2, 1, 3}), {1, seq, -1}), a + ".o_proj");
            norm = rms(x, p + ".post_attention_layernorm");
            auto gate = linear(norm, p + ".mlp.gate_proj");
            x = x + linear(silu(gate) * linear(norm, p + ".mlp.up_proj"), p + ".mlp.down_proj");
        }
        x = rms(mx::slice(x, {0, seq - 1, 0}, {1, seq, int(tc.at("hidden_size"))}), "model.norm");
        return linear(x, "model.embed_tokens");
    }
    std::vector<int> tokens(const std::string &s) {
        const std::vector<std::pair<std::string, int>> special = {{"<|im_start|>", 151644},
                                                                  {"<|im_end|>", 151645},
                                                                  {"<|audio_start|>", 151669},
                                                                  {"<|audio_end|>", 151670},
                                                                  {"<asr_text>", 151704}};
        std::vector<int> result;
        size_t pos = 0;
        while (pos < s.size()) {
            size_t next = s.size();
            int token = -1;
            size_t span = 0;
            for (auto &p : special) {
                auto found = s.find(p.first, pos);
                if (found < next) {
                    next = found;
                    token = p.second;
                    span = p.first.size();
                }
            }
            if (next > pos) {
                int n = 0;
                auto plain = s.substr(pos, next - pos);
                std::unique_ptr<int, decltype(&free)> ids(
                    qwen_tokenizer_encode(tok.get(), plain.c_str(), &n), free);
                if (!ids)
                    throw std::runtime_error("encode prompt failed");
                result.insert(result.end(), ids.get(), ids.get() + n);
            }
            if (token < 0)
                break;
            result.push_back(token);
            pos = next + span;
        }
        return result;
    }
    std::string transcribe(const float *audio, int n, const std::string &language,
                           const std::string &prompt, const asr_cancel_flag *cancel) {
        mx::set_default_device(device);
        decode_seconds = 0;
        diagnostics = {{"decode_mode",async_decode ? "async" : "sync"},{"audio_samples",n},{"forced_language",language},{"generated_tokens",0},{"input_tokens",0},{"audio_tokens",0},{"last_token_id",nullptr},{"eos_token_id",nullptr},{"max_same_token_run",0},{"repeated_suffix_length",0},{"stop_reason","error"}};
        check_cancel(cancel);
        last_language = language;
        if (n < 400 || n > 16000 * 28)
            throw std::runtime_error("expected 25 ms to 28 s mono 16 kHz");
        int frames = n / 160;
        array features = this->features(audio, n);
        auto encoded = encode(features, frames, cancel);
        std::string prefix = "<|im_start|>system\n" + (prompt.empty() ? "" : prompt + "\n") +
                             "<|im_end|>\n<|im_start|>user\n<|audio_start|>";
        std::string suffix = "<|audio_end|><|im_end|>\n<|im_start|>assistant\n" +
                             (language.empty() ? "" : "language " + language + "<asr_text>");
        check_cancel(cancel);
        auto before = tokens(prefix), after = tokens(suffix);

        auto x = mx::concatenate(
            {embed(array(before.data(), {1, int(before.size())})),
             mx::expand_dims(mx::astype(encoded, get("model.embed_tokens.scales").dtype()), 0),
             embed(array(after.data(), {1, int(after.size())}))},
            1);
        diagnostics["input_tokens"] = x.shape(1);
        diagnostics["audio_tokens"] = encoded.shape(0);
        std::vector<KV> cache(int(tc.at("num_hidden_layers")));
        std::string text;
        int offset = 0;
        // Match mlx-audio: prefill all but the final prompt token, then decode it.
        if (x.shape(1) > 1) {
            int count = x.shape(1) - 1, dim = x.shape(2);
            step(mx::slice(x, {0, 0, 0}, {1, count, dim}), cache, 0);
            std::vector<array> states;
            for (auto &c : cache) {
                states.push_back(*c.k);
                states.push_back(*c.v);
            }
            mx::eval(states);
            x = mx::slice(x, {0, count, 0}, {1, count + 1, dim});
            offset = count;
        }
        bool finished = false;
        auto started = std::chrono::steady_clock::now();
        std::vector<int> generated;
        int same_run=0, last_token=-1;
        auto observe = [&](int token) {
            generated.push_back(token);
            same_run = token == last_token ? same_run+1 : 1;
            last_token=token;
            diagnostics["generated_tokens"]=generated.size();
            diagnostics["last_token_id"]=token;
            diagnostics["max_same_token_run"]=std::max(int(diagnostics["max_same_token_run"]),same_run);
            if (token==151645 || token==151643) {
                diagnostics["eos_token_id"]=token;
                diagnostics["stop_reason"]="eos";
            }
            for (size_t width=1;width<=16 && width*3<=generated.size();width++) {
                auto end=generated.end();
                if (std::equal(end-width,end,end-2*width) && std::equal(end-width,end,end-3*width))
                    diagnostics["repeated_suffix_length"]=width;
            }
        };
        auto append = [&](int token) {
            if (token == 151704)
                text += "<asr_text>";
            else {
                const char *piece = qwen_tokenizer_decode(tok.get(), token);
                if (piece)
                    text += piece;
            }
        };
        auto predict = [&](const array &input, int position) {
            check_cancel(cancel);
            return mx::argmax(mx::reshape(step(input, cache, position), {-1}));
        };
        if (!async_decode) {
            for (int count = 0; count < 512; count++) {
                auto next = predict(x, offset);
                mx::eval(next);
                int token = next.item<int>();
                observe(token);
                check_cancel(cancel);
                if (token == 151645 || token == 151643) {
                    finished = true;
                    break;
                }
                append(token);
                offset += x.shape(1);
                x = embed(array(&token, {1, 1}));
            }
        } else {
            // Keep tokens on GPU: submit t+1 before waiting for token t on CPU.
            // Exactly one token of lookahead; this does not parallelize calls.
            try {
                auto pending = predict(x, offset);
                mx::async_eval(pending);
                offset += x.shape(1);
                for (int count = 0; count < 512; count++) {
                    auto next = predict(embed(mx::reshape(pending, {1, 1})), offset);
                    mx::async_eval(next);
                    int token = pending.item<int>();
                    observe(token);
                    check_cancel(cancel);
                    if (token == 151645 || token == 151643) {
                        finished = true;
                        break;
                    }
                    append(token);
                    pending = next;
                    offset++;
                }
                // Drain the extra lookahead after EOS before releasing its KV cache.
                // Include that work in timings; no hidden GPU work in the next request.
                mx::synchronize();
            } catch (...) {
                mx::synchronize();
                throw;
            }
        }
        decode_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        diagnostics["decode_seconds"] = decode_seconds;
        if (!finished) {
            diagnostics["stop_reason"]="generation_limit";
            throw std::runtime_error("generation exceeded 512 tokens");
        }
        auto marker = text.find("<asr_text>");
        if (marker != std::string::npos) {
            if (language.empty() && text.rfind("language ", 0) == 0)
                last_language = text.substr(9, marker - 9);
            text = text.substr(marker + 10);
        }
        return text;
    }
};
extern "C" void *asr_load(const char *path, const char *device, const char *mode, char **error) {
    try {
        return new Model(path, device, mode);
    } catch (const std::exception &e) {
        *error = strdup(e.what());
        return nullptr;
    }
}
extern "C" char *asr_transcribe(void *state, const float *audio, int samples, const char *language,
                                const char *prompt, const asr_cancel_flag *cancel, char **error) {
    try {
        auto *model = static_cast<Model *>(state);
        return strdup(model->transcribe(audio, samples, language, prompt, cancel).c_str());
    } catch (const std::exception &e) {
        // Drain pending operations before request buffers and cancellation flag are released.
        try {
            mx::synchronize();
        } catch (...) {
        }
        auto *model = static_cast<Model *>(state);
        if (cancel && cancel->canceled.load(std::memory_order_relaxed)) model->diagnostics["stop_reason"]="canceled";
        *error = strdup(e.what());
        return nullptr;
    }
}
extern "C" const char *asr_diagnostics(void *state) {
    auto *model=static_cast<Model *>(state);
    model->diagnostics_json=model->diagnostics.dump();
    return model->diagnostics_json.c_str();
}
extern "C" void asr_free(void *state) { delete static_cast<Model *>(state); }
extern "C" void asr_string_free(char *text) { free(text); }
extern "C" double asr_decode_seconds(void *state) {
    return static_cast<Model *>(state)->decode_seconds;
}
extern "C" const char *asr_device(void *state) {
    return static_cast<Model *>(state)->backend.c_str();
}
extern "C" const char *asr_language(void *state) {
    return static_cast<Model *>(state)->last_language.c_str();
}
extern "C" asr_cancel_flag *asr_cancel_new() { return new (std::nothrow) asr_cancel_flag; }
extern "C" void asr_cancel_set(asr_cancel_flag *flag) {
    flag->canceled.store(true, std::memory_order_relaxed);
}
extern "C" void asr_cancel_free(asr_cancel_flag *flag) { delete flag; }
