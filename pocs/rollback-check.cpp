// Checks recurrent-state rollback (n_rs_seq) against a reference context that never saw the
// rolled-back tokens. Greedy text is not a usable oracle for this: batched and single-token
// steps go through different kernels and flip near-ties, so compare logits instead.
//
//   ctx_rs  (n_rs_seq = 4): prompt | c1..c5 | seq_rm(keep c1,c2) | x  -> logits A
//   ctx_ref (n_rs_seq = 0): prompt | c1,c2  |                    | x  -> logits B
//
// then a second round on top (y1..y4, keep y1, then z) so the pending-rollback index is
// exercised twice and the positions cross 4-token block boundaries of the QSA pooled cache.
// A wrong slot or plane shows up as O(0.1..1) differences; kernel noise is ~1e-4.
//
// build: g++ -O2 -std=c++17 -Iinclude -Iggml/include pocs/rollback-check.cpp \
//            -Lbuild-release/bin -lllama -lggml -lggml-base -Wl,-rpath,$PWD/build-release/bin \
//            -o build-release/bin/rollback-check

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <map>
#include <cstdlib>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool add_special) {
    std::vector<llama_token> out(text.size() + 16);
    const int n = llama_tokenize(vocab, text.c_str(), (int) text.size(), out.data(), (int) out.size(), add_special, true);
    out.resize(n < 0 ? 0 : n);
    return out;
}

static bool decode(llama_context * ctx, const std::vector<llama_token> & toks, llama_pos pos0, bool logits_last) {
    llama_batch batch = llama_batch_init((int) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        batch.token[i]     = toks[i];
        batch.pos[i]       = pos0 + (llama_pos) i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (logits_last && i + 1 == toks.size()) ? 1 : 0;
    }
    batch.n_tokens = (int) toks.size();
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        fprintf(stderr, "llama_decode failed: %d\n", rc);
        return false;
    }
    return true;
}

static std::vector<float> last_logits(llama_context * ctx, int n_vocab) {
    const float * p = llama_get_logits_ith(ctx, -1);
    return std::vector<float>(p, p + n_vocab);
}

struct cmp_result {
    float  max_abs;
    int    argmax_a;
    int    argmax_b;
};

static cmp_result compare(const std::vector<float> & a, const std::vector<float> & b) {
    cmp_result r = { 0.0f, 0, 0 };
    for (size_t i = 0; i < a.size(); ++i) {
        r.max_abs = std::fmax(r.max_abs, std::fabs(a[i] - b[i]));
        if (a[i] > a[r.argmax_a]) r.argmax_a = (int) i;
        if (b[i] > b[r.argmax_b]) r.argmax_b = (int) i;
    }
    return r;
}

// recurrent state only (conv + GDN + PLE conv rows); a pending rollback is resolved by the getter
static std::vector<float> rec_state(llama_context * ctx) {
    const size_t n = llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    std::vector<uint8_t> buf(n);
    llama_state_seq_get_data_ext(ctx, buf.data(), n, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    std::vector<float> f(n / sizeof(float));
    memcpy(f.data(), buf.data(), f.size() * sizeof(float));
    return f;
}

static void report_state(const char * what, const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        printf("%-34s state size differs: %zu vs %zu floats\n", what, a.size(), b.size());
        return;
    }
    double max_abs = 0.0, sum_abs = 0.0, sum_ref = 0.0;
    size_t n_diff = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs((double) a[i] - (double) b[i]);
        max_abs = std::fmax(max_abs, d);
        sum_abs += d;
        sum_ref += std::fabs((double) b[i]);
        n_diff  += d > 1e-6;
    }
    printf("%-34s state: max|d| = %.3e  mean|d| = %.3e  mean|ref| = %.3e  differing = %zu/%zu\n",
            what, max_abs, sum_abs / a.size(), sum_ref / a.size(), n_diff, a.size());
}

static void report(const char * what, const cmp_result & r, const llama_vocab * vocab) {
    char sa[64] = {0};
    char sb[64] = {0};
    llama_token_to_piece(vocab, r.argmax_a, sa, sizeof(sa) - 1, 0, true);
    llama_token_to_piece(vocab, r.argmax_b, sb, sizeof(sb) - 1, 0, true);
    printf("%-34s max|dlogit| = %9.5f   top1 rs=%d '%s'  ref=%d '%s'  %s\n",
            what, r.max_abs, r.argmax_a, sa, r.argmax_b, sb, r.argmax_a == r.argmax_b ? "" : "<-- top-1 differs");
}


// ---- trace mode: per-tensor comparison of "c2 as the 2nd token of a batch" vs "c2 alone" ----

struct capture {
    bool             enabled = false;
    int              n_tok   = 1;                    // tokens in the ubatch being captured
    std::vector<std::string>                 order;  // names in graph order
    std::map<std::string, std::vector<float>> data;  // last-token slice per name
};

static bool cb_capture(ggml_tensor * t, bool ask, void * ud) {
    capture * c = (capture *) ud;
    if (!c->enabled) return false;
    const char * name = ggml_get_name(t);
    if (ask) {
        return t->type == GGML_TYPE_F32 && name[0] != '\0' && strstr(name, "(view)") == nullptr && strstr(name, "(reshaped)") == nullptr;
    }
    // find the token dim: the last dim whose size equals n_tok (n_tok == 1: whole tensor)
    int tdim = -1;
    for (int d = 3; d >= 1; --d) {
        if (t->ne[d] == c->n_tok && c->n_tok > 1) { tdim = d; break; }
    }
    std::vector<float> all(ggml_nelements(t));
    if (!ggml_is_contiguous(t)) return true;
    ggml_backend_tensor_get(t, all.data(), 0, ggml_nbytes(t));
    std::vector<float> slice;
    if (tdim < 0) {
        slice = std::move(all);
    } else {
        // slice index n_tok-1 along tdim
        int64_t inner = 1; for (int d = 0; d < tdim; ++d) inner *= t->ne[d];
        int64_t outer = 1; for (int d = tdim + 1; d < 4; ++d) outer *= t->ne[d];
        slice.reserve(inner * outer);
        for (int64_t o = 0; o < outer; ++o) {
            const float * src = all.data() + (o * t->ne[tdim] + (c->n_tok - 1)) * inner;
            slice.insert(slice.end(), src, src + inner);
        }
    }
    std::string key = name;
    if (c->data.count(key)) {
        // duplicate names (e.g. per-call helpers): keep the first, tag the rest
        int k = 2; while (c->data.count(key + "#" + std::to_string(k))) ++k;
        key += "#" + std::to_string(k);
    }
    c->order.push_back(key);
    c->data[key] = std::move(slice);
    return true;
}

static int run_trace(llama_model * model, const llama_vocab * vocab, const std::string & prompt) {
    capture cap_s, cap_b;
    auto make = [&](capture * cap) {
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = 512; cparams.n_batch = 64; cparams.n_ubatch = 64; cparams.n_seq_max = 1;
        cparams.type_k = GGML_TYPE_Q8_0; cparams.type_v = GGML_TYPE_Q8_0;
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cparams.cb_eval = cb_capture; cparams.cb_eval_user_data = cap;
        return llama_init_from_model(model, cparams);
    };
    llama_context * ctx_s = make(&cap_s);
    llama_context * ctx_b = make(&cap_b);
    if (!ctx_s || !ctx_b) return 1;

    const auto toks_p = tokenize(vocab, prompt, true);
    auto toks_c = tokenize(vocab, " Here is a robust Python function", false);
    toks_c.resize(2);
    const llama_pos pos = (llama_pos) toks_p.size();

    if (!decode(ctx_s, toks_p, 0, true) || !decode(ctx_b, toks_p, 0, true)) return 1;
    if (!decode(ctx_s, { toks_c[0] }, pos, true)) return 1;

    cap_s.enabled = true; cap_s.n_tok = 1;
    if (!decode(ctx_s, { toks_c[1] }, pos + 1, true)) return 1;
    cap_s.enabled = false;

    cap_b.enabled = true; cap_b.n_tok = 2;
    if (!decode(ctx_b, toks_c, pos, true)) return 1;
    cap_b.enabled = false;

    printf("captured %zu tensors (single) / %zu (batch)\n", cap_s.order.size(), cap_b.order.size());
    printf("%-36s %10s %10s %10s\n", "tensor", "max|d|", "mean|d|", "mean|ref|");
    int shown = 0;
    for (const auto & name : cap_b.order) {
        auto it = cap_s.data.find(name);
        if (it == cap_s.data.end()) continue;
        const auto & a = it->second;
        const auto & b = cap_b.data[name];
        if (a.size() != b.size()) { printf("%-36s size %zu vs %zu\n", name.c_str(), a.size(), b.size()); continue; }
        double mx = 0, sd = 0, sr = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            const double d = std::fabs((double) a[i] - (double) b[i]);
            mx = std::fmax(mx, d); sd += d; sr += std::fabs((double) b[i]);
        }
        const double rel = sr > 0 ? sd / sr : 0.0;
        // print everything for the first layers, then only notable ones
        if (shown < 60 || rel > 0.02) {
            printf("%-36s %10.3e %10.3e %10.3e  rel %.4f%s\n", name.c_str(), mx, sd / a.size(), sr / a.size(), rel, rel > 0.02 ? "  <--" : "");
            shown++;
        }
    }
    llama_free(ctx_s); llama_free(ctx_b);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s model.gguf \"prompt\" [n_rs_seq=4]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const std::string prompt = argv[2];
    const bool     trace    = argc > 3 && strcmp(argv[3], "trace") == 0;
    const uint32_t n_rs_seq = (argc > 3 && !trace) ? (uint32_t) atoi(argv[3]) : 4;

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;
    mparams.ple_on_disk  = true;
    mparams.ple_cache_mb = 8192;

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    if (trace) {
        return run_trace(model, vocab, prompt);
    }

    auto make_ctx = [&](uint32_t rs) {
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx     = 512;
        cparams.n_batch   = 64;
        cparams.n_ubatch  = 64;
        cparams.n_seq_max = 1;
        cparams.n_rs_seq  = rs;
        cparams.type_k    = GGML_TYPE_Q8_0;
        cparams.type_v    = GGML_TYPE_Q8_0;
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        return llama_init_from_model(model, cparams);
    };

    llama_context * ctx_rs  = make_ctx(n_rs_seq);
    llama_context * ctx_ref = make_ctx(0);
    if (!ctx_rs || !ctx_ref) {
        fprintf(stderr, "failed to create contexts\n");
        return 1;
    }
    printf("n_rs_seq: rs=%u ref=%u\n", llama_n_rs_seq(ctx_rs), llama_n_rs_seq(ctx_ref));

    const auto toks_p = tokenize(vocab, prompt, true);
    // continuation tokens: five, then one, then four, then one - only their count matters
    auto toks_c = tokenize(vocab, " Here is a robust Python function", false);
    auto toks_y = tokenize(vocab, " that parses a CSV file", false);
    if (toks_c.size() < 5 || toks_y.size() < 4) {
        fprintf(stderr, "continuation too short\n");
        return 1;
    }
    toks_c.resize(5);
    toks_y.resize(4);
    const std::vector<llama_token> tok_x = { toks_c[2] };
    const std::vector<llama_token> tok_z = { toks_y[1] };

    llama_pos pos = 0;

    // prompt on both, compare: the kernel-noise floor for identical inputs
    if (!decode(ctx_rs, toks_p, pos, true) || !decode(ctx_ref, toks_p, pos, true)) return 1;
    report("prompt (noise floor)", compare(last_logits(ctx_rs, n_vocab), last_logits(ctx_ref, n_vocab)), vocab);
    pos += (llama_pos) toks_p.size();

    // shape-noise control: c1,c2 as one batch vs as two single steps, no rollback anywhere
    {
        llama_context * ctx_single = make_ctx(0);
        if (!ctx_single || !decode(ctx_single, toks_p, 0, true)) return 1;
        if (!decode(ctx_single, { toks_c[0] }, pos, true) || !decode(ctx_single, { toks_c[1] }, pos + 1, true)) return 1;
        if (!decode(ctx_ref, { toks_c[0], toks_c[1] }, pos, true)) return 1;
        report_state("control: 2-batch vs 2 singles", rec_state(ctx_single), rec_state(ctx_ref));
        if (!decode(ctx_single, tok_x, pos + 2, true) || !decode(ctx_ref, tok_x, pos + 2, true)) return 1;
        report("control: 2-batch vs 2 singles, +1", compare(last_logits(ctx_single, n_vocab), last_logits(ctx_ref, n_vocab)), vocab);
        llama_free(ctx_single);
        // ref is now at prompt + c1,c2,x; rebuild it at prompt + c1,c2 for the real test
        llama_free(ctx_ref);
        ctx_ref = make_ctx(0);
        if (!ctx_ref || !decode(ctx_ref, toks_p, 0, true)) return 1;
        if (!decode(ctx_ref, { toks_c[0], toks_c[1] }, pos, true)) return 1;
    }

    // round 1: rs sees 5, keeps 2 (rollback 3); ref saw only 2
    if (!decode(ctx_rs, toks_c, pos, true)) return 1;
    if (!llama_memory_seq_rm(llama_get_memory(ctx_rs), 0, pos + 2, -1)) {
        fprintf(stderr, "seq_rm (rollback 3) refused\n");
        return 1;
    }
    report_state("after rollback 3 of 5", rec_state(ctx_rs), rec_state(ctx_ref));
    pos += 2;
    if (!decode(ctx_rs, tok_x, pos, true) || !decode(ctx_ref, tok_x, pos, true)) return 1;
    report("after rollback 3 of 5, +1 token", compare(last_logits(ctx_rs, n_vocab), last_logits(ctx_ref, n_vocab)), vocab);
    pos += 1;

    // round 2: rs sees 4, keeps 1; ref sees 1
    if (!decode(ctx_rs, toks_y, pos, true)) return 1;
    if (!llama_memory_seq_rm(llama_get_memory(ctx_rs), 0, pos + 1, -1)) {
        fprintf(stderr, "seq_rm (rollback 3) refused\n");
        return 1;
    }
    if (!decode(ctx_ref, { toks_y[0] }, pos, true)) return 1;
    pos += 1;
    if (!decode(ctx_rs, tok_z, pos, true) || !decode(ctx_ref, tok_z, pos, true)) return 1;
    report("after rollback 3 of 4, +1 token", compare(last_logits(ctx_rs, n_vocab), last_logits(ctx_ref, n_vocab)), vocab);
    pos += 1;

    // round 3: a few plain single-token steps on both, the states must stay in sync
    for (int i = 0; i < 3; ++i) {
        const std::vector<llama_token> t = { toks_y[2 + (i % 2)] };
        if (!decode(ctx_rs, t, pos, true) || !decode(ctx_ref, t, pos, true)) return 1;
        pos += 1;
    }
    report("3 single steps later", compare(last_logits(ctx_rs, n_vocab), last_logits(ctx_ref, n_vocab)), vocab);

    // round 4: rollback of the maximum depth (n_rs_seq) out of a 5-token batch
    if (!decode(ctx_rs, toks_c, pos, true)) return 1;
    if (!llama_memory_seq_rm(llama_get_memory(ctx_rs), 0, pos + 1, -1)) {
        fprintf(stderr, "seq_rm (rollback 4) refused\n");
        return 1;
    }
    if (!decode(ctx_ref, { toks_c[0] }, pos, true)) return 1;
    pos += 1;
    if (!decode(ctx_rs, tok_x, pos, true) || !decode(ctx_ref, tok_x, pos, true)) return 1;
    report("after rollback 4 of 5, +1 token", compare(last_logits(ctx_rs, n_vocab), last_logits(ctx_ref, n_vocab)), vocab);

    llama_free(ctx_rs);
    llama_free(ctx_ref);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
