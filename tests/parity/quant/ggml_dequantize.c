// ggml's own dequantization of GGUF blocks: the oracle gen_block_cases.py writes tests/fixtures/quant/blocks with.
// Built with ggml/src/ggml-quants.c of a llama.cpp checkout (gen_block_cases.py does that, without fused
// multiply-adds).
//
//   ggml_dequantize layout                     one line per format: name, bytes and weights of a block
//   ggml_dequantize tables <out>               the codebook tables, one after another (test_block_decode.cpp's order)
//   ggml_dequantize <format> <blocks> <values> the float values of the blocks in file <blocks>, as f32 into <values>

#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "ggml-quants.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ggml-quants.c refers to these from code this program never runs.
void ggml_abort(const char* file, int line, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "ggml_abort at %s:%d: ", file, line);
    vfprintf(stderr, fmt, args);
    va_end(args);
    abort();
}
size_t ggml_row_size(enum ggml_type type, int64_t ne) {
    (void)type;
    (void)ne;
    abort();
}
const char* ggml_type_name(enum ggml_type type) {
    (void)type;
    abort();
}
size_t ggml_type_size(enum ggml_type type) {
    (void)type;
    abort();
}

typedef void (*dequantize_fn)(const void* blocks, float* values, int64_t k);

struct format {
    const char* name;
    size_t bytes;
    int weights;
    dequantize_fn fn;
};

#define FORMAT(name, block, qk, fn) {name, sizeof(block), qk, (dequantize_fn)fn}

static const struct format formats[] = {
    FORMAT("Q4_0", block_q4_0, QK4_0, dequantize_row_q4_0),
    FORMAT("Q4_1", block_q4_1, QK4_1, dequantize_row_q4_1),
    FORMAT("Q5_0", block_q5_0, QK5_0, dequantize_row_q5_0),
    FORMAT("Q5_1", block_q5_1, QK5_1, dequantize_row_q5_1),
    FORMAT("Q8_0", block_q8_0, QK8_0, dequantize_row_q8_0),
    FORMAT("Q2_K", block_q2_K, QK_K, dequantize_row_q2_K),
    FORMAT("Q3_K", block_q3_K, QK_K, dequantize_row_q3_K),
    FORMAT("Q4_K", block_q4_K, QK_K, dequantize_row_q4_K),
    FORMAT("Q5_K", block_q5_K, QK_K, dequantize_row_q5_K),
    FORMAT("Q6_K", block_q6_K, QK_K, dequantize_row_q6_K),
    FORMAT("IQ1_S", block_iq1_s, QK_K, dequantize_row_iq1_s),
    FORMAT("IQ1_M", block_iq1_m, QK_K, dequantize_row_iq1_m),
    FORMAT("IQ2_XXS", block_iq2_xxs, QK_K, dequantize_row_iq2_xxs),
    FORMAT("IQ2_XS", block_iq2_xs, QK_K, dequantize_row_iq2_xs),
    FORMAT("IQ2_S", block_iq2_s, QK_K, dequantize_row_iq2_s),
    FORMAT("IQ3_XXS", block_iq3_xxs, QK_K, dequantize_row_iq3_xxs),
    FORMAT("IQ3_S", block_iq3_s, QK_K, dequantize_row_iq3_s),
    FORMAT("IQ4_NL", block_iq4_nl, QK4_NL, dequantize_row_iq4_nl),
    FORMAT("IQ4_XS", block_iq4_xs, QK_K, dequantize_row_iq4_xs),
};
#define FORMATS (sizeof(formats) / sizeof(formats[0]))

static void* read_file(const char* path, size_t* size) {
    FILE* f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "cannot read %s\n", path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    *size = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    void* data = malloc(*size + 1);
    if (fread(data, 1, *size, f) != *size) exit(1);
    fclose(f);
    return data;
}

static void write_file(const char* path, const void* data, size_t size) {
    FILE* f = fopen(path, "wb");
    if (f == NULL || fwrite(data, 1, size, f) != size) {
        fprintf(stderr, "cannot write %s\n", path);
        exit(1);
    }
    fclose(f);
}

#define TABLE(t) fwrite(t, sizeof(t), 1, f)

int main(int argc, char** argv) {
    if (argc == 2 && strcmp(argv[1], "layout") == 0) {
        for (size_t i = 0; i < FORMATS; ++i)
            printf("%s %zu %d\n", formats[i].name, formats[i].bytes, formats[i].weights);
        return 0;
    }
    if (argc == 3 && strcmp(argv[1], "tables") == 0) {
        FILE* f = fopen(argv[2], "wb");
        if (f == NULL) return 1;
        TABLE(kmask_iq2xs);
        TABLE(ksigns_iq2xs);
        TABLE(kvalues_iq4nl);
        TABLE(iq2xxs_grid);
        TABLE(iq2xs_grid);
        TABLE(iq2s_grid);
        TABLE(iq3xxs_grid);
        TABLE(iq3s_grid);
        TABLE(iq1s_grid);
        return fclose(f) == 0 ? 0 : 1;
    }
    if (argc == 4) {
        for (size_t i = 0; i < FORMATS; ++i) {
            if (strcmp(argv[1], formats[i].name) != 0) continue;
            size_t size = 0;
            void* blocks = read_file(argv[2], &size);
            if (size % formats[i].bytes != 0) {
                fprintf(stderr, "%s: %zu bytes are not whole blocks\n", argv[2], size);
                return 1;
            }
            const int64_t k = (int64_t)(size / formats[i].bytes) * formats[i].weights;
            float* values = malloc(sizeof(float) * (size_t)k + 1);
            formats[i].fn(blocks, values, k);
            write_file(argv[3], values, sizeof(float) * (size_t)k);
            return 0;
        }
        fprintf(stderr, "unknown format %s\n", argv[1]);
        return 1;
    }
    fprintf(stderr, "usage: ggml_dequantize layout | tables <out> | <format> <blocks> <values>\n");
    return 1;
}
