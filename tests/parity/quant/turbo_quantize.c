// The TurboQuant KV formats as their reference implementation writes them: the oracle gen_turbo_cases.py writes
// tests/fixtures/quant/turbo with. Built with ggml/src/ggml-turbo-quant.c of a TheTom/llama-cpp-turboquant checkout.
//
//   turbo_quantize layout                  one line per format: bits, bytes of a block, values of a block
//   turbo_quantize <bits> <rows> <blocks>  the block of each row of 128 f32 values in file <rows>, into <blocks>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml-quants.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ggml-turbo-quant.c refers to these from code this program never runs.
void ggml_abort(const char* file, int line, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "ggml_abort at %s:%d: ", file, line);
    vfprintf(stderr, fmt, args);
    va_end(args);
    abort();
}
float ggml_table_f32_f16[1 << 16];

#define GROUP 128

int main(int argc, char** argv) {
    if (argc == 2 && strcmp(argv[1], "layout") == 0) {
        printf("4 %zu %d\n3 %zu %d\n2 %zu %d\n", sizeof(block_turbo4_0), QK_TURBO4, sizeof(block_turbo3_0), QK_TURBO3,
               sizeof(block_turbo2_0), QK_TURBO2);
        return 0;
    }
    if (argc != 4) {
        fprintf(stderr, "usage: turbo_quantize layout | <bits> <rows> <blocks>\n");
        return 2;
    }
    const int bits = atoi(argv[1]);
    FILE* in = fopen(argv[2], "rb");
    FILE* out = fopen(argv[3], "wb");
    if (in == NULL || out == NULL || bits < 2 || bits > 4) {
        fprintf(stderr, "turbo_quantize: bad arguments\n");
        return 2;
    }
    float row[GROUP];
    while (fread(row, sizeof(float), GROUP, in) == GROUP) {
        if (bits == 4) {
            block_turbo4_0 b;
            quantize_row_turbo4_0_ref(row, &b, GROUP);
            fwrite(&b, sizeof(b), 1, out);
        } else if (bits == 3) {
            block_turbo3_0 b;
            quantize_row_turbo3_0_ref(row, &b, GROUP);
            fwrite(&b, sizeof(b), 1, out);
        } else {
            block_turbo2_0 b;
            quantize_row_turbo2_0_ref(row, &b, GROUP);
            fwrite(&b, sizeof(b), 1, out);
        }
    }
    fclose(in);
    fclose(out);
    return 0;
}
