#include "nnue.h"

#define NDEBUG
#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Linear algebra functions (not efficient-update optimized yet) */

void mat16_mul_bitvec_efficient(int m, const int16_t A[769][m], const uint64_t x[13],
                                uint64_t prev_x[13], int16_t y[m]) {
    for (int j = 0; j < 13; j++) {
        uint64_t changed_bits = x[j] ^ prev_x[j];
        prev_x[j] = x[j];
        for (int k = 0; k < 64 && changed_bits != 0; k++) {
            if (changed_bits & 1) {
                for (int i = 0; i < m; i++) {
                    int16_t delta = (x[j] & (1ULL << k)) ? A[j * 64 + k][i] : -A[j * 64 + k][i];
                    y[i] += delta;
                }
            }
            changed_bits >>= 1;
        }
    }
}

void mat16_mul(int m, int n, const int16_t A[m][n], const int16_t x[n], int16_t y[m], int divisor) {
    for (int i = 0; i < m; i++) {
        int32_t sum = 0;
        for (int j = 0; j < n; j++) {
            sum += (int32_t)A[i][j] * x[j];
        }
        assert(sum >= INT16_MIN * divisor && sum <= INT16_MAX * divisor);
        y[i] = (int16_t)(sum / divisor);
    }
}

void vec16_add(int size, const int16_t x[size], const int16_t y[size], int16_t z[size]) {
    for (int i = 0; i < size; i++) {
        assert((int32_t)x[i] + y[i] >= INT16_MIN && (int32_t)x[i] + y[i] <= INT16_MAX);
        z[i] = x[i] + y[i];
    }
}

void vec16_clamp(int size, int16_t x[size], int16_t y[size], int16_t min, int16_t max) {
    for (int i = 0; i < size; i++) {
        if (x[i] < min)
            y[i] = min;
        else if (x[i] > max)
            y[i] = max;
        else
            y[i] = x[i];
    }
}

/* Helper functions */

void print_vec16(const int16_t* x, int size) {
    printf("[");
    for (int i = 0; i < size; i++) {
        printf("%d", x[i]);
        if (i < size - 1) printf(", ");
    }
    printf("]\n");
}

/* Neural network functions */

#define MODEL_TINY

// Convert piece type to index (0-11)
const int piece_to_plane[128] = {
    ['p'] = 0, ['P'] = 1, ['n'] = 2, ['N'] = 3, ['b'] = 4,  ['B'] = 5,
    ['r'] = 6, ['R'] = 7, ['q'] = 8, ['Q'] = 9, ['k'] = 10, ['K'] = 11,
};

#ifdef MODEL_ARCH1

// Constants and parameters defined in params.c for Arch1 model
extern const int16_t arch1_fc1_weight[769][256];
extern const int16_t arch1_fc1_bias[256];
extern const int16_t arch1_fc2_weight[64][256];
extern const int16_t arch1_fc2_bias[64];
extern const int16_t arch1_fc3_weight[1][64];
extern const int16_t arch1_fc3_bias[1];
extern const int arch1_fc1_k;
extern const int arch1_fc2_k;
extern const int arch1_fc3_k;

void init_nnue(Chess* chess) {
    memset(chess->nnue.input, 0, sizeof(chess->nnue.input));  // Fill input accumulator with 0
    memcpy(chess->nnue.y1, arch1_fc1_bias, sizeof(arch1_fc1_bias));  // Start with bias values
}

int forward(Chess* chess) {
    uint64_t input[13] = {
        chess->bb.black_pawns,
        chess->bb.white_pawns,
        chess->bb.black_knights,
        chess->bb.white_knights,
        chess->bb.black_bishops,
        chess->bb.white_bishops,
        chess->bb.black_rooks,
        chess->bb.white_rooks,
        chess->bb.black_queens,
        chess->bb.white_queens,
        chess->bb.black_kings,
        chess->bb.white_kings,
        !chess->turn,  // Turn is flipped in NNUE, white is 1, black is 0
    };

    int16_t x1[256], x2[64], output[1];
    mat16_mul_bitvec_efficient(256, arch1_fc1_weight, input, chess->nnue.input, chess->nnue.y1);
    vec16_clamp(256, chess->nnue.y1, x1, 0, arch1_fc1_k);

    mat16_mul(64, 256, arch1_fc2_weight, x1, x2, arch1_fc1_k);
    vec16_add(64, x2, arch1_fc2_bias, x2);
    vec16_clamp(64, x2, x2, 0, arch1_fc2_k);

    mat16_mul(1, 64, arch1_fc3_weight, x2, output, arch1_fc2_k);
    vec16_add(1, output, arch1_fc3_bias, output);
    return (int)output[0] * 100 / arch1_fc3_k;
}

#elif defined MODEL_ARCH2

extern const int16_t arch2_fc1_weight[769][1024];
extern const int16_t arch2_fc1_bias[1024];
extern const int16_t arch2_fc2_weight[256][1024];
extern const int16_t arch2_fc2_bias[256];
extern const int16_t arch2_fc3_weight[128][256];
extern const int16_t arch2_fc3_bias[128];
extern const int16_t arch2_fc4_weight[1][128];
extern const int16_t arch2_fc4_bias[1];
extern const int arch2_fc1_k;
extern const int arch2_fc2_k;
extern const int arch2_fc3_k;
extern const int arch2_fc4_k;

void init_nnue(Chess* chess) {
    memset(chess->nnue.input, 0, sizeof(chess->nnue.input));  // Fill input accumulator with 0
    memcpy(chess->nnue.y1, arch2_fc1_bias, sizeof(arch2_fc1_bias));  // Start with bias values
}

int forward(Chess* chess) {
    uint64_t input[13] = {
        chess->bb.black_pawns,
        chess->bb.white_pawns,
        chess->bb.black_knights,
        chess->bb.white_knights,
        chess->bb.black_bishops,
        chess->bb.white_bishops,
        chess->bb.black_rooks,
        chess->bb.white_rooks,
        chess->bb.black_queens,
        chess->bb.white_queens,
        chess->bb.black_kings,
        chess->bb.white_kings,
        !chess->turn,  // Turn is flipped in NNUE, white is 1, black is 0
    };

    int16_t x1[1024], x2[256], x3[128], output[1];
    mat16_mul_bitvec_efficient(1024, arch2_fc1_weight, input, chess->nnue.input, chess->nnue.y1);
    vec16_clamp(1024, chess->nnue.y1, x1, 0, arch2_fc1_k);

    mat16_mul(256, 1024, arch2_fc2_weight, x1, x2, arch2_fc1_k);
    vec16_add(256, x2, arch2_fc2_bias, x2);
    vec16_clamp(256, x2, x2, 0, arch2_fc2_k);

    mat16_mul(128, 256, arch2_fc3_weight, x2, x3, arch2_fc2_k);
    vec16_add(128, x3, arch2_fc3_bias, x3);
    vec16_clamp(128, x3, x3, 0, arch2_fc3_k);

    mat16_mul(1, 128, arch2_fc4_weight, x3, output, arch2_fc3_k);
    vec16_add(1, output, arch2_fc4_bias, output);
    return (int)output[0] * 100 / arch2_fc4_k;
}

#elif defined MODEL_TINY

extern const int16_t tiny_fc1_weight[769][32];
extern const int16_t tiny_fc1_bias[32];
extern const int16_t tiny_fc2_weight[64][32];
extern const int16_t tiny_fc2_bias[64];
extern const int16_t tiny_fc3_weight[1][64];
extern const int16_t tiny_fc3_bias[1];
extern const int tiny_fc1_k;
extern const int tiny_fc2_k;
extern const int tiny_fc3_k;

void init_nnue(Chess* chess) {
    memset(chess->nnue.input, 0, sizeof(chess->nnue.input));  // Fill input accumulator with 0
    memcpy(chess->nnue.y1, tiny_fc1_bias, sizeof(tiny_fc1_bias));  // Start with bias values
}

int forward(Chess* chess) {
    uint64_t input[13] = {
        chess->bb.black_pawns,
        chess->bb.white_pawns,
        chess->bb.black_knights,
        chess->bb.white_knights,
        chess->bb.black_bishops,
        chess->bb.white_bishops,
        chess->bb.black_rooks,
        chess->bb.white_rooks,
        chess->bb.black_queens,
        chess->bb.white_queens,
        chess->bb.black_kings,
        chess->bb.white_kings,
        !chess->turn,  // Turn is flipped in NNUE, white is 1, black is 0
    };

    int16_t x1[32], x2[64], output[1];
    mat16_mul_bitvec_efficient(32, tiny_fc1_weight, input, chess->nnue.input, chess->nnue.y1);
    vec16_clamp(32, chess->nnue.y1, x1, 0, tiny_fc1_k);

    mat16_mul(64, 32, tiny_fc2_weight, x1, x2, tiny_fc1_k);
    vec16_add(64, x2, tiny_fc2_bias, x2);
    vec16_clamp(64, x2, x2, 0, tiny_fc2_k);

    mat16_mul(1, 64, tiny_fc3_weight, x2, output, tiny_fc2_k);
    vec16_add(1, output, tiny_fc3_bias, output);
    return (int)output[0] * 100 / tiny_fc3_k;
}

#elif defined MODEL_RESNET

extern const int16_t resnet_input_proj_weight[769][512];
extern const int16_t resnet_input_proj_bias[512];
extern const int16_t resnet_blocks_0_fc1_weight[512][512];
extern const int16_t resnet_blocks_0_fc1_bias[512];
extern const int16_t resnet_blocks_0_fc2_weight[512][512];
extern const int16_t resnet_blocks_0_fc2_bias[512];
extern const int16_t resnet_blocks_0_norm1_weight[512];
extern const int16_t resnet_blocks_0_norm1_bias[512];
extern const int16_t resnet_blocks_0_norm2_weight[512];
extern const int16_t resnet_blocks_0_norm2_bias[512];
extern const int16_t resnet_blocks_1_fc1_weight[512][512];
extern const int16_t resnet_blocks_1_fc1_bias[512];
extern const int16_t resnet_blocks_1_fc2_weight[512][512];
extern const int16_t resnet_blocks_1_fc2_bias[512];
extern const int16_t resnet_blocks_1_norm1_weight[512];
extern const int16_t resnet_blocks_1_norm1_bias[512];
extern const int16_t resnet_blocks_1_norm2_weight[512];
extern const int16_t resnet_blocks_1_norm2_bias[512];
extern const int16_t resnet_blocks_2_fc1_weight[512][512];
extern const int16_t resnet_blocks_2_fc1_bias[512];
extern const int16_t resnet_blocks_2_fc2_weight[512][512];
extern const int16_t resnet_blocks_2_fc2_bias[512];
extern const int16_t resnet_blocks_2_norm1_weight[512];
extern const int16_t resnet_blocks_2_norm1_bias[512];
extern const int16_t resnet_blocks_2_norm2_weight[512];
extern const int16_t resnet_blocks_2_norm2_bias[512];
extern const int16_t resnet_blocks_3_fc1_weight[512][512];
extern const int16_t resnet_blocks_3_fc1_bias[512];
extern const int16_t resnet_blocks_3_fc2_weight[512][512];
extern const int16_t resnet_blocks_3_fc2_bias[512];
extern const int16_t resnet_blocks_3_norm1_weight[512];
extern const int16_t resnet_blocks_3_norm1_bias[512];
extern const int16_t resnet_blocks_3_norm2_weight[512];
extern const int16_t resnet_blocks_3_norm2_bias[512];
extern const int16_t resnet_blocks_4_fc1_weight[512][512];
extern const int16_t resnet_blocks_4_fc1_bias[512];
extern const int16_t resnet_blocks_4_fc2_weight[512][512];
extern const int16_t resnet_blocks_4_fc2_bias[512];
extern const int16_t resnet_blocks_4_norm1_weight[512];
extern const int16_t resnet_blocks_4_norm1_bias[512];
extern const int16_t resnet_blocks_4_norm2_weight[512];
extern const int16_t resnet_blocks_4_norm2_bias[512];
extern const int16_t resnet_blocks_5_fc1_weight[512][512];
extern const int16_t resnet_blocks_5_fc1_bias[512];
extern const int16_t resnet_blocks_5_fc2_weight[512][512];
extern const int16_t resnet_blocks_5_fc2_bias[512];
extern const int16_t resnet_blocks_5_norm1_weight[512];
extern const int16_t resnet_blocks_5_norm1_bias[512];
extern const int16_t resnet_blocks_5_norm2_weight[512];
extern const int16_t resnet_blocks_5_norm2_bias[512];
extern const int16_t resnet_output_weight[1][512];
extern const int16_t resnet_output_bias[1];
extern const int resnet_input_proj_k;
extern const int resnet_blocks_0_fc1_k;
extern const int resnet_blocks_0_fc2_k;
extern const int resnet_blocks_0_norm1_k;
extern const int resnet_blocks_0_norm2_k;
extern const int resnet_blocks_1_fc1_k;
extern const int resnet_blocks_1_fc2_k;
extern const int resnet_blocks_1_norm1_k;
extern const int resnet_blocks_1_norm2_k;
extern const int resnet_blocks_2_fc1_k;
extern const int resnet_blocks_2_fc2_k;
extern const int resnet_blocks_2_norm1_k;
extern const int resnet_blocks_2_norm2_k;
extern const int resnet_blocks_3_fc1_k;
extern const int resnet_blocks_3_fc2_k;
extern const int resnet_blocks_3_norm1_k;
extern const int resnet_blocks_3_norm2_k;
extern const int resnet_blocks_4_fc1_k;
extern const int resnet_blocks_4_fc2_k;
extern const int resnet_blocks_4_norm1_k;
extern const int resnet_blocks_4_norm2_k;
extern const int resnet_blocks_5_fc1_k;
extern const int resnet_blocks_5_fc2_k;
extern const int resnet_blocks_5_norm1_k;
extern const int resnet_blocks_5_norm2_k;
extern const int resnet_output_k;

void init_nnue(Chess* chess) {
    memset(chess->nnue.input, 0, sizeof(chess->nnue.input));  // Fill input accumulator with 0
    memcpy(chess->nnue.y1, resnet_input_proj_bias,
           sizeof(resnet_input_proj_bias));  // Start with bias values
}

enum { RESNET_Q = 1024 };

static uint32_t resnet_isqrt(uint64_t value) {
    uint64_t result = 0;
    uint64_t bit = (uint64_t)1 << 62;
    while (bit > value) bit >>= 2;
    while (bit != 0) {
        if (value >= result + bit) {
            value -= result + bit;
            result = (result >> 1) + bit;
        } else {
            result >>= 1;
        }
        bit >>= 2;
    }
    return (uint32_t)result;
}

static void resnet_layer_norm(const int32_t x[512], int32_t y[512], const int16_t weight[512],
                              const int16_t bias[512], int factor) {
    int64_t sum = 0;
    for (int i = 0; i < 512; i++) sum += x[i];
    int32_t mean = (int32_t)(sum / 512);

    uint64_t variance_sum = 0;
    for (int i = 0; i < 512; i++) {
        int64_t centered = (int64_t)x[i] - mean;
        variance_sum += (uint64_t)(centered * centered);
    }
    uint32_t stddev = resnet_isqrt(variance_sum / 512 + 10);
    if (stddev == 0) stddev = 1;

    for (int i = 0; i < 512; i++) {
        int64_t centered = (int64_t)x[i] - mean;
        int64_t normalized = centered * RESNET_Q / stddev;
        int64_t value = normalized * weight[i] / factor + (int64_t)bias[i] * RESNET_Q / factor;
        assert(value >= INT32_MIN && value <= INT32_MAX);
        y[i] = (int32_t)value;
    }
}

static void resnet_linear(const int16_t weight[512][512], const int16_t bias[512], int factor,
                          const int32_t x[512], int32_t y[512]) {
    for (int i = 0; i < 512; i++) {
        int64_t sum = (int64_t)bias[i] * RESNET_Q;
        for (int j = 0; j < 512; j++) sum += (int64_t)weight[i][j] * x[j];
        sum /= factor;
        assert(sum >= INT32_MIN && sum <= INT32_MAX);
        y[i] = (int32_t)sum;
    }
}

// ResNet block forward pass in a common fixed-point activation domain.
void _block_forward(const int16_t fc1_weight[512][512], const int16_t fc1_bias[512],
                   const int16_t fc2_weight[512][512], const int16_t fc2_bias[512],
                   const int16_t norm1_weight[512], const int16_t norm1_bias[512],
                   const int16_t norm2_weight[512], const int16_t norm2_bias[512], int fc1_k,
                   int fc2_k, int norm1_k, int norm2_k, const int32_t x_in[512],
                   int32_t x_out[512]) {
    int32_t norm1[512];
    int32_t fc1[512];
    int32_t norm2[512];
    int32_t fc2[512];

    resnet_layer_norm(x_in, norm1, norm1_weight, norm1_bias, norm1_k);
    resnet_linear(fc1_weight, fc1_bias, fc1_k, norm1, fc1);
    for (int i = 0; i < 512; i++) {
        if (fc1[i] < 0) fc1[i] = 0;
        else if (fc1[i] > RESNET_Q) fc1[i] = RESNET_Q;
    }

    resnet_layer_norm(fc1, norm2, norm2_weight, norm2_bias, norm2_k);
    resnet_linear(fc2_weight, fc2_bias, fc2_k, norm2, fc2);
    for (int i = 0; i < 512; i++) {
        int64_t value = (int64_t)x_in[i] + fc2[i];
        assert(value >= INT32_MIN && value <= INT32_MAX);
        x_out[i] = (int32_t)value;
    }
}

#define block_forward(block_id, x_in, x_out) \
    _block_forward(resnet_blocks_##block_id##_fc1_weight, resnet_blocks_##block_id##_fc1_bias, \
                   resnet_blocks_##block_id##_fc2_weight, resnet_blocks_##block_id##_fc2_bias, \
                   resnet_blocks_##block_id##_norm1_weight, resnet_blocks_##block_id##_norm1_bias, \
                   resnet_blocks_##block_id##_norm2_weight, resnet_blocks_##block_id##_norm2_bias, \
                   resnet_blocks_##block_id##_fc1_k, resnet_blocks_##block_id##_fc2_k, \
                   resnet_blocks_##block_id##_norm1_k, resnet_blocks_##block_id##_norm2_k, x_in, x_out)

int forward(Chess* chess) {
    uint64_t input[13] = {
        chess->bb.black_pawns, chess->bb.white_pawns, chess->bb.black_knights,
        chess->bb.white_knights, chess->bb.black_bishops, chess->bb.white_bishops,
        chess->bb.black_rooks, chess->bb.white_rooks, chess->bb.black_queens,
        chess->bb.white_queens, chess->bb.black_kings, chess->bb.white_kings,
        !chess->turn,
    };

    int32_t x1[512], x2[512], x3[512], x4[512], x5[512], x6[512], x7[512];
    mat16_mul_bitvec_efficient(512, resnet_input_proj_weight, input, chess->nnue.input,
                               chess->nnue.y1);
    for (int i = 0; i < 512; i++) {
        int64_t value = (int64_t)chess->nnue.y1[i] * RESNET_Q / resnet_input_proj_k;
        if (value < 0) value = 0;
        else if (value > RESNET_Q) value = RESNET_Q;
        x1[i] = (int32_t)value;
    }

    block_forward(0, x1, x2);
    block_forward(1, x2, x3);
    block_forward(2, x3, x4);
    block_forward(3, x4, x5);
    block_forward(4, x5, x6);
    block_forward(5, x6, x7);

    int64_t output = (int64_t)resnet_output_bias[0] * RESNET_Q;
    for (int i = 0; i < 512; i++) output += (int64_t)resnet_output_weight[0][i] * x7[i];
    output /= resnet_output_k;
    return (int)(output * 100 / RESNET_Q);
}

#endif

/* Test function */

void test_nnue() {
    char* fens[] = {
        "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  // white up a queen
        "rnbqkbn1/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQq - 0 1",   // white up a rook
        "rnbqk1nr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  // white up a bishop
        "rnbqkb1r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  // white up a knight
        "rnbqkbnr/ppppppp1/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  // white up a pawn
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1",  // black up a queen
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1 w Qkq - 0 1",   // black up a rook
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK1NR w KQkq - 0 1",  // black up a bishop
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1",  // black up a knight
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPP1/RNBQKBNR w KQkq - 0 1",  // black up a pawn
        "8/4k3/8/8/4PK2/8/8/8 w - - 0 1",   // winning white endgame position
        "8/4k3/8/8/4P3/4K3/8/8 w - - 0 1",  // drawing endgame position
        "k7/4r3/8/8/8/2QK4/8/8 w - - 0 1",
    };

    for (int i = 0; i < sizeof(fens) / sizeof(fens[0]); i++) {
        Chess* chess = Chess_from_fen(fens[i]);
        int score = forward(chess);
        printf("FEN: %s -> Score: %d\n", fens[i], score);
        free(chess);
    }

    // Testing efficient updates
    // Without efficient updates Depth: 4.67 ± 0.12 (25515)
    // Efficient updates on first layer Depth: 8.24 ± 0.10 (17696)
    // Memory layout optimization Depth: 9.45 ± 0.15

    char* starting = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    char* e4e5 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2";

    Chess* chess = Chess_from_fen(starting);
    printf("Starting eval: %d\n", forward(chess));

    Chess_user_move(chess, "e2e4");
    forward(chess);
    Chess_user_move(chess, "e7e5");
    printf("Efficient update eval: %d\n", forward(chess));
    free(chess);

    chess = Chess_from_fen(e4e5);
    printf("Normal forward eval: %d\n", forward(chess));
}