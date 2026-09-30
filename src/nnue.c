#include "nnue.h"

#define NDEBUG
#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MODEL_ARCH1

typedef int16_t param_t;
typedef int32_t value_t;
typedef int64_t accumulator_t;

static __int128 round_divide(__int128 numerator, int64_t divisor) {
    __int128 half = divisor / 2;
    return numerator >= 0 ? (numerator + half) / divisor : (numerator - half) / divisor;
}

void mat_mul_bitvec_efficient(int m, const param_t A[769][m], const uint64_t x[13],
                              uint64_t prev_x[13], accumulator_t y[m]) {
    for (int j = 0; j < 13; j++) {
        uint64_t changed_bits = x[j] ^ prev_x[j];
        prev_x[j] = x[j];
        for (int k = 0; k < 64 && changed_bits; k++) {
            if (changed_bits & 1)
                for (int i = 0; i < m; i++)
                    y[i] += (x[j] & (1ULL << k)) ? (accumulator_t)A[j * 64 + k][i]
                                                 : -(accumulator_t)A[j * 64 + k][i];
            changed_bits >>= 1;
        }
    }
}

void mat_mul(int m, int n, const param_t A[m][n], const value_t x[n], value_t y[m],
             int64_t divisor) {
    for (int i = 0; i < m; i++) {
        __int128 sum = 0;
        for (int j = 0; j < n; j++) sum += (__int128)A[i][j] * x[j];
        __int128 quotient = round_divide(sum, divisor);
        assert(quotient >= INT64_MIN && quotient <= INT64_MAX);
        y[i] = (value_t)quotient;
    }
}

void vec_add(int size, const value_t x[size], const param_t y[size], value_t z[size]) {
    for (int i = 0; i < size; i++) {
        __int128 sum = (__int128)x[i] + y[i];
        assert(sum >= INT64_MIN && sum <= INT64_MAX);
        z[i] = (value_t)sum;
    }
}

void vec_clamp(int size, const value_t x[size], value_t y[size], value_t min, value_t max) {
    for (int i = 0; i < size; i++) y[i] = x[i] < min ? min : (x[i] > max ? max : x[i]);
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

// Convert piece type to index (0-11)
const int piece_to_plane[128] = {
    ['p'] = 0, ['P'] = 1, ['n'] = 2, ['N'] = 3, ['b'] = 4,  ['B'] = 5,
    ['r'] = 6, ['R'] = 7, ['q'] = 8, ['Q'] = 9, ['k'] = 10, ['K'] = 11,
};

#ifdef MODEL_ARCH1

// Constants and parameters defined in params.c for Arch1 model
const value_t fc1_k = 104030;
const value_t fc2_k = 81024;
const value_t fc3_k = 18184;
extern const int16_t fc1_weight[769][256];
extern const int16_t fc1_bias[256];
extern const int16_t fc2_weight[64][256];
extern const int16_t fc2_bias[64];
extern const int16_t fc3_weight[1][64];
extern const int16_t fc3_bias[1];

void init_nnue(Chess* chess) {
    memset(chess->nnue.input, 0, sizeof(chess->nnue.input));  // Fill input accumulator with 0
    for (int i = 0; i < 256; i++) chess->nnue.y1[i] = fc1_bias[i];  // Fill y1 accumulator with bias
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

    value_t x1[256], x2[64], output[1];
    mat_mul_bitvec_efficient(256, fc1_weight, input, chess->nnue.input,
                             (accumulator_t*)chess->nnue.y1);
    for (int i = 0; i < 256; i++) {
        accumulator_t value = ((accumulator_t*)chess->nnue.y1)[i];
        x1[i] = value < 0 ? 0 : (value > fc1_k ? fc1_k : (value_t)value);
    }

    mat_mul(64, 256, fc2_weight, x1, x2, fc1_k);
    vec_add(64, x2, fc2_bias, x2);
    vec_clamp(64, x2, x2, 0, fc2_k);

    mat_mul(1, 64, fc3_weight, x2, output, fc2_k);
    vec_add(1, output, fc3_bias, output);
    return (int)output[0] * 100 / fc3_k;
}

#elif defined MODEL_ARCH2

const int64_t fc1_k = 7444;
const int64_t fc2_k = 6146;
const int64_t fc3_k = 18325;
const int64_t fc4_k = 3384;
extern const param_t fc1_weight[769][1024];
extern const param_t fc1_bias[1024];
extern const param_t fc2_weight[256][1024];
extern const param_t fc2_bias[256];
extern const param_t fc3_weight[128][256];
extern const param_t fc3_bias[128];
extern const param_t fc4_weight[1][128];
extern const param_t fc4_bias[1];

void init_nnue(Chess* chess) {
    memset(chess->nnue.input, 0, sizeof(chess->nnue.input));  // Fill input accumulator with 0
    if (chess->nnue.y1 == NULL) {
        chess->nnue.y1 = malloc(sizeof(accumulator_t) * 1024);  // Allocate memory for y1
    }
    for (int i = 0; i < 1024; i++) ((accumulator_t*)chess->nnue.y1)[i] = fc1_bias[i];
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

    value_t x1[1024], x2[256], x3[128], output[1];
    mat_mul_bitvec_efficient(1024, fc1_weight, input, chess->nnue.input,
                             (accumulator_t*)chess->nnue.y1);
    for (int i = 0; i < 1024; i++) {
        accumulator_t value = ((accumulator_t*)chess->nnue.y1)[i];
        x1[i] = value < 0 ? 0 : (value > fc1_k ? fc1_k : (value_t)value);
    }

    mat_mul(256, 1024, fc2_weight, x1, x2, fc1_k);
    vec_add(256, x2, fc2_bias, x2);
    vec_clamp(256, x2, x2, 0, fc2_k);

    mat_mul(128, 256, fc3_weight, x2, x3, fc2_k);
    vec_add(128, x3, fc3_bias, x3);
    vec_clamp(128, x3, x3, 0, fc3_k);

    mat_mul(1, 128, fc4_weight, x3, output, fc3_k);
    vec_add(1, output, fc4_bias, output);
    return (int)output[0] * 100 / fc4_k;
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
        // "8/4k3/8/8/4PK2/8/8/8 w - - 0 1",   // winning white endgame position
        // "8/4k3/8/8/4P3/4K3/8/8 w - - 0 1",  // drawing endgame position
        // "k7/4r3/8/8/8/2QK4/8/8 w - - 0 1",
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

    // char* starting = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    // char* e4e5 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2";

    // Chess* chess = Chess_from_fen(starting);
    // printf("Starting eval: %d\n", forward(chess));

    // Chess_user_move(chess, "e2e4");
    // forward(chess);
    // Chess_user_move(chess, "e7e5");
    // printf("Efficient update eval: %d\n", forward(chess));
    // free(chess);

    // chess = Chess_from_fen(e4e5);
    // printf("Normal forward eval: %d\n", forward(chess));
}