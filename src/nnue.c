#include "nnue.h"

#define NDEBUG
#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// #define COUNT_MATH_OPS  // Uncomment to count number of math operations for testing purposes

int math_ops = 0;  // Count number of math operations for testing purposes
#ifdef COUNT_MATH_OPS
#define MATHOP(op) \
    math_ops++;    \
    op
#else
#define MATHOP(op) op
#endif

/* Linear algebra functions (not efficient-update optimized yet) */

void mat16_mul_bitvec(int m, const int16_t A[m][769], const uint64_t x[13], int16_t y[m]) {
    for (int i = 0; i < m; i++) {
        int32_t sum = 0;
        for (int j = 0; j < 13; j++) {
            uint64_t bits = x[j];
            for (int k = 0; k < 64 && bits != 0; k++) {
                if (bits & 1) {
                    MATHOP(sum += A[i][j * 64 + k]);
                }
                MATHOP(bits >>= 1);
            }
        }
        assert(sum >= INT16_MIN && sum <= INT16_MAX);
        y[i] = (int16_t)sum;
    }
}

void mat16_mul_bitvec_efficient(int m, const int16_t A[769][m], const uint64_t x[13],
                                uint64_t prev_x[13], int16_t y[m]) {
    for (int j = 0; j < 13; j++) {
        uint64_t changed_bits = x[j] ^ prev_x[j];
        prev_x[j] = x[j];
        for (int k = 0; k < 64 && changed_bits != 0; k++) {
            if (changed_bits & 1) {
                for (int i = 0; i < m; i++) {
                    int16_t delta = (x[j] & (1ULL << k)) ? A[j * 64 + k][i] : -A[j * 64 + k][i];
                    MATHOP(y[i] += delta);
                }
            }
            MATHOP(changed_bits >>= 1);
        }
    }
}

void mat16_mul(int m, int n, const int16_t A[m][n], const int16_t x[n], int16_t y[m], int divisor) {
    for (int i = 0; i < m; i++) {
        int32_t sum = 0;
        for (int j = 0; j < n; j++) {
            MATHOP(sum += (int32_t)A[i][j] * x[j]);
        }
        assert(sum >= INT16_MIN * divisor && sum <= INT16_MAX * divisor);
        MATHOP(y[i] = (int16_t)(sum / divisor));
    }
}

void vec16_add(int size, const int16_t x[size], const int16_t y[size], int16_t z[size]) {
    for (int i = 0; i < size; i++) {
        assert((int32_t)x[i] + y[i] >= INT16_MIN && (int32_t)x[i] + y[i] <= INT16_MAX);
        MATHOP(z[i] = x[i] + y[i]);
    }
}

void vec16_sub(int size, const int16_t x[size], const int16_t y[size], int16_t z[size]) {
    for (int i = 0; i < size; i++) {
        assert((int32_t)x[i] - y[i] >= INT16_MIN && (int32_t)x[i] - y[i] <= INT16_MAX);
        MATHOP(z[i] = x[i] - y[i]);
    }
}

void clamp16(int size, int16_t x[size], int16_t y[size], int16_t min, int16_t max) {
    for (int i = 0; i < size; i++) {
        MATHOP(if (x[i] < min) y[i] = min; else if (x[i] > max) y[i] = max; else y[i] = x[i];)
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

// Constants and parameters defined in params.c for Arch1 model
const int fc1_k = 6872;
const int fc2_k = 5463;
const int fc3_k = 1338;
extern const int16_t fc1_weight[769][256];
extern const int16_t fc1_bias[256];
extern const int16_t fc2_weight[64][256];
extern const int16_t fc2_bias[64];
extern const int16_t fc3_weight[1][64];
extern const int16_t fc3_bias[1];

// Convert piece type to index (0-11)
const int piece_to_plane[128] = {
    ['p'] = 0, ['P'] = 1, ['n'] = 2, ['N'] = 3, ['b'] = 4,  ['B'] = 5,
    ['r'] = 6, ['R'] = 7, ['q'] = 8, ['Q'] = 9, ['k'] = 10, ['K'] = 11,
};

void init_nnue(Chess* chess) {
    memset(chess->nnue.input, 0, sizeof(chess->nnue.input));  // Fill input accumulator with 0
    memcpy(chess->nnue.y1, fc1_bias, sizeof(fc1_bias));       // Start with bias values
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
    // mat16_mul_bitvec(256, fc1_weight, input, x1);
    mat16_mul_bitvec_efficient(256, fc1_weight, input, chess->nnue.input, chess->nnue.y1);
    // vec16_add(256, x1, fc1_bias, x1);
    clamp16(256, chess->nnue.y1, x1, 0, fc1_k);

    mat16_mul(64, 256, fc2_weight, x1, x2, fc1_k);
    vec16_add(64, x2, fc2_bias, x2);
    clamp16(64, x2, x2, 0, fc2_k);

    mat16_mul(1, 64, fc3_weight, x2, output, fc2_k);
    vec16_add(1, output, fc3_bias, output);
    return (int)output[0] * 100 / fc3_k;
}

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
    math_ops = 0;
    Chess_user_move(chess, "e7e5");
    printf("Efficient update eval: %d\n", forward(chess));
    printf("Number of math operations: %d\n", math_ops);
    free(chess);

    chess = Chess_from_fen(e4e5);
    math_ops = 0;
    printf("Normal forward eval: %d\n", forward(chess));
    printf("Number of math operations: %d\n", math_ops);
}