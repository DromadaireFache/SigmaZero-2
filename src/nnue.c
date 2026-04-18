#include "nnue.h"

#define NDEBUG
#include <assert.h>
#include <inttypes.h>
#include <stdio.h>

/* Linear algebra functions (not efficient-update optimized yet) */

void mat16_mul_bitvec(int m, const int16_t A[m][769], const uint64_t x[13], int16_t y[m]) {
    for (int i = 0; i < m; i++) {
        int32_t sum = 0;
        for (int j = 0; j < 13; j++) {
            uint64_t bits = x[j];
            for (int k = 0; k < 64 && bits != 0; k++) {
                if (bits & 1) {
                    sum += A[i][j * 64 + k];
                }
                bits >>= 1;
            }
        }
        assert(sum >= INT16_MIN && sum <= INT16_MAX);
        y[i] = (int16_t)sum;
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

void vec16_sub(int size, const int16_t x[size], const int16_t y[size], int16_t z[size]) {
    for (int i = 0; i < size; i++) {
        assert((int32_t)x[i] - y[i] >= INT16_MIN && (int32_t)x[i] - y[i] <= INT16_MAX);
        z[i] = x[i] - y[i];
    }
}

void clamp16(int size, int16_t x[size], int16_t min, int16_t max) {
    for (int i = 0; i < size; i++) {
        if (x[i] < min)
            x[i] = min;
        else if (x[i] > max)
            x[i] = max;
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

// Constants and parameters defined in params.c
const int fc1_k = 9574;
const int fc2_k = 3171;
const int fc3_k = 19928;
const int fc4_k = 2978;
extern const int16_t fc1_weight[1024][769];
extern const int16_t fc1_bias[1024];
extern const int16_t fc2_weight[256][1024];
extern const int16_t fc2_bias[256];
extern const int16_t fc3_weight[128][256];
extern const int16_t fc3_bias[128];
extern const int16_t fc4_weight[1][128];
extern const int16_t fc4_bias[1];

// Convert piece type to index (0-11)
const int piece_to_plane[128] = {
    ['p'] = 0, ['P'] = 1, ['n'] = 2, ['N'] = 3, ['b'] = 4,  ['B'] = 5,
    ['r'] = 6, ['R'] = 7, ['q'] = 8, ['Q'] = 9, ['k'] = 10, ['K'] = 11,
};

// input: 12 planes of 64 squares + 1 plane for side to move = 769
int forward(const uint64_t input[13]) {
    // Layer 1: x = clamp16(fc1_weights * input + fc1_biases, 0, fc1_k)
    // Layer 2: x = clamp16(fc2_weights * x + fc2_biases, 0, fc2_k)
    // Layer 3: x = clamp16(fc3_weights * x + fc3_biases, 0, fc3_k)
    // Layer 4: x = fc4_weights * x + fc4_biases
    // Output: return x / fc4_k
    int16_t x1[1024], x2[256], x3[128], output[1];
    mat16_mul_bitvec(1024, fc1_weight, input, x1);
    vec16_add(1024, x1, fc1_bias, x1);
    clamp16(1024, x1, 0, fc1_k);
    // printf("x1 = ");
    // print_vec16(x1, 1024); // debug
    mat16_mul(256, 1024, fc2_weight, x1, x2, fc1_k);
    vec16_add(256, x2, fc2_bias, x2);
    clamp16(256, x2, 0, fc2_k);
    // printf("x2 = ");
    // print_vec16(x2, 256); // debug
    mat16_mul(128, 256, fc3_weight, x2, x3, fc2_k);
    vec16_add(128, x3, fc3_bias, x3);
    clamp16(128, x3, 0, fc3_k);
    // printf("x3 = ");
    // print_vec16(x3, 128); // debug
    mat16_mul(1, 128, fc4_weight, x3, output, fc3_k);
    vec16_add(1, output, fc4_bias, output);
    // printf("output = ");
    // print_vec16(output, 1); // debug
    return (int)output[0] * 100 / fc4_k;
}

/* Test function */

// Fen to input vector
void fen_to_input(const char* fen, uint64_t input[13]) {
    for (int i = 0; i < 13; i++) input[i] = 0;
    int row = 7, col = 0;
    int phase = 0;  // 0: board, 1: side-to-move, 2: remaining fields
    for (const char* p = fen; *p; p++) {
        char c = *p;
        if (phase == 0) {
            if (c == ' ') {
                phase = 1;
            } else if (c == '/') {
                row--;
                col = 0;
            } else if (c >= '1' && c <= '8') {
                col += c - '0';
            } else {
                int index = piece_to_plane[(int)c];
                if (index >= 0) {
                    input[index] |= (1ULL << (row * 8 + col));
                    col++;
                }
            }
        } else if (phase == 1) {
            input[12] = (c == 'w') ? 1 : 0;
            phase = 2;
        } else if (c == ' ') {
            break;
        }
    }
}

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

    uint64_t input[13];
    for (int i = 0; i < sizeof(fens) / sizeof(fens[0]); i++) {
        fen_to_input(fens[i], input);
        int score = forward(input);
        printf("FEN: %s -> Score: %d\n", fens[i], score);
    }

    // Testing efficient updates

    uint64_t input_a[13];
    fen_to_input("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", input_a);
    int16_t x1_a[1024], x2_a[256], x3_a[128];
    mat16_mul_bitvec(1024, fc1_weight, input_a, x1_a);
    vec16_add(1024, x1_a, fc1_bias, x1_a);
    clamp16(1024, x1_a, 0, fc1_k);
    mat16_mul(256, 1024, fc2_weight, x1_a, x2_a, fc1_k);
    vec16_add(256, x2_a, fc2_bias, x2_a);
    clamp16(256, x2_a, 0, fc2_k);
    mat16_mul(128, 256, fc3_weight, x2_a, x3_a, fc2_k);
    vec16_add(128, x3_a, fc3_bias, x3_a);
    clamp16(128, x3_a, 0, fc3_k);

    // Make a move
    uint64_t input_b[13];
    fen_to_input("r2q1rk1/5ppp/1pppnn2/b6b/P2PP3/1N2QN1P/2B2PP1/1RB2RK1 b - - 5 21", input_b);
    int16_t x1_b[1024], x2_b[256], x3_b[128];
    mat16_mul_bitvec(1024, fc1_weight, input_b, x1_b);
    vec16_add(1024, x1_b, fc1_bias, x1_b);
    clamp16(1024, x1_b, 0, fc1_k);
    mat16_mul(256, 1024, fc2_weight, x1_b, x2_b, fc1_k);
    vec16_add(256, x2_b, fc2_bias, x2_b);
    clamp16(256, x2_b, 0, fc2_k);
    mat16_mul(128, 256, fc3_weight, x2_b, x3_b, fc2_k);
    vec16_add(128, x3_b, fc3_bias, x3_b);
    clamp16(128, x3_b, 0, fc3_k);

    // Difference on input layer
    uint64_t input_diff[13];
    int non_zero = 0, total_non_zero = 0, total_neurons = 0;
    for (int i = 0; i < 13; i++) {
        input_diff[i] = input_a[i] ^ input_b[i];  // should be zero if inputs are identical
        non_zero += __builtin_popcountll(input_diff[i]);
    }
    printf("Percentage of non-zero differences in input: %.2f%%\n", (float)non_zero * 100 / 769);
    total_non_zero += non_zero;
    total_neurons += 769;

    // Difference on first layer
    int16_t diff[1024];
    vec16_sub(1024, x1_b, x1_a, diff);
    non_zero = 0;
    for (int i = 0; i < 1024; i++) {
        if (diff[i] != 0) non_zero++;
    }
    printf("Percentage of non-zero differences in layer 1: %.2f%%\n", (float)non_zero * 100 / 1024);
    total_non_zero += non_zero;
    total_neurons += 1024;

    // Difference on second layer
    vec16_sub(256, x2_b, x2_a, diff);
    non_zero = 0;
    for (int i = 0; i < 256; i++) {
        if (diff[i] != 0) non_zero++;
    }
    printf("Percentage of non-zero differences in layer 2: %.2f%%\n", (float)non_zero * 100 / 256);
    total_non_zero += non_zero;
    total_neurons += 256;

    // Difference on third layer
    vec16_sub(128, x3_b, x3_a, diff);
    non_zero = 0;
    for (int i = 0; i < 128; i++) {
        if (diff[i] != 0) non_zero++;
    }
    printf("Percentage of non-zero differences in layer 3: %.2f%%\n", (float)non_zero * 100 / 128);
    total_non_zero += non_zero;
    total_neurons += 128;

    printf("Overall percentage of non-zero differences: %.2f%%\n", (float)total_non_zero * 100 / total_neurons);
}