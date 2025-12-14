#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_STALL 600  // Number of seconds until stop stall

// A bitboard is a 64-bit integer where each bit represents a square on the
// chessboard (from a1 to h8). The least significant bit (LSB) represents a1,
// and the most significant bit (MSB) represents h8.
typedef uint64_t bitboard_t;

// Print a bitboard as a 64-character string of 0s and 1s
void bitboard_print(bitboard_t bb) {
    for (int rank = 7; rank >= 0; rank--) {
        for (int file = 0; file < 8; file++) {
            int index = rank * 8 + file;
            putchar((bb >> index) & 1 ? '1' : '0');
            putchar(' ');
        }
        putchar('\n');
    }
    putchar('\n');
}

#define EDGE_BB 0xFF818181818181FF

// Convert an index (0-63) to a bitboard with only that bit set
bitboard_t bitboard_from_index(int i) { return 1ULL << i; }

int index_col(int index) { return index % 8; }

int index_row(int index) { return index / 8; }

bitboard_t bitboard_row(int i) { return 0xFFULL << (i - index_col(i)); }

bitboard_t bitboard_col(int i) { return 0x0101010101010101ULL << index_col(i); }

bitboard_t bitboard_row_no_edge(int i) { return 0x7EULL << (i - index_col(i)); }

bitboard_t bitboard_col_no_edge(int i) { return 0x0001010101010100ULL << index_col(i); }

bitboard_t bitboard_rook_mask(int i) {
    return (bitboard_row_no_edge(i) ^ bitboard_col_no_edge(i)) & ~bitboard_from_index(i);
}

const bitboard_t BISHOP_MASKS[64] = {
    0x0040201008040200ULL, 0x0000402010080400ULL, 0x0000004020100a00ULL, 0x0000000040221400ULL,
    0x0000000002442800ULL, 0x0000000204085000ULL, 0x0000020408102000ULL, 0x0002040810204000ULL,
    0x0020100804020000ULL, 0x0040201008040000ULL, 0x00004020100a0000ULL, 0x0000004022140000ULL,
    0x0000000244280000ULL, 0x0000020408500000ULL, 0x0002040810200000ULL, 0x0004081020400000ULL,
    0x0010080402000200ULL, 0x0020100804000400ULL, 0x004020100a000a00ULL, 0x0000402214001400ULL,
    0x0000024428002800ULL, 0x0002040850005000ULL, 0x0004081020002000ULL, 0x0008102040004000ULL,
    0x0008040200020400ULL, 0x0010080400040800ULL, 0x0020100a000a1000ULL, 0x0040221400142200ULL,
    0x0002442800284400ULL, 0x0004085000500800ULL, 0x0008102000201000ULL, 0x0010204000402000ULL,
    0x0004020002040800ULL, 0x0008040004081000ULL, 0x00100a000a102000ULL, 0x0022140014224000ULL,
    0x0044280028440200ULL, 0x0008500050080400ULL, 0x0010200020100800ULL, 0x0020400040201000ULL,
    0x0002000204081000ULL, 0x0004000408102000ULL, 0x000a000a10204000ULL, 0x0014001422400000ULL,
    0x0028002844020000ULL, 0x0050005008040200ULL, 0x0020002010080400ULL, 0x0040004020100800ULL,
    0x0000020408102000ULL, 0x0000040810204000ULL, 0x00000a1020400000ULL, 0x0000142240000000ULL,
    0x0000284402000000ULL, 0x0000500804020000ULL, 0x0000201008040200ULL, 0x0000402010080400ULL,
    0x0002040810204000ULL, 0x0004081020400000ULL, 0x000a102040000000ULL, 0x0014224000000000ULL,
    0x0028440200000000ULL, 0x0050080402000000ULL, 0x0020100804020000ULL, 0x0040201008040200ULL};

bitboard_t bitboard_bishop_mask(int i) { return BISHOP_MASKS[i]; }

int bitboard_bit_count(bitboard_t bb) { return __builtin_popcountll(bb); }

bitboard_t bitboard_target_mask(bitboard_t bb, int target_idx) {
    bitboard_t result = 0;

    // Iterate through each bit set in the mask
    while (bb) {
        // Get the index of the least significant bit
        int bit_pos = __builtin_ctzll(bb);

        // If the corresponding bit in target_idx is set, set it in result
        if (target_idx & 1) result |= (1ULL << bit_pos);

        // Clear the least significant bit from bb
        bb &= bb - 1;

        // Move to next bit in target_idx
        target_idx >>= 1;
    }

    return result;
}

#define SLIDING_PIECE_ADD_MOVE(condition, offset)                \
    for (i = 1; (condition); i++) {                              \
        bitboard_t to_bit = bitboard_from_index(pos + (offset)); \
        bb |= to_bit;                                            \
        if (target_mask & to_bit) break;                         \
    }

bitboard_t rook_move_bb(bitboard_t target_mask, int pos) {
    bitboard_t bb = 0;
    int row = index_row(pos);
    int col = index_col(pos);
    int i;

    SLIDING_PIECE_ADD_MOVE(col + i < 8, i);
    SLIDING_PIECE_ADD_MOVE(row + i < 8, 8 * i);
    SLIDING_PIECE_ADD_MOVE(col - i >= 0, -i);
    SLIDING_PIECE_ADD_MOVE(row - i >= 0, -8 * i);

    return bb;
}

bitboard_t bishop_move_bb(bitboard_t target_mask, int pos) {
    bitboard_t bb = 0;
    int row = index_row(pos);
    int col = index_col(pos);
    int i;

    SLIDING_PIECE_ADD_MOVE(col + i < 8 && row + i < 8, 9 * i);
    SLIDING_PIECE_ADD_MOVE(col - i >= 0 && row + i < 8, 7 * i);
    SLIDING_PIECE_ADD_MOVE(col + i < 8 && row - i >= 0, -7 * i);
    SLIDING_PIECE_ADD_MOVE(col - i >= 0 && row - i >= 0, -9 * i);

    return bb;
}

uint64_t random_uint64() {
    uint64_t result = 0xFFFFFFFFFFFFFFFFULL;
    for (int i = 0; i < 3; i++) {
        uint64_t r1 = rand() & 0xFFFF;
        uint64_t r2 = rand() & 0xFFFF;
        uint64_t r3 = rand() & 0xFFFF;
        uint64_t r4 = rand() & 0xFFFF;
        result &= (r1 << 48) | (r2 << 32) | (r3 << 16) | r4;
    }
    return result;
}

/*
ROOK_MOVE_BB[square][(target_mask * MAGIC_NUMS[square]) >> MAGIC_SHIFTS[square]]

For each square:
1.  Generate magic number (random number)
2.  initially set magic shift to smallest discovered so far (15 by default)
3.  loop through each possible target mask at this square:
3.1.    multiply target mask by magic num and shift right
3.2.    check if this result is unique (array of bool)
3.3.    if result is not unique break
4.  if all results are unique, decrease magic shift by 1

Same for bishop
*/

bitboard_t ROOK_MAGIC_NUMS[64];
int ROOK_MAGIC_SHIFTS[64] = {[0 ... 63] = 44};
bitboard_t BISHOP_MAGIC_NUMS[64];
int BISHOP_MAGIC_SHIFTS[64] = {[0 ... 63] = 44};
bitboard_t encountered[1 << 20];
bitboard_t* ROOK_MOVES[64];
bitboard_t* BISHOP_MOVES[64];

size_t piece_magic_iteration(int square, int MAGIC_SHIFTS[64], bitboard_t MAGIC_NUMS[64],
                             bitboard_t (*bitboard_piece_mask)(int),
                             bitboard_t (*piece_move_bb)(bitboard_t, int),
                             bitboard_t* MAGIC_MOVES[64]) {
    int magic_shift = MAGIC_SHIFTS[square] + 1;
    bitboard_t bb = bitboard_piece_mask(square);
    int num_targets = 1 << bitboard_bit_count(bb);
    bitboard_t magic_num = random_uint64();

    // Moves array to determine uniqueness
    memset(encountered, 0, sizeof(encountered));
    bool unique = true;

    for (int i = 0; i < num_targets; i++) {
        bitboard_t target_mask = bitboard_target_mask(bb, i);
        int index = (target_mask * magic_num) >> magic_shift;
        bitboard_t moves;

        if (magic_shift == 45) {
            moves = piece_move_bb(target_mask, square);
        } else {
            int index = (target_mask * MAGIC_NUMS[square]) >> MAGIC_SHIFTS[square];
            moves = MAGIC_MOVES[square][index];
        }

        if (encountered[index] != 0 && encountered[index] != moves) {
            unique = false;
            break;
        }

        encountered[index] = moves;
    }

    // if we found a magic number that works return
    if (unique) {
        MAGIC_NUMS[square] = magic_num;
        MAGIC_SHIFTS[square]++;
        if (magic_shift > 45) free(MAGIC_MOVES[square]);
        MAGIC_MOVES[square] = malloc(sizeof(bitboard_t) * (1 << (64 - magic_shift)));

        for (int i = 0; i < num_targets; i++) {
            bitboard_t target_mask = bitboard_target_mask(bb, i);
            int index = (target_mask * MAGIC_NUMS[square]) >> MAGIC_SHIFTS[square];
            bitboard_t moves = piece_move_bb(target_mask, square);
            MAGIC_MOVES[square][index] = moves;
        }

        return sizeof(bitboard_t) * (1 << (64 - magic_shift));
    }

    return sizeof(bitboard_t) * (1 << (65 - magic_shift));
}

void piece_write_iteration(FILE* f, char* piece_name, int square, int MAGIC_SHIFTS[64],
                           bitboard_t MAGIC_NUMS[64], bitboard_t (*bitboard_piece_mask)(int),
                           bitboard_t (*piece_move_bb)(bitboard_t, int)) {
    int magic_shift = MAGIC_SHIFTS[square];
    bitboard_t magic_num = MAGIC_NUMS[square];
    bitboard_t bb = bitboard_piece_mask(square);
    int num_targets = 1 << bitboard_bit_count(bb);
    memset(encountered, 0, sizeof(encountered));

    fprintf(f, "const bitboard_t %s_MOVES_%d[%d] = {", piece_name, square, 1 << (64 - magic_shift));

    for (int i = 0; i < num_targets; i++) {
        bitboard_t target_mask = bitboard_target_mask(bb, i);
        int index = (target_mask * magic_num) >> magic_shift;

        if (!encountered[index]) {
            bitboard_t moves = piece_move_bb(target_mask, square);
            fprintf(f, "[%d]=0x%" PRIx64 "ULL,", index, moves);
            encountered[index] = moves;
        }
    }

    fprintf(f, "};\n");
}

size_t rook_magic_iteration(int square) {
    return piece_magic_iteration(square, ROOK_MAGIC_SHIFTS, ROOK_MAGIC_NUMS, bitboard_rook_mask,
                                 rook_move_bb, ROOK_MOVES);
}

size_t bishop_magic_iteration(int square) {
    return piece_magic_iteration(square, BISHOP_MAGIC_SHIFTS, BISHOP_MAGIC_NUMS,
                                 bitboard_bishop_mask, bishop_move_bb, BISHOP_MOVES);
}

void rook_write_iteration(FILE* f, int square) {
    return piece_write_iteration(f, "ROOK", square, ROOK_MAGIC_SHIFTS, ROOK_MAGIC_NUMS,
                                 bitboard_rook_mask, rook_move_bb);
}

void bishop_write_iteration(FILE* f, int square) {
    return piece_write_iteration(f, "BISHOP", square, BISHOP_MAGIC_SHIFTS, BISHOP_MAGIC_NUMS,
                                 bitboard_bishop_mask, bishop_move_bb);
}

#define WRITE64X(beginning, iteration, end)       \
    fprintf(f, beginning);                        \
    for (int square = 0; square < 64; square++) { \
        iteration;                                \
    }                                             \
    fprintf(f, end);

int main() {
    srand(time(NULL));
    size_t best_total_size_bishop = 0xFFFFFFFF;
    size_t best_total_size_rook = 0xFFFFFFFF;
    time_t last_improvement = time(NULL);

    while (1) {
        size_t total_size_bishop = 0;
        size_t total_size_rook = 0;
        for (int square = 0; square < 64; square++) {
            total_size_bishop += bishop_magic_iteration(square);
            total_size_rook += rook_magic_iteration(square);
        }

        if (total_size_bishop < best_total_size_bishop) {
            best_total_size_bishop = total_size_bishop;
            last_improvement = time(NULL);
            printf("Bishop: %10lu bytes, Rook: %10lu bytes\n", (unsigned long)total_size_bishop,
                   (unsigned long)total_size_rook);
        } else if (total_size_rook < best_total_size_rook) {
            best_total_size_rook = total_size_rook;
            last_improvement = time(NULL);
            printf("Bishop: %10lu bytes, Rook: %10lu bytes\n", (unsigned long)total_size_bishop,
                   (unsigned long)total_size_rook);
        }

        // Check if we've been stalling for more than 600 seconds
        if (difftime(time(NULL), last_improvement) > MAX_STALL) {
            printf("No improvement for %d seconds. Stopping.\n", MAX_STALL);
            break;
        }
    }

    // Write data to a C files (.c_no_format extension so that VS doesn't crash when opening it)
    FILE* f = fopen("magicbb/moves.c_no_format", "wt");
    if (!f) exit(1);

    fprintf(f, "#include <stdint.h>\n\n");
    fprintf(f, "typedef uint64_t bitboard_t;\n\n");

    for (int square = 0; square < 64; square++) {
        bishop_write_iteration(f, square);
        rook_write_iteration(f, square);
    }

    WRITE64X("const bitboard_t *ROOK_MOVES[64] = {", fprintf(f, "ROOK_MOVES_%d,", square), "};\n");

    WRITE64X("const bitboard_t *BISHOP_MOVES[64] = {", fprintf(f, "BISHOP_MOVES_%d,", square),
             "};\n");

    WRITE64X("const bitboard_t ROOK_MAGIC_NUMS[64] = {",
             fprintf(f, "0x%" PRIx64 "ULL,", ROOK_MAGIC_NUMS[square]), "};\n");

    WRITE64X("const int ROOK_MAGIC_SHIFTS[64] = {", fprintf(f, "%d,", ROOK_MAGIC_SHIFTS[square]),
             "};\n");

    WRITE64X("const bitboard_t BISHOP_MAGIC_NUMS[64] = {",
             fprintf(f, "0x%" PRIx64 "ULL,", BISHOP_MAGIC_NUMS[square]), "};\n");

    WRITE64X("const int BISHOP_MAGIC_SHIFTS[64] = {",
             fprintf(f, "%d,", BISHOP_MAGIC_SHIFTS[square]), "};\n");

    fclose(f);
}