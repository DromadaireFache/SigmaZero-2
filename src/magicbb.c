#include <stdint.h>
#include <stdio.h>

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
        if (target_idx & 1) {
            result |= (1ULL << bit_pos);
        }

        // Clear the least significant bit from bb
        bb &= bb - 1;

        // Move to next bit in target_idx
        target_idx >>= 1;
    }

    return result;
}

int main() {
    // bitboard_t bb = bitboard_rook_mask(0);
    // bitboard_print(bb);
    // printf("count: %d\n", bitboard_bit_count(bb));
    // bitboard_print(bitboard_target_mask(bb, 127));

    bitboard_print(bitboard_bishop_mask(28));
}