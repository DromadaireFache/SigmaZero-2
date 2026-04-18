#pragma once
#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <ctype.h>

#include "bitboard.h"

// A position on the chessboard (from (0,0) to (7,7))
// (Position){row, col}
typedef struct {
    uint8_t row;
    uint8_t col;
} Position;

// Check if a position is valid (on the board)
static inline bool Position_valid(Position* pos) {
    return 0 <= pos->col && pos->col < 8 && 0 <= pos->row && pos->row < 8;
}

// Convert a string (e.g. "e4") to a position
// Returns (Position){-1, -1} if invalid
static inline Position Position_from_string(char* s) {
    if (strlen(s) != 2) {
        fprintf(stderr, "Invalid position string: %s\n", s);
        return (Position){-1, -1};
    }

    char col = s[0];
    char row = s[1];

    // Convert to 0-7 range
    col = tolower(col) - 'a';
    row = row - '1';

    if (col < 0 || col >= 8 || row < 0 || row >= 8) {
        fprintf(stderr, "Invalid position string: %s\n", s);
        return (Position){-1, -1};
    }

    return (Position){row, col};
}

// Convert a bitboard with only one bit set to a position
// Returns (Position){-1, -1} if invalid
static inline Position Position_from_bitboard(bitboard_t b) {
    if (__builtin_popcountll(b) != 1) {
        return (Position){-1, -1};
    }
    for (int i = 0; i < 64; i++) {
        if ((b >> i) & 1) {
            return (Position){i / 8, i % 8};
        }
    }
    return (Position){-1, -1};
}

// Convert a position to a string (e.g. "e4")
// Returns a pointer to a static buffer
static inline char* Position_to_string(Position* pos) {
    static char buffer[3];
    buffer[0] = 'a' + pos->col;
    buffer[1] = '1' + pos->row;
    buffer[2] = 0;
    return buffer;
}

// Convert a position to an index (0-63)
static inline int Position_to_index(Position pos) { return pos.row * 8 + pos.col; }

static inline Position Position_from_index(int index) {
    return (Position){.col = index_col(index), .row = index_row(index)};
}

// Print a position
static inline void Position_print(Position pos) {
    printf("Position: %s (row: %d, col: %d)\n", Position_to_string(&pos), pos.row, pos.col);
}
