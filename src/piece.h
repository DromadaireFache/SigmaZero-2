#pragma once

#include <ctype.h>
#include <stdbool.h>
#include <stdlib.h>

#include "consts.h"

// All pieces
typedef enum __attribute__((__packed__)) {
    EMPTY = '.',
    WHITE_PAWN = 'P',
    BLACK_PAWN = 'p',
    WHITE_KNIGHT = 'N',
    BLACK_KNIGHT = 'n',
    WHITE_BISHOP = 'B',
    BLACK_BISHOP = 'b',
    WHITE_ROOK = 'R',
    BLACK_ROOK = 'r',
    WHITE_QUEEN = 'Q',
    BLACK_QUEEN = 'q',
    WHITE_KING = 'K',
    BLACK_KING = 'k',
} Piece;

static inline int Piece_victim_score(Piece piece) {
    switch (piece) {
        case WHITE_PAWN:
        case BLACK_PAWN:
            return PAWN_VICTIM_SCORE;
        case WHITE_KNIGHT:
        case BLACK_KNIGHT:
            return KNIGHT_VICTIM_SCORE;
        case WHITE_BISHOP:
        case BLACK_BISHOP:
            return BISHOP_VICTIM_SCORE;
        case WHITE_ROOK:
        case BLACK_ROOK:
            return ROOK_VICTIM_SCORE;
        case WHITE_QUEEN:
        case BLACK_QUEEN:
            return QUEEN_VICTIM_SCORE;
        case WHITE_KING:
        case BLACK_KING:
            return KING_VICTIM_SCORE;
        default:
            return 0;
    }
}

static inline int Piece_aggro_score(Piece piece) {
    switch (piece) {
        case WHITE_PAWN:
        case BLACK_PAWN:
            return PAWN_AGGRO_SCORE;
        case WHITE_KNIGHT:
        case BLACK_KNIGHT:
            return KNIGHT_AGGRO_SCORE;
        case WHITE_BISHOP:
        case BLACK_BISHOP:
            return BISHOP_AGGRO_SCORE;
        case WHITE_ROOK:
        case BLACK_ROOK:
            return ROOK_AGGRO_SCORE;
        case WHITE_QUEEN:
        case BLACK_QUEEN:
            return QUEEN_AGGRO_SCORE;
        case WHITE_KING:
        case BLACK_KING:
            return KING_AGGRO_SCORE;
        default:
            return 0;
    }
}

static inline int Piece_value(Piece piece) {
    switch (piece) {
        case WHITE_PAWN:
            return PAWN_VALUE;
        case BLACK_PAWN:
            return -PAWN_VALUE;
        case WHITE_KNIGHT:
            return KNIGHT_VALUE;
        case BLACK_KNIGHT:
            return -KNIGHT_VALUE;
        case WHITE_BISHOP:
            return BISHOP_VALUE;
        case BLACK_BISHOP:
            return -BISHOP_VALUE;
        case WHITE_ROOK:
            return ROOK_VALUE;
        case BLACK_ROOK:
            return -ROOK_VALUE;
        case WHITE_QUEEN:
            return QUEEN_VALUE;
        case BLACK_QUEEN:
            return -QUEEN_VALUE;
        case WHITE_KING:
            return KING_VALUE;
        case BLACK_KING:
            return -KING_VALUE;
        default:
            return 0;
    }
}

static inline int Piece_value_at(Piece piece, int i) {
    switch (piece) {
        case WHITE_PAWN:
            return PAWN_VALUE + PS_WHITE_PAWN[i];
        case BLACK_PAWN:
            return -PAWN_VALUE + PS_BLACK_PAWN[i];
        case WHITE_KNIGHT:
            return KNIGHT_VALUE + PS_WHITE_KNIGHT[i];
        case BLACK_KNIGHT:
            return -KNIGHT_VALUE + PS_BLACK_KNIGHT[i];
        case WHITE_BISHOP:
            return BISHOP_VALUE + PS_WHITE_BISHOP[i];
        case BLACK_BISHOP:
            return -BISHOP_VALUE + PS_BLACK_BISHOP[i];
        case WHITE_ROOK:
            return ROOK_VALUE + PS_WHITE_ROOK[i];
        case BLACK_ROOK:
            return -ROOK_VALUE + PS_BLACK_ROOK[i];
        case WHITE_QUEEN:
            return QUEEN_VALUE + PS_WHITE_QUEEN[i];
        case BLACK_QUEEN:
            return -QUEEN_VALUE + PS_BLACK_QUEEN[i];
        default:
            return 0;
    }
}

static inline uint64_t Piece_zhash_at(Piece piece, int i) {
    switch (piece) {
        case WHITE_PAWN:
            return ZHASH_WHITE_PAWN[i];
        case BLACK_PAWN:
            return ZHASH_BLACK_PAWN[i];
        case WHITE_KNIGHT:
            return ZHASH_WHITE_KNIGHT[i];
        case BLACK_KNIGHT:
            return ZHASH_BLACK_KNIGHT[i];
        case WHITE_BISHOP:
            return ZHASH_WHITE_BISHOP[i];
        case BLACK_BISHOP:
            return ZHASH_BLACK_BISHOP[i];
        case WHITE_ROOK:
            return ZHASH_WHITE_ROOK[i];
        case BLACK_ROOK:
            return ZHASH_BLACK_ROOK[i];
        case WHITE_QUEEN:
            return ZHASH_WHITE_QUEEN[i];
        case BLACK_QUEEN:
            return ZHASH_BLACK_QUEEN[i];
        case WHITE_KING:
            return ZHASH_WHITE_KING[i];
        case BLACK_KING:
            return ZHASH_BLACK_KING[i];
        default:
            return 0;
    }
}

static inline bool Piece_is_white(Piece piece) { return isupper(piece); }

static inline bool Piece_is_black(Piece piece) { return islower(piece); }

static inline bool Piece_is_pawn(Piece piece) { return piece == WHITE_PAWN || piece == BLACK_PAWN; }

static inline bool Piece_is_king(Piece piece) { return piece == WHITE_KING || piece == BLACK_KING; }

static inline bool Piece_is_queen(Piece piece) {
    return piece == WHITE_QUEEN || piece == BLACK_QUEEN;
}

static inline bool Piece_is_rook(Piece piece) { return piece == WHITE_ROOK || piece == BLACK_ROOK; }

static inline bool Piece_is_bishop(Piece piece) {
    return piece == WHITE_BISHOP || piece == BLACK_BISHOP;
}

static inline bool Piece_is_knight(Piece piece) {
    return piece == WHITE_KNIGHT || piece == BLACK_KNIGHT;
}

// Convert a character to its piece representation
static inline Piece Piece_from_char(char c) {
    switch (c) {
        case 'P':
            return WHITE_PAWN;
        case 'p':
            return BLACK_PAWN;
        case 'N':
            return WHITE_KNIGHT;
        case 'n':
            return BLACK_KNIGHT;
        case 'B':
            return WHITE_BISHOP;
        case 'b':
            return BLACK_BISHOP;
        case 'R':
            return WHITE_ROOK;
        case 'r':
            return BLACK_ROOK;
        case 'Q':
            return WHITE_QUEEN;
        case 'q':
            return BLACK_QUEEN;
        case 'K':
            return WHITE_KING;
        case 'k':
            return BLACK_KING;
        default:
            return EMPTY;
    }
}

static inline int Piece_index(Piece piece) {
    switch (piece) {
        case WHITE_PAWN:
            return 0;
        case BLACK_PAWN:
            return 1;
        case WHITE_KNIGHT:
            return 2;
        case BLACK_KNIGHT:
            return 3;
        case WHITE_BISHOP:
            return 4;
        case BLACK_BISHOP:
            return 5;
        case WHITE_ROOK:
            return 6;
        case BLACK_ROOK:
            return 7;
        case WHITE_QUEEN:
            return 8;
        case BLACK_QUEEN:
            return 9;
        case WHITE_KING:
            return 10;
        case BLACK_KING:
            return 11;
        default:
            fprintf(stderr, "Error in Piece_index\n");
            exit(69);
    }
}