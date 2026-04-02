#pragma once
#include "chess.h"

extern const bitboard_t ROOK_MAGIC_NUMS[64];
extern const int ROOK_MAGIC_SHIFTS[64];
extern const bitboard_t BISHOP_MAGIC_NUMS[64];
extern const int BISHOP_MAGIC_SHIFTS[64];
extern const bitboard_t* ROOK_MOVES[64];
extern const bitboard_t* BISHOP_MOVES[64];

size_t Chess_legal_moves_scored(Chess* chess, Move* moves, int* scores, bool captures_only);
size_t Chess_legal_moves(Chess* chess, Move* moves, bool captures_only);
bool Chess_friendly_check(Chess* chess);