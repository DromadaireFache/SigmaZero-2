#pragma once
#include "chess.h"

extern const bitboard_t ROOK_MAGIC_NUMS[64];
extern const int ROOK_MAGIC_SHIFTS[64];
extern const bitboard_t BISHOP_MAGIC_NUMS[64];
extern const int BISHOP_MAGIC_SHIFTS[64];
extern const bitboard_t* ROOK_MOVES[64];
extern const bitboard_t* BISHOP_MOVES[64];

// Change this to have one function for bishop and one for rook
static inline bitboard_t Chess_magic_moves_bb(Chess* chess, int from, bool is_bishop,
                                              const bitboard_t MAGIC_NUMS[64],
                                              const int MAGIC_SHIFTS[64],
                                              const bitboard_t* MOVES[64]) {
    bitboard_t friendly_bb = Chess_friendly_bb(chess);
    bitboard_t all_bb = chess->bb_white | chess->bb_black;

    // Get piece mask from table (all pseudo-legal moves, exluding moves to the edge of board)
    bitboard_t piece_mask = is_bishop ? bitboard_bishop_mask(from) : bitboard_rook_mask(from);

    // Get target mask (blockers in the path of the sliding piece)
    bitboard_t target_mask = piece_mask & all_bb;

    // Get move index from magic bitboard multiplication
    int index = (target_mask * MAGIC_NUMS[from]) >> MAGIC_SHIFTS[from];

    // Get pseudo-legal moves from table
    bitboard_t moves = MOVES[from][index] & ~friendly_bb;

    return moves;
}

size_t Chess_legal_moves_scored(Chess* chess, Move* moves, int* scores, bool captures_only);
size_t Chess_legal_moves(Chess* chess, Move* moves, bool captures_only);
bool Chess_friendly_check(Chess* chess);

void Chess_fill_attack_map(Chess* chess);