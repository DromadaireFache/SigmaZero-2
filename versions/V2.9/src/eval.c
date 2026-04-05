#include "chess.h"
#include "movegen.h"

int king_mobility(Chess* chess, bool is_white, int king_i, bool only_attacks) {
    bitboard_t friendly_bb = is_white ? chess->bb_white : chess->bb_black;
    bitboard_t enemy_bb = is_white ? chess->bb_black : chess->bb_white;
    bitboard_t all_bb = chess->bb_white | chess->bb_black;

    // Rook moves
    bitboard_t piece_mask = bitboard_rook_mask(king_i);
    bitboard_t target_mask = piece_mask & all_bb;
    int index = (target_mask * ROOK_MAGIC_NUMS[king_i]) >> ROOK_MAGIC_SHIFTS[king_i];
    bitboard_t moves = ROOK_MOVES[king_i][index];

    // Bishop moves
    piece_mask = bitboard_bishop_mask(king_i);
    target_mask = piece_mask & all_bb;
    index = (target_mask * BISHOP_MAGIC_NUMS[king_i]) >> BISHOP_MAGIC_SHIFTS[king_i];
    moves |= BISHOP_MOVES[king_i][index];
    moves &= ~friendly_bb;

    bitboard_t attacks = moves & enemy_bb;
    if (only_attacks) return KING_SAFETY_FACTOR1 * __builtin_popcountll(attacks);
    return KING_SAFETY_FACTOR2 * __builtin_popcountll(moves) +
           KING_SAFETY_FACTOR3 * __builtin_popcountll(attacks);
}

int king_safety(Chess* chess) {
    if (chess->fullmoves >= FULLMOVES_ENDGAME) return 0;
    uint8_t fullmoves_min = chess->fullmoves;
    uint8_t fullmoves_max = FULLMOVES_ENDGAME - chess->fullmoves;
    uint8_t fullmoves_score = fullmoves_min < fullmoves_max ? fullmoves_min : fullmoves_max;
    uint8_t king_white_col = index_col(chess->king_white);
    uint8_t king_black_col = index_col(chess->king_black);
    int e = 0;

    // at position
    e -= king_mobility(chess, true, chess->king_white, false);
    e += king_mobility(chess, false, chess->king_black, false);

    // left
    if (king_white_col > 0 && chess->board[chess->king_white - 1] == EMPTY)
        e -= king_mobility(chess, true, chess->king_white - 1, true);
    if (king_black_col > 0 && chess->board[chess->king_black - 1] == EMPTY)
        e += king_mobility(chess, false, chess->king_black - 1, true);

    // right
    if (king_white_col < 7 && chess->board[chess->king_white + 1] == EMPTY)
        e -= king_mobility(chess, true, chess->king_white + 1, true);
    if (king_black_col < 7 && chess->board[chess->king_black + 1] == EMPTY)
        e += king_mobility(chess, false, chess->king_black + 1, true);

    e = e * fullmoves_score / 64;
    return e;
}

int eval(Chess* chess) {
    uint8_t fullmoves = chess->fullmoves > FULLMOVES_ENDGAME ? FULLMOVES_ENDGAME : chess->fullmoves;
    int e = chess->eval;

    // Pawn rank bonus
    e += chess->pawn_row_sum * fullmoves / PAWN_RANK_BONUS;

    // King square value
    int white_king_value = PS_WHITE_KING[chess->king_white] * (FULLMOVES_ENDGAME - fullmoves);
    white_king_value += PS_WHITE_KING_ENDGAME[chess->king_white] * fullmoves;
    e += white_king_value / FULLMOVES_ENDGAME;

    int black_king_value = PS_BLACK_KING[chess->king_black] * (FULLMOVES_ENDGAME - fullmoves);
    black_king_value += PS_BLACK_KING_ENDGAME[chess->king_black] * fullmoves;
    e += black_king_value / FULLMOVES_ENDGAME;

    // Add bonus if king has castled
    e += chess->white_has_castled * CASTLE_BONUS;
    e -= chess->black_has_castled * CASTLE_BONUS;

    e += king_safety(chess);
    return e;
}