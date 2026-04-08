#include <assert.h>

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

// TODO: move this into bitboards.h
const bitboard_t NEAR_KING_MASKS[64] = {
    0x0000000003070f0fULL, 0x00000000070f1f1fULL, 0x000000000e1f3f3fULL, 0x000000001c3e7f7fULL,
    0x00000000387cfefeULL, 0x0000000070f8fcfcULL, 0x00000000e0f0f8f8ULL, 0x00000000c0e0f0f0ULL,
    0x00000003070f0f0fULL, 0x000000070f1f1f1fULL, 0x0000000e1f3f3f3fULL, 0x0000001c3e7f7f7fULL,
    0x000000387cfefefeULL, 0x00000070f8fcfcfcULL, 0x000000e0f0f8f8f8ULL, 0x000000c0e0f0f0f0ULL,
    0x000003070f0f0f07ULL, 0x0000070f1f1f1f0fULL, 0x00000e1f3f3f3f1fULL, 0x00001c3e7f7f7f3eULL,
    0x0000387cfefefe7cULL, 0x000070f8fcfcfcf8ULL, 0x0000e0f0f8f8f8f0ULL, 0x0000c0e0f0f0f0e0ULL,
    0x0003070f0f0f0703ULL, 0x00070f1f1f1f0f07ULL, 0x000e1f3f3f3f1f0eULL, 0x001c3e7f7f7f3e1cULL,
    0x00387cfefefe7c38ULL, 0x0070f8fcfcfcf870ULL, 0x00e0f0f8f8f8f0e0ULL, 0x00c0e0f0f0f0e0c0ULL,
    0x03070f0f0f070300ULL, 0x070f1f1f1f0f0700ULL, 0x0e1f3f3f3f1f0e00ULL, 0x1c3e7f7f7f3e1c00ULL,
    0x387cfefefe7c3800ULL, 0x70f8fcfcfcf87000ULL, 0xe0f0f8f8f8f0e000ULL, 0xc0e0f0f0f0e0c000ULL,
    0x070f0f0f07030000ULL, 0x0f1f1f1f0f070000ULL, 0x1f3f3f3f1f0e0000ULL, 0x3e7f7f7f3e1c0000ULL,
    0x7cfefefe7c380000ULL, 0xf8fcfcfcf8700000ULL, 0xf0f8f8f8f0e00000ULL, 0xe0f0f0f0e0c00000ULL,
    0x0f0f0f0703000000ULL, 0x1f1f1f0f07000000ULL, 0x3f3f3f1f0e000000ULL, 0x7f7f7f3e1c000000ULL,
    0xfefefe7c38000000ULL, 0xfcfcfcf870000000ULL, 0xf8f8f8f0e0000000ULL, 0xf0f0f0e0c0000000ULL,
    0x0f0f070300000000ULL, 0x1f1f0f0700000000ULL, 0x3f3f1f0e00000000ULL, 0x7f7f3e1c00000000ULL,
    0xfefe7c3800000000ULL, 0xfcfcf87000000000ULL, 0xf8f8f0e000000000ULL, 0xf0f0e0c000000000ULL};

static inline int near_king_penalty(int king_i, bitboard_t friendly_queens,
                                    bitboard_t friendly_rooks, bitboard_t friendly_minor_pieces,
                                    bitboard_t enemy_queens, bitboard_t enemy_rooks,
                                    bitboard_t enemy_minor_pieces) {
    bitboard_t near_king_mask = NEAR_KING_MASKS[king_i];
    int penalty = 0;
    penalty += 4 * bitboard_popcount(near_king_mask & enemy_queens);
    penalty += 2 * bitboard_popcount(near_king_mask & enemy_rooks);
    penalty += bitboard_popcount(near_king_mask & enemy_minor_pieces);
    penalty -= 2 * bitboard_popcount(near_king_mask & friendly_queens);
    penalty -= bitboard_popcount(near_king_mask & friendly_rooks);
    penalty -= bitboard_popcount(near_king_mask & friendly_minor_pieces);
    return penalty;
}

int king_safety(Chess* chess, int endness) {
    if (endness >= FULLMOVES_ENDGAME) return 0;
    uint8_t fullmoves_min = endness;
    uint8_t fullmoves_max = FULLMOVES_ENDGAME - endness;
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

    e -= 30 * near_king_penalty(chess->king_white, chess->bb.white_queens, chess->bb.white_rooks,
                                chess->bb.white_bishops | chess->bb.white_knights,
                                chess->bb.black_queens, chess->bb.black_rooks,
                                chess->bb.black_bishops | chess->bb.black_knights);
    e += 30 * near_king_penalty(chess->king_black, chess->bb.black_queens, chess->bb.black_rooks,
                                chess->bb.black_bishops | chess->bb.black_knights,
                                chess->bb.white_queens, chess->bb.white_rooks,
                                chess->bb.white_bishops | chess->bb.white_knights);

    e = e * fullmoves_score / 64;
    return e;
}

void assert_consistency(Chess* chess) {
    bitboard_t bb_white_check = 0;
    bitboard_t bb_black_check = 0;
    BitboardMap bb_check = {0};
    int eval_check = 0, pawn_row_sum_check = 0;
    int white_king_i = -1, black_king_i = -1;

    for (int i = 0; i < 64; i++) {
        Piece piece = chess->board[i];
        bitboard_t bit = 1ULL << i;
        eval_check += Piece_value_at(piece, i);
        switch (piece) {
            case WHITE_PAWN:
                bb_check.white_pawns |= bit;
                bb_white_check |= bit;
                pawn_row_sum_check += index_row(i) - 1;
                break;
            case BLACK_PAWN:
                bb_check.black_pawns |= bit;
                bb_black_check |= bit;
                pawn_row_sum_check += index_row(i) - 6;
                break;
            case WHITE_KNIGHT:
                bb_check.white_knights |= bit;
                bb_white_check |= bit;
                break;
            case BLACK_KNIGHT:
                bb_check.black_knights |= bit;
                bb_black_check |= bit;
                break;
            case WHITE_BISHOP:
                bb_check.white_bishops |= bit;
                bb_white_check |= bit;
                break;
            case BLACK_BISHOP:
                bb_check.black_bishops |= bit;
                bb_black_check |= bit;
                break;
            case WHITE_ROOK:
                bb_check.white_rooks |= bit;
                bb_white_check |= bit;
                break;
            case BLACK_ROOK:
                bb_check.black_rooks |= bit;
                bb_black_check |= bit;
                break;
            case WHITE_QUEEN:
                bb_check.white_queens |= bit;
                bb_white_check |= bit;
                break;
            case BLACK_QUEEN:
                bb_check.black_queens |= bit;
                bb_black_check |= bit;
                break;
            case WHITE_KING:
                bb_check.white_kings |= bit;
                bb_white_check |= bit;
                white_king_i = i;
                break;
            case BLACK_KING:
                bb_check.black_kings |= bit;
                bb_black_check |= bit;
                black_king_i = i;
                break;
            default:
                break;
        }
    }

    assert(eval_check == chess->eval);
    assert(white_king_i == chess->king_white);
    assert(black_king_i == chess->king_black);
    assert(pawn_row_sum_check == chess->pawn_row_sum);
    assert(bb_white_check == chess->bb_white);
    assert(bb_black_check == chess->bb_black);
    assert(bb_check.white_pawns == chess->bb.white_pawns);
    assert(bb_check.black_pawns == chess->bb.black_pawns);
    assert(bb_check.white_knights == chess->bb.white_knights);
    assert(bb_check.black_knights == chess->bb.black_knights);
    assert(bb_check.white_bishops == chess->bb.white_bishops);
    assert(bb_check.black_bishops == chess->bb.black_bishops);
    assert(bb_check.white_rooks == chess->bb.white_rooks);
    assert(bb_check.black_rooks == chess->bb.black_rooks);
    assert(bb_check.white_queens == chess->bb.white_queens);
    assert(bb_check.black_queens == chess->bb.black_queens);
    assert(bb_check.white_kings == chess->bb.white_kings);
    assert(bb_check.black_kings == chess->bb.black_kings);
    assert(chess->bb_white ==
           (chess->bb.white_pawns | chess->bb.white_knights | chess->bb.white_bishops |
            chess->bb.white_rooks | chess->bb.white_queens | chess->bb.white_kings));
    assert(chess->bb_black ==
           (chess->bb.black_pawns | chess->bb.black_knights | chess->bb.black_bishops |
            chess->bb.black_rooks | chess->bb.black_queens | chess->bb.black_kings));
}

int bishop_pawn_penalty(bitboard_t bishops, bitboard_t pawns) {
    int penalty = 0;
    while (bishops) {
        int bishop_i = bitboard_pop_lsb(&bishops);
        bitboard_t mask = bishop_i % 2 == 0 ? DARK_SQUARES : LIGHT_SQUARES;
        penalty += bitboard_popcount(pawns & mask) * BISHOP_PAWN_PENALTY;
    }
    return penalty;
}

int eval(Chess* chess) {
    // npm is a measure from 0-24, 0 being kings/pawns and 24 is full board. This will now be used
    // to interpolate between opening and endgame positions instead of using fullmoves, but to be
    // compatible with have to "convert" to a fullmoves-based endgame score. An npm of 10 is
    // considered full endgame. Therefore, we convert npm to fullmoves (24-10) ->
    // (0-FULLMOVES_ENDGAME)
    int npm = Chess_non_pawn_material(chess);
    int endness = (24 - npm) * FULLMOVES_ENDGAME / (24 - 10);
    endness = endness > FULLMOVES_ENDGAME ? FULLMOVES_ENDGAME : endness;
    int e = chess->eval;

    // Pawn rank bonus
    e += chess->pawn_row_sum * endness / PAWN_RANK_BONUS;

    // King square value
    int white_king_value = PS_WHITE_KING[chess->king_white] * (FULLMOVES_ENDGAME - endness);
    white_king_value += PS_WHITE_KING_ENDGAME[chess->king_white] * endness;
    e += white_king_value / FULLMOVES_ENDGAME;

    int black_king_value = PS_BLACK_KING[chess->king_black] * (FULLMOVES_ENDGAME - endness);
    black_king_value += PS_BLACK_KING_ENDGAME[chess->king_black] * endness;
    e += black_king_value / FULLMOVES_ENDGAME;

    // Add bonus if king has castled
    e += chess->white_has_castled * CASTLE_BONUS;
    e -= chess->black_has_castled * CASTLE_BONUS;

    e += king_safety(chess, endness);

    // Light/dark square bonuses
    if (bitboard_popcount(chess->bb.white_bishops) == 1)  // No penalty for bishop pair
        e -= bishop_pawn_penalty(chess->bb.white_bishops, chess->bb.white_pawns);
    if (bitboard_popcount(chess->bb.black_bishops) == 1)
        e += bishop_pawn_penalty(chess->bb.black_bishops, chess->bb.black_pawns);

    // Check that the bitboards and board array are consistent
#ifdef DEBUG
    assert_consistency(chess);
#endif

    return e;
}