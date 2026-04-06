#include "movegen.h"

#include <stdlib.h>

// Returns true if double check, otherwise fills the enemy attack map with check and pin information
static inline bool sliding_piece_attack_map(Chess* chess, bool is_bishop,
                                            const bitboard_t MAGIC_NUMS[64],
                                            const int MAGIC_SHIFTS[64],
                                            const bitboard_t* MOVES[64]) {
    // New approach using bitboards for bishop/queen attacks
    bitboard_t friendly_bb = Chess_friendly_bb(chess);
    int king_i = Chess_friendly_king_i(chess);
    EnemyAttackMap* eam = &chess->enemy_attack_map;

    // Get the piece mask for the bishop moves from the king
    // X-ray through friendly pieces to find potential pins
    bitboard_t enemies = (is_bishop ? Chess_enemy_bishops_bb(chess) : Chess_enemy_rooks_bb(chess)) |
                         Chess_enemy_queens_bb(chess);
    if (!enemies) return false;  // no potential attackers, so skip
    bitboard_t piece_mask = is_bishop ? bitboard_bishop_mask(king_i) : bitboard_rook_mask(king_i);
    bitboard_t target_mask = piece_mask & Chess_enemy_bb(chess);
    int index = (target_mask * MAGIC_NUMS[king_i]) >> MAGIC_SHIFTS[king_i];
    bitboard_t bishop_xray = MOVES[king_i][index];

    for (int i = 0; i < 4; i++) {
        // Get the X-ray in this quadrant
        bitboard_t ray = QUADRANTS[i][king_i] & bishop_xray;

        // Get the bitboard of a potential attacker in this quadrant
        bitboard_t attacker = ray & enemies;
        if (!attacker) continue;  // no attacker in this quadrant, so skip

        // If there's an attacker, check the number of pieces between the king and the attacker to
        // determine if it's a check (0 pieces), a pin (1 piece), or no threat (2+ pieces)
        bitboard_t blockers = ray & friendly_bb;
        int n_blockers = bitboard_popcount(blockers);
        if (n_blockers == 0) {  // check
            eam->n_checks++;
            if (eam->n_checks == 1) eam->block_attack_map = ray;
            if (eam->n_checks >= 2) return true;  // double check, so return immediately
        } else if (n_blockers == 1) {             // pin
            uint8_t pinned_piece = __builtin_ctzll(blockers);
            eam->valid_map[pinned_piece] = ray;
            eam->pinned_piece_map |= bitboard_from_index(pinned_piece);
        }
    }
    return false;  // not double check
}

void Chess_fill_attack_map(Chess* chess) {
    EnemyAttackMap* eam = &chess->enemy_attack_map;
    eam->n_checks = 0;

    uint8_t king_i = Chess_friendly_king_i(chess);

    // Look for pawn attacks
    bitboard_t pawn_attacks = PAWN_CAPTURES[chess->turn][king_i] & Chess_enemy_pawns_bb(chess);
    if (pawn_attacks) {
        eam->n_checks += bitboard_popcount(pawn_attacks);
        if (eam->n_checks == 1) eam->block_attack_map = pawn_attacks;
        if (eam->n_checks >= 2) return;
    }

    // Look for knight attacks
    bitboard_t knight_attacks = KNIGHT_MOVES[king_i] & Chess_enemy_knights_bb(chess);
    if (knight_attacks) {
        eam->n_checks += __builtin_popcountll(knight_attacks);
        if (eam->n_checks == 1) eam->block_attack_map = knight_attacks;
        if (eam->n_checks >= 2) return;
    }

    // Reset the pinned piece map
    eam->pinned_piece_map = 0;

    // Look for bishop/queen attacks
    bool double_check =
        sliding_piece_attack_map(chess, true, BISHOP_MAGIC_NUMS, BISHOP_MAGIC_SHIFTS, BISHOP_MOVES);
    if (double_check) return;

    // Look for rook/queen attacks
    double_check =
        sliding_piece_attack_map(chess, false, ROOK_MAGIC_NUMS, ROOK_MAGIC_SHIFTS, ROOK_MOVES);
    if (double_check) return;
}

Piece Chess_set_friendly_king_i(Chess* chess, uint8_t index) {
    if (chess->turn == TURN_WHITE) {
        chess->king_white = index;
        chess->bb_white &= ~chess->bb.white_kings;  // remove old king from bitboard
        chess->bb.white_kings = bitboard_from_index(index);
        chess->bb_white |= chess->bb.white_kings;  // add new king to bitboard
        return WHITE_KING;
    } else {
        chess->king_black = index;
        chess->bb_black &= ~chess->bb.black_kings;  // remove old king from bitboard
        chess->bb.black_kings = bitboard_from_index(index);
        chess->bb_black |= chess->bb.black_kings;  // add new king to bitboard
        return BLACK_KING;
    }
}

bool Chess_friendly_check(Chess* chess) {
    uint8_t king_i = Chess_friendly_king_i(chess);

    // Look for pawn attacks
    if (PAWN_CAPTURES[chess->turn][king_i] & Chess_enemy_pawns_bb(chess)) return true;

    // Look for knight attacks
    if (KNIGHT_MOVES[king_i] & Chess_enemy_knights_bb(chess)) return true;

    // Look for king attacks (not really necessary since we don't expect illegal moves)
    // uint8_t enemy_king_i = Chess_enemy_king_i(chess);
    // Position enemy_king_pos = Position_from_index(enemy_king_i);
    // if (abs(enemy_king_pos.row - king_pos.row) <= 1 &&
    //     abs(enemy_king_pos.col - king_pos.col) <= 1) {
    //     return true;
    // }

    // Look for bishop/queen attacks
    bitboard_t bishop_moves = Chess_magic_moves_bb(chess, king_i, true, BISHOP_MAGIC_NUMS,
                                                   BISHOP_MAGIC_SHIFTS, BISHOP_MOVES);
    if (bishop_moves & (Chess_enemy_bishops_bb(chess) | Chess_enemy_queens_bb(chess))) return true;

    // Look for rook/queen attacks
    bitboard_t rook_moves =
        Chess_magic_moves_bb(chess, king_i, false, ROOK_MAGIC_NUMS, ROOK_MAGIC_SHIFTS, ROOK_MOVES);
    if (rook_moves & (Chess_enemy_rooks_bb(chess) | Chess_enemy_queens_bb(chess))) return true;

    return false;
}

// This will check if the king is in check after the move
bool Chess_is_move_legal(Chess* chess, Move* move) {
    EnemyAttackMap* eam = &chess->enemy_attack_map;
    uint8_t king_i = Chess_friendly_king_i(chess);
    bitboard_t to_bb = bitboard_from_index(move->to);
    bool is_king = move->from == king_i;

    if (is_king) {
        bool still_attacked = to_bb & eam->block_attack_map;
        bool captured_attacker = to_bb == eam->block_attack_map;
        if (eam->n_checks >= 1 && still_attacked && !captured_attacker) return false;

        // Check if the king is in check after its move
        Chess_set_friendly_king_i(chess, move->to);
        chess->board[move->from] = EMPTY;
        bool in_check = Chess_friendly_check(chess);
        chess->board[move->from] = Chess_set_friendly_king_i(chess, move->from);
        if (in_check) return false;

    } else {
        // no checks: pieces don't have to move to protect king
        // but they will have to confirm they are not pinned
        bitboard_t from_bb = bitboard_from_index(move->from);
        bool is_pinned = from_bb & eam->pinned_piece_map;
        if (is_pinned) {
            // if pinned and king is in check, can't move to block the attack
            if (eam->n_checks == 1) return false;

            // if pinned, limit movement to stay pinned
            bool still_pinned = to_bb & eam->valid_map[move->from];
            if (!still_pinned) return false;
        } else {
            // if it's not pinned and there's no check, can move freely
            if (eam->n_checks == 0) return true;

            // single check: has to block the attack with the piece
            bool blocking_attack = to_bb & eam->block_attack_map;
            if (eam->n_checks == 1 && !blocking_attack) return false;
        }
    }

    return true;
}

// This will mask out moves that are illegal due to checks or pins, but won't check for legality
// of the move itself
static inline bitboard_t Chess_legal_moves_mask(Chess* chess, int from, bitboard_t moves) {
    EnemyAttackMap* eam = &chess->enemy_attack_map;
    bitboard_t from_bb = bitboard_from_index(from);
    bool is_pinned = from_bb & eam->pinned_piece_map;

    if (is_pinned) {
        // if pinned and king is in check, can't move to block the attack
        if (eam->n_checks == 1) moves = 0;
        // if pinned, limit movement to stay pinned
        moves &= eam->valid_map[from];
    } else if (eam->n_checks == 1) {
        // single check: has to block the attack with the piece
        moves &= eam->block_attack_map;
    }
    return moves;
}

// Converts a bitboard of moves into Move structs, returns number of moves added
static inline size_t moves_from_bb(Move* move, int from, bitboard_t moves) {
    size_t n_moves = __builtin_popcountll(moves);
    while (moves) {
        move->from = from;
        move->to = __builtin_ctzll(moves);  // Get the index
        moves &= moves - 1;                 // Clear the least significant bit
        move->promotion = NO_PROMOTION;
        move++;
    }
    return n_moves;
}

size_t Chess_knight_moves(Chess* chess, Move* move, int from, bool captures_only) {
    // Get pseudo-legal moves from table
    bitboard_t moves = KNIGHT_MOVES[from];

    // Remove friendly captures
    moves &= ~(Chess_friendly_bb(chess));

    // If only looking for captures, remove non-captures
    if (captures_only) moves &= Chess_enemy_bb(chess);

    // Mask out illegal moves that don't resolve checks or pins
    moves = Chess_legal_moves_mask(chess, from, moves);

    // Convert bitboard to move list
    return moves_from_bb(move, from, moves);
}

static inline size_t Chess_sliding_piece_moves(Chess* chess, Move* move, int from,
                                               bool captures_only, bool is_bishop,
                                               const bitboard_t MAGIC_NUMS[64],
                                               const int MAGIC_SHIFTS[64],
                                               const bitboard_t* MOVES[64]) {
    // Get pseudo-legal moves from magic bitboard
    bitboard_t moves =
        Chess_magic_moves_bb(chess, from, is_bishop, MAGIC_NUMS, MAGIC_SHIFTS, MOVES);

    // If only looking for captures, remove non-captures
    if (captures_only) moves &= Chess_enemy_bb(chess);

    // Mask out illegal moves that don't resolve checks or pins
    moves = Chess_legal_moves_mask(chess, from, moves);

    // Convert bitboard to move list
    return moves_from_bb(move, from, moves);
}

size_t Chess_bishop_moves(Chess* chess, Move* move, int from, bool captures_only) {
    return Chess_sliding_piece_moves(chess, move, from, captures_only, true, BISHOP_MAGIC_NUMS,
                                     BISHOP_MAGIC_SHIFTS, BISHOP_MOVES);
}

size_t Chess_rook_moves(Chess* chess, Move* move, int from, bool captures_only) {
    return Chess_sliding_piece_moves(chess, move, from, captures_only, false, ROOK_MAGIC_NUMS,
                                     ROOK_MAGIC_SHIFTS, ROOK_MOVES);
}

size_t Chess_queen_moves(Chess* chess, Move* move, int from, bool captures_only) {
    size_t n_moves = Chess_rook_moves(chess, move, from, captures_only);
    return n_moves + Chess_bishop_moves(chess, move + n_moves, from, captures_only);
}

// TODO: optimize this with bitboards like the others
size_t Chess_pawn_moves(Chess* chess, Move* move, int from, bool captures_only) {
    Position pos = Position_from_index(from);
    size_t n_moves = 0;
    int one_forward, left_capture, right_capture;
    bool at_home_rank, at_last_rank, at_en_passant_rank;

    if (chess->turn == TURN_WHITE) {
        one_forward = from + 8, left_capture = from + 7, right_capture = from + 9;
        at_home_rank = pos.row == 1, at_last_rank = pos.row == 6, at_en_passant_rank = pos.row == 4;
    } else {
        one_forward = from - 8, left_capture = from - 9, right_capture = from - 7;
        at_home_rank = pos.row == 6, at_last_rank = pos.row == 1, at_en_passant_rank = pos.row == 3;
    }

#define PAWN_ADD_MOVE_PROMOTE(to_square)    \
    move->from = from;                      \
    move->to = (to_square);                 \
    if (Chess_is_move_legal(chess, move)) { \
        move->promotion = PROMOTE_QUEEN;    \
        move++;                             \
        n_moves++;                          \
        move->from = from;                  \
        move->to = (to_square);             \
        move->promotion = PROMOTE_ROOK;     \
        move++;                             \
        n_moves++;                          \
        move->from = from;                  \
        move->to = (to_square);             \
        move->promotion = PROMOTE_KNIGHT;   \
        move++;                             \
        n_moves++;                          \
        move->from = from;                  \
        move->to = (to_square);             \
        move->promotion = PROMOTE_BISHOP;   \
        move++;                             \
        n_moves++;                          \
    }

#define PAWN_ADD_MOVE(to_square)            \
    move->from = from;                      \
    move->to = (to_square);                 \
    if (Chess_is_move_legal(chess, move)) { \
        move->promotion = NO_PROMOTION;     \
        move++;                             \
        n_moves++;                          \
    }

    // 1 row up
    if (chess->board[one_forward] == EMPTY && !captures_only) {
        if (at_last_rank) {
            PAWN_ADD_MOVE_PROMOTE(one_forward)
        } else {
            PAWN_ADD_MOVE(one_forward)

            // 2 rows up
            int two_forward = chess->turn == TURN_WHITE ? from + 16 : from - 16;
            if (at_home_rank && chess->board[two_forward] == EMPTY) {
                PAWN_ADD_MOVE(two_forward)
            }
        }
    }

    // normal captures
    if (pos.col > 0 && Chess_enemy_piece_at(chess, left_capture)) {
        if (at_last_rank) {
            PAWN_ADD_MOVE_PROMOTE(left_capture)
        } else {
            PAWN_ADD_MOVE(left_capture)
        }
    }
    if (pos.col < 7 && Chess_enemy_piece_at(chess, right_capture)) {
        if (at_last_rank) {
            PAWN_ADD_MOVE_PROMOTE(right_capture)
        } else {
            PAWN_ADD_MOVE(right_capture)
        }
    }

#define PAWN_EN_PASSANT(to_square)                \
    move->from = from;                            \
    move->to = (to_square);                       \
    move->promotion = NO_PROMOTION;               \
    gamestate_t gamestate = chess->gamestate;     \
    uint64_t hash = chess->zhash;                 \
    Piece capture = Chess_make_move(chess, move); \
    chess->turn = !chess->turn;                   \
    bool in_check = Chess_friendly_check(chess);  \
    chess->turn = !chess->turn;                   \
    Chess_unmake_move(chess, move, capture);      \
    chess->gamestate = gamestate;                 \
    chess->zhash = hash;                          \
    if (!in_check) {                              \
        move++;                                   \
        n_moves++;                                \
    }

    // en passant capture
    uint8_t en_passant_col = Chess_en_passant(chess);
    if (at_en_passant_rank && en_passant_col != NO_ENPASSANT) {
        if (en_passant_col == pos.col - 1) {
            PAWN_EN_PASSANT(left_capture)
        } else if (en_passant_col == pos.col + 1) {
            PAWN_EN_PASSANT(right_capture)
        }
    }

    return n_moves;
}

// New approach using bitboards
size_t Chess_king_moves(Chess* chess, Move* move, int from, bool captures_only) {
    int king_i = Chess_friendly_king_i(chess);
    bitboard_t moves = KING_MOVES[king_i];  // Get pseudo-legal moves from table
    bitboard_t castles = 0;                 // Hold potential castling moves
    bitboard_t friendly_bb = Chess_friendly_bb(chess);
    bitboard_t enemy_bb = Chess_enemy_bb(chess);
    bitboard_t all_bb = friendly_bb | enemy_bb;

    // Add castling moves if available
    int n_checks = chess->enemy_attack_map.n_checks;
    if (n_checks == 0) {
        bitboard_t kingside_empty = king_i == 4 ? 0x0000000000000060ULL : 0x6000000000000000ULL;
        bitboard_t queenside_empty = king_i == 4 ? 0x000000000000000EULL : 0x0E00000000000000ULL;
        if (Chess_castle_king_side(chess) && !(all_bb & kingside_empty))
            castles |= king_i == 4 ? 0x0000000000000040ULL : 0x4000000000000000ULL;
        if (Chess_castle_queen_side(chess) && !(all_bb & queenside_empty))
            castles |= king_i == 4 ? 0x0000000000000004ULL : 0x0400000000000000ULL;
    }

    /* First, filter out moves with the pre-filled attack map */
    // Remove squares that are occupied by friendly pieces
    bitboard_t illegal_squares = friendly_bb;
    if (n_checks > 0) {
        // A king can't block it's own attack, so remove squares of the attack blocking bitboard
        // Except if those squares are occupied by enemy pieces, in which case the king can capture
        illegal_squares |= chess->enemy_attack_map.block_attack_map & ~enemy_bb;
    }
    // Remove illegal squares from moves
    moves &= ~illegal_squares;
    castles &= ~illegal_squares;
    // Add legal castling moves back in
    moves |= castles;

    // Remove king from bitboards to avoid self-attacks
    if (chess->turn == TURN_WHITE) {
        chess->bb_white &= ~chess->bb.white_kings;
    } else {
        chess->bb_black &= ~chess->bb.black_kings;
    }

    // Filter for non-sliding pieces (king, pawns, and knights)

    // Knight attacks
    bitboard_t knight_attacks = 0;
    bitboard_t remaining_squares = Chess_enemy_knights_bb(chess);
    while (remaining_squares) knight_attacks |= KNIGHT_MOVES[bitboard_pop_lsb(&remaining_squares)];
    moves &= ~knight_attacks;

    // King attacks
    bitboard_t king_attacks = KING_MOVES[Chess_enemy_king_i(chess)];
    moves &= ~king_attacks;

    // Pawn attacks
    bitboard_t enemy_pawns = Chess_enemy_pawns_bb(chess);
    bitboard_t pawn_danger;
    if (chess->turn == TURN_WHITE) {
        // Enemy is black, black pawns attack downward
        pawn_danger = ((enemy_pawns >> 7) & ~0x0101010101010101ULL)     // attack left
                      | ((enemy_pawns >> 9) & ~0x8080808080808080ULL);  // attack right
    } else {
        // Enemy is white, white pawns attack upward
        pawn_danger = ((enemy_pawns << 9) & ~0x0101010101010101ULL)     // attack right
                      | ((enemy_pawns << 7) & ~0x8080808080808080ULL);  // attack left
    }
    moves &= ~pawn_danger;

    /* Secondly, filter move squares that are attacked by sliding pieces using magic bitboards */
    remaining_squares = moves;
    bitboard_t enemy_bishops = Chess_enemy_bishops_bb(chess) | Chess_enemy_queens_bb(chess);
    bitboard_t enemy_rooks = Chess_enemy_rooks_bb(chess) | Chess_enemy_queens_bb(chess);
    // If there are no sliding attackers, we can skip this step
    if (!enemy_bishops && !enemy_rooks) remaining_squares = 0;
    while (remaining_squares) {
        int i = bitboard_pop_lsb(&remaining_squares);  // Get the index of the lsb and clear it
        bitboard_t move_bb = bitboard_from_index(i);   // Bitboard of the move square
        bitboard_t sector1 = QUADRANTS[0][i] | QUADRANTS[2][i];  // Up-right diagonal and colmun
        bitboard_t sector2 = QUADRANTS[1][i] | QUADRANTS[3][i];  // Up-left diagonal and row

        // Filter for bishop/queen attacks
        if (enemy_bishops) {
            bitboard_t bishop_moves = Chess_magic_moves_bb(chess, i, true, BISHOP_MAGIC_NUMS,
                                                           BISHOP_MAGIC_SHIFTS, BISHOP_MOVES);
            bitboard_t bishop_attackers = bishop_moves & enemy_bishops;

            // Bishop/queen attack on the up-right diagonal
            if (bishop_attackers & sector1) {
                if (enemy_bb & move_bb) {
                    // If there's an attacker on the move square, it might be blocking attacks on
                    // other squares so only remove this square
                    moves &= ~move_bb;
                } else {
                    // Otherwise, remove all squares in this sector, since the king can't move there
                    // without being attacked by the bishop/queen
                    moves &= ~(bishop_moves & sector1 & ~enemy_bishops) & ~move_bb;
                }
            }

            // Bishop/queen attack on the up-left diagonal
            if (bishop_attackers & sector2) {
                if (enemy_bb & move_bb) {  // Same logic as above
                    moves &= ~move_bb;
                } else {
                    moves &= ~(bishop_moves & sector2 & ~enemy_bishops) & ~move_bb;
                }
            }

            if (bishop_attackers) {
                // update remaining squares after filtering bishop attacks
                remaining_squares &= moves;
                // skip rook checks since the square is already illegal
                continue;
            }
        }

        // Filter for rook/queen attacks
        if (!enemy_rooks) continue;  // skip rook checks if no enemy rooks/queens
        bitboard_t rook_moves =
            Chess_magic_moves_bb(chess, i, false, ROOK_MAGIC_NUMS, ROOK_MAGIC_SHIFTS, ROOK_MOVES);
        bitboard_t rook_attackers = rook_moves & enemy_rooks;

        // Rook/queen attack on the column
        if (rook_attackers & sector1) {
            if (enemy_bb & move_bb) {  // Same logic as above
                moves &= ~move_bb;
            } else {
                moves &= ~(rook_moves & sector1 & ~enemy_rooks) & ~move_bb;
            }
        }

        // Rook/queen attack on the row
        if (rook_attackers & sector2) {
            if (enemy_bb & move_bb) {  // Same logic as above
                moves &= ~move_bb;
            } else {
                moves &= ~(rook_moves & sector2 & ~enemy_rooks) & ~move_bb;
            }
        }

        remaining_squares &= moves;  // update remaining squares after filtering rook attacks
    }

    // Remove castling moves if the king would be in check in the intermediate square
    if (king_i == 4) {
        // if f1 is attacked, remove g1
        if (!(moves & 0x0000000000000020ULL)) moves &= ~0x0000000000000040ULL;
        // if d1 is attacked, remove c1
        if (!(moves & 0x0000000000000008ULL)) moves &= ~0x0000000000000004ULL;
    } else if (king_i == 60) {
        // if f8 is attacked, remove g8
        if (!(moves & 0x2000000000000000ULL)) moves &= ~0x4000000000000000ULL;
        // if d8 is attacked, remove c8
        if (!(moves & 0x0800000000000000ULL)) moves &= ~0x0400000000000000ULL;
    }

    // Restore king bitboards
    if (chess->turn == TURN_WHITE) {
        chess->bb_white |= chess->bb.white_kings;
    } else {
        chess->bb_black |= chess->bb.black_kings;
    }

    // Filter out non-captures if only looking for captures
    if (captures_only) moves &= enemy_bb;

    // Convert bitboard to move list
    return moves_from_bb(move, from, moves);
}

size_t Chess_legal_moves(Chess* chess, Move* moves, bool captures_only) {
    // make the enemy attack map to check legality
    Chess_fill_attack_map(chess);
    size_t n_moves = 0;

    // If double check, only consider king moves
    if (chess->enemy_attack_map.n_checks >= 2) {
        int i = Chess_friendly_king_i(chess);
        return Chess_king_moves(chess, moves, i, captures_only);
    }

    // Process king first since there is always a king
    int king_i = Chess_friendly_king_i(chess);
    n_moves += Chess_king_moves(chess, &moves[n_moves], king_i, captures_only);

    // Iterate over pieces by type using bitboards
    bitboard_t bb;
#define MOVES_FOR_PIECE(piece_type)                                                      \
    bb = Chess_friendly_##piece_type##s_bb(chess);                                       \
    while (bb) {                                                                         \
        int i = __builtin_ctzll(bb); /* Get the index */                                 \
        bb &= bb - 1;                /* Clear the least significant bit */               \
        n_moves += Chess_##piece_type##_moves(chess, &moves[n_moves], i, captures_only); \
    }

    MOVES_FOR_PIECE(pawn);
    MOVES_FOR_PIECE(knight);
    MOVES_FOR_PIECE(bishop);
    MOVES_FOR_PIECE(rook);
    MOVES_FOR_PIECE(queen);

    return n_moves;
}

void Chess_score_move(Chess* chess, Move* move, int* score) {
    // Give very high scores to promotions
    if (move->promotion == PROMOTE_QUEEN) {
        *score = PROMOTION_MOVE_SCORE;
        return;
    }

    Piece aggressor = chess->board[move->from];
    Piece victim = chess->board[move->to];

    // MVV - LVA
    if (victim != EMPTY) {
        *score = Piece_victim_score(victim) - Piece_aggro_score(aggressor);
    } else {
        // +2.12% improvement in first_move_cutoff_%
        *score = (abs(Piece_value_at(aggressor, move->to)) -
                  abs(Piece_value_at(aggressor, move->from))) /
                 4;
    }
}

size_t Chess_legal_moves_scored(Chess* chess, Move* moves, int* scores, bool captures_only) {
    size_t n_moves = Chess_legal_moves(chess, moves, captures_only);

    // Give a score to each move
    for (int i = 0; i < n_moves; i++) {
        Chess_score_move(chess, &moves[i], &scores[i]);
    }

    return n_moves;
}
