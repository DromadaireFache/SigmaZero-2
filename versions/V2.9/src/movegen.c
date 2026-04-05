#include "movegen.h"

#include <stdlib.h>

void Chess_fill_attack_map(Chess* chess) {
    EnemyAttackMap* eam = &chess->enemy_attack_map;
    eam->n_checks = 0;

    uint8_t king_i = Chess_friendly_king_i(chess);
    Position king_pos = Position_from_index(king_i);
    bitboard_t occupied = chess->bb_white | chess->bb_black;

#define ENEMY_ATTACK(condition1, condition2, attack_map)              \
    if ((condition1) && (condition2)) {                               \
        eam->n_checks++;                                              \
        if (eam->n_checks == 1) eam->block_attack_map = (attack_map); \
        /* return if 2 checks, because only king can move! */         \
        if (eam->n_checks >= 2) return;                               \
    }

// Look for pawn attacks
#define PAWN_ATTACK(condition, offset)                                     \
    ENEMY_ATTACK(condition, Chess_enemy_pawn_at(chess, king_i + (offset)), \
                 bitboard_from_index(king_i + (offset)))
    if (chess->turn == TURN_WHITE) {
        PAWN_ATTACK(king_pos.row < 7 && king_pos.col < 7, 9)
        PAWN_ATTACK(king_pos.row < 7 && king_pos.col > 0, 7)
    } else {
        PAWN_ATTACK(king_pos.row > 0 && king_pos.col > 0, -9)
        PAWN_ATTACK(king_pos.row > 0 && king_pos.col < 7, -7)
    }

    // TODO FIX THIS WITH BITBOARDS
// Look for knight attacks
#define KNIGHT_ATTACK(condition, offset)                                     \
    ENEMY_ATTACK(condition, Chess_enemy_knight_at(chess, king_i + (offset)), \
                 bitboard_from_index(king_i + (offset)))
    if (chess->turn == TURN_WHITE) {
        KNIGHT_ATTACK(king_pos.row < 7 && king_pos.col > 1, 6)
        KNIGHT_ATTACK(king_pos.row < 7 && king_pos.col < 6, 10)
        KNIGHT_ATTACK(king_pos.row < 6 && king_pos.col > 0, 15)
        KNIGHT_ATTACK(king_pos.row < 6 && king_pos.col < 7, 17)
        KNIGHT_ATTACK(king_pos.row > 1 && king_pos.col > 0, -17)
        KNIGHT_ATTACK(king_pos.row > 1 && king_pos.col < 7, -15)
        KNIGHT_ATTACK(king_pos.row > 0 && king_pos.col > 1, -10)
        KNIGHT_ATTACK(king_pos.row > 0 && king_pos.col < 6, -6)
    } else {
        KNIGHT_ATTACK(king_pos.row > 1 && king_pos.col > 0, -17)
        KNIGHT_ATTACK(king_pos.row > 1 && king_pos.col < 7, -15)
        KNIGHT_ATTACK(king_pos.row > 0 && king_pos.col > 1, -10)
        KNIGHT_ATTACK(king_pos.row > 0 && king_pos.col < 6, -6)
        KNIGHT_ATTACK(king_pos.row < 7 && king_pos.col > 1, 6)
        KNIGHT_ATTACK(king_pos.row < 7 && king_pos.col < 6, 10)
        KNIGHT_ATTACK(king_pos.row < 6 && king_pos.col > 0, 15)
        KNIGHT_ATTACK(king_pos.row < 6 && king_pos.col < 7, 17)
    }

    // Look for king attacks
    uint8_t enemy_king_i = Chess_enemy_king_i(chess);
    Position enemy_king_pos = Position_from_index(enemy_king_i);
    ENEMY_ATTACK(abs(enemy_king_pos.row - king_pos.row) <= 1,
                 abs(enemy_king_pos.col - king_pos.col) <= 1, 0)

    int i;
    bitboard_t attack_map;
    bool found_pinned_piece;
    uint8_t pinned_piece = -1;

    // Reset the pinned piece map
    eam->pinned_piece_map = 0;

    // TODO: fix this with bitboards
#define SLIDING_PIECE_ATTACK(fn, condition, offset)                            \
    attack_map = 0;                                                            \
    found_pinned_piece = false;                                                \
    for (i = 0; (condition); i++) {                                            \
        int square = king_i + (offset);                                        \
        bitboard_t square_bit = bitboard_from_index(square);                   \
        attack_map |= square_bit;                                              \
        if (Chess_friendly_piece_at(chess, square)) {                          \
            if (found_pinned_piece) {                                          \
                break; /* two pieces stack so pin possible*/                   \
            } else {                                                           \
                /* found a friendly piece, will keep looking for a pin */      \
                found_pinned_piece = true;                                     \
                pinned_piece = square;                                         \
            }                                                                  \
        } else if (fn(chess, square) || Chess_enemy_queen_at(chess, square)) { \
            if (found_pinned_piece) {                                          \
                /* found an enemy behind friendly, so piece pinned */          \
                eam->valid_map[pinned_piece] = attack_map;                     \
                eam->pinned_piece_map |= bitboard_from_index(pinned_piece);    \
            } else {                                                           \
                /* found an enemy without a pin, so it's a check */            \
                eam->n_checks++;                                               \
                if (eam->n_checks == 1) eam->block_attack_map = attack_map;    \
                if (eam->n_checks >= 2) return;                                \
                break;                                                         \
            }                                                                  \
        } else if (occupied & square_bit)                                      \
            break;                                                             \
    }
#define kp king_pos

// Look for bishop/queen attacks
#define BISHOP_ATTACK(condition, offset) \
    SLIDING_PIECE_ATTACK(Chess_enemy_bishop_at, condition, offset)
    if (chess->turn == TURN_WHITE) {
        BISHOP_ATTACK(kp.col - i > 0 && kp.row + i < 7, (i + 1) * 7)
        BISHOP_ATTACK(kp.col + i < 7 && kp.row + i < 7, (i + 1) * 9)
        BISHOP_ATTACK(kp.col - i > 0 && kp.row - i > 0, (i + 1) * -9)
        BISHOP_ATTACK(kp.col + i < 7 && kp.row - i > 0, (i + 1) * -7)
    } else {
        BISHOP_ATTACK(kp.col - i > 0 && kp.row - i > 0, (i + 1) * -9)
        BISHOP_ATTACK(kp.col + i < 7 && kp.row - i > 0, (i + 1) * -7)
        BISHOP_ATTACK(kp.col - i > 0 && kp.row + i < 7, (i + 1) * 7)
        BISHOP_ATTACK(kp.col + i < 7 && kp.row + i < 7, (i + 1) * 9)
    }

// Look for rook/queen attacks
#define ROOK_ATTACK(condition, offset) SLIDING_PIECE_ATTACK(Chess_enemy_rook_at, condition, offset)
    if (chess->turn == TURN_WHITE) {
        ROOK_ATTACK(king_pos.row + i < 7, (i + 1) * 8)
        ROOK_ATTACK(king_pos.col + i < 7, (i + 1))
        ROOK_ATTACK(king_pos.col - i > 0, (i + 1) * -1)
        ROOK_ATTACK(king_pos.row - i > 0, (i + 1) * -8)
    } else {
        ROOK_ATTACK(king_pos.row - i > 0, (i + 1) * -8)
        ROOK_ATTACK(king_pos.col + i < 7, (i + 1))
        ROOK_ATTACK(king_pos.col - i > 0, (i + 1) * -1)
        ROOK_ATTACK(king_pos.row + i < 7, (i + 1) * 8)
    }
}

Piece Chess_set_friendly_king_i(Chess* chess, uint8_t index) {
    if (chess->turn == TURN_WHITE) {
        chess->king_white = index;
        return WHITE_KING;
    } else {
        chess->king_black = index;
        return BLACK_KING;
    }
}

bool Chess_friendly_check(Chess* chess) {
    // Find the king
    uint8_t king_i = Chess_friendly_king_i(chess);
    Position king_pos = Position_from_index(king_i);

#define ENEMY_CHECK(condition1, condition2) \
    if ((condition1) && (condition2)) return true;

    // Look for pawn attacks
#define PAWN_CHECK(condition, offset) \
    ENEMY_CHECK(condition, Chess_enemy_pawn_at(chess, king_i + (offset)))
    if (chess->turn == TURN_WHITE) {
        PAWN_CHECK(king_pos.row < 7 && king_pos.col < 7, 9)
        PAWN_CHECK(king_pos.row < 7 && king_pos.col > 0, 7)
    } else {
        PAWN_CHECK(king_pos.row > 0 && king_pos.col > 0, -9)
        PAWN_CHECK(king_pos.row > 0 && king_pos.col < 7, -7)
    }

    // Look for knight attacks
#define KNIGHT_CHECK(condition, offset) \
    ENEMY_CHECK(condition, Chess_enemy_knight_at(chess, king_i + (offset)))
    if (chess->turn == TURN_WHITE) {
        KNIGHT_CHECK(king_pos.row < 7 && king_pos.col > 1, 6)
        KNIGHT_CHECK(king_pos.row < 7 && king_pos.col < 6, 10)
        KNIGHT_CHECK(king_pos.row < 6 && king_pos.col > 0, 15)
        KNIGHT_CHECK(king_pos.row < 6 && king_pos.col < 7, 17)
        KNIGHT_CHECK(king_pos.row > 1 && king_pos.col > 0, -17)
        KNIGHT_CHECK(king_pos.row > 1 && king_pos.col < 7, -15)
        KNIGHT_CHECK(king_pos.row > 0 && king_pos.col > 1, -10)
        KNIGHT_CHECK(king_pos.row > 0 && king_pos.col < 6, -6)
    } else {
        KNIGHT_CHECK(king_pos.row > 1 && king_pos.col > 0, -17)
        KNIGHT_CHECK(king_pos.row > 1 && king_pos.col < 7, -15)
        KNIGHT_CHECK(king_pos.row > 0 && king_pos.col > 1, -10)
        KNIGHT_CHECK(king_pos.row > 0 && king_pos.col < 6, -6)
        KNIGHT_CHECK(king_pos.row < 7 && king_pos.col > 1, 6)
        KNIGHT_CHECK(king_pos.row < 7 && king_pos.col < 6, 10)
        KNIGHT_CHECK(king_pos.row < 6 && king_pos.col > 0, 15)
        KNIGHT_CHECK(king_pos.row < 6 && king_pos.col < 7, 17)
    }

    // Look for king attacks
    uint8_t enemy_king_i = Chess_enemy_king_i(chess);
    Position enemy_king_pos = Position_from_index(enemy_king_i);
    if (abs(enemy_king_pos.row - king_pos.row) <= 1 &&
        abs(enemy_king_pos.col - king_pos.col) <= 1) {
        return true;
    }

    int i;
#define SLIDING_PIECE_CHECK(fn, condition, offset)                                            \
    for (i = 0; (condition); i++) {                                                           \
        if (fn(chess, king_i + (offset)) || Chess_enemy_queen_at(chess, king_i + (offset))) { \
            return true;                                                                      \
        }                                                                                     \
        if (chess->board[king_i + (offset)] != EMPTY) break;                                  \
    }
#define kp king_pos

    // Look for bishop/queen attacks
#define BISHOP_CHECK(condition, offset) \
    SLIDING_PIECE_CHECK(Chess_enemy_bishop_at, condition, offset)
    if (chess->turn == TURN_WHITE) {
        BISHOP_CHECK(kp.col - i > 0 && kp.row + i < 7, (i + 1) * 7)
        BISHOP_CHECK(kp.col + i < 7 && kp.row + i < 7, (i + 1) * 9)
        BISHOP_CHECK(kp.col - i > 0 && kp.row - i > 0, (i + 1) * -9)
        BISHOP_CHECK(kp.col + i < 7 && kp.row - i > 0, (i + 1) * -7)
    } else {
        BISHOP_CHECK(kp.col - i > 0 && kp.row - i > 0, (i + 1) * -9)
        BISHOP_CHECK(kp.col + i < 7 && kp.row - i > 0, (i + 1) * -7)
        BISHOP_CHECK(kp.col - i > 0 && kp.row + i < 7, (i + 1) * 7)
        BISHOP_CHECK(kp.col + i < 7 && kp.row + i < 7, (i + 1) * 9)
    }

    // Look for rook/queen attacks
#define ROOK_CHECK(condition, offset) SLIDING_PIECE_CHECK(Chess_enemy_rook_at, condition, offset)
    if (chess->turn == TURN_WHITE) {
        ROOK_CHECK(king_pos.row + i < 7, (i + 1) * 8)
        ROOK_CHECK(king_pos.col + i < 7, (i + 1))
        ROOK_CHECK(king_pos.col - i > 0, (i + 1) * -1)
        ROOK_CHECK(king_pos.row - i > 0, (i + 1) * -8)
    } else {
        ROOK_CHECK(king_pos.row - i > 0, (i + 1) * -8)
        ROOK_CHECK(king_pos.col + i < 7, (i + 1))
        ROOK_CHECK(king_pos.col - i > 0, (i + 1) * -1)
        ROOK_CHECK(king_pos.row + i < 7, (i + 1) * 8)
    }

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

// Move *move, size_t n_moves NEED to be defined within scope
#define ADD_MOVE_IF(condition, offset)                                \
    if (condition) {                                                  \
        move->from = from;                                            \
        move->to = from + (offset);                                   \
        move->promotion = NO_PROMOTION;                               \
        if (Chess_square_available(chess, move->to, captures_only) && \
            Chess_is_move_legal(chess, move)) {                       \
            move++;                                                   \
            n_moves++;                                                \
        }                                                             \
    }

size_t Chess_knight_moves(Chess* chess, Move* move, int from, bool captures_only) {
    size_t n_moves = 0;
    Position pos = Position_from_index(from);

    ADD_MOVE_IF(pos.row < 6 && pos.col < 7, 17)   // 2 up 1 right
    ADD_MOVE_IF(pos.row > 1 && pos.col > 0, -17)  // 2 down 1 left
    ADD_MOVE_IF(pos.row < 6 && pos.col > 0, 15)   // 2 up 1 left
    ADD_MOVE_IF(pos.row > 1 && pos.col < 7, -15)  // 2 down 1 right
    ADD_MOVE_IF(pos.row < 7 && pos.col < 6, 10)   // 1 up 2 right
    ADD_MOVE_IF(pos.row > 0 && pos.col > 1, -10)  // 1 down 2 left
    ADD_MOVE_IF(pos.row < 7 && pos.col > 1, 6)    // 1 up 2 left
    ADD_MOVE_IF(pos.row > 0 && pos.col < 6, -6)   // 1 down 2 right

    return n_moves;
}

__attribute__((always_inline)) static inline size_t  //
Chess_sliding_piece_moves(Chess* chess, Move* move, int from, bool captures_only, bool is_bishop,
                          const bitboard_t MAGIC_NUMS[64], const int MAGIC_SHIFTS[64],
                          const bitboard_t* MOVES[64]) {
    EnemyAttackMap* eam = &chess->enemy_attack_map;

    bitboard_t piece_mask = is_bishop ? bitboard_bishop_mask(from) : bitboard_rook_mask(from);
    bitboard_t friendly_bb = chess->turn == TURN_WHITE ? chess->bb_white : chess->bb_black;
    bitboard_t all_bb = chess->bb_white | chess->bb_black;
    bitboard_t target_mask = piece_mask & all_bb;
    int index = (target_mask * MAGIC_NUMS[from]) >> MAGIC_SHIFTS[from];
    bitboard_t moves = MOVES[from][index] & ~friendly_bb;
    if (captures_only) moves &= chess->turn == TURN_WHITE ? chess->bb_black : chess->bb_white;

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
    int e = chess->eval;                          \
    int pawn_row_sum = chess->pawn_row_sum;       \
    bitboard_t bb_white = chess->bb_white;        \
    bitboard_t bb_black = chess->bb_black;        \
    Piece capture = Chess_make_move(chess, move); \
    chess->turn = !chess->turn;                   \
    bool in_check = Chess_friendly_check(chess);  \
    chess->turn = !chess->turn;                   \
    Chess_unmake_move(chess, move, capture);      \
    chess->gamestate = gamestate;                 \
    chess->zhash = hash;                          \
    chess->eval = e;                              \
    chess->pawn_row_sum = pawn_row_sum;           \
    chess->bb_white = bb_white;                   \
    chess->bb_black = bb_black;                   \
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

size_t Chess_king_moves(Chess* chess, Move* move, int from, bool captures_only) {
    Position pos = Position_from_index(from);
    size_t n_moves = 0;
    bool left_safe = false;
    bool right_safe = false;

    ADD_MOVE_IF(pos.row > 0 && pos.col > 0, -9)
    ADD_MOVE_IF(pos.row > 0 && pos.col < 7, -7)
    ADD_MOVE_IF(pos.row < 7 && pos.col > 0, 7)
    ADD_MOVE_IF(pos.row < 7 && pos.col < 7, 9)
    ADD_MOVE_IF(pos.row > 0, -8)
    ADD_MOVE_IF(pos.row < 7, 8)
    if (pos.col > 0) {  // left
        move->from = from;
        move->to = from - 1;
        move->promotion = NO_PROMOTION;
        if (Chess_square_available(chess, move->to, captures_only) &&
            Chess_is_move_legal(chess, move)) {
            left_safe = true;
            move++;
            n_moves++;
        }
    }
    if (pos.col < 7) {  // right
        move->from = from;
        move->to = from + 1;
        move->promotion = NO_PROMOTION;
        if (Chess_square_available(chess, move->to, captures_only) &&
            Chess_is_move_legal(chess, move)) {
            right_safe = true;
            move++;
            n_moves++;
        }
    }
    if (captures_only) return n_moves;

#define ADD_KING_MOVE_IF(offset)                           \
    if (!Chess_friendly_check(chess)) {                    \
        Chess_set_friendly_king_i(chess, from + (offset)); \
        if (!Chess_friendly_check(chess)) {                \
            move->from = from;                             \
            move->to = from + (offset);                    \
            move->promotion = NO_PROMOTION;                \
            move++;                                        \
            n_moves++;                                     \
        }                                                  \
        Chess_set_friendly_king_i(chess, from);            \
    }

    // Castling
#define ADD_CASTLE_MOVE(k1, k2, q1, q2, q3)                                         \
    /* King side castling */                                                        \
    if (right_safe && Chess_castle_king_side(chess) && chess->board[k1] == EMPTY && \
        chess->board[k2] == EMPTY) {                                                \
        ADD_KING_MOVE_IF(2)                                                         \
    } /* Queen side castling */                                                     \
    if (left_safe && Chess_castle_queen_side(chess) && chess->board[q1] == EMPTY && \
        chess->board[q2] == EMPTY && chess->board[q3] == EMPTY) {                   \
        ADD_KING_MOVE_IF(-2)                                                        \
    }

    bool king_in_check = chess->enemy_attack_map.n_checks > 0;
    if (!king_in_check) {
        if (chess->turn == TURN_WHITE) {
            ADD_CASTLE_MOVE(5, 6, 1, 2, 3)
        } else {
            ADD_CASTLE_MOVE(61, 62, 57, 58, 59)
        }
    }

    return n_moves;
}


typedef size_t (*MoveFn)(Chess*, Move*, int, bool);

// Lookup table for move generation functions
static const MoveFn move_fns[256] = {
    [WHITE_PAWN] = Chess_pawn_moves,     [WHITE_KNIGHT] = Chess_knight_moves,
    [WHITE_BISHOP] = Chess_bishop_moves, [WHITE_ROOK] = Chess_rook_moves,
    [WHITE_QUEEN] = Chess_queen_moves,   [WHITE_KING] = Chess_king_moves,
    [BLACK_PAWN] = Chess_pawn_moves,     [BLACK_KNIGHT] = Chess_knight_moves,
    [BLACK_BISHOP] = Chess_bishop_moves, [BLACK_ROOK] = Chess_rook_moves,
    [BLACK_QUEEN] = Chess_queen_moves,   [BLACK_KING] = Chess_king_moves,
};

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

    // Remove king from bitboard before iterating
    bitboard_t friendly_bb = chess->turn == TURN_WHITE ? chess->bb_white : chess->bb_black;
    friendly_bb &= ~bitboard_from_index(king_i);

    // Iterate over friendly pieces using bitboard
    while (friendly_bb) {
        int i = __builtin_ctzll(friendly_bb);  // Get the index
        friendly_bb &= friendly_bb - 1;        // Clear the least significant bit

        Piece piece = chess->board[i];

        // A piece is represented by its character in the board array
        // Grab the move generation function from the lookup table
        // and generate moves for that piece
        n_moves += move_fns[piece](chess, &moves[n_moves], i, captures_only);
    }
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
