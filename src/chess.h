#pragma once
#include <inttypes.h>
#include <stdbool.h>

#include "move.h"
#include "piece.h"

#define TURN_BLACK true
#define TURN_WHITE false
typedef bool turn_t;

/*
bit 1: white castling kingside, 0 if canCastleed
bit 2: white castling queenside, 0 if canCastleed
bit 3: black castling kingside, 0 if canCastleed
bit 4: black castling queenside, 0 if canCastleed
bit 5: 0 if no en passant, 1 if en passant is available
bit 6: next 3 bits are for the en-passant column
bit 7: ...
bit 8: ...
*/
typedef uint8_t gamestate_t;

#define Z_HASH_STACK_SIZE 1024
typedef struct {
    uint64_t hashes[Z_HASH_STACK_SIZE];
    int sp;
} ZHashStack;

static inline void ZHashStack_push(ZHashStack* zhstack, uint64_t hash) {
    zhstack->hashes[zhstack->sp++] = hash;
}

static inline uint64_t ZHashStack_pop(ZHashStack* zhstack) {
    return zhstack->hashes[--zhstack->sp];
}

static inline uint64_t ZHashStack_peek(ZHashStack* zhstack) {
    return zhstack->hashes[zhstack->sp - 1];
}

typedef struct {
    // 0: king not in check
    // 1: in check
    // 2: double check
    int n_checks;
    // n_checks=1: tells pieces where to move to protect the king
    bitboard_t block_attack_map;
    // xor this with the location of a piece to check if it's pinned
    bitboard_t pinned_piece_map;
    // once you know a piece is pinned, check this to find legal moves map
    bitboard_t valid_map[64];
} EnemyAttackMap;

typedef struct {
    bitboard_t white_pawns, black_pawns, white_knights, black_knights, white_bishops, black_bishops,
        white_rooks, black_rooks, white_queens, black_queens, white_kings, black_kings;
} BitboardMap;

// The chessboard
typedef struct {
    Piece board[64];        // Array of pieces, index 0 is a1, index 63 is h8
    turn_t turn;            // true for black, false for white
    gamestate_t gamestate;  // Game state bits
    uint8_t halfmoves;      // Halfmoves since last capture or pawn move
    uint8_t fullmoves;      // Full moves (incremented after every move)
    uint8_t king_white;     // Position of white king
    uint8_t king_black;     // Position of black king
    ZHashStack zhstack;     // Stack to store zobrist hash of previous positions
    uint64_t zhash;         // Current zobrist hash of the position
    EnemyAttackMap enemy_attack_map;
    int eval;                  // Cache a partial value of the eval that doesn't depend on fullmoves
    int pawn_row_sum;          // Sum pawn rows to use in final eval calculation
    bitboard_t bb_white;       // Bitboard of all white pieces
    bitboard_t bb_black;       // Bitboard of all black pieces
    Move killer_moves[2][64];  // Used for move ordering [id][depth]
    bool white_has_castled;
    bool black_has_castled;
    BitboardMap bb;  // Bitboards to store the pieces
} Chess;

#define BITMASK(nbit) (1 << (nbit))

// Set white kingside castling right
static inline void Chess_castle_K_set(Chess* board, bool canCastle) {
    if (canCastle)
        board->gamestate &= ~BITMASK(0);
    else
        board->gamestate |= BITMASK(0);
}

// Set white queenside castling right
static inline void Chess_castle_Q_set(Chess* board, bool canCastle) {
    if (canCastle)
        board->gamestate &= ~BITMASK(1);
    else
        board->gamestate |= BITMASK(1);
}

// Set black kingside castling right
static inline void Chess_castle_k_set(Chess* board, bool canCastle) {
    if (canCastle)
        board->gamestate &= ~BITMASK(2);
    else
        board->gamestate |= BITMASK(2);
}

// Set black queenside castling right
static inline void Chess_castle_q_set(Chess* board, bool canCastle) {
    if (canCastle)
        board->gamestate &= ~BITMASK(3);
    else
        board->gamestate |= BITMASK(3);
}

// Get kingside castling right
static inline bool Chess_castle_king_side(Chess* chess) {
    if (chess->turn == TURN_WHITE) {
        return !(chess->gamestate & BITMASK(0));
    } else {
        return !(chess->gamestate & BITMASK(2));
    }
}

// Get queenside castling right
static inline bool Chess_castle_queen_side(Chess* chess) {
    if (chess->turn == TURN_WHITE) {
        return !(chess->gamestate & BITMASK(1));
    } else {
        return !(chess->gamestate & BITMASK(3));
    }
}

// Set en passant column (0-7) or disable (-1)
static inline void Chess_en_passant_set(Chess* board, uint8_t col) {
    if (0 <= col && col < 8) {
        board->gamestate |= BITMASK(4);
        board->gamestate &= 0b00011111;
        board->gamestate |= col << 5;
    } else {
        board->gamestate &= 0b11101111;
    }
}

#define NO_ENPASSANT 255

// Get en passant column (or NO_ENPASSANT if not available)
static inline uint8_t Chess_en_passant(Chess* chess) {
    if (chess->gamestate & BITMASK(4)) {
        return chess->gamestate >> 5;
    } else {
        return NO_ENPASSANT;
    }
}

static inline bool Chess_friendly_piece_at(Chess* chess, int index) {
    return (chess->turn == TURN_WHITE ? Piece_is_white(chess->board[index])
                                      : Piece_is_black(chess->board[index]));
}

static inline bool Chess_enemy_piece_at(Chess* chess, int index) {
    return (chess->turn == TURN_WHITE ? Piece_is_black(chess->board[index])
                                      : Piece_is_white(chess->board[index]));
}

static inline bool Chess_friendly_king_at(Chess* chess, int index) {
    Piece piece = chess->board[index];
    if (chess->turn == TURN_WHITE) {
        return piece == WHITE_KING;
    } else {
        return piece == BLACK_KING;
    }
}

static inline bool Chess_enemy_king_at(Chess* chess, int index) {
    Piece piece = chess->board[index];
    if (chess->turn == TURN_WHITE) {
        return piece == BLACK_KING;
    } else {
        return piece == WHITE_KING;
    }
}

static inline bool Chess_friendly_pawn_at(Chess* chess, int index) {
    Piece piece = chess->board[index];
    if (chess->turn == TURN_WHITE) {
        return piece == WHITE_PAWN;
    } else {
        return piece == BLACK_PAWN;
    }
}

static inline bool Chess_enemy_pawn_at(Chess* chess, int index) {
    Piece piece = chess->board[index];
    if (chess->turn == TURN_WHITE) {
        return piece == BLACK_PAWN;
    } else {
        return piece == WHITE_PAWN;
    }
}

static inline bool Chess_friendly_knight_at(Chess* chess, int index) {
    Piece piece = chess->board[index];
    if (chess->turn == TURN_WHITE) {
        return piece == WHITE_KNIGHT;
    } else {
        return piece == BLACK_KNIGHT;
    }
}

static inline bool Chess_enemy_knight_at(Chess* chess, int index) {
    Piece piece = chess->board[index];
    if (chess->turn == TURN_WHITE) {
        return piece == BLACK_KNIGHT;
    } else {
        return piece == WHITE_KNIGHT;
    }
}

static inline bool Chess_friendly_bishop_at(Chess* chess, int index) {
    Piece piece = chess->board[index];
    if (chess->turn == TURN_WHITE) {
        return piece == WHITE_BISHOP;
    } else {
        return piece == BLACK_BISHOP;
    }
}

static inline bool Chess_enemy_bishop_at(Chess* chess, int index) {
    Piece piece = chess->board[index];
    if (chess->turn == TURN_WHITE) {
        return piece == BLACK_BISHOP;
    } else {
        return piece == WHITE_BISHOP;
    }
}

static inline bool Chess_friendly_rook_at(Chess* chess, int index) {
    Piece piece = chess->board[index];
    if (chess->turn == TURN_WHITE) {
        return piece == WHITE_ROOK;
    } else {
        return piece == BLACK_ROOK;
    }
}

static inline bool Chess_enemy_rook_at(Chess* chess, int index) {
    Piece piece = chess->board[index];
    if (chess->turn == TURN_WHITE) {
        return piece == BLACK_ROOK;
    } else {
        return piece == WHITE_ROOK;
    }
}

static inline bool Chess_friendly_queen_at(Chess* chess, int index) {
    Piece piece = chess->board[index];
    if (chess->turn == TURN_WHITE) {
        return piece == WHITE_QUEEN;
    } else {
        return piece == BLACK_QUEEN;
    }
}

static inline bool Chess_enemy_queen_at(Chess* chess, int index) {
    Piece piece = chess->board[index];
    if (chess->turn == TURN_WHITE) {
        return piece == BLACK_QUEEN;
    } else {
        return piece == WHITE_QUEEN;
    }
}

static inline uint8_t Chess_friendly_king_i(Chess* chess) {
    return chess->turn == TURN_WHITE ? chess->king_white : chess->king_black;
}

static inline uint8_t Chess_enemy_king_i(Chess* chess) {
    return chess->turn == TURN_WHITE ? chess->king_black : chess->king_white;
}

// Remove a piece from the board at a given position (updates board, eval, zhash, and bitboards
// accordingly)
static inline void Chess_remove(Chess* chess, int i) {
    Piece piece = chess->board[i];
    chess->eval -= Piece_value_at(piece, i);
    chess->zhash ^= Piece_zhash_at(piece, i);
    chess->board[i] = EMPTY;
    bitboard_t bb = bitboard_from_index(i);
    switch (piece) {
        case WHITE_PAWN:
            chess->bb_white &= ~bb;
            chess->bb.white_pawns &= ~bb;
            chess->pawn_row_sum -= index_row(i) - 1;
            break;
        case BLACK_PAWN:
            chess->bb_black &= ~bb;
            chess->bb.black_pawns &= ~bb;
            chess->pawn_row_sum -= index_row(i) - 6;
            break;
        case WHITE_KNIGHT:
            chess->bb_white &= ~bb;
            chess->bb.white_knights &= ~bb;
            break;
        case BLACK_KNIGHT:
            chess->bb_black &= ~bb;
            chess->bb.black_knights &= ~bb;
            break;
        case WHITE_BISHOP:
            chess->bb_white &= ~bb;
            chess->bb.white_bishops &= ~bb;
            break;
        case BLACK_BISHOP:
            chess->bb_black &= ~bb;
            chess->bb.black_bishops &= ~bb;
            break;
        case WHITE_ROOK:
            chess->bb_white &= ~bb;
            chess->bb.white_rooks &= ~bb;
            break;
        case BLACK_ROOK:
            chess->bb_black &= ~bb;
            chess->bb.black_rooks &= ~bb;
            break;
        case WHITE_QUEEN:
            chess->bb_white &= ~bb;
            chess->bb.white_queens &= ~bb;
            break;
        case BLACK_QUEEN:
            chess->bb_black &= ~bb;
            chess->bb.black_queens &= ~bb;
            break;
        case WHITE_KING:
            chess->bb_white &= ~bb;
            chess->bb.white_kings &= ~bb;
            break;
        case BLACK_KING:
            chess->bb_black &= ~bb;
            chess->bb.black_kings &= ~bb;
            break;
        default:
            break;
    }
}

// Add a piece to the board at a given position (updates board, eval, zhash, and bitboards
// accordingly) Will overwrite piece at square i
static inline Piece Chess_add(Chess* chess, Piece piece, int i) {
    // if board[i] != EMPTY, remove that piece first
    Piece capture = chess->board[i];
    if (capture != EMPTY) Chess_remove(chess, i);
    chess->board[i] = piece;
    chess->eval += Piece_value_at(piece, i);
    chess->zhash ^= Piece_zhash_at(piece, i);
    bitboard_t bb = bitboard_from_index(i);
    switch (piece) {
        case WHITE_PAWN:
            chess->bb_white |= bb;
            chess->bb.white_pawns |= bb;
            chess->pawn_row_sum += index_row(i) - 1;
            break;
        case BLACK_PAWN:
            chess->bb_black |= bb;
            chess->bb.black_pawns |= bb;
            chess->pawn_row_sum += index_row(i) - 6;
            break;
        case WHITE_KNIGHT:
            chess->bb_white |= bb;
            chess->bb.white_knights |= bb;
            break;
        case BLACK_KNIGHT:
            chess->bb_black |= bb;
            chess->bb.black_knights |= bb;
            break;
        case WHITE_BISHOP:
            chess->bb_white |= bb;
            chess->bb.white_bishops |= bb;
            break;
        case BLACK_BISHOP:
            chess->bb_black |= bb;
            chess->bb.black_bishops |= bb;
            break;
        case WHITE_ROOK:
            chess->bb_white |= bb;
            chess->bb.white_rooks |= bb;
            break;
        case BLACK_ROOK:
            chess->bb_black |= bb;
            chess->bb.black_rooks |= bb;
            break;
        case WHITE_QUEEN:
            chess->bb_white |= bb;
            chess->bb.white_queens |= bb;
            break;
        case BLACK_QUEEN:
            chess->bb_black |= bb;
            chess->bb.black_queens |= bb;
            break;
        case WHITE_KING:
            chess->bb_white |= bb;
            chess->bb.white_kings |= bb;
            chess->king_white = i;
            break;
        case BLACK_KING:
            chess->bb_black |= bb;
            chess->bb.black_kings |= bb;
            chess->king_black = i;
            break;
        default:
            break;
    }
    return capture;
}

static inline void Chess_empty_board(Chess* chess) {
    for (int i = 0; i < 64; i++) {
        chess->board[i] = EMPTY;
    }
    chess->turn = TURN_WHITE;
    chess->gamestate = 0b00001111;  // All castling rights available, no en passant
    chess->halfmoves = 0;
    chess->fullmoves = 1;
    chess->zhstack = (ZHashStack){0};
}

static inline void Chess_find_kings(Chess* chess) {
    for (int i = 0; i < 64; i++) {
        Piece piece = chess->board[i];
        if (piece == WHITE_KING) {
            chess->king_white = i;
        } else if (piece == BLACK_KING) {
            chess->king_black = i;
        }
    }
}

// Get a string representation of the board (64 characters)
static inline char* Chess_to_string(Chess* chess) {
    static char board_s[65];
    for (int i = 0; i < 64; i++) {
        board_s[i] = chess->board[i];
    }
    board_s[64] = 0;
    return board_s;
}

// Dump the board state (for debugging)
static inline void Chess_dump(Chess* chess) {
    printf("Board: %s\n", Chess_to_string(chess));
    printf("Game state: %02x\n", chess->gamestate);
    printf("Turn: %s\n", chess->turn == TURN_WHITE ? "White" : "Black");
    printf("Castling rights: ");
    printf("%s", chess->gamestate & BITMASK(0) ? "" : "K");
    printf("%s", chess->gamestate & BITMASK(1) ? "" : "Q");
    printf("%s", chess->gamestate & BITMASK(2) ? "" : "k");
    printf("%s\n", chess->gamestate & BITMASK(3) ? "" : "q");
    bitboard_t en_passant = Chess_en_passant(chess);
    Position pos = Position_from_bitboard(en_passant);
    printf("En passant: %s\n", en_passant == -1 ? "NA" : Position_to_string(&pos));
    printf("Half moves: %d\n", chess->halfmoves);
    printf("Full moves: %d\n", chess->fullmoves);
    printf("White king: %d\n", chess->king_white);
    printf("Black king: %d\n", chess->king_black);
}

// Print the board in a human-readable format
static inline void Chess_print(Chess* board) {
    char* board_s = Chess_to_string(board);
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            putchar(board_s[(7 - i) * 8 + j]);
            putchar(' ');
        }
        putchar('\n');
    }
}

static inline uint64_t Chess_zhash(Chess* chess) {
    uint64_t hash = ZHASH_STATE[chess->gamestate];
    hash ^= chess->turn == TURN_WHITE ? ZHASH_WHITE : ZHASH_BLACK;
    for (int i = 0; i < 64; i++) {
        Piece piece = chess->board[i];
        hash ^= Piece_zhash_at(piece, i);
    }
    return hash;
}

static inline bool Chess_has_non_pawn_material(Chess* chess) {
    int number_of_piece = __builtin_popcountll(chess->bb_white | chess->bb_black);
    int number_of_pawns = __builtin_popcountll(chess->bb.white_pawns | chess->bb.black_pawns);
    return number_of_piece - number_of_pawns > 2;  // 2 because of the kings
}

static inline bool Chess_square_available(Chess* chess, int index, bool captures_only) {
    return captures_only ? Chess_enemy_piece_at(chess, index)
                         : !Chess_friendly_piece_at(chess, index);
}

static inline int Chess_3fold_repetition(Chess* chess) {
    uint64_t hash = ZHashStack_peek(&chess->zhstack);
    int count = 1;
    int range_end = chess->zhstack.sp - chess->halfmoves - 1;
    if (range_end < 0) range_end = 0;

    // Loop through the stack backwards
    for (int i = chess->zhstack.sp - 1; i >= range_end; i -= 2) {
        if (chess->zhstack.hashes[i] == hash) {
            count++;
            if (count >= 3) return 3;
        }
    }

    return count;
}

static inline bitboard_t Chess_friendly_pawns_bb(Chess* chess) {
    return chess->turn == TURN_WHITE ? chess->bb.white_pawns : chess->bb.black_pawns;
}

static inline bitboard_t Chess_enemy_pawns_bb(Chess* chess) {
    return chess->turn == TURN_WHITE ? chess->bb.black_pawns : chess->bb.white_pawns;
}

static inline bitboard_t Chess_friendly_knights_bb(Chess* chess) {
    return chess->turn == TURN_WHITE ? chess->bb.white_knights : chess->bb.black_knights;
}

static inline bitboard_t Chess_enemy_knights_bb(Chess* chess) {
    return chess->turn == TURN_WHITE ? chess->bb.black_knights : chess->bb.white_knights;
}

static inline bitboard_t Chess_friendly_bishops_bb(Chess* chess) {
    return chess->turn == TURN_WHITE ? chess->bb.white_bishops : chess->bb.black_bishops;
}

static inline bitboard_t Chess_enemy_bishops_bb(Chess* chess) {
    return chess->turn == TURN_WHITE ? chess->bb.black_bishops : chess->bb.white_bishops;
}

static inline bitboard_t Chess_friendly_rooks_bb(Chess* chess) {
    return chess->turn == TURN_WHITE ? chess->bb.white_rooks : chess->bb.black_rooks;
}

static inline bitboard_t Chess_enemy_rooks_bb(Chess* chess) {
    return chess->turn == TURN_WHITE ? chess->bb.black_rooks : chess->bb.white_rooks;
}

static inline bitboard_t Chess_friendly_queens_bb(Chess* chess) {
    return chess->turn == TURN_WHITE ? chess->bb.white_queens : chess->bb.black_queens;
}

static inline bitboard_t Chess_enemy_queens_bb(Chess* chess) {
    return chess->turn == TURN_WHITE ? chess->bb.black_queens : chess->bb.white_queens;
}

static inline bitboard_t Chess_friendly_kings_bb(Chess* chess) {
    return chess->turn == TURN_WHITE ? chess->bb.white_kings : chess->bb.black_kings;
}

static inline bitboard_t Chess_enemy_kings_bb(Chess* chess) {
    return chess->turn == TURN_WHITE ? chess->bb.black_kings : chess->bb.white_kings;
}

static inline bitboard_t Chess_friendly_bb(Chess* chess) {
    return chess->turn == TURN_WHITE ? chess->bb_white : chess->bb_black;
}

static inline bitboard_t Chess_enemy_bb(Chess* chess) {
    return chess->turn == TURN_WHITE ? chess->bb_black : chess->bb_white;
}

static inline int Chess_non_pawn_material(Chess* chess) {
    return 4 * bitboard_popcount(chess->bb.white_queens | chess->bb.black_queens) +
           2 * bitboard_popcount(chess->bb.white_rooks | chess->bb.black_rooks) +
           1 * bitboard_popcount(chess->bb.white_bishops | chess->bb.black_bishops |
                                 chess->bb.white_knights | chess->bb.black_knights);
}

Piece Chess_make_move(Chess* chess, Move* move);
void Chess_unmake_move(Chess* chess, Move* move, Piece capture);
gamestate_t Chess_make_null_move(Chess* chess);
void Chess_unmake_null_move(Chess* chess, gamestate_t gamestate);
Piece Chess_user_move(Chess* chess, char* move_input);
Chess* Chess_from_fen(char* fen);
void Chess_print_fen(Chess* chess);
void Chess_game_history(Chess* chess, char* game_history);