#include <ctype.h>
#include <inttypes.h>
#include <limits.h>
#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../consts.c"

#define class typedef struct
#define INF 1000000000

// A bitboard is a 64-bit integer where each bit represents a square on the
// chessboard (from a1 to h8). The least significant bit (LSB) represents a1,
// and the most significant bit (MSB) represents h8.
typedef uint64_t bitboard_t;

// Print a bitboard as a 64-character string of 0s and 1s
void bitboard_print(bitboard_t bb) {
    for (int i = 63; i >= 0; i--) {
        putchar((bb >> i) & 1 ? '1' : '0');
    }
    putchar('\n');
}

// Convert an index (0-63) to a bitboard with only that bit set
static inline bitboard_t bitboard_from_index(int i) {
    return (bitboard_t)(1ULL << i);
}

static inline int index_col(int index) { return index % 8; }

static inline int index_row(int index) { return index / 8; }

// All pieces
typedef enum {
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

int Piece_value(Piece piece) {
    switch (piece) {
        case WHITE_PAWN:
            return 100;
        case BLACK_PAWN:
            return -100;
        case WHITE_KNIGHT:
            return 320;
        case BLACK_KNIGHT:
            return -320;
        case WHITE_BISHOP:
            return 330;
        case BLACK_BISHOP:
            return -330;
        case WHITE_ROOK:
            return 500;
        case BLACK_ROOK:
            return -500;
        case WHITE_QUEEN:
            return 900;
        case BLACK_QUEEN:
            return -900;
        case WHITE_KING:
            return 20000;
        case BLACK_KING:
            return -20000;
        default:
            return 0;
    }
}

int Piece_value_at(Piece piece, int i) {
    switch (piece) {
        case WHITE_PAWN:
            return 100 + PS_WHITE_PAWN[i];
        case BLACK_PAWN:
            return -100 + PS_BLACK_PAWN[i];
        case WHITE_KNIGHT:
            return 320 + PS_WHITE_KNIGHT[i];
        case BLACK_KNIGHT:
            return -320 + PS_BLACK_KNIGHT[i];
        case WHITE_BISHOP:
            return 330 + PS_WHITE_BISHOP[i];
        case BLACK_BISHOP:
            return -330 + PS_BLACK_BISHOP[i];
        case WHITE_ROOK:
            return 500 + PS_WHITE_ROOK[i];
        case BLACK_ROOK:
            return -500 + PS_BLACK_ROOK[i];
        case WHITE_QUEEN:
            return 900 + PS_WHITE_QUEEN[i];
        case BLACK_QUEEN:
            return -900 + PS_BLACK_QUEEN[i];
        case WHITE_KING:
            return 20000 + PS_WHITE_KING[i];
        case BLACK_KING:
            return -20000 + PS_BLACK_KING[i];
        default:
            return 0;
    }
}

uint64_t Piece_zhash_at(Piece piece, int i) {
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

static inline bool Piece_is_white(Piece piece) {
    return isupper(piece) && piece != EMPTY;
}

static inline bool Piece_is_black(Piece piece) {
    return islower(piece) && piece != EMPTY;
}

static inline bool Piece_is_pawn(Piece piece) {
    return piece == WHITE_PAWN || piece == BLACK_PAWN;
}

static inline bool Piece_is_king(Piece piece) {
    return piece == WHITE_KING || piece == BLACK_KING;
}

static inline bool Piece_is_queen(Piece piece) {
    return piece == WHITE_QUEEN || piece == BLACK_QUEEN;
}

static inline bool Piece_is_rook(Piece piece) {
    return piece == WHITE_ROOK || piece == BLACK_ROOK;
}

static inline bool Piece_is_bishop(Piece piece) {
    return piece == WHITE_BISHOP || piece == BLACK_BISHOP;
}

static inline bool Piece_is_knight(Piece piece) {
    return piece == WHITE_KNIGHT || piece == BLACK_KNIGHT;
}

// Convert a character to its piece representation
Piece Piece_from_char(char c) {
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

// A position on the chessboard (from (0,0) to (7,7))
// (Position){row, col}
class {
    uint8_t row;
    uint8_t col;
}
Position;

// Check if a position is valid (on the board)
static inline bool Position_valid(Position *pos) {
    return 0 <= pos->col && pos->col < 8 && 0 <= pos->row && pos->row < 8;
}

// Convert a string (e.g. "e4") to a position
// Returns (Position){-1, -1} if invalid
Position Position_from_string(char *s) {
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
Position Position_from_bitboard(bitboard_t b) {
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
char *Position_to_string(Position *pos) {
    static char buffer[3];
    buffer[0] = 'a' + pos->col;
    buffer[1] = '1' + pos->row;
    buffer[2] = 0;
    return buffer;
}

// Convert a position to an index (0-63)
static inline int Position_to_index(Position *pos) {
    return pos->row * 8 + pos->col;
}

Position Position_from_index(int index) {
    return (Position){index / 8, index % 8};
}

// Print a position
void Position_print(Position pos) {
    printf("Position: %s (row: %d, col: %d)\n", Position_to_string(&pos),
           pos.row, pos.col);
}

typedef enum {
    PROMOTE_QUEEN = 'q',
    PROMOTE_ROOK = 'r',
    PROMOTE_BISHOP = 'b',
    PROMOTE_KNIGHT = 'n',
    NO_PROMOTION = 0
} Promotion;

class {
    uint8_t from;
    uint8_t to;
    Promotion promotion;  // 'q', 'r', 'b', 'n' or 0 for no promotion
    int score;
}
Move;

#define MAX_LEGAL_MOVES 218

char *Move_string(Move *move) {
    static char buffer[6];
    Position from = Position_from_index(move->from);
    Position to = Position_from_index(move->to);
    if (Position_valid(&from) && Position_valid(&to)) {
        strcpy(buffer, Position_to_string(&from));
        strcat(buffer, Position_to_string(&to));
        if (move->promotion) {
            buffer[4] = move->promotion;
        } else {
            buffer[4] = 0;
        }
        buffer[5] = 0;
    } else {
        printf("Invalid move: Move(from=%d, to=%d)\n", move->from, move->to);
        strcpy(buffer, "????");
    }
    return buffer;
}

void Move_print(Move *move) { printf("%s\n", Move_string(move)); }

#define TURN_BLACK true
#define TURN_WHITE false
typedef bool turn_t;

/*
bit 1: white castling kingside, 0 if allowed
bit 2: white castling queenside, 0 if allowed
bit 3: black castling kingside, 0 if allowed
bit 4: black castling queenside, 0 if allowed
bit 5: 0 if no en passant, 1 if en passant is available
bit 6: next 3 bits are for the en-passant column
bit 7: ...
bit 8: ...
*/
typedef uint8_t gamestate_t;

#define Z_HASH_STACK_SIZE 1024
class {
    uint64_t hashes[Z_HASH_STACK_SIZE];
    int sp;
}
ZHashStack;

static inline void ZHashStack_push(ZHashStack *zhstack, uint64_t hash) {
    zhstack->hashes[zhstack->sp++] = hash;
}

static inline uint64_t ZHashStack_pop(ZHashStack *zhstack) {
    return zhstack->hashes[--zhstack->sp];
}

static inline uint64_t ZHashStack_peek(ZHashStack *zhstack) {
    return zhstack->hashes[zhstack->sp - 1];
}

// The chessboard
class {
    Piece board[64];        // Array of pieces, index 0 is a1, index 63 is h8
    turn_t turn;            // true for black, false for white
    gamestate_t gamestate;  // Game state bits
    uint8_t halfmoves;      // Halfmoves since last capture or pawn move
    uint8_t fullmoves;      // Full moves (incremented after every move)
    uint8_t king_white;     // Position of white king
    uint8_t king_black;     // Position of black king
    ZHashStack zhstack;     // Stack to store zobrist hash of previous positions
}
Chess;

#define BITMASK(nbit) (1 << (nbit))

// Set white kingside castling right
static inline void Chess_castle_K_set(Chess *board, bool allow) {
    if (allow)
        board->gamestate &= ~BITMASK(0);
    else
        board->gamestate |= BITMASK(0);
}

// Set white queenside castling right
static inline void Chess_castle_Q_set(Chess *board, bool allow) {
    if (allow)
        board->gamestate &= ~BITMASK(1);
    else
        board->gamestate |= BITMASK(1);
}

// Set black kingside castling right
static inline void Chess_castle_k_set(Chess *board, bool allow) {
    if (allow)
        board->gamestate &= ~BITMASK(2);
    else
        board->gamestate |= BITMASK(2);
}

// Set black queenside castling right
static inline void Chess_castle_q_set(Chess *board, bool allow) {
    if (allow)
        board->gamestate &= ~BITMASK(3);
    else
        board->gamestate |= BITMASK(3);
}

// Get kingside castling right
static inline bool Chess_castle_king_side(Chess *chess) {
    if (chess->turn == TURN_WHITE) {
        return !(chess->gamestate & BITMASK(0));
    } else {
        return !(chess->gamestate & BITMASK(2));
    }
}

// Get queenside castling right
static inline bool Chess_castle_queen_side(Chess *chess) {
    if (chess->turn == TURN_WHITE) {
        return !(chess->gamestate & BITMASK(1));
    } else {
        return !(chess->gamestate & BITMASK(3));
    }
}

// Set en passant column (0-7) or disable (-1)
static inline void Chess_en_passant_set(Chess *board, uint8_t col) {
    if (0 <= col && col < 8) {
        board->gamestate |= BITMASK(4);
        board->gamestate &= 0b00011111;
        board->gamestate |= col << 5;
    } else {
        board->gamestate &= 0b11101111;
    }
}

// Get en passant column (or -1 if not available)
static inline uint8_t Chess_en_passant(Chess *chess) {
    if (chess->gamestate & BITMASK(4)) {
        return chess->gamestate >> 5;
    } else {
        return -1;
    }
}

// Add a piece to the board at a given position
// Only meant for initializing the board
void Chess_add(Chess *chess, Piece piece, Position pos) {
    if (!Position_valid(&pos)) return;
    int i = Position_to_index(&pos);
    chess->board[i] = piece;
}

void Chess_empty_board(Chess *chess) {
    for (int i = 0; i < 64; i++) {
        chess->board[i] = EMPTY;
    }
    chess->turn = TURN_WHITE;
    chess->gamestate =
        0b00001111;  // All castling rights available, no en passant
    chess->halfmoves = 0;
    chess->fullmoves = 1;
    chess->zhstack = (ZHashStack){0};
}

void Chess_find_kings(Chess *chess) {
    for (int i = 0; i < 64; i++) {
        Piece piece = chess->board[i];
        if (piece == WHITE_KING) {
            chess->king_white = i;
        } else if (piece == BLACK_KING) {
            chess->king_black = i;
        }
    }
}

// Create a new board with the initial chess position
Chess *Chess_new() {
    static Chess *chess;
    Chess_empty_board(chess);

    // Pawns
    for (int i = 0; i < 8; i++) {
        Chess_add(chess, WHITE_PAWN, (Position){1, i});
        Chess_add(chess, BLACK_PAWN, (Position){6, i});
    }

    // Rooks
    Chess_add(chess, WHITE_ROOK, (Position){0, 0});
    Chess_add(chess, WHITE_ROOK, (Position){0, 7});
    Chess_add(chess, BLACK_ROOK, (Position){7, 0});
    Chess_add(chess, BLACK_ROOK, (Position){7, 7});

    // Knights
    Chess_add(chess, WHITE_KNIGHT, (Position){0, 1});
    Chess_add(chess, WHITE_KNIGHT, (Position){0, 6});
    Chess_add(chess, BLACK_KNIGHT, (Position){7, 1});
    Chess_add(chess, BLACK_KNIGHT, (Position){7, 6});

    // Bishops
    Chess_add(chess, WHITE_BISHOP, (Position){0, 2});
    Chess_add(chess, WHITE_BISHOP, (Position){0, 5});
    Chess_add(chess, BLACK_BISHOP, (Position){7, 2});
    Chess_add(chess, BLACK_BISHOP, (Position){7, 5});

    // Kings and queens
    Chess_add(chess, WHITE_QUEEN, (Position){0, 3});
    Chess_add(chess, WHITE_KING, (Position){0, 4});
    Chess_add(chess, BLACK_QUEEN, (Position){7, 3});
    Chess_add(chess, BLACK_KING, (Position){7, 4});

    Chess_find_kings(chess);

    return chess;
}

// Get a string representation of the board (64 characters)
char *Chess_to_string(Chess *chess) {
    static char board_s[65];
    for (int i = 0; i < 64; i++) {
        board_s[i] = chess->board[i];
    }
    board_s[64] = 0;
    return board_s;
}

// Dump the board state (for debugging)
void Chess_dump(Chess *chess) {
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
    printf("En passant: %s\n",
           en_passant == -1 ? "NA" : Position_to_string(&pos));
    printf("Half moves: %d\n", chess->halfmoves);
    printf("Full moves: %d\n", chess->fullmoves);
}

// Print the board in a human-readable format
void Chess_print(Chess *board) {
    char *board_s = Chess_to_string(board);
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            putchar(board_s[(7 - i) * 8 + j]);
            putchar(' ');
        }
        putchar('\n');
    }
}

bool Chess_friendly_piece_at(Chess *chess, int index) {
    Piece piece = chess->board[index];
    if (piece == EMPTY) return false;
    if (chess->turn == TURN_WHITE) {
        return isupper(piece);
    } else {
        return islower(piece);
    }
}

bool Chess_enemy_piece_at(Chess *chess, int index) {
    Piece piece = chess->board[index];
    if (piece == EMPTY) return false;
    if (chess->turn == TURN_WHITE) {
        return islower(piece);
    } else {
        return isupper(piece);
    }
}

bool Chess_friendly_king_at(Chess *chess, int index) {
    Piece piece = chess->board[index];
    if (chess->turn == TURN_WHITE) {
        return piece == WHITE_KING;
    } else {
        return piece == BLACK_KING;
    }
}

bool Chess_enemy_king_at(Chess *chess, int index) {
    Piece piece = chess->board[index];
    if (chess->turn == TURN_WHITE) {
        return piece == BLACK_KING;
    } else {
        return piece == WHITE_KING;
    }
}

bool Chess_friendly_pawn_at(Chess *chess, int index) {
    Piece piece = chess->board[index];
    if (chess->turn == TURN_WHITE) {
        return piece == WHITE_PAWN;
    } else {
        return piece == BLACK_PAWN;
    }
}

bool Chess_enemy_pawn_at(Chess *chess, int index) {
    Piece piece = chess->board[index];
    if (chess->turn == TURN_WHITE) {
        return piece == BLACK_PAWN;
    } else {
        return piece == WHITE_PAWN;
    }
}

bool Chess_friendly_knight_at(Chess *chess, int index) {
    Piece piece = chess->board[index];
    if (chess->turn == TURN_WHITE) {
        return piece == WHITE_KNIGHT;
    } else {
        return piece == BLACK_KNIGHT;
    }
}

bool Chess_enemy_knight_at(Chess *chess, int index) {
    Piece piece = chess->board[index];
    if (chess->turn == TURN_WHITE) {
        return piece == BLACK_KNIGHT;
    } else {
        return piece == WHITE_KNIGHT;
    }
}

bool Chess_friendly_bishop_at(Chess *chess, int index) {
    Piece piece = chess->board[index];
    if (chess->turn == TURN_WHITE) {
        return piece == WHITE_BISHOP;
    } else {
        return piece == BLACK_BISHOP;
    }
}

bool Chess_enemy_bishop_at(Chess *chess, int index) {
    Piece piece = chess->board[index];
    if (chess->turn == TURN_WHITE) {
        return piece == BLACK_BISHOP;
    } else {
        return piece == WHITE_BISHOP;
    }
}

bool Chess_friendly_rook_at(Chess *chess, int index) {
    Piece piece = chess->board[index];
    if (chess->turn == TURN_WHITE) {
        return piece == WHITE_ROOK;
    } else {
        return piece == BLACK_ROOK;
    }
}

bool Chess_enemy_rook_at(Chess *chess, int index) {
    Piece piece = chess->board[index];
    if (chess->turn == TURN_WHITE) {
        return piece == BLACK_ROOK;
    } else {
        return piece == WHITE_ROOK;
    }
}

bool Chess_friendly_queen_at(Chess *chess, int index) {
    Piece piece = chess->board[index];
    if (chess->turn == TURN_WHITE) {
        return piece == WHITE_QUEEN;
    } else {
        return piece == BLACK_QUEEN;
    }
}

bool Chess_enemy_queen_at(Chess *chess, int index) {
    Piece piece = chess->board[index];
    if (chess->turn == TURN_WHITE) {
        return piece == BLACK_QUEEN;
    } else {
        return piece == WHITE_QUEEN;
    }
}

uint64_t Chess_zhash(Chess *chess) {
    uint64_t hash = ZHASH_STATE[chess->gamestate];
    hash ^= chess->turn == TURN_WHITE ? ZHASH_WHITE : ZHASH_BLACK;
    for (int i = 0; i < 64; i++) {
        Piece piece = chess->board[i];
        hash ^= Piece_zhash_at(piece, i);
    }
    return hash;
}

// Returns the piece that was captured, or EMPTY if no capture
Piece Chess_make_move(Chess *chess, Move *move) {
    Piece moving_piece = chess->board[move->from];
    Piece target_piece = chess->board[move->to];

    // Update halfmove clock
    // Reset if a pawn moved or a capture was made
    if (!Piece_is_pawn(moving_piece) && target_piece == EMPTY) {
        chess->halfmoves++;
    } else {
        chess->halfmoves = 0;
    }

    // Update fullmove number
    if (chess->turn == TURN_BLACK) {
        chess->fullmoves++;
    }

    // Update en passant status
    if (Piece_is_pawn(moving_piece) && abs(move->to - move->from) == 16) {
        // Pawn moved two squares
        Chess_en_passant_set(chess, index_col(move->from));
    } else {
        Chess_en_passant_set(chess, -1);
    }

    // Update castling rights if a rook or king moved
    if (moving_piece == WHITE_KING) {
        Chess_castle_K_set(chess, false);
        Chess_castle_Q_set(chess, false);
        chess->king_white = move->to;
    } else if (moving_piece == BLACK_KING) {
        Chess_castle_k_set(chess, false);
        Chess_castle_q_set(chess, false);
        chess->king_black = move->to;
    } else if (moving_piece == WHITE_ROOK) {
        if (move->from == 0) {
            Chess_castle_Q_set(chess, false);
        } else if (move->from == 7) {
            Chess_castle_K_set(chess, false);
        }
    } else if (moving_piece == BLACK_ROOK) {
        if (move->from == 56) {
            Chess_castle_q_set(chess, false);
        } else if (move->from == 63) {
            Chess_castle_k_set(chess, false);
        }
    }

    // Update castling rights if a rook was captured
    if (target_piece == WHITE_ROOK) {
        if (move->to == 0) {
            Chess_castle_Q_set(chess, false);
        } else if (move->to == 7) {
            Chess_castle_K_set(chess, false);
        }
    } else if (target_piece == BLACK_ROOK) {
        if (move->to == 56) {
            Chess_castle_q_set(chess, false);
        } else if (move->to == 63) {
            Chess_castle_k_set(chess, false);
        }
    }

    // Move the rook if castling
    if (moving_piece == WHITE_KING && move->from == 4 && move->to == 6) {
        // White kingside
        chess->board[5] = WHITE_ROOK;
        chess->board[7] = EMPTY;
    } else if (moving_piece == WHITE_KING && move->from == 4 && move->to == 2) {
        // White queenside
        chess->board[3] = WHITE_ROOK;
        chess->board[0] = EMPTY;
    } else if (moving_piece == BLACK_KING && move->from == 60 &&
               move->to == 62) {
        // Black kingside
        chess->board[61] = BLACK_ROOK;
        chess->board[63] = EMPTY;
    } else if (moving_piece == BLACK_KING && move->from == 60 &&
               move->to == 58) {
        // Black queenside
        chess->board[59] = BLACK_ROOK;
        chess->board[56] = EMPTY;
    }

    // Handle en passant capture
    if (moving_piece == WHITE_PAWN &&
        index_col(move->from) != index_col(move->to) && target_piece == EMPTY) {
        // White pawn capturing en passant
        chess->board[move->to - 8] = EMPTY;
    } else if (moving_piece == BLACK_PAWN &&
               index_col(move->from) != index_col(move->to) &&
               target_piece == EMPTY) {
        // Black pawn capturing en passant
        chess->board[move->to + 8] = EMPTY;
    }

    // Handle promotion
    if (move->promotion) {
        if (chess->turn == TURN_WHITE) {
            switch (move->promotion) {
                case PROMOTE_QUEEN:
                    moving_piece = WHITE_QUEEN;
                    break;
                case PROMOTE_ROOK:
                    moving_piece = WHITE_ROOK;
                    break;
                case PROMOTE_BISHOP:
                    moving_piece = WHITE_BISHOP;
                    break;
                case PROMOTE_KNIGHT:
                    moving_piece = WHITE_KNIGHT;
                    break;
                default:
                    break;
            }
        } else {
            switch (move->promotion) {
                case PROMOTE_QUEEN:
                    moving_piece = BLACK_QUEEN;
                    break;
                case PROMOTE_ROOK:
                    moving_piece = BLACK_ROOK;
                    break;
                case PROMOTE_BISHOP:
                    moving_piece = BLACK_BISHOP;
                    break;
                case PROMOTE_KNIGHT:
                    moving_piece = BLACK_KNIGHT;
                    break;
                default:
                    break;
            }
        }
    }

    // Switch turn
    chess->turn = !chess->turn;

    chess->board[move->to] = moving_piece;
    chess->board[move->from] = EMPTY;
    uint64_t hash = Chess_zhash(chess);
    ZHashStack_push(&chess->zhstack, hash);
    return target_piece;
}

// NEED to reset gamestate manually afterwards
void Chess_unmake_move(Chess *chess, Move *move, Piece capture) {
    ZHashStack_pop(&chess->zhstack);
    chess->turn = !chess->turn;

    // Reset the board
    Piece moving_piece;
    switch (move->promotion) {
        case PROMOTE_BISHOP:
        case PROMOTE_KNIGHT:
        case PROMOTE_QUEEN:
        case PROMOTE_ROOK:
            moving_piece = chess->turn == TURN_WHITE ? WHITE_PAWN : BLACK_PAWN;
            break;
        default:
            moving_piece = chess->board[move->to];
            break;
    }
    chess->board[move->from] = moving_piece;
    chess->board[move->to] = capture;

    if (Piece_is_king(moving_piece)) {
        // castling
        uint8_t king_move = abs(move->to - move->from);
        if (king_move == 2) {
            Position pos = Position_from_index(move->to);
            if (pos.col < 4) {  // queen side castling
                chess->board[8 * pos.row] = chess->board[8 * pos.row + 3];
                chess->board[8 * pos.row + 3] = EMPTY;
            } else {  // king side castling
                chess->board[8 * pos.row + 7] = chess->board[8 * pos.row + 5];
                chess->board[8 * pos.row + 5] = EMPTY;
            }
        }

        // Reset king position
        if (moving_piece == WHITE_KING) {
            chess->king_white = move->from;
        } else {
            chess->king_black = move->from;
        }

    } else if (Piece_is_pawn(moving_piece) && capture == EMPTY) {
        // en passant capture
        uint8_t pawn_move = abs(move->to - move->from);
        if (pawn_move == 7 || pawn_move == 9) {
            int col = index_col(move->to);
            if (chess->turn == TURN_WHITE) {
                chess->board[col + 32] = BLACK_PAWN;
            } else {
                chess->board[col + 24] = WHITE_PAWN;
            }
        }
    }

    // Update halfmove clock
    // Reset if a pawn moved or a capture was made
    // TODO: this aint working
    if (!Piece_is_pawn(moving_piece) && capture == EMPTY) {
        chess->halfmoves--;
    } else {
        chess->halfmoves = 0;
    }

    // Update fullmove number
    if (chess->turn == TURN_BLACK) {
        chess->fullmoves--;
    }
}

// Parse and make a user move in algebraic notation (e.g. "e2e4")
// No validation is done, so the move must be legal
Piece Chess_user_move(Chess *chess, char *move_input) {
#define INVALID_MOVE(details)                                 \
    fprintf(stderr, "Invalid move: " details ": %s\n", move); \
    return EMPTY

    // Create a local copy of the move string
    char move[6];
    strncpy(move, move_input, 5);
    move[5] = 0;

    Promotion promotion = NO_PROMOTION;
    if (strlen(move) == 5) {
        promotion = move[4];
        move[4] = 0;  // Temporarily terminate the string
        if (promotion != PROMOTE_QUEEN && promotion != PROMOTE_ROOK &&
            promotion != PROMOTE_BISHOP && promotion != PROMOTE_KNIGHT) {
            INVALID_MOVE("Invalid promotion piece");
        }
    }

    if (strlen(move) != 4) {
        INVALID_MOVE("Invalid length");
    }

    Position from = Position_from_string((char[]){move[0], move[1], 0});
    Position to = Position_from_string((char[]){move[2], move[3], 0});

    if (!Position_valid(&from) || !Position_valid(&to)) {
        INVALID_MOVE("Invalid position");
    }

    int from_i = Position_to_index(&from);
    int to_i = Position_to_index(&to);
    Move move_ = {.from = from_i, .to = to_i, .promotion = promotion};

    // if piece at 'from' is empty or not friendly
    // or piece at 'to' is friendly, invalid move
    if (!Chess_friendly_piece_at(chess, move_.from)) {
        INVALID_MOVE("No friendly piece at 'from' position");
    }
    if (Chess_friendly_piece_at(chess, move_.to)) {
        INVALID_MOVE("Cannot capture friendly piece");
    }

    return Chess_make_move(chess, &move_);
}

// Check if a string is a valid non-empty digit string
bool string_isdigit(const char *s) {
    return s[0] != '\0' && strspn(s, "0123456789") == strlen(s);
}

// FEN fields used for parsing
typedef enum {
    FEN_PLACEMENT,
    FEN_TURN,
    FEN_CASTLING,
    FEN_EN_PASSANT,
    FEN_HALFMOVE,
    FEN_FULLMOVE,
    FEN_END
} FENField;

Chess *Chess_from_fen(char *fen_arg) {
#define FEN_PARSING_ERROR(details)                                \
    fprintf(stderr, "FEN Parsing error: " details ": %s\n", fen); \
    return NULL

    char fen[128];
    strncpy(fen, fen_arg, sizeof(fen) - 1);

    static Chess chess_struct;
    Chess *board = &chess_struct;  // empty board
    Chess_empty_board(board);

    // Split FEN into fields
    char *fields[6];
    int i = 0;
    char *token = strtok(fen, " ");
    while (token && i < 6) {
        fields[i++] = token;
        token = strtok(NULL, " ");
    }
    if (i < 6) {
        FEN_PARSING_ERROR("Not enough fields in FEN");
    }

    // 1. Piece placement
    Position pos = {.col = 0, .row = 7};
    for (char *c = fields[0]; *c; ++c) {
        if (*c == '/') {
            if (pos.col != 0) {
                FEN_PARSING_ERROR("Invalid FEN format");
            }
            continue;
        }
        if (isdigit(*c)) {
            int skip = *c - '0';
            if (*c == '0' || *c == '9' || pos.col + skip > 8) {
                FEN_PARSING_ERROR("Invalid empty spacing");
            }
            pos.col += skip;
        } else {
            Piece piece = Piece_from_char(*c);
            if (piece == -1) {
                FEN_PARSING_ERROR("Invalid piece");
            }
            Chess_add(board, piece, pos);
            pos.col += 1;
        }
        if (pos.col == 8) {
            pos.col = 0;
            pos.row -= 1;
        }
    }

    // 2. Turn
    if (strcmp(fields[1], "w") == 0) {
        board->turn = TURN_WHITE;
    } else if (strcmp(fields[1], "b") == 0) {
        board->turn = TURN_BLACK;
    } else {
        FEN_PARSING_ERROR("Turn must be 'w' or 'b'");
    }

    // 3. Castling rights
    Chess_castle_Q_set(board, strchr(fields[2], 'Q') != NULL);
    Chess_castle_q_set(board, strchr(fields[2], 'q') != NULL);
    Chess_castle_K_set(board, strchr(fields[2], 'K') != NULL);
    Chess_castle_k_set(board, strchr(fields[2], 'k') != NULL);

    // 4. En passant
    if (strcmp(fields[3], "-") == 0) {
        Chess_en_passant_set(board, -1);
    } else {
        Position ep = Position_from_string(fields[3]);
        if (!Position_valid(&ep)) {
            FEN_PARSING_ERROR("Invalid en passant position");
        }
        Chess_en_passant_set(board, ep.col);
    }

    // 5. Halfmove clock
    if (!string_isdigit(fields[4])) {
        FEN_PARSING_ERROR("Half move clock NaN");
    }
    int halfmoves = strtoul(fields[4], NULL, 10);
    if (halfmoves > 99) {
        FEN_PARSING_ERROR("Half move clock overflow");
    }
    board->halfmoves = (uint8_t)halfmoves;

    // 6. Fullmove number
    if (!string_isdigit(fields[5])) {
        FEN_PARSING_ERROR("Full move clock NaN");
    }
    int fullmoves = strtoul(fields[5], NULL, 10);
    if (fullmoves > 255) {
        FEN_PARSING_ERROR("Full move clock overflow");
    }
    board->fullmoves = (uint8_t)fullmoves;
    Chess_find_kings(board);
    return board;
}

static inline uint8_t Chess_friendly_king_i(Chess *chess) {
    return chess->turn == TURN_WHITE ? chess->king_white : chess->king_black;
}

static inline uint8_t Chess_enemy_king_i(Chess *chess) {
    return chess->turn == TURN_WHITE ? chess->king_black : chess->king_white;
}

bool Chess_friendly_check(Chess *chess) {
    // Find the king
    uint8_t king_i = Chess_friendly_king_i(chess);
    Position king_pos = Position_from_index(king_i);

#define ENEMY_ATTACK(condition1, condition2) \
    if ((condition1) && (condition2)) return true;

    // Look for pawn attacks
#define PAWN_ATTACK(condition, offset) \
    ENEMY_ATTACK(condition, Chess_enemy_pawn_at(chess, king_i + (offset)))
    if (chess->turn == TURN_WHITE) {
        PAWN_ATTACK(king_pos.row < 7 && king_pos.col < 7, 9)
        PAWN_ATTACK(king_pos.row < 7 && king_pos.col > 0, 7)
    } else {
        PAWN_ATTACK(king_pos.row > 0 && king_pos.col > 0, -9)
        PAWN_ATTACK(king_pos.row > 0 && king_pos.col < 7, -7)
    }

    // Look for knight attacks
#define KNIGHT_ATTACK(condition, offset) \
    ENEMY_ATTACK(condition, Chess_enemy_knight_at(chess, king_i + (offset)))
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
    if (abs(enemy_king_pos.row - king_pos.row) <= 1 &&
        abs(enemy_king_pos.col - king_pos.col) <= 1) {
        return true;
    }

    int i;
#define SLIDING_PIECE_ATTACK(fn, condition, offset)           \
    for (i = 0; (condition); i++) {                           \
        if (fn(chess, king_i + (offset)) ||                   \
            Chess_enemy_queen_at(chess, king_i + (offset))) { \
            return true;                                      \
        }                                                     \
        if (chess->board[king_i + (offset)] != EMPTY) break;  \
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
#define ROOK_ATTACK(condition, offset) \
    SLIDING_PIECE_ATTACK(Chess_enemy_rook_at, condition, offset)
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

    return false;
}

bool captures_only = false;

bool Chess_square_available(Chess *chess, int index) {
    return captures_only ? Chess_enemy_piece_at(chess, index)
                         : !Chess_friendly_piece_at(chess, index);
}

// This will check if the king is in check after the move
bool Chess_is_move_legal(Chess *chess, Move *move) {
    // Make the move
    // Remember: make move changes turn so enemy pieces are 'friendly'
    gamestate_t gamestate = chess->gamestate;
    Piece capture = Chess_make_move(chess, move);
    chess->turn = !chess->turn;  // to make piece friendly

    bool king_under_attack = Chess_friendly_check(chess);

    chess->turn = !chess->turn;
    Chess_unmake_move(chess, move, capture);
    chess->gamestate = gamestate;
    return !king_under_attack;
}

// Move *move, size_t n_moves NEED to be defined within scope
#define ADD_MOVE_IF(condition, offset)                 \
    if (condition) {                                   \
        move->from = from;                             \
        move->to = from + (offset);                    \
        move->promotion = NO_PROMOTION;                \
        if (Chess_square_available(chess, move->to) && \
            Chess_is_move_legal(chess, move)) {        \
            move++;                                    \
            n_moves++;                                 \
        }                                              \
    }

size_t Chess_knight_moves(Chess *chess, Move *move, int from) {
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

#define SLIDING_PIECE_ADD_MOVE(condition, offset)          \
    for (i = 0; (condition); i++) {                        \
        move->from = from;                                 \
        move->to = from + (offset);                        \
        move->promotion = NO_PROMOTION;                    \
        if (Chess_square_available(chess, move->to) &&     \
            Chess_is_move_legal(chess, move)) {            \
            move++;                                        \
            n_moves++;                                     \
        }                                                  \
        if (chess->board[from + (offset)] != EMPTY) break; \
    }

size_t Chess_bishop_moves(Chess *chess, Move *move, int from) {
    size_t n_moves = 0;
    Position pos = Position_from_index(from);
    int i;

    SLIDING_PIECE_ADD_MOVE(pos.col + i < 7 && pos.row + i < 7, (i + 1) * 9)
    SLIDING_PIECE_ADD_MOVE(pos.col - i > 0 && pos.row - i > 0, (i + 1) * -9)
    SLIDING_PIECE_ADD_MOVE(pos.col + i < 7 && pos.row - i > 0, (i + 1) * -7)
    SLIDING_PIECE_ADD_MOVE(pos.col - i > 0 && pos.row + i < 7, (i + 1) * 7)

    return n_moves;
}

size_t Chess_rook_moves(Chess *chess, Move *move, int from) {
    size_t n_moves = 0;
    Position pos = Position_from_index(from);
    int i;

    SLIDING_PIECE_ADD_MOVE(pos.row + i < 7, (i + 1) * 8)
    SLIDING_PIECE_ADD_MOVE(pos.col + i < 7, (i + 1))
    SLIDING_PIECE_ADD_MOVE(pos.row - i > 0, (i + 1) * -8)
    SLIDING_PIECE_ADD_MOVE(pos.col - i > 0, (i + 1) * -1)

    return n_moves;
}

size_t Chess_queen_moves(Chess *chess, Move *move, int from) {
    size_t n_moves = Chess_rook_moves(chess, move, from);
    return n_moves + Chess_bishop_moves(chess, move + n_moves, from);
}

size_t Chess_pawn_moves(Chess *chess, Move *move, int from) {
    Position pos = Position_from_index(from);
    size_t n_moves = 0;
    int direction = chess->turn == TURN_WHITE ? 1 : -1;
    bool at_home_rank = chess->turn == TURN_WHITE ? pos.row == 1 : pos.row == 6;
    bool at_last_rank = chess->turn == TURN_WHITE ? pos.row == 6 : pos.row == 1;
    bool at_en_passant_rank =
        chess->turn == TURN_WHITE ? pos.row == 4 : pos.row == 3;

#define PAWN_ADD_MOVE_PROMOTE(offset, promote_to) \
    move->from = from;                            \
    move->to = from + (offset);                   \
    if (Chess_is_move_legal(chess, move)) {       \
        move->promotion = (promote_to);           \
        move++;                                   \
        n_moves++;                                \
    }

#define PAWN_ADD_MOVE(offset) PAWN_ADD_MOVE_PROMOTE(offset, NO_PROMOTION)
    // 1 row up
    if (chess->board[from + 8 * direction] == EMPTY && !captures_only) {
        if (at_last_rank) {
            PAWN_ADD_MOVE_PROMOTE(8 * direction, PROMOTE_BISHOP)
            PAWN_ADD_MOVE_PROMOTE(8 * direction, PROMOTE_KNIGHT)
            PAWN_ADD_MOVE_PROMOTE(8 * direction, PROMOTE_QUEEN)
            PAWN_ADD_MOVE_PROMOTE(8 * direction, PROMOTE_ROOK)
        } else {
            PAWN_ADD_MOVE(8 * direction)

            // 2 rows up
            if (at_home_rank && chess->board[from + 16 * direction] == EMPTY) {
                PAWN_ADD_MOVE(16 * direction)
            }
        }
    }

    // normal captures
    if (pos.col > 0 && Chess_enemy_piece_at(chess, from + 8 * direction - 1)) {
        if (at_last_rank) {
            PAWN_ADD_MOVE_PROMOTE(8 * direction - 1, PROMOTE_BISHOP)
            PAWN_ADD_MOVE_PROMOTE(8 * direction - 1, PROMOTE_KNIGHT)
            PAWN_ADD_MOVE_PROMOTE(8 * direction - 1, PROMOTE_QUEEN)
            PAWN_ADD_MOVE_PROMOTE(8 * direction - 1, PROMOTE_ROOK)
        } else {
            PAWN_ADD_MOVE(8 * direction - 1)
        }
    }
    if (pos.col < 7 && Chess_enemy_piece_at(chess, from + 8 * direction + 1)) {
        if (at_last_rank) {
            PAWN_ADD_MOVE_PROMOTE(8 * direction + 1, PROMOTE_BISHOP)
            PAWN_ADD_MOVE_PROMOTE(8 * direction + 1, PROMOTE_KNIGHT)
            PAWN_ADD_MOVE_PROMOTE(8 * direction + 1, PROMOTE_QUEEN)
            PAWN_ADD_MOVE_PROMOTE(8 * direction + 1, PROMOTE_ROOK)
        } else {
            PAWN_ADD_MOVE(8 * direction + 1)
        }
    }

    // en passant capture
    uint8_t en_passant_col = Chess_en_passant(chess);
    if (at_en_passant_rank && en_passant_col != -1) {
        if (en_passant_col == pos.col - 1) {
            PAWN_ADD_MOVE(8 * direction - 1)
        } else if (en_passant_col == pos.col + 1) {
            PAWN_ADD_MOVE(8 * direction + 1)
        }
    }

    return n_moves;
}

size_t Chess_king_moves(Chess *chess, Move *move, int from) {
    Position pos = Position_from_index(from);
    size_t n_moves = 0;

    ADD_MOVE_IF(pos.row > 0 && pos.col > 0, -9)
    ADD_MOVE_IF(pos.row > 0 && pos.col < 7, -7)
    ADD_MOVE_IF(pos.row < 7 && pos.col > 0, 7)
    ADD_MOVE_IF(pos.row < 7 && pos.col < 7, 9)
    ADD_MOVE_IF(pos.row > 0, -8)
    ADD_MOVE_IF(pos.col > 0, -1)
    ADD_MOVE_IF(pos.row < 7, 8)
    ADD_MOVE_IF(pos.col < 7, 1)
    if (captures_only) return n_moves;

#define ADD_KING_MOVE_IF(offset)                \
    if (Chess_is_move_legal(chess, move)) {     \
        move->from = from;                      \
        move->to = from + (offset);             \
        move->promotion = NO_PROMOTION;         \
        if (Chess_is_move_legal(chess, move)) { \
            move++;                             \
            n_moves++;                          \
        }                                       \
    }

    // Castling
#define ADD_CASTLE_MOVE(k1, k2, q1, q2, q3)                             \
    /* King side castling */                                            \
    if ((Chess_castle_king_side(chess) && chess->board[k1] == EMPTY &&  \
         chess->board[k2] == EMPTY)) {                                  \
        move->from = from;                                              \
        move->to = k1;                                                  \
        ADD_KING_MOVE_IF(2)                                             \
    }                                                                   \
    /* Queen side castling */                                           \
    if ((Chess_castle_queen_side(chess) && chess->board[q1] == EMPTY && \
         chess->board[q2] == EMPTY && chess->board[q3] == EMPTY)) {     \
        move->from = from;                                              \
        move->to = q3;                                                  \
        ADD_KING_MOVE_IF(-2)                                            \
    }

    if (!Chess_friendly_check(chess)) {
        if (chess->turn == TURN_WHITE) {
            ADD_CASTLE_MOVE(5, 6, 1, 2, 3)
        } else {
            ADD_CASTLE_MOVE(61, 62, 57, 58, 59)
        }
    }

    return n_moves;
}

size_t Chess_legal_moves(Chess *chess, Move *moves) {
    size_t n_moves = 0;
    for (int i = 0; i < 64; i++) {
        if (!Chess_friendly_piece_at(chess, i)) continue;
        Piece piece = chess->board[i];
        Move *move_p = &moves[n_moves];

#define PIECE_PAWN                                     \
    if (Piece_is_pawn(piece)) {                        \
        n_moves += Chess_pawn_moves(chess, move_p, i); \
    }
#define PIECE_KNIGHT                                     \
    if (Piece_is_knight(piece)) {                        \
        n_moves += Chess_knight_moves(chess, move_p, i); \
    }
#define PIECE_BISHOP                                     \
    if (Piece_is_bishop(piece)) {                        \
        n_moves += Chess_bishop_moves(chess, move_p, i); \
    }
#define PIECE_ROOK                                     \
    if (Piece_is_rook(piece)) {                        \
        n_moves += Chess_rook_moves(chess, move_p, i); \
    }
#define PIECE_KING                                     \
    if (Piece_is_king(piece)) {                        \
        n_moves += Chess_king_moves(chess, move_p, i); \
    }
#define PIECE_QUEEN                                     \
    if (Piece_is_queen(piece)) {                        \
        n_moves += Chess_queen_moves(chess, move_p, i); \
    }

        // TEST PIECES HERE
        PIECE_PAWN
        PIECE_KNIGHT
        PIECE_BISHOP
        PIECE_ROOK
        PIECE_KING
        PIECE_QUEEN
    }
    return n_moves;
}

#define SCORE_VICTIM_MULTIPLIER 1

void Chess_score_move(Chess *chess, Move *move) {
    // Give very high scores to promotions
    if (move->promotion == PROMOTE_QUEEN) {
        move->score = 100;
        return;
    }

    move->score = 0;
    Position pos = Position_from_index(move->to);
    Piece aggressor = chess->board[move->from];
    Piece victim = chess->board[move->to];

    // MVV - LVA
    if (victim != EMPTY) {
        move->score += abs(SCORE_VICTIM_MULTIPLIER * Piece_value(victim) -
                           Piece_value(aggressor));
    } else {
        // Deduct points if attacked by enemy pawns
#define ATTACKED_BY_ENEMY_PAWN(condition, offset, pawn)             \
    if ((condition) && chess->board[move->to + (offset)] == (pawn)) \
        move->score -= abs(Piece_value(aggressor));

        if (chess->turn == TURN_WHITE && aggressor != WHITE_PAWN) {
            ATTACKED_BY_ENEMY_PAWN(pos.row < 6 && pos.col < 7, 9, BLACK_PAWN)
            ATTACKED_BY_ENEMY_PAWN(pos.row < 6 && pos.col > 0, 7, BLACK_PAWN)
        } else if (chess->turn == TURN_BLACK && aggressor != BLACK_PAWN) {
            ATTACKED_BY_ENEMY_PAWN(pos.row > 1 && pos.col < 7, -7, WHITE_PAWN)
            ATTACKED_BY_ENEMY_PAWN(pos.row > 1 && pos.col > 0, -9, WHITE_PAWN)
        }
    }
}

int compare_moves(const void *a, const void *b) {
    const Move *ma = (const Move *)a;
    const Move *mb = (const Move *)b;
    return mb->score - ma->score;
}

size_t Chess_legal_moves_sorted(Chess *chess, Move *moves) {
    size_t n_moves = Chess_legal_moves(chess, moves);

    // Give a score to each move
    for (int i = 0; i < n_moves; i++) {
        Move *move = &moves[i];
        Chess_score_move(chess, move);
    }

    // C lib sort
    qsort(moves, n_moves, sizeof(Move), compare_moves);

    return n_moves;
}

size_t Chess_count_moves(Chess *chess, int depth) {
    if (depth == 0) return 1;

    Move moves[MAX_LEGAL_MOVES];
    size_t n_moves = Chess_legal_moves(chess, moves);
    if (depth == 1) return n_moves;

    size_t nodes = 0;
    for (int i = 0; i < n_moves; i++) {
        gamestate_t gamestate = chess->gamestate;
        Piece capture = Chess_make_move(chess, &moves[i]);
        nodes += Chess_count_moves(chess, depth - 1);
        Chess_unmake_move(chess, &moves[i], capture);
        chess->gamestate = gamestate;
    }

    return nodes;
}

int Chess_3fold_repetition(Chess *chess) {
    uint64_t hash = ZHashStack_peek(&chess->zhstack);
    int count = 1;

    // Loop through the stack backwards
    for (int i = chess->zhstack.sp - 2; i >= 0; i--) {
        if (chess->zhstack.hashes[i] == hash) {
            count++;
            if (count >= 3) return 3;
        }
    }

    return count;
}

size_t nodes_total = 0;
pthread_mutex_t lock;
class {
    Chess chess;
    int depth;
    Move move;
}
ChessThread;

void *Chess_count_moves_thread(void *arg_void) {
    ChessThread *arg = (ChessThread *)arg_void;
    Chess *chess = &arg->chess;
    int depth = arg->depth;
    Move *move = &arg->move;

    gamestate_t gamestate = chess->gamestate;
    Piece capture = Chess_make_move(chess, move);
    size_t nodes = Chess_count_moves(chess, depth - 1);
    Chess_unmake_move(chess, move, capture);
    chess->gamestate = gamestate;

    pthread_mutex_lock(&lock);    // enter critical section
    nodes_total += nodes;         // update shared variable
    pthread_mutex_unlock(&lock);  // leave critical section

    return NULL;
}

size_t Chess_count_moves_multi(Chess *chess, int depth) {
    Move moves[MAX_LEGAL_MOVES];
    size_t n_moves = Chess_legal_moves(chess, moves);
    nodes_total = 0;

    pthread_t *threads = calloc(n_moves, sizeof(pthread_t));
    ChessThread *args = calloc(n_moves, sizeof(ChessThread));

    // Initialize the mutex
    if (pthread_mutex_init(&lock, NULL) != 0) {
        perror("mutex init failed");
        return 1;
    }

    // Create threads
    for (int i = 0; i < n_moves; i++) {
        ChessThread *arg = &args[i];
        arg->depth = depth;
        memcpy(&arg->chess, chess, sizeof(Chess));
        memcpy(&arg->move, &moves[i], sizeof(Move));

        if (pthread_create(&threads[i], NULL, Chess_count_moves_thread, arg) !=
            0) {
            perror("pthread_create failed");
            return 1;
        }
    }

    // Wait for threads to finish
    for (int i = 0; i < n_moves; i++) {
        pthread_join(threads[i], NULL);
    }

    // Clean up
    pthread_mutex_destroy(&lock);
    free(threads);
    free(args);

    return nodes_total;
}

int moves(char *fen, int depth) {
    Chess *chess = Chess_from_fen(fen);
    if (!chess) return 1;
    if (depth > 1) {
        clock_t start = clock();
        size_t n_moves = Chess_count_moves_multi(chess, depth);
        clock_t end = clock();
        double cpu_time = ((double)end - start) / CLOCKS_PER_SEC;
        double nps = cpu_time > 0.0 ? n_moves / cpu_time : -0.0;
        puts("{");
        printf("  \"depth\": %d,\n", depth);
        printf("  \"nodes\": %lu,\n", (unsigned long)n_moves);
        printf("  \"time\": %.3lf,\n", cpu_time);
        printf("  \"nps\": %.3lf\n", nps);
        puts("}");
    } else {
        Move moves[MAX_LEGAL_MOVES];
        size_t n_moves = Chess_legal_moves(chess, moves);
        puts("{");
        printf("  \"nodes\": %lu,\n", (unsigned long)n_moves);
        puts("  \"moves\": [");
        for (int i = 0; i < n_moves; i++) {
            printf("    \"%s\"", Move_string(&moves[i]));
            if (i < n_moves - 1)
                puts(",");
            else
                puts("");
        }
        puts("  ]");
        puts("}");
    }
    return 0;
}

int eval(Chess *chess) {
    int e = 0;

    for (int i = 0; i < 64; i++) {
        e += Piece_value_at(chess->board[i], i);
    }

    return e;
}

int minimax_captures_only(Chess *chess, clock_t endtime, int depth, int a, int b) {
    int best_score = chess->turn == TURN_WHITE ? eval(chess) : -eval(chess);

    // Stand Pat
    if (depth == 0 || best_score >= b || clock() > endtime) {
        nodes_total++;
        return best_score;
    }
    if (best_score > a) a = best_score;

    Move moves[MAX_LEGAL_MOVES];
    captures_only = true;
    size_t n_moves = Chess_legal_moves_sorted(chess, moves);
    captures_only = false;

    for (int i = 0; i < n_moves; i++) {
        Move *move = &moves[i];
        gamestate_t gamestate = chess->gamestate;
        Piece capture = Chess_make_move(chess, move);

        int score = -minimax_captures_only(chess, endtime, depth - 1, -b, -a);

        Chess_unmake_move(chess, move, capture);
        chess->gamestate = gamestate;

        if (score >= b) return score;
        if (score > best_score) best_score = score;
        if (score > a) a = score;
    }
    return best_score;
}

#define QUIES_DEPTH 5

int minimax(Chess *chess, clock_t endtime, int depth, int a, int b,
            Piece last_capture) {
    if (depth == 0 && last_capture != EMPTY)
        return minimax_captures_only(chess, endtime, QUIES_DEPTH, a, b);

    if (depth == 0 || clock() > endtime) {
        nodes_total++;
        return chess->turn == TURN_WHITE ? eval(chess) : -eval(chess);
    }

    // Check for 3 fold repetition
    if (Chess_3fold_repetition(chess) >= 3) {
        nodes_total++;
        return 0;
    }

    Move moves[MAX_LEGAL_MOVES];
    size_t n_moves = Chess_legal_moves_sorted(chess, moves);

    if (n_moves == 0) {
        nodes_total++;
        if (Chess_friendly_check(chess)) {
            // Checkmate
            return -1000000 - depth;
        } else {
            // draw by stalemate
            return 0;
        }
    }

    for (int i = 0; i < n_moves; i++) {
        Move *move = &moves[i];
        gamestate_t gamestate = chess->gamestate;
        Piece capture = Chess_make_move(chess, move);

        int score = -minimax(chess, endtime, depth - 1, -b, -a, capture);

        Chess_unmake_move(chess, move, capture);
        chess->gamestate = gamestate;

        if (score >= b) return b;
        if (score > a) a = score;
    }
    return a;
}

// if is_white sort in descending order, otherwise ascending
void bubble_sort(Move *moves, int *scores, size_t n_moves) {
    bool swapped;
    do {
        swapped = false;
        for (int i = 1; i < n_moves; i++) {
            if (scores[i - 1] < scores[i]) {
                Move tmp_move = moves[i - 1];
                moves[i - 1] = moves[i];
                moves[i] = tmp_move;
                int tmp_score = scores[i - 1];
                scores[i - 1] = scores[i];
                scores[i] = tmp_score;
                swapped = true;
            }
        }
    } while (swapped);
}

// Play a move given a FEN string
// Returns 0 on success, 1 on error
int play(char *fen, int millis) {
    Chess *chess = Chess_from_fen(fen);
    if (!chess) return 1;
    if (millis < 1) return 1;

    clock_t start = clock();
    clock_t endtime = start + CLOCKS_PER_SEC * (millis) / 1000;

    Move moves[MAX_LEGAL_MOVES];
    int scores[MAX_LEGAL_MOVES];
    size_t n_moves = Chess_legal_moves_sorted(chess, moves);
    if (n_moves < 1) return 1;

    Move *best_move = NULL;
    int best_score = -INF;
    int depth = 1;

    while (clock() < endtime) {
        int current_best_score = -INF;
        Move *current_best_move = NULL;
        nodes_total = 0;

        for (int i = 0; i < n_moves && clock() < endtime; i++) {
            Move *move = &moves[i];
            gamestate_t gamestate = chess->gamestate;
            Piece capture = Chess_make_move(chess, move);

            int score = -minimax(chess, endtime, depth, -INF, INF, capture);
            scores[i] = score;

            Chess_unmake_move(chess, move, capture);
            chess->gamestate = gamestate;

            if (score > current_best_score) {
                current_best_score = score;
                current_best_move = move;
            }
        }

        // If we finished this depth, update best move
        if (clock() < endtime && current_best_move) {
            bubble_sort(moves, scores, n_moves);
            best_score = current_best_score;
            best_move = current_best_move;
            depth++;
        }
    }

    clock_t end = clock();
    double cpu_time = ((double)end - start) / CLOCKS_PER_SEC;
    best_score *= chess->turn == TURN_WHITE ? 1 : -1;

    puts("{");
    printf("  \"scores\": {\n");
    int show_n_scores = (n_moves > 5 ? n_moves : n_moves);
    for (int i = 0; i < show_n_scores; i++) {
        if (i >= show_n_scores - 1) {
            printf("    \"%s\": %.2f\n", Move_string(moves + i),
                   (double)scores[i] / 100);
        } else {
            printf("    \"%s\": %.2f,\n", Move_string(moves + i),
                   (double)scores[i] / 100);
        }
    }
    printf("  },\n");
    printf("  \"millis\": %d,\n", millis);
    printf("  \"depth\": %d,\n", depth);
    printf("  \"time\": %.3lf,\n", cpu_time);
    printf("  \"nodes\": %lu,\n", (unsigned long)nodes_total);
    printf("  \"eval\": %.2f,\n", (double)best_score / 100);
    printf("  \"move\": \"%s\"\n", Move_string(best_move));
    puts("}");

    return 0;
}

int version() {
    printf("SigmaZero Chess Engine 2.0 (2025-09-06)\n");
    return 0;
}

void help(void) {
    printf("Usage: sigma-zero <command>\n");
    printf("Commands:\n");
#define HELP_WIDTH "  %-20s "
    printf(HELP_WIDTH "Show this help message\n", "help");
    printf(HELP_WIDTH "Show version information\n", "version");
    printf(HELP_WIDTH "Show legal moves for the given position\n",
           "moves <FEN> <depth>");
    printf(HELP_WIDTH "Get the evaluation of the given position\n",
           "eval <FEN>");
    printf(HELP_WIDTH "Bot plays a move based on the given position\n",
           "play <FEN> <millis>");
}

int test() {
    Chess *chess = Chess_from_fen(
        "rnbqk2r/ppppnpp1/4P2p/8/1P6/b6N/P1P1PPPP/RNBQKB1R b KQkq - 1 2");
    Position h8 = Position_from_string("h8");
    Position g8 = Position_from_string("g8");
    Move move = (Move){.from = Position_to_index(&h8),
                       .to = Position_to_index(&g8),
                       .promotion = NO_PROMOTION};
    gamestate_t gamestate = chess->gamestate;
    Piece capture = Chess_make_move(chess, &move);
    Chess_dump(chess);
    Chess_print(chess);
    puts("-------------------");
    Chess_unmake_move(chess, &move, capture);
    chess->gamestate = gamestate;
    Chess_dump(chess);
    Chess_print(chess);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2 || strcmp(argv[1], "help") == 0 ||
        strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        help();
        return argc > 1;
    } else if (strcmp(argv[1], "version") == 0 ||
               strcmp(argv[1], "--version") == 0 ||
               strcmp(argv[1], "-v") == 0) {
        return version();
    } else if (strcmp(argv[1], "test") == 0) {
        return test();
    } else if (argc == 4 && strcmp(argv[1], "play") == 0) {
        int millis = atoi(argv[3]);
        return play(argv[2], millis);
    } else if (argc == 4 && strcmp(argv[1], "moves") == 0) {
        int depth = atoi(argv[3]);
        return moves(argv[2], depth);
    } else if (argc == 3 && strcmp(argv[1], "eval") == 0) {
        Chess *chess = Chess_from_fen(argv[2]);
        if (!chess) return 1;
        printf("%f\n", (double)eval(chess) / 100);
        return 0;
    } else {
        help();
        return 1;
    }
}