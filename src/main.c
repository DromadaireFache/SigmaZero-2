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

#include "consts.c"

#define class typedef struct
#define INF 1000000000
#define ISLOWER(c) ((c) >= 'a' && (c) <= 'z')
#define ISUPPER(c) ((c) >= 'A' && (c) <= 'Z')

#ifdef _WIN32
#define TIME_TYPE clock_t
#define TIME_NOW() clock()
#define TIME_DIFF_S(end, start) ((double)((end) - (start)) / CLOCKS_PER_SEC)
#define TIME_PLUS_OFFSET_MS(start, millis) ((start) + CLOCKS_PER_SEC * (millis) / 1000)
#else
__attribute__((always_inline)) static inline uint64_t now_nanos() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_COARSE, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}
#define TIME_TYPE uint64_t
#define TIME_NOW() now_nanos()
#define TIME_DIFF_S(end, start) ((double)((end) - (start)) / 1000000000.0)
#define TIME_PLUS_OFFSET_MS(start, millis) ((start) + ((uint64_t)millis) * 1000000)
#endif

// A bitboard is a 64-bit integer where each bit represents a square on the
// chessboard (from a1 to h8). The least significant bit (LSB) represents a1,
// and the most significant bit (MSB) represents h8.
typedef uint64_t bitboard_t;

extern const bitboard_t ROOK_MAGIC_NUMS[64];
extern const int ROOK_MAGIC_SHIFTS[64];
extern const bitboard_t BISHOP_MAGIC_NUMS[64];
extern const int BISHOP_MAGIC_SHIFTS[64];
extern const bitboard_t* ROOK_MOVES[64];
extern const bitboard_t* BISHOP_MOVES[64];

// Print a bitboard as a 64-character string of 0s and 1s
void bitboard_print(bitboard_t bb) {
    for (int rank = 7; rank >= 0; rank--) {
        for (int file = 0; file < 8; file++) {
            int index = rank * 8 + file;
            putchar((bb >> index) & 1 ? '1' : '0');
            putchar(' ');
        }
        putchar('\n');
    }
}

// Convert an index (0-63) to a bitboard with only that bit set
static inline bitboard_t bitboard_from_index(int i) { return (bitboard_t)1ULL << i; }

static inline int index_col(int index) { return index % 8; }

static inline int index_row(int index) { return index / 8; }

static inline bitboard_t bitboard_row_no_edge(int i) { return 0x7EULL << (i - index_col(i)); }

static inline bitboard_t bitboard_col_no_edge(int i) {
    return 0x0001010101010100ULL << index_col(i);
}

static inline bitboard_t bitboard_rook_mask(int i) {
    return (bitboard_row_no_edge(i) ^ bitboard_col_no_edge(i)) & ~bitboard_from_index(i);
}

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

int Piece_value_at(Piece piece, int i) {
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

int Piece_king_proximity(Piece piece, int i, int white_king, int black_king) {
#define ROW_COL_VALUES(color_king)                   \
    x = abs(index_row(i) - index_row(color_king));   \
    tmp = abs(index_col(i) - index_col(color_king)); \
    y = tmp < x ? tmp : x;                           \
    x = tmp > x ? tmp : x;
    int x, tmp, y;

    switch (piece) {
        // case WHITE_KNIGHT: // TODO
        //     ROW_COL_VALUES(black_king);
        //     return 0;
        // case BLACK_KNIGHT:
        //     ROW_COL_VALUES(white_king);
        //     return 0;
        case WHITE_BISHOP:
            ROW_COL_VALUES(black_king);
            return BISHOP_KING_PROX * 2 * y / ((x + y) * (x + y));
        case BLACK_BISHOP:
            ROW_COL_VALUES(white_king);
            return -BISHOP_KING_PROX * 2 * y / ((x + y) * (x + y));
        case WHITE_ROOK:
            ROW_COL_VALUES(black_king);
            return ROOK_KING_PROX * (x - y) / ((x + y) * (x + y));
        case BLACK_ROOK:
            ROW_COL_VALUES(white_king);
            return -ROOK_KING_PROX * (x - y) / ((x + y) * (x + y));
        case WHITE_QUEEN:
            ROW_COL_VALUES(black_king);
            return QUEEN_KING_PROX / (x + y);
        case BLACK_QUEEN:
            ROW_COL_VALUES(white_king);
            return -QUEEN_KING_PROX / (x + y);
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

static inline bool Piece_is_white(Piece piece) { return ISUPPER(piece); }

static inline bool Piece_is_black(Piece piece) { return ISLOWER(piece); }

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
static inline bool Position_valid(Position* pos) {
    return 0 <= pos->col && pos->col < 8 && 0 <= pos->row && pos->row < 8;
}

// Convert a string (e.g. "e4") to a position
// Returns (Position){-1, -1} if invalid
Position Position_from_string(char* s) {
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
char* Position_to_string(Position* pos) {
    static char buffer[3];
    buffer[0] = 'a' + pos->col;
    buffer[1] = '1' + pos->row;
    buffer[2] = 0;
    return buffer;
}

// Convert a position to an index (0-63)
static inline int Position_to_index(Position* pos) { return pos->row * 8 + pos->col; }

// All positions on the board
const Position positions[] = {
    {0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}, {0, 6}, {0, 7}, {1, 0}, {1, 1}, {1, 2},
    {1, 3}, {1, 4}, {1, 5}, {1, 6}, {1, 7}, {2, 0}, {2, 1}, {2, 2}, {2, 3}, {2, 4}, {2, 5},
    {2, 6}, {2, 7}, {3, 0}, {3, 1}, {3, 2}, {3, 3}, {3, 4}, {3, 5}, {3, 6}, {3, 7}, {4, 0},
    {4, 1}, {4, 2}, {4, 3}, {4, 4}, {4, 5}, {4, 6}, {4, 7}, {5, 0}, {5, 1}, {5, 2}, {5, 3},
    {5, 4}, {5, 5}, {5, 6}, {5, 7}, {6, 0}, {6, 1}, {6, 2}, {6, 3}, {6, 4}, {6, 5}, {6, 6},
    {6, 7}, {7, 0}, {7, 1}, {7, 2}, {7, 3}, {7, 4}, {7, 5}, {7, 6}, {7, 7}};

Position Position_from_index(int index) { return positions[index]; }

// Print a position
void Position_print(Position pos) {
    printf("Position: %s (row: %d, col: %d)\n", Position_to_string(&pos), pos.row, pos.col);
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

char* Move_string(Move* move) {
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

void Move_print(Move* move) { printf("%s\n", Move_string(move)); }

bool Move_equals(Move* move1, Move* move2) {
    return move1->from == move2->from && move1->to == move2->to &&
           move1->promotion == move2->promotion;
}

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

static inline void ZHashStack_push(ZHashStack* zhstack, uint64_t hash) {
    zhstack->hashes[zhstack->sp++] = hash;
}

static inline uint64_t ZHashStack_pop(ZHashStack* zhstack) {
    return zhstack->hashes[--zhstack->sp];
}

static inline uint64_t ZHashStack_peek(ZHashStack* zhstack) {
    return zhstack->hashes[zhstack->sp - 1];
}

class {
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
}
EnemyAttackMap;

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
    uint64_t zhash;         // Current zobrist hash of the position
    EnemyAttackMap enemy_attack_map;
    int eval;                  // Cache a partial value of the eval that doesn't depend on fullmoves
    int pawn_row_sum;          // Sum pawn rows to use in final eval calculation
    bitboard_t bb_white;       // Bitboard of all white pieces
    bitboard_t bb_black;       // Bitboard of all black pieces
    Move killer_moves[2][64];  // Used for move ordering [id][depth]
    int history_table[2][64][64];  // Also used for move ordering [turn][from][to]
}
Chess;

#define BITMASK(nbit) (1 << (nbit))

// Set white kingside castling right
static inline void Chess_castle_K_set(Chess* board, bool allow) {
    if (allow)
        board->gamestate &= ~BITMASK(0);
    else
        board->gamestate |= BITMASK(0);
}

// Set white queenside castling right
static inline void Chess_castle_Q_set(Chess* board, bool allow) {
    if (allow)
        board->gamestate &= ~BITMASK(1);
    else
        board->gamestate |= BITMASK(1);
}

// Set black kingside castling right
static inline void Chess_castle_k_set(Chess* board, bool allow) {
    if (allow)
        board->gamestate &= ~BITMASK(2);
    else
        board->gamestate |= BITMASK(2);
}

// Set black queenside castling right
static inline void Chess_castle_q_set(Chess* board, bool allow) {
    if (allow)
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

// Get en passant column (or -1 if not available)
static inline uint8_t Chess_en_passant(Chess* chess) {
    if (chess->gamestate & BITMASK(4)) {
        return chess->gamestate >> 5;
    } else {
        return -1;
    }
}

// Add a piece to the board at a given position
// Only meant for initializing the board
void Chess_add(Chess* chess, Piece piece, Position pos) {
    if (!Position_valid(&pos)) return;
    int i = Position_to_index(&pos);
    chess->board[i] = piece;
}

void Chess_empty_board(Chess* chess) {
    for (int i = 0; i < 64; i++) {
        chess->board[i] = EMPTY;
    }
    chess->turn = TURN_WHITE;
    chess->gamestate = 0b00001111;  // All castling rights available, no en passant
    chess->halfmoves = 0;
    chess->fullmoves = 1;
    chess->zhstack = (ZHashStack){0};
}

void Chess_find_kings(Chess* chess) {
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
Chess* Chess_new() {
    static Chess* chess;
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
char* Chess_to_string(Chess* chess) {
    static char board_s[65];
    for (int i = 0; i < 64; i++) {
        board_s[i] = chess->board[i];
    }
    board_s[64] = 0;
    return board_s;
}

// Dump the board state (for debugging)
void Chess_dump(Chess* chess) {
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
void Chess_print(Chess* board) {
    char* board_s = Chess_to_string(board);
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            putchar(board_s[(7 - i) * 8 + j]);
            putchar(' ');
        }
        putchar('\n');
    }
}

#define Chess_friendly_piece_at(chess, index)                            \
    ((chess)->turn == TURN_WHITE ? Piece_is_white((chess)->board[index]) \
                                 : Piece_is_black((chess)->board[index]))

#define Chess_enemy_piece_at(chess, index)                               \
    ((chess)->turn == TURN_WHITE ? Piece_is_black((chess)->board[index]) \
                                 : Piece_is_white((chess)->board[index]))

bool Chess_friendly_king_at(Chess* chess, int index) {
    Piece piece = chess->board[index];
    if (chess->turn == TURN_WHITE) {
        return piece == WHITE_KING;
    } else {
        return piece == BLACK_KING;
    }
}

bool Chess_enemy_king_at(Chess* chess, int index) {
    Piece piece = chess->board[index];
    if (chess->turn == TURN_WHITE) {
        return piece == BLACK_KING;
    } else {
        return piece == WHITE_KING;
    }
}

bool Chess_friendly_pawn_at(Chess* chess, int index) {
    Piece piece = chess->board[index];
    if (chess->turn == TURN_WHITE) {
        return piece == WHITE_PAWN;
    } else {
        return piece == BLACK_PAWN;
    }
}

bool Chess_enemy_pawn_at(Chess* chess, int index) {
    Piece piece = chess->board[index];
    if (chess->turn == TURN_WHITE) {
        return piece == BLACK_PAWN;
    } else {
        return piece == WHITE_PAWN;
    }
}

bool Chess_friendly_knight_at(Chess* chess, int index) {
    Piece piece = chess->board[index];
    if (chess->turn == TURN_WHITE) {
        return piece == WHITE_KNIGHT;
    } else {
        return piece == BLACK_KNIGHT;
    }
}

bool Chess_enemy_knight_at(Chess* chess, int index) {
    Piece piece = chess->board[index];
    if (chess->turn == TURN_WHITE) {
        return piece == BLACK_KNIGHT;
    } else {
        return piece == WHITE_KNIGHT;
    }
}

bool Chess_friendly_bishop_at(Chess* chess, int index) {
    Piece piece = chess->board[index];
    if (chess->turn == TURN_WHITE) {
        return piece == WHITE_BISHOP;
    } else {
        return piece == BLACK_BISHOP;
    }
}

bool Chess_enemy_bishop_at(Chess* chess, int index) {
    Piece piece = chess->board[index];
    if (chess->turn == TURN_WHITE) {
        return piece == BLACK_BISHOP;
    } else {
        return piece == WHITE_BISHOP;
    }
}

bool Chess_friendly_rook_at(Chess* chess, int index) {
    Piece piece = chess->board[index];
    if (chess->turn == TURN_WHITE) {
        return piece == WHITE_ROOK;
    } else {
        return piece == BLACK_ROOK;
    }
}

bool Chess_enemy_rook_at(Chess* chess, int index) {
    Piece piece = chess->board[index];
    if (chess->turn == TURN_WHITE) {
        return piece == BLACK_ROOK;
    } else {
        return piece == WHITE_ROOK;
    }
}

bool Chess_friendly_queen_at(Chess* chess, int index) {
    Piece piece = chess->board[index];
    if (chess->turn == TURN_WHITE) {
        return piece == WHITE_QUEEN;
    } else {
        return piece == BLACK_QUEEN;
    }
}

bool Chess_enemy_queen_at(Chess* chess, int index) {
    Piece piece = chess->board[index];
    if (chess->turn == TURN_WHITE) {
        return piece == BLACK_QUEEN;
    } else {
        return piece == WHITE_QUEEN;
    }
}

uint64_t Chess_zhash(Chess* chess) {
    uint64_t hash = ZHASH_STATE[chess->gamestate];
    hash ^= chess->turn == TURN_WHITE ? ZHASH_WHITE : ZHASH_BLACK;
    for (int i = 0; i < 64; i++) {
        Piece piece = chess->board[i];
        hash ^= Piece_zhash_at(piece, i);
    }
    return hash;
}

// Returns the piece that was captured, or EMPTY if no capture
Piece Chess_make_move(Chess* chess, Move* move) {
    Piece moving_piece = chess->board[move->from];
    Piece target_piece = chess->board[move->to];

    bitboard_t from_bb = bitboard_from_index(move->from);
    bitboard_t to_bb = bitboard_from_index(move->to);

    // Update bitboards
    if (chess->turn == TURN_WHITE) {
        chess->bb_white &= ~from_bb;  // Remove from source
        chess->bb_white |= to_bb;     // Add to destination
        chess->bb_black &= ~to_bb;    // Remove captured piece
    } else {
        chess->bb_black &= ~from_bb;
        chess->bb_black |= to_bb;
        chess->bb_white &= ~to_bb;
    }

    // Remove piece from source square
    chess->zhash ^= Piece_zhash_at(moving_piece, move->from);
    chess->eval -= Piece_value_at(moving_piece, move->from);

    // Remove captured piece (if any)
    chess->zhash ^= Piece_zhash_at(target_piece, move->to);
    chess->eval -= Piece_value_at(target_piece, move->to);

    // Update gamestate hash
    chess->zhash ^= ZHASH_STATE[chess->gamestate];

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
        chess->zhash ^= Piece_zhash_at(WHITE_ROOK, 7);
        chess->zhash ^= Piece_zhash_at(WHITE_ROOK, 5);
        chess->eval -= Piece_value_at(WHITE_ROOK, 7);
        chess->eval += Piece_value_at(WHITE_ROOK, 5);
        chess->bb_white &= ~bitboard_from_index(7);  // Remove rook from h1
        chess->bb_white |= bitboard_from_index(5);   // Add rook to f1
    } else if (moving_piece == WHITE_KING && move->from == 4 && move->to == 2) {
        // White queenside
        chess->board[3] = WHITE_ROOK;
        chess->board[0] = EMPTY;
        chess->zhash ^= Piece_zhash_at(WHITE_ROOK, 0);
        chess->zhash ^= Piece_zhash_at(WHITE_ROOK, 3);
        chess->eval -= Piece_value_at(WHITE_ROOK, 0);
        chess->eval += Piece_value_at(WHITE_ROOK, 3);
        chess->bb_white &= ~bitboard_from_index(0);  // Remove rook from a1
        chess->bb_white |= bitboard_from_index(3);   // Add rook to d1
    } else if (moving_piece == BLACK_KING && move->from == 60 && move->to == 62) {
        // Black kingside
        chess->board[61] = BLACK_ROOK;
        chess->board[63] = EMPTY;
        chess->zhash ^= Piece_zhash_at(BLACK_ROOK, 63);
        chess->zhash ^= Piece_zhash_at(BLACK_ROOK, 61);
        chess->eval -= Piece_value_at(BLACK_ROOK, 63);
        chess->eval += Piece_value_at(BLACK_ROOK, 61);
        chess->bb_black &= ~bitboard_from_index(63);  // Remove rook from h8
        chess->bb_black |= bitboard_from_index(61);   // Add rook to f8
    } else if (moving_piece == BLACK_KING && move->from == 60 && move->to == 58) {
        // Black queenside
        chess->board[59] = BLACK_ROOK;
        chess->board[56] = EMPTY;
        chess->zhash ^= Piece_zhash_at(BLACK_ROOK, 56);
        chess->zhash ^= Piece_zhash_at(BLACK_ROOK, 59);
        chess->eval -= Piece_value_at(BLACK_ROOK, 56);
        chess->eval += Piece_value_at(BLACK_ROOK, 59);
        chess->bb_black &= ~bitboard_from_index(56);  // Remove rook from a8
        chess->bb_black |= bitboard_from_index(59);   // Add rook to d8
    }

    // Handle en passant capture
    if (moving_piece == WHITE_PAWN && index_col(move->from) != index_col(move->to) &&
        target_piece == EMPTY) {
        // White pawn capturing en passant
        chess->zhash ^= Piece_zhash_at(BLACK_PAWN, move->to - 8);
        chess->eval -= Piece_value_at(BLACK_PAWN, move->to - 8);
        chess->board[move->to - 8] = EMPTY;
        chess->pawn_row_sum += 2;
        chess->bb_black &= ~bitboard_from_index(move->to - 8);
    } else if (moving_piece == BLACK_PAWN && index_col(move->from) != index_col(move->to) &&
               target_piece == EMPTY) {
        // Black pawn capturing en passant
        chess->zhash ^= Piece_zhash_at(WHITE_PAWN, move->to + 8);
        chess->eval -= Piece_value_at(WHITE_PAWN, move->to + 8);
        chess->board[move->to + 8] = EMPTY;
        chess->pawn_row_sum -= 2;
        chess->bb_white &= ~bitboard_from_index(move->to + 8);
    }

    // Handle promotion and update pawn row sum number
    if (moving_piece == WHITE_PAWN) {
        chess->pawn_row_sum += index_row(move->to - move->from + 1);
        if (target_piece == BLACK_PAWN) chess->pawn_row_sum -= index_row(move->to) - 6;

        switch (move->promotion) {
            case PROMOTE_QUEEN:
                moving_piece = WHITE_QUEEN;
                chess->pawn_row_sum -= index_row(move->to) - 1;
                break;
            case PROMOTE_ROOK:
                moving_piece = WHITE_ROOK;
                chess->pawn_row_sum -= index_row(move->to) - 1;
                break;
            case PROMOTE_BISHOP:
                moving_piece = WHITE_BISHOP;
                chess->pawn_row_sum -= index_row(move->to) - 1;
                break;
            case PROMOTE_KNIGHT:
                moving_piece = WHITE_KNIGHT;
                chess->pawn_row_sum -= index_row(move->to) - 1;
                break;
            default:
                break;
        }
    } else if (moving_piece == BLACK_PAWN) {
        chess->pawn_row_sum += index_row(move->to - move->from - 1);
        if (target_piece == WHITE_PAWN) chess->pawn_row_sum -= index_row(move->to) - 1;

        switch (move->promotion) {
            case PROMOTE_QUEEN:
                moving_piece = BLACK_QUEEN;
                chess->pawn_row_sum -= index_row(move->to) - 6;
                break;
            case PROMOTE_ROOK:
                moving_piece = BLACK_ROOK;
                chess->pawn_row_sum -= index_row(move->to) - 6;
                break;
            case PROMOTE_BISHOP:
                moving_piece = BLACK_BISHOP;
                chess->pawn_row_sum -= index_row(move->to) - 6;
                break;
            case PROMOTE_KNIGHT:
                moving_piece = BLACK_KNIGHT;
                chess->pawn_row_sum -= index_row(move->to) - 6;
                break;
            default:
                break;
        }
    }

    // Switch turn
    chess->zhash ^= ZHASH_WHITE ^ ZHASH_BLACK;
    chess->turn = !chess->turn;

    chess->board[move->to] = moving_piece;
    chess->board[move->from] = EMPTY;

    // Update gamestate in hash
    chess->zhash ^= ZHASH_STATE[chess->gamestate];

    // Add piece to destination square
    chess->zhash ^= Piece_zhash_at(moving_piece, move->to);
    chess->eval += Piece_value_at(moving_piece, move->to);

    // uint64_t hash = Chess_zhash(chess);
    ZHashStack_push(&chess->zhstack, chess->zhash);
    // if (chess->zhash != Chess_zhash(chess)) {
    //     printf("Different zhash! %" PRIx64 " %" PRIx64 "\n", chess->zhash, Chess_zhash(chess));
    //     exit(1);
    // }
    return target_piece;
}

// NEED to reset gamestate manually afterwards
void Chess_unmake_move(Chess* chess, Move* move, Piece capture) {
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
Piece Chess_user_move(Chess* chess, char* move_input) {
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
bool string_isdigit(const char* s) { return s[0] != '\0' && strspn(s, "0123456789") == strlen(s); }

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

void Chess_init_eval(Chess* chess) {
    chess->eval = 0;
    chess->pawn_row_sum = 0;

    for (int i = 0; i < 64; i++) {
        Piece piece = chess->board[i];
        if (piece == EMPTY) continue;

        chess->eval += Piece_value_at(piece, i);

        if (piece == WHITE_PAWN) {
            chess->pawn_row_sum += index_row(i) - 1;
        } else if (piece == BLACK_PAWN) {
            chess->pawn_row_sum += index_row(i) - 6;
        }
    }
}

void Chess_init_bb(Chess* chess) {
    chess->bb_white = 0;
    chess->bb_black = 0;

    for (int i = 0; i < 64; i++) {
        Piece piece = chess->board[i];
        if (piece == EMPTY) continue;

        bitboard_t bit = 1ULL << i;
        if (Piece_is_white(piece)) {
            chess->bb_white |= bit;
        } else {
            chess->bb_black |= bit;
        }
    }
}

Chess* Chess_from_fen(char* fen_arg) {
#define FEN_PARSING_ERROR(details)                                \
    fprintf(stderr, "FEN Parsing error: " details ": %s\n", fen); \
    return NULL

    char fen[128];
    strncpy(fen, fen_arg, sizeof(fen) - 1);

    static Chess chess_struct;
    Chess* board = &chess_struct;  // empty board
    Chess_empty_board(board);

    // Split FEN into fields
    char* fields[6];
    int i = 0;
    char* token = strtok(fen, " ");
    while (token && i < 6) {
        fields[i++] = token;
        token = strtok(NULL, " ");
    }
    if (i < 6) {
        FEN_PARSING_ERROR("Not enough fields in FEN");
    }

    // 1. Piece placement
    Position pos = {.col = 0, .row = 7};
    for (char* c = fields[0]; *c; ++c) {
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
    Chess_init_eval(board);
    Chess_init_bb(board);
    board->zhash = Chess_zhash(board);
    return board;
}

void Chess_print_fen(Chess* chess) {
    // board
    for (int i = 0; i < 8; i++) {
        int empty_counter = 0;
        for (int j = 0; j < 8; j++) {
            int index = 8 * (7 - i) + j;
            if (chess->board[index] == EMPTY) {
                empty_counter++;
            } else {
                if (empty_counter > 0) putchar(empty_counter + '0');
                putchar(chess->board[index]);
                empty_counter = 0;
            }
        }
        if (empty_counter > 0) putchar(empty_counter + '0');
        if (i != 7) putchar('/');
    }

    // active color
    putchar(' ');
    if (chess->turn == TURN_WHITE) {
        putchar('w');
    } else {
        putchar('b');
    }

    // castling rights
    putchar(' ');
    if (!(chess->gamestate & BITMASK(0))) putchar('K');
    if (!(chess->gamestate & BITMASK(1))) putchar('Q');
    if (!(chess->gamestate & BITMASK(2))) putchar('k');
    if (!(chess->gamestate & BITMASK(3))) putchar('q');
    if ((chess->gamestate & BITMASK(0)) && (chess->gamestate & BITMASK(1)) &&
        (chess->gamestate & BITMASK(2)) && (chess->gamestate & BITMASK(3)))
        putchar('-');

    // en passant
    putchar(' ');
    uint8_t ep_col = Chess_en_passant(chess);
    if (ep_col == (uint8_t)-1) {
        putchar('-');
    } else {
        putchar('a' + ep_col);
        if (chess->turn == TURN_WHITE) {
            putchar('6');
        } else {
            putchar('3');
        }
    }

    // halfmove clock
    printf(" %d", chess->halfmoves);

    // fullmove number
    printf(" %d", chess->fullmoves);

    putchar('\n');
}

void ZHashStack_game_history(ZHashStack* zhstack, char* _game_history) {
    char* game_history = calloc(strlen(_game_history) + 1, sizeof(char));
    strcpy(game_history, _game_history);

    char* saveptr;
    char* fen = strtok_r(game_history, ",", &saveptr);

    while (fen) {
        Chess* chess = Chess_from_fen(fen);
        if (!chess) return;
        uint64_t hash = Chess_zhash(chess);
        ZHashStack_push(zhstack, hash);

        fen = strtok_r(NULL, ",", &saveptr);
    }

    free(game_history);
}

static inline uint8_t Chess_friendly_king_i(Chess* chess) {
    return chess->turn == TURN_WHITE ? chess->king_white : chess->king_black;
}

static inline uint8_t Chess_enemy_king_i(Chess* chess) {
    return chess->turn == TURN_WHITE ? chess->king_black : chess->king_white;
}

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

bool Chess_square_available(Chess* chess, int index, bool captures_only) {
    return captures_only ? Chess_enemy_piece_at(chess, index)
                         : !Chess_friendly_piece_at(chess, index);
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

const bitboard_t BISHOP_MASKS[64] = {
    0x0040201008040200ULL, 0x0000402010080400ULL, 0x0000004020100a00ULL, 0x0000000040221400ULL,
    0x0000000002442800ULL, 0x0000000204085000ULL, 0x0000020408102000ULL, 0x0002040810204000ULL,
    0x0020100804020000ULL, 0x0040201008040000ULL, 0x00004020100a0000ULL, 0x0000004022140000ULL,
    0x0000000244280000ULL, 0x0000020408500000ULL, 0x0002040810200000ULL, 0x0004081020400000ULL,
    0x0010080402000200ULL, 0x0020100804000400ULL, 0x004020100a000a00ULL, 0x0000402214001400ULL,
    0x0000024428002800ULL, 0x0002040850005000ULL, 0x0004081020002000ULL, 0x0008102040004000ULL,
    0x0008040200020400ULL, 0x0010080400040800ULL, 0x0020100a000a1000ULL, 0x0040221400142200ULL,
    0x0002442800284400ULL, 0x0004085000500800ULL, 0x0008102000201000ULL, 0x0010204000402000ULL,
    0x0004020002040800ULL, 0x0008040004081000ULL, 0x00100a000a102000ULL, 0x0022140014224000ULL,
    0x0044280028440200ULL, 0x0008500050080400ULL, 0x0010200020100800ULL, 0x0020400040201000ULL,
    0x0002000204081000ULL, 0x0004000408102000ULL, 0x000a000a10204000ULL, 0x0014001422400000ULL,
    0x0028002844020000ULL, 0x0050005008040200ULL, 0x0020002010080400ULL, 0x0040004020100800ULL,
    0x0000020408102000ULL, 0x0000040810204000ULL, 0x00000a1020400000ULL, 0x0000142240000000ULL,
    0x0000284402000000ULL, 0x0000500804020000ULL, 0x0000201008040200ULL, 0x0000402010080400ULL,
    0x0002040810204000ULL, 0x0004081020400000ULL, 0x000a102040000000ULL, 0x0014224000000000ULL,
    0x0028440200000000ULL, 0x0050080402000000ULL, 0x0020100804020000ULL, 0x0040201008040200ULL};

bitboard_t bitboard_bishop_mask(int i) { return BISHOP_MASKS[i]; }

__attribute__((always_inline)) static inline size_t  //
Chess_sliding_piece_moves(Chess* chess, Move* move, int from, bool captures_only,
                          bitboard_t (*bitboard_piece_mask)(int), const bitboard_t MAGIC_NUMS[64],
                          const int MAGIC_SHIFTS[64], const bitboard_t* MOVES[64]) {
    EnemyAttackMap* eam = &chess->enemy_attack_map;

    bitboard_t piece_mask = bitboard_piece_mask(from);
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
    return Chess_sliding_piece_moves(chess, move, from, captures_only, bitboard_bishop_mask,
                                     BISHOP_MAGIC_NUMS, BISHOP_MAGIC_SHIFTS, BISHOP_MOVES);
}

size_t Chess_rook_moves(Chess* chess, Move* move, int from, bool captures_only) {
    return Chess_sliding_piece_moves(chess, move, from, captures_only, bitboard_rook_mask,
                                     ROOK_MAGIC_NUMS, ROOK_MAGIC_SHIFTS, ROOK_MOVES);
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
    if (at_en_passant_rank && en_passant_col != -1) {
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
    if (__builtin_expect(chess->enemy_attack_map.n_checks >= 2, 0)) {
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

void Chess_score_move(Chess* chess, Move* move) {
    // Give very high scores to promotions
    if (move->promotion == PROMOTE_QUEEN) {
        move->score = PROMOTION_MOVE_SCORE;
        return;
    }

    Piece aggressor = chess->board[move->from];
    Piece victim = chess->board[move->to];

    // MVV - LVA
    if (victim != EMPTY) {
        if (chess->turn == TURN_WHITE) {
            move->score = -Piece_value(aggressor) - Piece_value(victim);
        } else {
            move->score = Piece_value(aggressor) + Piece_value(victim);
        }
    } else {
        // Deduct points if attacked by enemy pawns
#define ATTACKED_BY_ENEMY_PAWN(condition, offset, pawn)               \
    if ((condition) && chess->board[move->to + (offset)] == (pawn)) { \
        move->score = -abs(Piece_value(aggressor));                   \
        return;                                                       \
    }
        Position pos = Position_from_index(move->to);

        if (chess->turn == TURN_WHITE && aggressor != WHITE_PAWN) {
            ATTACKED_BY_ENEMY_PAWN(pos.row < 6 && pos.col < 7, 9, BLACK_PAWN)
            ATTACKED_BY_ENEMY_PAWN(pos.row < 6 && pos.col > 0, 7, BLACK_PAWN)
        } else if (chess->turn == TURN_BLACK && aggressor != BLACK_PAWN) {
            ATTACKED_BY_ENEMY_PAWN(pos.row > 1 && pos.col < 7, -7, WHITE_PAWN)
            ATTACKED_BY_ENEMY_PAWN(pos.row > 1 && pos.col > 0, -9, WHITE_PAWN)
        }

        // Not attacked by enemy pawns
        move->score = 0;
    }
}

int compare_moves(const void* a, const void* b) {
    const Move* ma = (const Move*)a;
    const Move* mb = (const Move*)b;
    return mb->score - ma->score;
}

// Partial sort - only find N best moves
void partial_sort_moves(Move* moves, size_t n_moves, size_t n_best) {
    if (n_best > n_moves) n_best = n_moves;

    for (int i = 0; i < n_best; i++) {
        int best_idx = i;
        for (int j = i + 1; j < n_moves; j++) {
            if (moves[j].score > moves[best_idx].score) {
                best_idx = j;
            }
        }
        if (best_idx != i) {
            Move temp = moves[i];
            moves[i] = moves[best_idx];
            moves[best_idx] = temp;
        }
    }
}

size_t Chess_legal_moves_scored(Chess* chess, Move* moves, bool captures_only) {
    size_t n_moves = Chess_legal_moves(chess, moves, captures_only);

    // Give a score to each move
    for (int i = 0; i < n_moves; i++) {
        Chess_score_move(chess, &moves[i]);
    }

    return n_moves;
}

static inline void select_best_move(Move* moves, int start, int n_moves) {
    int best = start;
    for (int i = start + 1; i < n_moves; i++) {
        if (moves[i].score > moves[best].score) {
            best = i;
        }
    }

    if (best != start) {
        Move temp = moves[start];
        moves[start] = moves[best];
        moves[best] = temp;
    }
}

bool Chess_equal(Chess* chess, char* board_fen) {
    for (int i = 0; i < 64; i++) {
        if (chess->board[i] != board_fen[i]) return false;
    }
    return true;
}

size_t Chess_count_moves(Chess* chess, int depth) {
    if (depth == 0) return 1;

    Move moves[MAX_LEGAL_MOVES];
    size_t n_moves = Chess_legal_moves(chess, moves, false);
    // if (n_moves == 32 && chess->board[54] == WHITE_BISHOP) {
    //     printf("%lu ", (unsigned long)n_moves);
    //     Chess_print_fen(chess);
    //     Chess_print(chess);
    //     Chess_dump(chess);
    // }
    if (depth == 1) return n_moves;

    size_t nodes = 0;
    for (int i = 0; i < n_moves; i++) {
        gamestate_t gamestate = chess->gamestate;
        uint64_t hash = chess->zhash;
        bitboard_t bb_white = chess->bb_white;
        bitboard_t bb_black = chess->bb_black;
        Piece capture = Chess_make_move(chess, &moves[i]);
        nodes += Chess_count_moves(chess, depth - 1);
        Chess_unmake_move(chess, &moves[i], capture);
        chess->gamestate = gamestate;
        chess->zhash = hash;
        chess->bb_white = bb_white;
        chess->bb_black = bb_black;
    }

    return nodes;
}

int Chess_3fold_repetition(Chess* chess) {
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
    TIME_TYPE endtime;
    int* score;
    bool* search_cancelled;
}
ChessThread;

void* Chess_count_moves_thread(void* arg_void) {
    ChessThread* arg = (ChessThread*)arg_void;
    Chess* chess = &arg->chess;
    int depth = arg->depth;
    Move* move = &arg->move;

    Chess_make_move(chess, move);
    size_t nodes = Chess_count_moves(chess, depth - 1);

    pthread_mutex_lock(&lock);    // enter critical section
    nodes_total += nodes;         // update shared variable
    pthread_mutex_unlock(&lock);  // leave critical section

    return NULL;
}

size_t Chess_count_moves_multi(Chess* chess, int depth) {
    Move moves[MAX_LEGAL_MOVES];
    size_t n_moves = Chess_legal_moves(chess, moves, false);
    nodes_total = 0;

    pthread_t* threads = calloc(n_moves, sizeof(pthread_t));
    ChessThread* args = calloc(n_moves, sizeof(ChessThread));

    // Initialize the mutex
    if (pthread_mutex_init(&lock, NULL) != 0) {
        perror("mutex init failed");
        return 1;
    }

    // Create threads
    for (int i = 0; i < n_moves; i++) {
        ChessThread* arg = &args[i];
        arg->depth = depth;
        memcpy(&arg->chess, chess, sizeof(Chess));
        memcpy(&arg->move, &moves[i], sizeof(Move));

        if (pthread_create(&threads[i], NULL, Chess_count_moves_thread, arg) != 0) {
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

// Transposition table
typedef enum { TT_EXACT, TT_LOWER, TT_UPPER } TTNodeType;

class {
    uint64_t key;
    int eval;
    uint8_t depth;
    uint8_t type;  // TTNodeType
}
TTItem;

// Will give ~100MB array
#define TT_LENGTH (1 << 22)

// Transposition table array
TTItem tt[TT_LENGTH] = {0};

// Store an entry with fine-grained locking
void TT_store(uint64_t key, int eval, int depth, TTNodeType node_type) {
    size_t i = key & (TT_LENGTH - 1);
    TTItem* item = &tt[i];

    if (depth > item->depth) {
        item->key = key;
        item->eval = eval;
        item->depth = depth;
        item->type = node_type;
    }
}

// Retrieve an entry with fine-grained locking
bool TT_get(uint64_t key, int* eval_p, int depth, int a, int b) {
    size_t i = key & (TT_LENGTH - 1);
    TTItem* item = &tt[i];

    if (item->key == key && depth <= item->depth &&
        ((item->type == TT_EXACT) || (item->type == TT_LOWER && item->eval >= b) ||
         (item->type == TT_UPPER && item->eval <= a))) {
        *eval_p = item->eval;
        return true;
    }

    return false;
}

void TT_occupancy(void) {
    const size_t tt_size = sizeof(tt) / 1024 / 1024;
    size_t tt_use = 0;

    for (int i = 0; i < TT_LENGTH; i++) {
        TTItem item = tt[i];
        if (i == (item.key & (TT_LENGTH - 1))) tt_use++;
    }

    double tt_use_pc = (double)tt_use * 100.0 / TT_LENGTH;
    printf("Transposition table (%.2lf%% of %luMB)\n", tt_use_pc, (unsigned long)tt_size);
}

int moves(char* fen, int depth) {
    Chess* chess = Chess_from_fen(fen);
    if (!chess) return 1;
    if (depth > 1) {
        TIME_TYPE start = TIME_NOW();
        size_t n_moves = Chess_count_moves_multi(chess, depth);
        TIME_TYPE end = TIME_NOW();
        double cpu_time = TIME_DIFF_S(end, start);
        double nps = cpu_time > 0.0 ? n_moves / cpu_time : -0.0;
        puts("{");
        printf("  \"depth\": %d,\n", depth);
        printf("  \"nodes\": %lu,\n", (unsigned long)n_moves);
        printf("  \"time\": %.3lf,\n", cpu_time);
        printf("  \"nps\": %.3lf\n", nps);
        puts("}");
    } else {
        Move moves[MAX_LEGAL_MOVES];
        size_t n_moves = Chess_legal_moves(chess, moves, false);
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

    return e;
}

int minimax_captures_only(Chess* chess, TIME_TYPE endtime, int depth, int a, int b) {
    int best_score = chess->turn == TURN_WHITE ? eval(chess) : -eval(chess);

    // Stand Pat
    if (depth == 0 || best_score >= b) {
        // nodes_total++;
        return best_score;
    }
    if (best_score > a) a = best_score;

    Move moves[MAX_LEGAL_MOVES];
    size_t n_moves = Chess_legal_moves_scored(chess, moves, true);

    for (int i = 0; i < n_moves; i++) {
        if (i < 8) select_best_move(moves, i, n_moves);
        Move* move = &moves[i];

        gamestate_t gamestate = chess->gamestate;
        uint64_t hash = chess->zhash;
        int e = chess->eval;
        int pawn_row_sum = chess->pawn_row_sum;
        bitboard_t bb_white = chess->bb_white;
        bitboard_t bb_black = chess->bb_black;
        Piece capture = Chess_make_move(chess, move);

        int score = -minimax_captures_only(chess, endtime, depth - 1, -b, -a);

        Chess_unmake_move(chess, move, capture);
        chess->gamestate = gamestate;
        chess->zhash = hash;
        chess->eval = e;
        chess->pawn_row_sum = pawn_row_sum;
        chess->bb_white = bb_white;
        chess->bb_black = bb_black;

        if (score >= b) return score;
        if (score > best_score) best_score = score;
        if (score > a) a = score;
    }
    return best_score;
}

int minimax(Chess* chess, TIME_TYPE endtime, int depth, int a, int b, Piece last_capture,
            int extensions) {
    if (depth == 0 && last_capture != EMPTY) {
        return minimax_captures_only(chess, endtime, QUIES_DEPTH, a, b);
    }

    // Look for existing eval in transposition table
    uint64_t hash = ZHashStack_peek(&chess->zhstack);
    int tt_eval;
    if (TT_get(hash, &tt_eval, depth, a, b)) {
        return tt_eval;
    }

#define RETURN_AND_STORE_TT(e, node_type)         \
    int evaluation = (e);                         \
    TT_store(hash, evaluation, depth, node_type); \
    return evaluation;

    // Extend search if in check, otherwise don't
    if (depth == 0) {
        if (extensions < MAX_EXTENSION && Chess_friendly_check(chess)) {
            depth++;
            extensions++;
        } else {
            RETURN_AND_STORE_TT(chess->turn == TURN_WHITE ? eval(chess) : -eval(chess), TT_EXACT)
        }
    }

    // Time cutoff
    if (TIME_NOW() > endtime) return 0;

    // Check for 3 fold repetition
    if (Chess_3fold_repetition(chess) >= 3) {
        // nodes_total++;
        return 0;
    }

    Move moves[MAX_LEGAL_MOVES];
    size_t n_moves = Chess_legal_moves_scored(chess, moves, false);

    if (n_moves == 0) {
        bool in_check = chess->enemy_attack_map.n_checks > 0;
        if (in_check) {
            // Checkmate
            RETURN_AND_STORE_TT(-1000000 - depth, TT_EXACT)
        } else {
            // draw by stalemate
            RETURN_AND_STORE_TT(0, TT_EXACT)
        }
    }

    // Add score for killer move heuristic
    // for (int i = 0; i < n_moves; i++) {
    //     Move* move = &moves[i];
    //     int bonus = chess->history_table[chess->turn][move->from][move->to];

    //     // Check both killer slots
    //     if ((chess->killer_moves[0][depth].from == move->from &&
    //          chess->killer_moves[0][depth].to == move->to) ||
    //         (chess->killer_moves[1][depth].from == move->from &&
    //          chess->killer_moves[1][depth].to == move->to)) {
    //         bonus += KILLER_MOVE_BONUS;
    //     }

    //     // if (bonus != 0) printf("bonus: %d\n", bonus);
    //     move->score += bonus;
    // }

    int original_a = a;
    int original_b = b;
    int best_score = -INF;
    for (int i = 0; i < n_moves; i++) {
        if (i < 8) select_best_move(moves, i, n_moves);
        Move* move = &moves[i];

        gamestate_t gamestate = chess->gamestate;
        uint64_t hash = chess->zhash;
        int e = chess->eval;
        int pawn_row_sum = chess->pawn_row_sum;
        bitboard_t bb_white = chess->bb_white;
        bitboard_t bb_black = chess->bb_black;
        Piece capture = Chess_make_move(chess, move);

        int score = -minimax(chess, endtime, depth - 1, -b, -a, capture, extensions);

        Chess_unmake_move(chess, move, capture);
        chess->gamestate = gamestate;
        chess->zhash = hash;
        chess->eval = e;
        chess->pawn_row_sum = pawn_row_sum;
        chess->bb_white = bb_white;
        chess->bb_black = bb_black;

        if (score > best_score) {
            best_score = score;
            if (score > a) a = score;
        }
        if (score >= b) {
            // if (capture == EMPTY) {
            //     // Shift killers: move primary to secondary, new move to primary
            //     if (!(chess->killer_moves[0][depth].from == move->from &&
            //           chess->killer_moves[0][depth].to == move->to)) {
            //         chess->killer_moves[1][depth] = chess->killer_moves[0][depth];
            //         chess->killer_moves[0][depth] = *move;
            //     }

            //     // Cap history table to prevent overflow
            //     if (chess->history_table[chess->turn][move->from][move->to] < 10000) {
            //         chess->history_table[chess->turn][move->from][move->to] += depth * depth;
            //     }
            // }
            break;
        }
    }

    TTNodeType node_type;
    if (best_score <= original_a)
        node_type = TT_UPPER;  // Failed low
    else if (best_score >= original_b)
        node_type = TT_LOWER;  // Failed high
    else
        node_type = TT_EXACT;
    RETURN_AND_STORE_TT(best_score, node_type)
}

// if is_white sort in descending order, otherwise ascending
void bubble_sort(Move* moves, int* scores, size_t n_moves) {
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

void* play_thread(void* arg_void) {
    ChessThread* arg = (ChessThread*)arg_void;
    Chess* chess = &arg->chess;
    TIME_TYPE endtime = arg->endtime;
    Move* move = &arg->move;
    int depth = arg->depth;
    memset(chess->killer_moves, 0, sizeof(chess->killer_moves));
    memset(chess->history_table, 0, sizeof(chess->history_table));
    Piece capture = Chess_make_move(chess, move);

    int score = -minimax(chess, endtime, depth, -INF, INF, capture, 0);
    *arg->search_cancelled = TIME_NOW() > endtime;
    *arg->score = score;

    return NULL;
}

bool openings_db(Chess* chess) {
    char s[100];
    sprintf(s, "%" PRIx64, Chess_zhash(chess));
    srand((unsigned int)time(NULL));

    // Opening the database file openings.db
    // The file should contain lines of the form:
    // <hash>,<n_options>,<option1>,<option2>,...
    // options are in UCI format, e.g. e2e4, g1f3, etc.
    FILE* file = fopen("openings.db", "r");
    if (!file) return false;

    char line[256];
    while (fgets(line, sizeof(line), file)) {
        if (line[0] == '#') continue;  // skip comments
        char* hash_str = strtok(line, ",");
        if (!hash_str) continue;
        if (strcmp(hash_str, s) != 0) continue;

        // Found the hash, now get the number of options
        char* n_options_str = strtok(NULL, ",");
        if (!n_options_str) continue;

        int n_options = atoi(n_options_str);
        if (n_options < 1) continue;

        // Get a random option
        int option_index = rand() % n_options;
        char* move_str = NULL;

        for (int i = 0; i < option_index + 1; i++) {
            move_str = strtok(NULL, ",");  // skip to the chosen option
        }
        if (!move_str) continue;

        // Remove trailing newline
        move_str[strcspn(move_str, "\n")] = 0;

        puts("{");
        printf("  \"scores\": {\n");
        printf("    \"%s\": 0.00\n", move_str);
        printf("  },\n");
        printf("  \"millis\": 0,\n");
        printf("  \"depth\": 0,\n");
        printf("  \"time\": %.3lf,\n", 0.0);
        printf("  \"eval\": 0.00,\n");
        printf("  \"move\": \"%s\"\n", move_str);
        puts("}");
        fclose(file);
        return true;
    }

    fclose(file);
    return false;
}

// Play a move given a FEN string
// Returns 0 on success, 1 on error
int play(char* fen, int millis, char* game_history, bool fancy) {
    ZHashStack zhstack = {0};
    if (game_history != NULL) {
        ZHashStack_game_history(&zhstack, game_history);
    }

    Chess* chess = Chess_from_fen(fen);
    if (!chess) return 1;
    if (millis < 1) return 1;
    memcpy(&chess->zhstack, &zhstack, sizeof(ZHashStack));

    if (chess->fullmoves <= 5 && openings_db(chess)) {
        return 0;
    }

    TIME_TYPE start = TIME_NOW();
    TIME_TYPE endtime = TIME_PLUS_OFFSET_MS(start, millis);
    Move moves[MAX_LEGAL_MOVES];
    int scores[MAX_LEGAL_MOVES];
    bool search_cancelled[MAX_LEGAL_MOVES];
    Move moves_at_depth2[MAX_LEGAL_MOVES];
    int scores_at_depth2[MAX_LEGAL_MOVES];
    size_t n_moves = Chess_legal_moves_scored(chess, moves, false);
    if (n_moves < 1) return 1;

    Move* best_move = NULL;
    int best_score = -INF;
    int depth = 1;

    pthread_t* threads = calloc(n_moves, sizeof(pthread_t));
    ChessThread* args = calloc(n_moves, sizeof(ChessThread));

    while (TIME_NOW() < endtime) {
        // nodes_total = 0;

        for (int i = 0; i < n_moves; i++) {
            ChessThread* arg = &args[i];
            memcpy(&arg->chess, chess, sizeof(Chess));
            memcpy(&arg->move, &moves[i], sizeof(Move));
            arg->endtime = endtime;
            arg->depth = depth;
            arg->score = &scores[i];

            // Use this in case the search stops because of time limit to know if we can use this
            // score
            arg->search_cancelled = &search_cancelled[i];

            if (pthread_create(&threads[i], NULL, play_thread, arg) != 0) {
                perror("pthread_create failed");
                return 1;
            }
        }

        // Wait for threads to finish
        for (int i = 0; i < n_moves; i++) {
            pthread_join(threads[i], NULL);
        }

        if (fancy && depth == 2) {
            memcpy(scores_at_depth2, scores, sizeof(scores));
            memcpy(moves_at_depth2, moves, sizeof(moves));
        }

        // If we finished this depth, update best move
        if (TIME_NOW() < endtime) {
            bubble_sort(moves, scores, n_moves);
            best_score = scores[0];
            best_move = &moves[0];

            // Give more points to a move if there is a large difference with depth 2 score
            if (fancy && depth > 2) {
                for (int i = 0; i < n_moves / 2; i++) {
                    if (scores[i] <= 0 || scores[i] > 500) continue;
                    int score_depth2 = scores[i];

                    for (int j = 0; j < n_moves; j++) {
                        if (Move_equals(&moves[i], &moves_at_depth2[j])) {
                            score_depth2 = scores_at_depth2[j];
                            break;
                        }
                    }

                    int improvement = scores[i] - score_depth2;
                    scores[i] += improvement / 2;
                }

                bubble_sort(moves, scores, n_moves);
                best_move = &moves[0];
            }

            depth++;

        } else if (!search_cancelled[0]) {
            // In case search was cancelled, try using results from this iteration
            // Give a null score to moves that didn't complete search
            for (int i = 1; i < n_moves; i++) {
                if (search_cancelled[i]) scores[i] = -INF;
            }

            // Partial sort for the best move at this depth
            bubble_sort(moves, scores, n_moves);
            best_score = scores[0];
            best_move = &moves[0];
        }

        // Show transposition table occupancy
        // TT_occupancy();
    }

    free(threads);
    free(args);

    TIME_TYPE end = TIME_NOW();
    double cpu_time = TIME_DIFF_S(end, start);
    best_score *= chess->turn == TURN_WHITE ? 1 : -1;

    puts("{");
    printf("  \"scores\": {\n");
    for (int i = 0; i < n_moves; i++) {
        if (i >= n_moves - 1) {
            printf("    \"%s\": %.2f\n", Move_string(moves + i), (double)scores[i] / 100);
        } else {
            printf("    \"%s\": %.2f,\n", Move_string(moves + i), (double)scores[i] / 100);
        }
    }
    printf("  },\n");
    printf("  \"millis\": %d,\n", millis);
    printf("  \"depth\": %d,\n", depth);
    printf("  \"time\": %.3lf,\n", cpu_time);
    // printf("  \"nodes\": %lu,\n", (unsigned long)nodes_total);
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
    printf(HELP_WIDTH "Show legal moves for the given position\n", "moves <FEN> <depth>");
    printf(HELP_WIDTH "Get the evaluation of the given position\n", "eval <FEN>");
    printf(HELP_WIDTH "Bot plays a move based on the given position\n", "play <FEN> <millis>");
}

int king_safety_command(Chess* chess) {
    int white_score = 0, black_score = 0;

    for (int i = 0; i < 64; i++) {
        Piece piece = chess->board[i];
        if (piece == EMPTY) continue;

        if (Piece_is_white(piece)) {
            white_score += Piece_king_proximity(piece, i, chess->king_white, chess->king_black);
        } else {
            black_score -= Piece_king_proximity(piece, i, chess->king_white, chess->king_black);
        }
    }

    printf("White king danger score: %d\n", black_score);
    printf("Black king danger score: %d\n", white_score);
    return 0;
}

int move_scores_command(Chess* chess) {
    Move moves[MAX_LEGAL_MOVES];
    size_t n_moves = Chess_legal_moves_scored(chess, moves, false);
    bool elipses = false;

    for (int i = 0; i < n_moves; i++) {
        select_best_move(moves, i, n_moves);
        if (moves[i].score != 0) {
            printf("%-5s %6d\n", Move_string(moves + i), moves[i].score);
        } else if (!elipses) {
            printf("...   %6d\n", 0);
            elipses = true;
        }
    }
    return 0;
}

int test() {
    Chess* chess = Chess_from_fen("8/1k6/8/4R3/8/8/4K3/8 w - - 0 1");
    if (!chess) return 1;

    Chess_fill_attack_map(chess);
    Move moves[MAX_LEGAL_MOVES];

    size_t n_moves = Chess_rook_moves(chess, moves, 36, false);

    printf("%lu moves\n", (unsigned long)n_moves);
    for (int i = 0; i < n_moves; i++) {
        Move_print(&moves[i]);
    }

    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2 || strcmp(argv[1], "help") == 0 || strcmp(argv[1], "--help") == 0 ||
        strcmp(argv[1], "-h") == 0) {
        help();
        return argc > 1;
    } else if (strcmp(argv[1], "version") == 0 || strcmp(argv[1], "--version") == 0 ||
               strcmp(argv[1], "-v") == 0) {
        return version();
    } else if (strcmp(argv[1], "test") == 0) {
        return test();
    } else if ((argc == 4 || argc == 5) && strcmp(argv[1], "play") == 0) {
        int millis = atoi(argv[3]);
        if (argc == 4) {
            return play(argv[2], millis, NULL, false);
        } else {
            return play(argv[2], millis, argv[4], false);
        }
    } else if ((argc == 4 || argc == 5) && strcmp(argv[1], "fancy") == 0) {
        int millis = atoi(argv[3]);
        if (argc == 4) {
            return play(argv[2], millis, NULL, true);
        } else {
            return play(argv[2], millis, argv[4], true);
        }
    } else if (argc == 4 && strcmp(argv[1], "moves") == 0) {
        int depth = atoi(argv[3]);
        return moves(argv[2], depth);
    } else if (argc == 3 && strcmp(argv[1], "eval") == 0) {
        Chess* chess = Chess_from_fen(argv[2]);
        if (!chess) return 1;
        printf("%f\n", (double)eval(chess) / 100);
        return 0;
    } else if (argc == 3 && strcmp(argv[1], "hash") == 0) {
        Chess* chess = Chess_from_fen(argv[2]);
        if (!chess) return 1;
        printf("%" PRIx64 "\n", Chess_zhash(chess));
        return 0;
    } else if (argc == 3 && strcmp(argv[1], "kingsafety") == 0) {
        Chess* chess = Chess_from_fen(argv[2]);
        if (!chess) return 1;
        return king_safety_command(chess);
    } else if (argc == 3 && strcmp(argv[1], "scores") == 0) {
        Chess* chess = Chess_from_fen(argv[2]);
        if (!chess) return 1;
        return move_scores_command(chess);
    } else {
        help();
        return 1;
    }
}