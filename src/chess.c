#include "chess.h"

#include <stdlib.h>

#include "move.h"
#include "piece.h"

// Returns the piece that was captured, or EMPTY if no capture
Piece Chess_make_move(Chess* chess, Move* move) {
    Piece moving_piece = chess->board[move->from];
    Piece target_piece = chess->board[move->to];

    // Remove piece from source square
    Chess_remove(chess, move->from);

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
    } else if (moving_piece == BLACK_KING) {
        Chess_castle_k_set(chess, false);
        Chess_castle_q_set(chess, false);
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
        Chess_remove(chess, 7);           // Remove rook from h1
        Chess_add(chess, WHITE_ROOK, 5);  // Add rook to f1
        chess->white_has_castled = true;
    } else if (moving_piece == WHITE_KING && move->from == 4 && move->to == 2) {
        // White queenside
        Chess_remove(chess, 0);           // Remove rook from a1
        Chess_add(chess, WHITE_ROOK, 3);  // Add rook to d1
        chess->white_has_castled = true;
    } else if (moving_piece == BLACK_KING && move->from == 60 && move->to == 62) {
        // Black kingside
        Chess_remove(chess, 63);           // Remove rook from h8
        Chess_add(chess, BLACK_ROOK, 61);  // Add rook to f8
        chess->black_has_castled = true;
    } else if (moving_piece == BLACK_KING && move->from == 60 && move->to == 58) {
        // Black queenside
        Chess_remove(chess, 56);           // Remove rook from a8
        Chess_add(chess, BLACK_ROOK, 59);  // Add rook to d8
        chess->black_has_castled = true;
    }

    // Handle en passant capture
    if (moving_piece == WHITE_PAWN && index_col(move->from) != index_col(move->to) &&
        target_piece == EMPTY) {
        // White pawn capturing en passant
        Chess_remove(chess, move->to - 8);
    } else if (moving_piece == BLACK_PAWN && index_col(move->from) != index_col(move->to) &&
               target_piece == EMPTY) {
        // Black pawn capturing en passant
        Chess_remove(chess, move->to + 8);
    }

    // Handle promotion and update pawn row sum number
    if (moving_piece == WHITE_PAWN) {
        // chess->pawn_row_sum += index_row(move->to - move->from + 1);
        // if (target_piece == BLACK_PAWN) chess->pawn_row_sum -= index_row(move->to) - 6;

        // if (move->promotion != NO_PROMOTION) {
        //     chess->pawn_row_sum -= index_row(move->to) - 1;
        // }

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
    } else if (moving_piece == BLACK_PAWN) {
        // chess->pawn_row_sum += index_row(move->to - move->from - 1);
        // if (target_piece == WHITE_PAWN) chess->pawn_row_sum -= index_row(move->to) - 1;

        // if (move->promotion != NO_PROMOTION) {
        //     chess->pawn_row_sum -= index_row(move->to) - 6;
        // }

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

    // Switch turn
    chess->zhash ^= ZHASH_WHITE ^ ZHASH_BLACK;
    chess->turn = !chess->turn;

    // Add piece to destination square (also remove target piece if there is one)
    Chess_add(chess, moving_piece, move->to);

    // Update gamestate in hash
    chess->zhash ^= ZHASH_STATE[chess->gamestate];
    ZHashStack_push(&chess->zhstack, chess->zhash);

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

    // Add the captured piece back to the board (this will also remove the piece from the 'to' square)
    Chess_add(chess, capture, move->to);

    // Put the moving piece back to its original square
    Chess_add(chess, moving_piece, move->from);

    if (Piece_is_king(moving_piece)) {
        // castling
        if (move->from == 4 && move->to == 6) {
            // White kingside
            Chess_remove(chess, 5);           // Remove rook from f1
            Chess_add(chess, WHITE_ROOK, 7);  // Add rook to h1
            chess->white_has_castled = false;
        } else if (move->from == 4 && move->to == 2) {
            // White queenside
            Chess_remove(chess, 3);           // Remove rook from d1
            Chess_add(chess, WHITE_ROOK, 0);  // Add rook to a1
            chess->white_has_castled = false;
        } else if (move->from == 60 && move->to == 62) {
            // Black kingside
            Chess_remove(chess, 61);           // Remove rook from f8
            Chess_add(chess, BLACK_ROOK, 63);  // Add rook to h8
            chess->black_has_castled = false;
        } else if (move->from == 60 && move->to == 58) {
            // Black queenside
            Chess_remove(chess, 59);           // Remove rook from d8
            Chess_add(chess, BLACK_ROOK, 56);  // Add rook to a8
            chess->black_has_castled = false;
        }

    } else if (Piece_is_pawn(moving_piece) && capture == EMPTY) {
        // en passant capture
        uint8_t pawn_move = abs(move->to - move->from);
        if (pawn_move == 7 || pawn_move == 9) {
            int col = index_col(move->to);
            if (chess->turn == TURN_WHITE) {
                Chess_add(chess, BLACK_PAWN, col + 32);
            } else {
                Chess_add(chess, WHITE_PAWN, col + 24);
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

gamestate_t Chess_make_null_move(Chess* chess) {
    gamestate_t gamestate = chess->gamestate;
    Chess_en_passant_set(chess, -1);
    chess->zhash ^= ZHASH_STATE[chess->gamestate] ^ ZHASH_STATE[gamestate];

    // Switch turn
    chess->zhash ^= ZHASH_WHITE ^ ZHASH_BLACK;
    chess->turn = !chess->turn;

    // Push zhash
    ZHashStack_push(&chess->zhstack, chess->zhash);

    // No need to update half move or full move clock since it won't matter much
    return gamestate;
}

void Chess_unmake_null_move(Chess* chess, gamestate_t gamestate) {
    ZHashStack_pop(&chess->zhstack);

    chess->zhash ^= ZHASH_STATE[chess->gamestate] ^ ZHASH_STATE[gamestate];
    chess->gamestate = gamestate;

    // Switch turn
    chess->zhash ^= ZHASH_WHITE ^ ZHASH_BLACK;
    chess->turn = !chess->turn;
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

    int from_i = Position_to_index(from);
    int to_i = Position_to_index(to);
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

Chess* Chess_from_fen(char* fen) {
#define FEN_PARSING_ERROR(details)                                \
    fprintf(stderr, "FEN Parsing error: " details ": %s\n", fen); \
    return NULL

    Chess* board = malloc(sizeof(Chess));  // empty board
    memset(board, 0, sizeof(Chess));
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
            if (piece == EMPTY) {
                FEN_PARSING_ERROR("Invalid piece");
            }
            Chess_add(board, piece, Position_to_index(pos));
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
    if (halfmoves > 255) {
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
    board->zhash = Chess_zhash(board);
    board->white_has_castled = false;
    board->black_has_castled = false;
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

void Chess_game_history(Chess* chess, char* game_history) {
    char* saveptr;
    char* fen = strtok_r(game_history, ",", &saveptr);
    int prev_king_pos = 255;

    while (fen) {
        Chess* prev = Chess_from_fen(fen);
        if (!prev) {
            chess->zhstack.sp++;
            continue;
        }
        uint64_t hash = Chess_zhash(prev);
        ZHashStack_push(&chess->zhstack, hash);

        // Update castling info
        if (prev->turn == TURN_BLACK) {  // Last move was white
            if (abs(prev->king_white - prev_king_pos) == 2) chess->white_has_castled = true;
            prev_king_pos = prev->king_black;
        } else {
            if (abs(prev->king_black - prev_king_pos) == 2) chess->black_has_castled = true;
            prev_king_pos = prev->king_white;
        }

        fen = strtok_r(NULL, ",", &saveptr);
    }
}