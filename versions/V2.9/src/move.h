#pragma once

#include <string.h>

#include "position.h"

typedef enum __attribute__((__packed__)) {
    PROMOTE_QUEEN = 'q',
    PROMOTE_ROOK = 'r',
    PROMOTE_BISHOP = 'b',
    PROMOTE_KNIGHT = 'n',
    NO_PROMOTION = 0
} Promotion;

typedef struct {
    uint8_t from;
    uint8_t to;
    Promotion promotion;  // 'q', 'r', 'b', 'n' or 0 for no promotion
}
Move;

#define MAX_LEGAL_MOVES 218

static inline char* Move_string(Move* move) {
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

static inline void Move_print(Move* move) { printf("%s\n", Move_string(move)); }

static inline bool Move_equals(Move* move1, Move* move2) {
    return move1->from == move2->from && move1->to == move2->to &&
           move1->promotion == move2->promotion;
}