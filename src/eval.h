#pragma once
#include "chess.h"

int eval(Chess* chess);
int king_safety(Chess* chess);

int bishop_pawn_penalty(bitboard_t bishops, bitboard_t pawns);