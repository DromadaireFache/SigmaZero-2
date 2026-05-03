#pragma once
#include "chess.h"

int eval(Chess* chess);
int king_safety(Chess* chess, int endness);
int nnue_eval(Chess* chess);