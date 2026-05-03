#pragma once
#include <stdint.h>
#include "chess.h"

void test_nnue();
void init_nnue(Chess* chess);
int forward(Chess* chess);