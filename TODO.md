# Quick Improvements for Your Chess Bot

Looking at your implementation, here are high-impact changes you could make quickly:

## 1. Improve Move Ordering
```c
void Chess_score_move(Chess *chess, Move *move) {
    // Add history heuristic
    static int history[64][64] = {0}; // From-To squares
    
    // Keep MVV-LVA but also add:
    if (move->promotion == PROMOTE_QUEEN) move->score += 1000;
    else if (move->promotion) move->score += 800;
    
    // Killer move heuristic (store 2 good moves at each ply)
    // Bonus for checks
    // Add history score
    move->score += history[move->from][move->to] / 100;
    
    // Update history after a move causes beta cutoff
    // history[from][to] += depth * depth;
}
```

## 2. Implement Null Move Pruning
```c
int minimax(Chess *chess, TIME_TYPE endtime, int depth, int a, int b, Piece last_capture) {
    // Add before normal search:
    if (depth >= 3 && !chess->enemy_attack_map.n_checks && !last_capture) {
        // Try skipping a move - if opponent still can't beat beta, we're good
        chess->turn = !chess->turn;
        int null_score = -minimax(chess, endtime, depth-3, -b, -b+1, EMPTY);
        chess->turn = !chess->turn;
        if (null_score >= b) return b; // Pruned - position is very good
    }
    // ... existing code ...
}
```

## 3. Add Late Move Reduction
```c
// Inside minimax search loop:
for (int i = 0; i < n_moves; i++) {
    // First few moves get full depth
    int new_depth = depth - 1;
    
    // Later moves with low scores get reduced depth if they're not captures
    if (i >= 3 && depth >= 3 && !Chess_enemy_piece_at(chess, move->to)) {
        new_depth = depth - 2;  // Search with reduced depth
    }
    
    int score = -minimax(chess, endtime, new_depth, -b, -a, capture);
    
    // If reduced search indicates this might be good, research at full depth
    if (new_depth < depth-1 && score > a) {
        score = -minimax(chess, endtime, depth-1, -b, -a, capture);
    }
}
```

## 4. Better King Safety Evaluation
```c
int eval(Chess *chess) {
    // Add to your evaluation:
    int king_safety_white = 0, king_safety_black = 0;
    
    // Penalize exposed kings
    Position w_king = Position_from_index(chess->king_white);
    Position b_king = Position_from_index(chess->king_black);
    
    // Check pawn shield
    for (int offset = -1; offset <= 1; offset++) {
        if (w_king.col+offset >= 0 && w_king.col+offset < 8 && w_king.row+1 < 8) {
            if (chess->board[(w_king.row+1)*8 + w_king.col+offset] == WHITE_PAWN)
                king_safety_white += 15;
        }
        if (b_king.col+offset >= 0 && b_king.col+offset < 8 && b_king.row-1 >= 0) {
            if (chess->board[(b_king.row-1)*8 + b_king.col+offset] == BLACK_PAWN)
                king_safety_black += 15;
        }
    }
    
    e += king_safety_white - king_safety_black;
    return e;
}
```

## 5. Use Aspiration Windows
```c
// In your main search loop:
int prev_score = 0;
int window = 50; // 0.5 pawns

while (TIME_NOW() < endtime) {
    // Start with narrow window around previous score
    int alpha = depth > 1 ? prev_score - window : -INF;
    int beta = depth > 1 ? prev_score + window : INF;
    
    // If window is too narrow, research with wider window
    if (score <= alpha || score >= beta) {
        alpha = -INF;
        beta = INF;
        score = -minimax(chess, endtime, depth, -beta, -alpha, EMPTY);
    }
    
    prev_score = score;
    depth++;
}
```

## 6. Extend Search for Checks
```c
// In minimax function:
if (Chess_friendly_check(chess)) {
    depth++; // Search one ply deeper when in check
}
```

## 7. Fix Transposition Table Usage
```c
// When storing TT entries, add the best move:
typedef struct {
    uint64_t key;
    int eval;
    uint8_t depth;
    TTNodeType type;
    Move best_move; // Store the best move found
} TTItem;

// When retrieving entries, use the move ordering:
if (TT_get(hash, &tt_eval, depth, a, b)) {
    // Also use the stored best move to improve ordering
    return tt_eval;
}
```

These improvements should give you significant ELO gains for minimal coding effort. Start with move ordering and null move pruning for the biggest immediate gains.