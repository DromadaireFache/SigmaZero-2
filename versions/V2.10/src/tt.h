#pragma once
#include <stdbool.h>
#include <stdlib.h>

// Transposition table
typedef enum { TT_EXACT, TT_LOWER, TT_UPPER } TTNodeType;

typedef struct {
    uint64_t key;
    int eval;
    uint8_t depth;
    uint8_t type;  // TTNodeType
    uint8_t best_from;
    uint8_t best_to;
} TTItem;

// Will give ~64MB array
#define TT_LENGTH (1 << 22)

// Transposition table array
extern TTItem tt[TT_LENGTH];

// Store an entry with fine-grained locking
static inline int TT_store(uint64_t key, int eval, int depth, TTNodeType node_type,
                           Move best_move) {
    size_t i = key & (TT_LENGTH - 1);
    TTItem* item = &tt[i];

#ifdef TRACK_TT
    atomic_fetch_add(&tt_stores, 1);
    if (item->depth > 0 && item->key != key) {
        atomic_fetch_add(&tt_collisions, 1);
    }
#endif

    if (depth > item->depth) {
        item->key = key;
        item->eval = eval;
        item->depth = depth;
        item->type = node_type;
        item->best_from = best_move.from;
        item->best_to = best_move.to;
    }

    return eval;
}

// Retrieve an entry with fine-grained locking
static inline bool TT_get(uint64_t key, int* eval_p, int depth, int a, int b) {
    size_t i = key & (TT_LENGTH - 1);
    TTItem* item = &tt[i];

#ifdef TRACK_TT
    atomic_fetch_add(&tt_lookups, 1);
#endif

    if (item->key == key && depth <= item->depth) {
        switch (item->type) {
            case TT_EXACT:
                break;
            case TT_LOWER:
                if (item->eval < b) return false;
                break;
            case TT_UPPER:
                if (item->eval > a) return false;
                break;
        }

#ifdef TRACK_TT
        atomic_fetch_add(&tt_hits, 1);
#endif
        *eval_p = item->eval;
        return true;
    }

    return false;
}

static inline double TT_occupancy(void) {
    size_t tt_use = 0;

    for (int i = 0; i < TT_LENGTH; i++) {
        TTItem item = tt[i];
        if (i == (item.key & (TT_LENGTH - 1))) tt_use++;
    }

    return (double)tt_use / TT_LENGTH;
}
