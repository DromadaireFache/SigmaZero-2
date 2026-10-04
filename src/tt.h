#pragma once
#include <stdbool.h>
#include <stdatomic.h>
#include <stdlib.h>

// Transposition table
typedef enum { TT_EMPTY, TT_EXACT, TT_LOWER, TT_UPPER } TTNodeType;

typedef struct {
    atomic_uint sequence;
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

typedef struct {
    uint64_t key;
    int eval;
    uint8_t depth;
    uint8_t type;
    uint8_t best_from;
    uint8_t best_to;
} TTEntry;

// Read a TTItem atomically and return a TTEntry
static inline TTEntry TT_read(const TTItem* item) {
    TTEntry entry;
    for (;;) {
        unsigned int sequence = atomic_load_explicit(&item->sequence, memory_order_acquire);
        if (sequence & 1) continue;
        entry.key = item->key;
        entry.eval = item->eval;
        entry.depth = item->depth;
        entry.type = item->type;
        entry.best_from = item->best_from;
        entry.best_to = item->best_to;
        if (sequence == atomic_load_explicit(&item->sequence, memory_order_acquire)) return entry;
    }
}

// Write a TTEntry to a TTItem atomically
static inline void TT_write(TTItem* item, TTEntry entry) {
    unsigned int sequence;
    for (;;) {
        sequence = atomic_load_explicit(&item->sequence, memory_order_relaxed);
        if (sequence & 1) continue;
        if (atomic_compare_exchange_weak_explicit(&item->sequence, &sequence, sequence + 1,
                                                  memory_order_acquire, memory_order_relaxed)) {
            break;
        }
    }

    item->key = entry.key;
    item->eval = entry.eval;
    item->depth = entry.depth;
    item->type = entry.type;
    item->best_from = entry.best_from;
    item->best_to = entry.best_to;
    atomic_store_explicit(&item->sequence, sequence + 2, memory_order_release);
}

// Store an entry in the transposition table
static inline int TT_store(uint64_t key, int eval, int depth, TTNodeType node_type,
                           Move best_move) {
    size_t i = key & (TT_LENGTH - 1);
    TTItem* item = &tt[i];
    TTEntry old = TT_read(item);

#ifdef TRACK_TT
    atomic_fetch_add(&tt_stores, 1);
    if (old.type != TT_EMPTY && old.key != key) {
        atomic_fetch_add(&tt_collisions, 1);
    }
#endif

    if (old.type == TT_EMPTY || old.key == key || depth >= old.depth) {
        TT_write(item, (TTEntry){.key = key,
                                .eval = eval,
                                .depth = (uint8_t)depth,
                                .type = node_type,
                                .best_from = best_move.from,
                                .best_to = best_move.to});
    }

    return eval;
}

// Retrieve an entry from the transposition table
static inline bool TT_get(uint64_t key, int* eval_p, int depth, int a, int b) {
    size_t i = key & (TT_LENGTH - 1);
    TTEntry entry = TT_read(&tt[i]);

#ifdef TRACK_TT
    atomic_fetch_add(&tt_lookups, 1);
#endif

    if (entry.type != TT_EMPTY && entry.key == key && depth <= entry.depth) {
        switch (entry.type) {
            case TT_EXACT:
                break;
            case TT_LOWER:
                if (entry.eval < b) return false;
                break;
            case TT_UPPER:
                if (entry.eval > a) return false;
                break;
            default:
                return false;
        }

#ifdef TRACK_TT
        atomic_fetch_add(&tt_hits, 1);
#endif
        *eval_p = entry.eval;
        return true;
    }

    return false;
}

// Retrieve the best move from the transposition table
static inline bool TT_get_best_move(uint64_t key, Move* move) {
    TTEntry entry = TT_read(&tt[key & (TT_LENGTH - 1)]);
    if (entry.type == TT_EMPTY || entry.key != key) return false;

    move->from = entry.best_from;
    move->to = entry.best_to;
    return true;
}

static inline void TT_clear(void) {
    for (size_t i = 0; i < TT_LENGTH; i++) {
        TT_write(&tt[i], (TTEntry){0});
    }
}

static inline double TT_occupancy(void) {
    size_t tt_use = 0;

    for (size_t i = 0; i < TT_LENGTH; i++) {
        TTEntry entry = TT_read(&tt[i]);
        if (entry.type != TT_EMPTY) tt_use++;
    }

    return (double)tt_use / TT_LENGTH;
}
