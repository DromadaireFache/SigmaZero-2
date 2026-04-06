#pragma once
#include <stdbool.h>
#include <stdatomic.h>

#include "chess.h"

#define INF 1000000000

// Uncomment to enable beta cutoff tracking (has performance cost)
// #define TRACK_BETA_CUTOFFS

#ifdef TRACK_BETA_CUTOFFS
extern atomic_size_t total_nodes;
extern atomic_size_t beta_cutoffs;
extern atomic_size_t first_move_cutoffs;
extern atomic_size_t total_cutoff_index;
#endif

// Uncomment to enable TT tracking (has performance cost)
// #define TRACK_TT

#ifdef TRACK_TT
extern atomic_size_t tt_lookups;
extern atomic_size_t tt_hits;
extern atomic_size_t tt_collisions;
extern atomic_size_t tt_stores;
#endif

// Uncomment to enable node tracking (has performance cost)
// #define TRACK_NODES

#ifdef TRACK_NODES
extern atomic_size_t nodes_searched;
#endif

#ifdef _WIN32
#define TIME_TYPE clock_t
#define TIME_NOW() clock()
#define TIME_DIFF_S(end, start) ((double)((end) - (start)) / CLOCKS_PER_SEC)
#define TIME_TO_S(t) ((double)(t) / CLOCKS_PER_SEC)
#define TIME_PLUS_OFFSET_MS(start, millis) ((start) + CLOCKS_PER_SEC * (millis) / 1000)
#include <windows.h>
static inline int cpu_count(void) {
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return sysinfo.dwNumberOfProcessors;
}

#else
#include <unistd.h>
#include <time.h>
static inline int cpu_count(void) {
    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
    return (nprocs > 0) ? nprocs : 1;
}
__attribute__((always_inline)) static inline uint64_t now_nanos() {
    struct timespec ts;
#ifdef __linux__
    clock_gettime(CLOCK_MONOTONIC_COARSE, &ts);
#else
    clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}
#define TIME_TYPE uint64_t
#define TIME_NOW() now_nanos()
#define TIME_DIFF_S(end, start) ((double)((end) - (start)) / 1000000000.0)
#define TIME_TO_S(t) ((double)(t) / 1000000000.0)
#define TIME_PLUS_OFFSET_MS(start, millis) ((start) + ((uint64_t)millis) * 1000000)
#endif

typedef struct {
    int score;
    bool reached;
} result_t;

typedef struct {
    Chess chess;
    int depth;
    Piece capture;
    Move move;
    int move_index;
    result_t* result;
    bool dont_push_next;
} task_t;

int minimax(Chess* chess, TIME_TYPE endtime, int depth, int a, int b, Piece last_capture,
            int extensions);
void select_best_move(Move* moves, int* scores, int start, int n_moves);
size_t Chess_count_moves(Chess* chess, int depth);
size_t Chess_count_moves_multi(Chess* chess, int depth);
void task_destroy(void);
void* play_thread(void* arg);
void task_push(task_t task);
void task_init(result_t (*results)[64], int n_moves);
bool openings_db(Chess* chess);