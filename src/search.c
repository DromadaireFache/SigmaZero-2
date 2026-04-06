#include "search.h"

#include <stdatomic.h>

#include "chess.h"
#include "eval.h"
#include "movegen.h"
#include "pthread.h"
#include "tt.h"

TTItem tt[TT_LENGTH] = {0};

#ifdef TRACK_BETA_CUTOFFS
atomic_size_t total_nodes = 0;
atomic_size_t beta_cutoffs = 0;
atomic_size_t first_move_cutoffs = 0;
atomic_size_t total_cutoff_index = 0;
#endif

#ifdef TRACK_TT
atomic_size_t tt_lookups = 0;
atomic_size_t tt_hits = 0;
atomic_size_t tt_collisions = 0;
atomic_size_t tt_stores = 0;
#endif

#ifdef TRACK_NODES
atomic_size_t nodes_searched = 0;
#endif

/* ~~~~~~~~~~~~~~~~~~~~~ Move counting ~~~~~~~~~~~~~~~~~~~~~ */

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
        Piece capture = Chess_make_move(chess, &moves[i]);
        nodes += Chess_count_moves(chess, depth - 1);
        Chess_unmake_move(chess, &moves[i], capture);
        chess->gamestate = gamestate;
        chess->zhash = hash;
    }

    return nodes;
}

size_t nodes_total = 0;
pthread_mutex_t lock;
typedef struct {
    Chess chess;
    int depth;
    Move move;
    TIME_TYPE endtime;
    int* score;
    bool* search_cancelled;
    size_t nodes;
} ChessThread;

void* Chess_count_moves_thread(void* arg_void) {
    ChessThread* arg = (ChessThread*)arg_void;
    Chess* chess = &arg->chess;
    int depth = arg->depth;
    Move* move = &arg->move;

    Chess_make_move(chess, move);
    arg->nodes = Chess_count_moves(chess, depth - 1);

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

    // Sum up results
    for (int i = 0; i < n_moves; i++) {
        nodes_total += args[i].nodes;
        // printf("%s: %lu\n", Move_string(&moves[i]), (unsigned long)args[i].nodes);
    }

    // Clean up
    pthread_mutex_destroy(&lock);
    free(threads);
    free(args);

    return nodes_total;
}

/* ~~~~~~~~~~~~~~~~~~~~~ Task queue ~~~~~~~~~~~~~~~~~~~~~ */

typedef void (*task_fn)(void*);

#define QUEUE_CAPACITY 1024

struct {
    result_t (*results)[64];  // results[n_moves][64]
    int n_moves;
    task_t tasks[QUEUE_CAPACITY];
    int sp;

    pthread_mutex_t mutex;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
    atomic_bool stop;
    atomic_size_t active_workers;
} task_stack;

void task_init(result_t (*results)[64], int n_moves) {
    task_stack.sp = 0;
    task_stack.results = results;
    task_stack.n_moves = n_moves;
    task_stack.stop = false;
    atomic_store(&task_stack.active_workers, 0);
    pthread_mutex_init(&task_stack.mutex, NULL);
    pthread_cond_init(&task_stack.not_empty, NULL);
    pthread_cond_init(&task_stack.not_full, NULL);
}

void task_destroy(void) {
    pthread_mutex_destroy(&task_stack.mutex);
    pthread_cond_destroy(&task_stack.not_empty);
    pthread_cond_destroy(&task_stack.not_full);
}

static inline void task_request_stop(void) {
    atomic_store(&task_stack.stop, true);
    pthread_cond_broadcast(&task_stack.not_empty);
    pthread_cond_broadcast(&task_stack.not_full);
}

void task_show(void) {
    printf("{");
    for (int i = 0; i < task_stack.sp; i++) {
        task_t task = task_stack.tasks[i];
        printf("%s(%d)", Move_string(&task.move), task.depth);
        if (i + 1 != task_stack.sp) printf(", ");
    }
    printf("}\n");
}

void task_push(task_t task) {
    pthread_mutex_lock(&task_stack.mutex);

    while (task_stack.sp >= QUEUE_CAPACITY && !atomic_load(&task_stack.stop)) {
        pthread_cond_wait(&task_stack.not_full, &task_stack.mutex);
    }
    if (atomic_load(&task_stack.stop)) {
        pthread_mutex_unlock(&task_stack.mutex);
        return;
    }

    task_stack.tasks[task_stack.sp++] = task;
    // task_show();

    pthread_cond_signal(&task_stack.not_empty);
    pthread_mutex_unlock(&task_stack.mutex);
}

size_t task_size(void) {
    pthread_mutex_lock(&task_stack.mutex);
    size_t sz = (size_t)task_stack.sp;
    pthread_mutex_unlock(&task_stack.mutex);
    return sz;
}

size_t task_max_pushes(void) {
    pthread_mutex_lock(&task_stack.mutex);
    if (task_stack.n_moves > cpu_count()) {
        pthread_mutex_unlock(&task_stack.mutex);
        return 1;
    }

    size_t max_pushes = (cpu_count() - task_stack.active_workers) / task_stack.n_moves + 1;
    // printf("workers: %zu/%d, max_pushes: %zu\n", task_stack.active_workers, cpu_count(),
    //        max_pushes);
    pthread_mutex_unlock(&task_stack.mutex);
    return max_pushes;
}

void task_maybe_stop_if_idle(void) {
    usleep(1000);
    if (atomic_load(&task_stack.active_workers) == 0 && task_size() == 0) {
        task_request_stop();
    }
}

void task_remove(int index) {
    if (index >= task_stack.sp) return;

    for (int i = index; i + 1 < task_stack.sp; i++) {
        task_stack.tasks[i] = task_stack.tasks[i + 1];
    }
    task_stack.sp--;
}

// Returns true if found a task at depth 1
bool task_find_depth_1(task_t* task) {
    for (int i = 0; i < task_stack.sp; i++) {
        task_t t = task_stack.tasks[i];
        if (t.depth <= 1) {
            *task = t;
            task_remove(i);
            return true;
        }
    }
    return false;
}

bool task_is_priority(int depth, int score) {
    int better_count = 0, total_count = 0;

    // Count how many moves have a better score at this depth
    for (int i = 0; i < task_stack.n_moves; i++) {
        result_t other = task_stack.results[i][depth];
        if (!other.reached) continue;
        total_count++;

        if (other.score > score) {
            better_count++;
            if (better_count >= PRIORITY_LINES) {
                return false;  // This move is not in top PRIORITY_LINES
            }
        }
    }

    return total_count >= PRIORITY_LINES;  // This move is in top PRIORITY_LINES
}

int task_find_best(bool priority_line) {
    int best_score = -INF, best_index = -1, best_depth = INF;

    for (int i = 0; i < task_stack.sp; i++) {
        task_t t = task_stack.tasks[i];
        result_t result = task_stack.results[t.move_index][t.depth - 1];
        int score = result.reached ? result.score : -INF;

        if (t.depth > best_depth) continue;
        if (priority_line && !result.reached) continue;
        if (priority_line && !task_is_priority(t.depth - 1, score)) continue;

        if (t.depth < best_depth || score > best_score) {
            best_depth = t.depth;
            best_index = i;
            best_score = score;
        }
    }

    return best_index;
}

// Will return true if there is time left
bool task_pop(task_t* task, TIME_TYPE endtime) {
    pthread_mutex_lock(&task_stack.mutex);

    while (task_stack.sp == 0 && !atomic_load(&task_stack.stop)) {
        // Wake up either when there is work or when time is up
        pthread_cond_wait(&task_stack.not_empty, &task_stack.mutex);
        if (TIME_NOW() > endtime) {
            task_request_stop();
        }
    }

    if (task_stack.sp == 0) {
        pthread_mutex_unlock(&task_stack.mutex);
        return false;  // nothing to do / stopping
    }

    // Complete all depth 1 tasks.
    if (task_find_depth_1(task)) {
        atomic_fetch_add(&task_stack.active_workers, 1);
        pthread_cond_signal(&task_stack.not_full);
        pthread_mutex_unlock(&task_stack.mutex);
        return true;
    }

    // Loop through all the tasks and find the lowest depth and best previous score at that depth.
    int i = task_find_best(false);

    // Loop through all the tasks and find the lowest depth where the score of the previous depth is
    // within the top PRIORITY_LINES best scores. Otherwise, do the task found in the previous step.
    int i2 = task_find_best(true);

    if (i2 != -1) i = i2;
    // if (i2 != -1) printf("%s\n", Move_string(&task_stack.tasks[i].move));
    *task = task_stack.tasks[i];
    task_remove(i);

    atomic_fetch_add(&task_stack.active_workers, 1);
    pthread_cond_signal(&task_stack.not_full);
    pthread_mutex_unlock(&task_stack.mutex);
    return TIME_NOW() < endtime;
}

/* ~~~~~~~~~~~~~~~~~~~~~ Minimax ~~~~~~~~~~~~~~~~~~~~~ */

void select_best_move(Move* moves, int* scores, int start, int n_moves) {
    int best = start;
    int best_score = scores[start];

    for (int i = start + 1; i < n_moves; i++) {
        if (scores[i] > best_score) {
            best_score = scores[i];
            best = i;
        }
    }

    if (best != start) {
        Move temp1 = moves[start];
        int temp2 = scores[start];
        moves[start] = moves[best];
        scores[start] = scores[best];
        moves[best] = temp1;
        scores[best] = temp2;
    }
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
    int scores[MAX_LEGAL_MOVES];
    size_t n_moves = Chess_legal_moves_scored(chess, moves, scores, true);

    for (int i = 0; i < n_moves; i++) {
        if (i < SELECT_MOVE_CUTOFF) select_best_move(moves, scores, i, n_moves);
        Move* move = &moves[i];

        gamestate_t gamestate = chess->gamestate;
        uint64_t hash = chess->zhash;
        Piece capture = Chess_make_move(chess, move);

        int score = -minimax_captures_only(chess, endtime, depth - 1, -b, -a);

        Chess_unmake_move(chess, move, capture);
        chess->gamestate = gamestate;
        chess->zhash = hash;

        if (score > best_score) {
            best_score = score;
            if (score > a) a = score;
        }
        if (score >= b) return best_score;
    }
    return best_score;
}

static inline int compute_reduction(int depth, int i) {
    int log_depth = 8 * sizeof(int) - __builtin_clz(depth) - 1;
    int log_i = 8 * sizeof(int) - __builtin_clz(i) - 1;
    return log_depth * log_i / 3;
}

int minimax(Chess* chess, TIME_TYPE endtime, int depth, int a, int b, Piece last_capture,
            int extensions) {
#ifdef TRACK_NODES
    atomic_fetch_add(&nodes_searched, 1);
#endif

    if (depth == 0 && last_capture != EMPTY) {
        return minimax_captures_only(chess, endtime, QUIES_DEPTH, a, b);
    }

    // Look for existing eval in transposition table
    uint64_t hash = ZHashStack_peek(&chess->zhstack);
    int tt_eval;
    if (TT_get(hash, &tt_eval, depth, a, b)) {
        return tt_eval;
    }

    // Extend search if in check, otherwise don't
    if (depth == 0) {
        if (extensions < MAX_EXTENSION && Chess_friendly_check(chess)) {
            depth++;
            extensions++;
        } else {
            return TT_store(hash, chess->turn == TURN_WHITE ? eval(chess) : -eval(chess), depth,
                            TT_EXACT, (Move){0});
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
    int scores[MAX_LEGAL_MOVES];
    size_t n_moves = Chess_legal_moves_scored(chess, moves, scores, false);
    bool in_check = chess->enemy_attack_map.n_checks > 0;

    if (n_moves == 0) {
        if (in_check) {
            // Checkmate
            return TT_store(hash, -1000000 - depth, depth, TT_EXACT, (Move){0});
        } else {
            // draw by stalemate
            return TT_store(hash, 0, depth, TT_EXACT, (Move){0});
        }
    }

    // Futility pruning
    if (!in_check && last_capture == EMPTY && depth < FP_DEPTH) {
        int e = chess->turn == TURN_WHITE ? chess->eval : -chess->eval;
        int margin = FP_BASE + depth * FP_FACTOR;
        if (e + margin <= a) {
            return TT_store(hash, a, depth, TT_UPPER, (Move){0});  // Failed low
        }

        margin = RFP_BASE + depth * RFP_FACTOR;
        if (e - 2 * margin >= b) {
            return TT_store(hash, b, depth, TT_LOWER, (Move){0});  // Failed high
        }
    }

    // Null move pruning
    bool is_null_move_allowed = extensions < MAX_EXTENSION;
    if (!in_check && depth >= 3 && is_null_move_allowed && Chess_has_non_pawn_material(chess)) {
        gamestate_t gamestate = Chess_make_null_move(chess);
        int R = (depth >= 6) ? 3 : 2;
        int score = -minimax(chess, endtime, depth - 1 - R, -b, -b + 1, EMPTY, MAX_EXTENSION);
        Chess_unmake_null_move(chess, gamestate);

        if (score >= b) return b;  // Null move cutoff
    }

    // Prioritize TT best move
    size_t tt_i = hash & (TT_LENGTH - 1);
    TTItem* tt_item = &tt[tt_i];
    if (tt_item->key == hash) {
        for (int i = 0; i < n_moves; i++) {
            if (moves[i].from == tt_item->best_from && moves[i].to == tt_item->best_to) {
                scores[i] += TT_MOVE_BONUS;
                break;
            }
        }
    }

    // Add score for killer move heuristic
    for (int i = 0; i < n_moves; i++) {
        Move* move = &moves[i];

        // Check both killer slots
        if (chess->board[move->to] == EMPTY) {
            bool is_killer_move = (chess->killer_moves[0][depth].from == move->from &&
                                   chess->killer_moves[0][depth].to == move->to) ||
                                  (chess->killer_moves[1][depth].from == move->from &&
                                   chess->killer_moves[1][depth].to == move->to);
            scores[i] += is_killer_move * KILLER_MOVE_BONUS;
        }
    }

#ifdef TRACK_BETA_CUTOFFS
    atomic_fetch_add(&total_nodes, 1);
#endif

    int original_a = a;
    int best_score = -INF;
    Move best_move = moves[0];
    for (int i = 0; i < n_moves; i++) {
        if (i < SELECT_MOVE_CUTOFF) select_best_move(moves, scores, i, n_moves);
        Move* move = &moves[i];

        gamestate_t gamestate = chess->gamestate;
        uint64_t hash = chess->zhash;
        int pawn_row_sum = chess->pawn_row_sum;
        Piece capture = Chess_make_move(chess, move);

        int score;
        if (i == 0) {
            // principal variation search
            score = -minimax(chess, endtime, depth - 1, -b, -a, capture, extensions);
        } else {
            // Late move reduction condition
            bool reduction_condition = depth >= 2 && !in_check && capture == EMPTY;
            int r = reduction_condition ? compute_reduction(depth, i) : 0;

            // Reduce less aggressively in endgames
            if (chess->fullmoves >= FULLMOVES_ENDGAME && r > 1) r = r / 2;

            // Clamp reduction so we don't go below depth 1
            int reduced_depth = depth - 1 - r;
            if (reduced_depth < 0) reduced_depth = 0;

            // search with a narrow window and reduction first
            score = -minimax(chess, endtime, reduced_depth, -a - 1, -a, capture, extensions);

            // if reduction caused potential improvement re-search
            if (r > 0 && score > a) {
                score = -minimax(chess, endtime, depth - 1, -a - 1, -a, capture, extensions);
            }

            // if score exceeds alpha do full search
            if (score > a) {
                score = -minimax(chess, endtime, depth - 1, -b, -a, capture, extensions);
            }
        }

        Chess_unmake_move(chess, move, capture);
        chess->gamestate = gamestate;
        chess->zhash = hash;
        chess->pawn_row_sum = pawn_row_sum;

        if (score > best_score) {
            best_score = score;
            best_move = *move;
            if (score > a) a = score;
        }
        if (score >= b) {
#ifdef TRACK_BETA_CUTOFFS
            atomic_fetch_add(&beta_cutoffs, 1);
            atomic_fetch_add(&total_cutoff_index, i);
            if (i == 0) atomic_fetch_add(&first_move_cutoffs, 1);
#endif
            if (capture == EMPTY) {
                // Shift killers: move primary to secondary, new move to primary
                if (!(chess->killer_moves[0][depth].from == move->from &&
                      chess->killer_moves[0][depth].to == move->to)) {
                    chess->killer_moves[1][depth] = chess->killer_moves[0][depth];
                    chess->killer_moves[0][depth] = *move;
                }
            }
            return TT_store(hash, best_score, depth, TT_LOWER, best_move);  // Failed high
        }
    }

#ifdef TRACK_BETA_CUTOFFS
    atomic_fetch_add(&total_cutoff_index, n_moves - 1);
#endif
    if (best_score <= original_a) {
        return TT_store(hash, best_score, depth, TT_UPPER, best_move);  // Failed low
    }
    return TT_store(hash, best_score, depth, TT_EXACT, best_move);
}

void* play_thread(void* arg) {
    TIME_TYPE endtime = *(TIME_TYPE*)arg;
    task_t task;
    int score;

    while (1) {
        if (!task_pop(&task, endtime)) break;
        if (task.depth >= 64) continue;
        Chess* chess = &task.chess;
        int depth = task.depth;
        Piece capture = task.capture;
        Move move = task.move;
        memset(chess->killer_moves, 0, sizeof(chess->killer_moves));

        if (task.depth > 1 && task.result[-1].reached) {  // aspiration window
            int window_alpha = ASP_WINDOW_ALPHA_INIT, window_beta = ASP_WINDOW_BETA_INIT;
            int prev_score = task.result[-1].score;

            while (1) {
                int alpha = prev_score - window_alpha;
                int beta = prev_score + window_beta;
                score = -minimax(chess, endtime, depth - 1, -beta, -alpha, capture, 0);
                if (TIME_NOW() > endtime) break;
                if (score <= alpha)
                    window_alpha *= 2;
                else if (score >= beta)
                    window_beta *= 2;
                else
                    break;
            }

        } else {
            score = -minimax(chess, endtime, depth - 1, -INF, INF, capture, 0);
        }

        if (TIME_NOW() > endtime) {
            task_request_stop();
            break;
        }

        // Add move score
        task.result->score = score;
        task.result->reached = true;
        atomic_fetch_sub(&task_stack.active_workers, 1);

        // Don't push moves that lead to checkmate
        bool is_checkmate = abs(score) >= 1000000;
        if (is_checkmate || task.dont_push_next || task.depth >= 62) {
            task_maybe_stop_if_idle();
            continue;
        }

        // Push the next depth to the queue
        // If there is still space push another depth, push task a second time
        // bool push_two_tasks = task_size() < cpu_count();
        size_t max_pushes = task_max_pushes();
        for (int i = 0; i < max_pushes; i++) {
            task_t task2 = {.chess = *chess,
                            .capture = capture,
                            .depth = ++depth,
                            .move = move,
                            .move_index = task.move_index,
                            .result = ++task.result,
                            .dont_push_next = i + 1 < max_pushes};
            task_push(task2);
        }
    }
    return NULL;
}

bool openings_db(Chess* chess) {
    char s[100];
    snprintf(s, 100, "%" PRIx64, Chess_zhash(chess));
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
