#include <math.h>
#include <pthread.h>
#include <stdio.h>

#define CONSTS_IMPL
#include "eval.h"
#include "movegen.h"
#include "search.h"
#include "tt.h"

// Play a move given a FEN string
// Returns 0 on success, 1 on error
int play(char* fen, int millis, char* game_history) {
    Chess* chess = Chess_from_fen(fen);
    if (!chess) return 1;
    if (millis < 1) return 1;

    if (game_history != NULL) {
        Chess_game_history(chess, game_history);
    }

#ifdef TRACK_BETA_CUTOFFS
    atomic_store(&total_nodes, 0);
    atomic_store(&beta_cutoffs, 0);
    atomic_store(&first_move_cutoffs, 0);
    atomic_store(&total_cutoff_index, 0);
#endif

#ifdef TRACK_TT
    atomic_store(&tt_lookups, 0);
    atomic_store(&tt_hits, 0);
    atomic_store(&tt_collisions, 0);
    atomic_store(&tt_stores, 0);
#endif

#ifdef TRACK_NODES
    atomic_store(&nodes_searched, 0);
#endif

    if (chess->fullmoves <= 5 && openings_db(chess)) {
        return 0;
    }

    TIME_TYPE start = TIME_NOW();
    TIME_TYPE endtime = TIME_PLUS_OFFSET_MS(start, millis);
    Move moves[MAX_LEGAL_MOVES];
    int scores[MAX_LEGAL_MOVES];
    result_t results[MAX_LEGAL_MOVES][64] = {0};
    size_t n_moves = Chess_legal_moves_scored(chess, moves, scores, false);
    if (n_moves < 1) return 1;
    task_init(results, n_moves);

    for (int i = 0; i < n_moves; i++) {
        Chess chess_cp = *chess;
        Move* move = &moves[i];
        Piece capture = Chess_make_move(&chess_cp, move);
        task_t task = {.chess = chess_cp,
                       .capture = capture,
                       .depth = 1,
                       .move = moves[i],
                       .move_index = i,
                       .result = &results[i][1]};
        task_push(task);
    }

    // printf("Finished filling up the task queue\n");
    // task_show();

    pthread_t* threads = calloc(cpu_count(), sizeof(pthread_t));
    for (int i = 0; i < cpu_count(); i++) {
        if (pthread_create(&threads[i], NULL, play_thread, &endtime) != 0) {
            perror("pthread_create failed");
            return 1;
        }
    }

    // Wait for threads to finish
    for (int i = 0; i < cpu_count(); i++) {
        pthread_join(threads[i], NULL);
    }

    free(threads);
    task_destroy();

    // Display results
    // for (int i = 0; i < n_moves; i++) {
    //     printf("%-2d %-5s ", i, Move_string(&moves[i]));
    //     for (int j = 1; results[i][j].reached; j++) {
    //         printf("%5d ", results[i][j].score);
    //     }
    //     printf("\n");
    // }

    // Select best move in results
    bool has_next = true;
    int depth = 0, best_score = -INF;
    Move best_move = moves[0];
    for (depth = 1; depth < 63 && has_next; depth++) {
        best_score = -INF;
        for (int i = 0; i < n_moves; i++) {
            result_t result = results[i][depth];
            if (result.reached && result.score > best_score) {
                best_score = result.score;
                best_move = moves[i];
                has_next = results[i][depth + 1].reached;
            }
        }
    }

    TIME_TYPE end = TIME_NOW();
    double cpu_time = TIME_DIFF_S(end, start);
    best_score = chess->turn == TURN_WHITE ? best_score : -best_score;
    depth--;

    puts("{");
    printf("  \"millis\": %d,\n", millis);
    printf("  \"depth\": %d,\n", depth);
    printf("  \"time\": %.3lf,\n", cpu_time);
#ifdef TRACK_BETA_CUTOFFS
    size_t cutoff_nodes = atomic_load(&total_nodes);
    size_t cutoffs = atomic_load(&beta_cutoffs);
    size_t first_cutoffs = atomic_load(&first_move_cutoffs);
    size_t cutoff_idx = atomic_load(&total_cutoff_index);
    double cutoff_rate = cutoff_nodes > 0 ? (double)cutoffs * 100.0 / cutoff_nodes : 0;
    double first_move_rate = cutoffs > 0 ? (double)first_cutoffs * 100.0 / cutoff_nodes : 0;
    double avg_cutoff_index = cutoffs > 0 ? (double)cutoff_idx / cutoffs : 0;
    printf("  \"cutoff_nodes\": %lu,\n", (unsigned long)cutoff_nodes);
    printf("  \"beta_cutoff_%%\": %.2f,\n", cutoff_rate);
    printf("  \"first_move_cutoff_%%\": %.2f,\n", first_move_rate);
    printf("  \"avg_cutoff_index\": %.2f,\n", avg_cutoff_index);
#endif
#ifdef TRACK_TT
    size_t lookups = atomic_load(&tt_lookups);
    size_t hits = atomic_load(&tt_hits);
    size_t collisions = atomic_load(&tt_collisions);
    size_t stores = atomic_load(&tt_stores);
    double hit_rate = lookups > 0 ? (double)hits * 100.0 / lookups : 0;
    double collision_rate = stores > 0 ? (double)collisions * 100.0 / stores : 0;
    printf("  \"tt_hit_rate_%%\": %.2f,\n", hit_rate);
    printf("  \"tt_collision_rate_%%\": %.2f,\n", collision_rate);
    printf("  \"tt_occupancy_%%\": %.2f,\n", TT_occupancy() * 100.0);
#endif
#ifdef TRACK_NODES
    size_t nodes = atomic_load(&nodes_searched);
    printf("  \"nodes\": %lu,\n", (unsigned long)nodes);
    printf("  \"nps\": %.0lf,\n", cpu_time > 0.0 ? (double)nodes / cpu_time : 0.0);
#endif
    printf("  \"eval\": %.2f,\n", (double)best_score / 100);
    printf("  \"move\": \"%s\"\n", Move_string(&best_move));
    puts("}");
    return 0;
}

int moves(char* fen, int depth) {
    Chess* chess = Chess_from_fen(fen);
    if (!chess) return 1;
    if (depth > 1) {
        TIME_TYPE start = TIME_NOW();
        size_t n_moves = Chess_count_moves_multi(chess, depth);
        TIME_TYPE end = TIME_NOW();
        double cpu_time = TIME_DIFF_S(end, start);
        double nps = cpu_time > 0.0 ? n_moves / cpu_time : -0.0;
        puts("{");
        printf("  \"depth\": %d,\n", depth);
        printf("  \"nodes\": %lu,\n", (unsigned long)n_moves);
        printf("  \"time\": %.3lf,\n", cpu_time);
        printf("  \"nps\": %.3lf\n", nps);
        puts("}");
    } else {
        Move moves[MAX_LEGAL_MOVES];
        size_t n_moves = Chess_legal_moves(chess, moves, false);
        puts("{");
        printf("  \"nodes\": %lu,\n", (unsigned long)n_moves);
        puts("  \"moves\": [");
        for (int i = 0; i < n_moves; i++) {
            printf("    \"%s\"", Move_string(&moves[i]));
            if (i < n_moves - 1)
                puts(",");
            else
                puts("");
        }
        puts("  ]");
        puts("}");
    }
    return 0;
}

int version() {
    printf("SigmaZero Chess Engine 2.10.0 (2026-04-05)\n");
    return 0;
}

void help(void) {
    printf("SigmaZero Chess Engine - Command Line Interface\n\n");
    printf("Usage: engine <command> [arguments]\n\n");
    printf("Commands:\n");
#define HELP_WIDTH "  %-30s"
    printf(HELP_WIDTH " %s\n", "help, -h, --help", "Show this help message");
    printf(HELP_WIDTH " %s\n", "version, -v, --version", "Show version information");
    printf("\n");
    printf(HELP_WIDTH " %s\n", "play <FEN> <millis> [history]", "Play best move for position");
    printf(HELP_WIDTH " %s\n", "", "  FEN: Position in FEN notation");
    printf(HELP_WIDTH " %s\n", "", "  millis: Time limit in milliseconds");
    printf(HELP_WIDTH " %s\n", "", "  history: Optional game history");
    printf("\n");
    printf(HELP_WIDTH " %s\n", "moves <FEN> <depth>", "Count legal moves at depth");
    printf(HELP_WIDTH " %s\n", "", "  depth: Search depth (perft)");
    printf("\n");
    printf(HELP_WIDTH " %s\n", "eval <FEN>", "Evaluate position");
    printf(HELP_WIDTH " %s\n", "hash <FEN>", "Show zobrist hash");
    printf(HELP_WIDTH " %s\n", "scores <FEN>", "Show move scores");
    printf(HELP_WIDTH " %s\n", "kingsafety <FEN>", "Show king danger scores");
    printf("\n");
    printf("Examples:\n");
    printf("  engine play \"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\" 1000\n");
    printf("  engine moves \"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\" 5\n");
    printf("  engine eval \"8/8/8/4k3/8/8/4K3/8 w - - 0 1\"\n");
}

int eval_command(Chess* chess, int depth) {
    if (depth == 0) {
        printf("%lg", (double)eval(chess) / 100.0);
    } else {
        int score = 0;

        for (int d = 0; d <= depth; d++) {
#ifdef TRACK_NODES
            size_t base_nodes = atomic_load(&nodes_searched);
#endif

            TIME_TYPE start = TIME_NOW();
            int window_alpha = ASP_WINDOW_ALPHA_INIT, window_beta = ASP_WINDOW_BETA_INIT;

            if (d == 0) {
                score = -minimax(chess, UINT64_MAX, d, -INF, INF, EMPTY, 0);
            } else {
                int prev_score = score;
                while (1) {
                    int alpha = prev_score - window_alpha;
                    int beta = prev_score + window_beta;
                    score = -minimax(chess, UINT64_MAX, d, -beta, -alpha, EMPTY, 0);
                    if (score <= alpha) {
                        if (window_alpha > 100) {
                            window_alpha = INF;
                        } else {
                            window_alpha *= 2;
                        }
                    } else if (score >= beta) {
                        if (window_beta > 100) {
                            window_beta = -INF;
                        } else {
                            window_beta *= 2;
                        }
                    } else
                        break;
                }
            }

            TIME_TYPE end = TIME_NOW();
            double cpu_time = TIME_DIFF_S(end, start);
            double e = (double)score / 100.0;

#ifdef TRACK_NODES
            size_t nodes = atomic_load(&nodes_searched) - base_nodes;
            printf("Depth %d: %lg reached in %.3lg seconds with %zu nodes\n", d, e, cpu_time,
                   nodes);
#else
            printf("Depth %d: %lg reached in %.3lg seconds\n", d, e, cpu_time);
#endif
        }
    }

#ifdef TRACK_NODES
    size_t nodes = atomic_load(&nodes_searched);
    printf("%zu total nodes\n", nodes);
#endif
    return 0;
}

int king_safety_command(Chess* chess) {
    printf("king_safety() -> %d\n", king_safety(chess));
    return 0;
}

int move_scores_command(Chess* chess) {
    Move moves[MAX_LEGAL_MOVES];
    int scores[MAX_LEGAL_MOVES];
    size_t n_moves = Chess_legal_moves_scored(chess, moves, scores, false);
    bool elipses = false;

    for (int i = 0; i < n_moves; i++) {
        select_best_move(moves, scores, i, n_moves);
        if (scores[i] != 0) {
            printf("%-5s %6d\n", Move_string(moves + i), scores[i]);
        } else if (!elipses) {
            printf("...   %6d\n", 0);
            elipses = true;
        }
    }
    return 0;
}

int minmax_command(int depth) {
    FILE* f = fopen("data/training.txt", "rt");
    if (f == NULL) return 1;
    char fen[1024];
    TIME_TYPE start = TIME_NOW();
    int fens = 0;

    while (fgets(fen, 1024, f)) {
        memset(tt, 0, sizeof(tt));
        fen[strcspn(fen, "\r\n")] = 0;
        Chess* chess = Chess_from_fen(fen);
        if (chess == NULL) continue;
        minimax(chess, UINT64_MAX, depth, -INF, INF, EMPTY, 0);
        fens++;
        if (TIME_DIFF_S(TIME_NOW(), start) > 60) break;
    }

    TIME_TYPE end = TIME_NOW();
    double cpu_time = TIME_DIFF_S(end, start);
    printf("{\n");
    printf("    \"time\" : %.3lf,\n", cpu_time);
    printf("    \"fens\" : %d,\n", fens);
#ifdef TRACK_NODES
    size_t nodes = atomic_load(&nodes_searched);
    printf("    \"nodes\": %zu,\n", nodes);
#endif
#ifdef TRACK_BETA_CUTOFFS
    size_t cutoff_nodes = atomic_load(&total_nodes);
    size_t cutoffs = atomic_load(&beta_cutoffs);
    size_t first_cutoffs = atomic_load(&first_move_cutoffs);
    double first_move_rate = cutoffs > 0 ? (double)first_cutoffs * 100.0 / cutoff_nodes : 0;
    printf("    \"first_move_cutoff_%%\": %.2f\n", first_move_rate);
#endif
    printf("}\n");
    return 0;
}

int compute_eval_loss() {
    FILE* f = fopen("data/chessData.csv", "r");
    if (f == NULL) {
        fprintf(stderr, "Could not open file: data/chessData.csv");
        return 1;
    }

    const int MAX_ERROR_CP = 300;  // clamp per-sample error to ±3.0 eval
    int fens_count = 0;
    uint64_t total_loss = 0;
    char line[1024];

    while (fgets(line, sizeof(line), f) && fens_count < 100000) {
        char* fen = strtok(line, ",");
        char* stockfish_eval_str = strtok(NULL, ",");
        if (stockfish_eval_str[0] == '#') continue;  // skip checkmates

        int stockfish_eval = atoi(stockfish_eval_str);
        if (abs(stockfish_eval) > 1000) continue;  // skip evals > 10

        Chess* chess = Chess_from_fen(fen);
        if (chess == NULL) continue;  // failed to parse FEN
        // int sigmazero_eval = eval(chess);
        int sigmazero_eval = minimax(chess, UINT64_MAX, 3, -INF, INF, EMPTY, 0);

        // clamp error
        int diff = sigmazero_eval - stockfish_eval;
        if (abs(diff) > MAX_ERROR_CP) continue;

        uint64_t loss = (uint64_t)(diff * diff);
        total_loss += loss;
        fens_count++;
        free(chess);
    }

    printf("%lg\n", fens_count ? (double)total_loss / fens_count : 0.0);
    fclose(f);
    return 0;
}

int compute_baseline_loss() {
    FILE* f = fopen("data/chessData.csv", "r");
    if (f == NULL) {
        fprintf(stderr, "Could not open file: data/chessData.csv");
        return 1;
    }

    uint64_t fens_count = 0;
    double total_loss = 0.0;
    char line[1024];

    while (fgets(line, sizeof(line), f)) {
        char* fen = strtok(line, ",");
        char* stockfish_eval_str = strtok(NULL, ",");
        if (stockfish_eval_str[0] == '#') continue;  // skip checkmates

        int stockfish_eval = atoi(stockfish_eval_str);

        Chess* chess = Chess_from_fen(fen);
        if (chess == NULL) continue;  // failed to parse FEN
        int sigmazero_eval = eval(chess);

        // clamp error
        double sigmazero_eval_sigmoid = 1 / (1 + exp(-sigmazero_eval / 300.0));
        double stockfish_eval_sigmoid = 1 / (1 + exp(-stockfish_eval / 300.0));
        double diff = sigmazero_eval_sigmoid - stockfish_eval_sigmoid;
        double loss = diff * diff;
        total_loss += loss;
        fens_count++;
        free(chess);
    }

    printf("%lg\n", fens_count ? (double)total_loss / fens_count : 0.0);
    fclose(f);
    return 0;
}

int test() {
    Chess* chess = Chess_from_fen(
        strdup("rnbqk2r/ppppnpp1/4P2p/6N1/1b6/8/P1PKPPPP/RNBQ1B1R w q - 0 5"));
    int king_i = Chess_friendly_king_i(chess);
    Chess_fill_attack_map(chess);
    bitboard_t moves = KING_MOVES[king_i], castles = 0;
    bitboard_t friendly_bb = Chess_friendly_bb(chess);
    bitboard_t enemy_bb = Chess_enemy_bb(chess);
    bitboard_t all_bb = friendly_bb | enemy_bb;

    // Add castling moves if available
    int n_checks = chess->enemy_attack_map.n_checks;
    if (n_checks == 0) {
        bitboard_t kingside_empty = king_i == 4 ? 0x0000000000000060ULL : 0x6000000000000000ULL;
        bitboard_t queenside_empty = king_i == 4 ? 0x000000000000000EULL : 0x0E00000000000000ULL;
        if (Chess_castle_king_side(chess) && !(all_bb & kingside_empty))
            castles |= king_i == 4 ? 0x0000000000000040ULL : 0x4000000000000000ULL;
        if (Chess_castle_queen_side(chess) && !(all_bb & queenside_empty))
            castles |= king_i == 4 ? 0x0000000000000004ULL : 0x0400000000000000ULL;
    }

    // Simple filtering with attack map
    bitboard_t illegal_squares = friendly_bb;
    if (n_checks > 0) {
        // A king can't block it's own attack, so remove squares of the attack blocking bitboard
        // Except if those squares are occupied by enemy pieces, in which case the king can capture
        bitboard_t blocks = chess->enemy_attack_map.block_attack_map & ~enemy_bb;
        // The mask from the attack map is symmetric, so we can flip the moves bitboard
        // and apply the same mask to catch illegal moves in the opposite direction
        illegal_squares |= blocks | bitboard_rotate180_center(blocks, king_i);
    }
    moves &= ~illegal_squares;
    castles &= ~illegal_squares;
    printf("Initial king moves:\n");
    bitboard_print(moves);
    moves |= castles;

    printf("Legal king moves after attack map filtering:\n");
    bitboard_print(moves);

    // Remove king from bitboards to avoid self-attacks
    if (chess->turn == TURN_WHITE) {
        chess->bb_white &= ~chess->bb.white_kings;
    } else {
        chess->bb_black &= ~chess->bb.black_kings;
    }

    // Filter move squares that are attacked by sliding pieces using magic bitboards
    bitboard_t remaining_squares = moves;
    bitboard_t enemy_bishops = Chess_enemy_bishops_bb(chess) | Chess_enemy_queens_bb(chess);
    bitboard_t enemy_rooks = Chess_enemy_rooks_bb(chess) | Chess_enemy_queens_bb(chess);
    while (remaining_squares) {
        int i = bitboard_pop_lsb(&remaining_squares);
        bitboard_t move_bb = bitboard_from_index(i);
        bitboard_t sector1 = QUADRANTS[0][i] | QUADRANTS[2][i];
        bitboard_t sector2 = QUADRANTS[1][i] | QUADRANTS[3][i];

        bitboard_t bishop_moves = Chess_magic_moves_bb(chess, i, true, BISHOP_MAGIC_NUMS,
                                                       BISHOP_MAGIC_SHIFTS, BISHOP_MOVES);
        bitboard_t bishop_attackers = bishop_moves & enemy_bishops;
        if (bishop_attackers & sector1) {
            if (enemy_bb & move_bb) {
                moves &= ~move_bb;
            } else {
                moves &= ~(bishop_moves & sector1 & ~enemy_bishops) & ~move_bb;
            }
        }
        if (bishop_attackers & sector2) {
            if (enemy_bb & move_bb) {
                moves &= ~move_bb;
            } else {
                moves &= ~(bishop_moves & sector2 & ~enemy_bishops) & ~move_bb;
            }
        }
        remaining_squares &= moves;      // update remaining squares after filtering bishop attacks
        if (bishop_attackers) continue;  // skip rook checks since the square is already illegal

        bitboard_t rook_moves =
            Chess_magic_moves_bb(chess, i, false, ROOK_MAGIC_NUMS, ROOK_MAGIC_SHIFTS, ROOK_MOVES);
        bitboard_t rook_attackers = rook_moves & enemy_rooks;
        if (rook_attackers & sector1) {
            if (enemy_bb & move_bb) {
                moves &= ~move_bb;
            } else {
                moves &= ~(rook_moves & sector1 & ~enemy_rooks) & ~move_bb;
            }
        }
        if (rook_attackers & sector2) {
            if (enemy_bb & move_bb) {
                moves &= ~move_bb;
            } else {
                moves &= ~(rook_moves & sector2 & ~enemy_rooks) & ~move_bb;
            }
        }
        remaining_squares &= moves;  // update remaining squares after filtering rook attacks
    }

    printf("Legal king moves after magic bitboard filtering:\n");
    bitboard_print(moves);

    // Filter for non-sliding pieces (king, pawns, and knights)
    remaining_squares = moves;
    while (remaining_squares) {
        int i = bitboard_pop_lsb(&remaining_squares);
        bitboard_t move_bb = bitboard_from_index(i);

        bitboard_t knight_attackers = KNIGHT_MOVES[i] & Chess_enemy_knights_bb(chess);
        if (knight_attackers) {
            moves &= ~move_bb;
            continue;
        }

        bitboard_t pawn_attackers = PAWN_CAPTURES[chess->turn][i] & Chess_enemy_pawns_bb(chess);
        if (pawn_attackers) {
            moves &= ~move_bb;
            continue;
        }

        bitboard_t king_attackers = KING_MOVES[i] & Chess_enemy_kings_bb(chess);
        if (king_attackers) {
            moves &= ~move_bb;
            continue;
        }
    }

    printf("Legal king moves after non-sliding piece filtering:\n");
    bitboard_print(moves);

    // Remove castling moves if the king would be in check in the intermediate square
    if (king_i == 4) {
        // if f1 is attacked, remove g1
        if (!(moves & 0x0000000000000020ULL)) moves &= ~0x0000000000000040ULL;
        // if d1 is attacked, remove c1
        if (!(moves & 0x0000000000000008ULL)) moves &= ~0x0000000000000004ULL;
    } else if (king_i == 60) {
        // if f8 is attacked, remove g8
        if (!(moves & 0x2000000000000000ULL)) moves &= ~0x4000000000000000ULL;
        // if d8 is attacked, remove c8
        if (!(moves & 0x0800000000000000ULL)) moves &= ~0x0400000000000000ULL;
    }

    printf("Legal king moves after castling intermediate square filtering:\n");
    bitboard_print(moves);

    // Restore king bitboards
    if (chess->turn == TURN_WHITE) {
        chess->bb_white |= chess->bb.white_kings;
    } else {
        chess->bb_black |= chess->bb.black_kings;
    }

    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2 || strcmp(argv[1], "help") == 0 || strcmp(argv[1], "--help") == 0 ||
        strcmp(argv[1], "-h") == 0) {
        help();
        return argc <= 1;
    } else if (strcmp(argv[1], "version") == 0 || strcmp(argv[1], "--version") == 0 ||
               strcmp(argv[1], "-v") == 0) {
        return version();
    } else if (strcmp(argv[1], "test") == 0) {
        return test();
    } else if ((argc == 4 || argc == 5) && strcmp(argv[1], "play") == 0) {
        int millis = atoi(argv[3]);
        if (argc == 4) {
            return play(argv[2], millis, NULL);
        } else {
            return play(argv[2], millis, argv[4]);
        }
    } else if (argc == 4 && strcmp(argv[1], "moves") == 0) {
        int depth = atoi(argv[3]);
        return moves(argv[2], depth);
    } else if (argc == 4 && strcmp(argv[1], "eval") == 0) {
        Chess* chess = Chess_from_fen(argv[2]);
        int depth = atoi(argv[3]);
        if (!chess) return 1;
        return eval_command(chess, depth);
    } else if (argc == 3 && strcmp(argv[1], "hash") == 0) {
        Chess* chess = Chess_from_fen(argv[2]);
        if (!chess) return 1;
        printf("%" PRIx64 "\n", Chess_zhash(chess));
        return 0;
    } else if (argc == 3 && strcmp(argv[1], "kingsafety") == 0) {
        Chess* chess = Chess_from_fen(argv[2]);
        if (!chess) return 1;
        return king_safety_command(chess);
    } else if (argc == 3 && strcmp(argv[1], "scores") == 0) {
        Chess* chess = Chess_from_fen(argv[2]);
        if (!chess) return 1;
        return move_scores_command(chess);
    } else if (argc == 3 && strcmp(argv[1], "minmax") == 0) {
        int depth = atoi(argv[2]);
        return minmax_command(depth);
    } else if (strcmp(argv[1], "eval_loss") == 0) {
        return compute_eval_loss();
    } else if (strcmp(argv[1], "baseline_loss") == 0) {
        return compute_baseline_loss();
    } else {
        for (int i = 0; i < argc; i++) {
            printf("\"%s\" ", argv[i]);
        }
        printf("\n");
        help();
        return 1;
    }
}
