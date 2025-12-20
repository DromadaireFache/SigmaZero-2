import copy
import math
import os
from pprint import pprint
import random
import shutil
import sys
import time

import chess

try:
    import sigma_zero
except ModuleNotFoundError:
    from . import sigma_zero

TIME = 100          # milliseconds per move
OLD = "V2.6"        # Old engine version to compete against
RAND_NOISE = 0.1    # Random noise to add to constant mutations

TIME_CUTOFF = 100   # milliseconds per move for cutoff training
CUTOFF_CONSTS = [
    "PROMOTION_MOVE_SCORE",
    "KILLER_MOVE_BONUS",
    "PAWN_VICTIM_SCORE",
    "KNIGHT_VICTIM_SCORE",
    "BISHOP_VICTIM_SCORE",
    "ROOK_VICTIM_SCORE",
    "QUEEN_VICTIM_SCORE",
    "KING_VICTIM_SCORE",
    "PAWN_AGGRO_SCORE",
    "KNIGHT_AGGRO_SCORE",
    "BISHOP_AGGRO_SCORE",
    "ROOK_AGGRO_SCORE",
    "QUEEN_AGGRO_SCORE",
    "KING_AGGRO_SCORE",
    "SELECT_MOVE_CUTOFF",
]
MINMAX_VALUES = ["depth", "nodes", "first_move_cutoff_%", "beta_cutoff_%", "avg_cutoff_index"]

# FENs to use for tournament
FENS = []
with open("data/FENs.txt", "r") as f:
    for line in f:
        fen = line.strip()
        if fen:  # append twice for both colors
            FENS.append(fen)
            FENS.append(fen)

# FENs to use for cutoff training
CUTOFF_FENS = []
with open("data/training.txt", "r") as f:
    for line in f:
        fen = line.strip().split(",", 1)[0]
        if fen:
            CUTOFF_FENS.append(fen)

best_consts = {
    "PAWN_RANK_BONUS": 7,
    "PAWN_VALUE": 113,
    "KNIGHT_VALUE": 340,
    "BISHOP_VALUE": 377,
    "ROOK_VALUE": 508,
    "QUEEN_VALUE": 958,
    "KING_VALUE": 400,
    "PROMOTION_MOVE_SCORE": 9232,
    "FULLMOVES_ENDGAME": 55,
    "QUIES_DEPTH": 9,
    "MAX_EXTENSION": 2,
    "BISHOP_KING_PROX": 97,
    "ROOK_KING_PROX": 126,
    "QUEEN_KING_PROX": 122,
    "KILLER_MOVE_BONUS": 59,
    "PAWN_VICTIM_SCORE": 88,
    "KNIGHT_VICTIM_SCORE": 127,
    "BISHOP_VICTIM_SCORE": 116,
    "ROOK_VICTIM_SCORE": 126,
    "QUEEN_VICTIM_SCORE": 167,
    "KING_VICTIM_SCORE": 88,
    "PAWN_AGGRO_SCORE": 28,
    "KNIGHT_AGGRO_SCORE": 55,
    "BISHOP_AGGRO_SCORE": 64,
    "ROOK_AGGRO_SCORE": 64,
    "QUEEN_AGGRO_SCORE": 96,
    "KING_AGGRO_SCORE": 37,
    "SELECT_MOVE_CUTOFF": 12,
    "PS_BLACK_PAWN": [0, 0, 0, 0, 0, 0, 0, 0, -50, -52, -50, -50, -50, -50, -50, -48, -9, -10, -20, -33, -30, -20, -11, -10, -4, -2, -10, -24, -26, -10, -6, -6, 0, 0, 0, -20, -20, 0, 0, 0, -4, 4, 9, 0, 0, 11, 5, -6, -5, -10, -10, 20, 20, -10, -10, -4, 0, 0, 0, 0, 0, 0, 0, 0],
    "PS_WHITE_PAWN": [0, 0, 0, 0, 0, 0, 0, 0, 6, 9, 10, -20, -20, 10, 10, 5, 5, -5, -10, 0, 0, -10, -5, 5, 0, 0, 0, 20, 20, 0, 0, 0, 5, 5, 11, 23, 25, 11, 5, 5, 10, 9, 19, 30, 30, 20, 10, 10, 50, 50, 48, 50, 50, 50, 50, 50, 0, 0, 0, 0, 0, 0, 0, 0],
    "PS_BLACK_KNIGHT": [50, 40, 31, 30, 30, 30, 40, 50, 40, 20, 0, 0, 0, 0, 20, 37, 30, 0, -10, -15, -15, -10, 0, 32, 33, -5, -15, -19, -20, -15, -5, 30, 30, 0, -15, -20, -20, -15, 0, 30, 33, -5, -10, -15, -15, -10, -5, 30, 40, 20, 0, -5, -5, 0, 20, 37, 50, 40, 30, 32, 30, 30, 40, 52],
    "PS_WHITE_KNIGHT": [-50, -40, -30, -30, -30, -30, -40, -50, -40, -20, 0, 5, 5, 0, -20, -40, -30, 5, 10, 15, 15, 10, 5, -30, -30, 0, 15, 21, 20, 16, 0, -31, -30, 5, 15, 20, 20, 15, 5, -30, -30, 0, 9, 15, 14, 11, 0, -31, -40, -20, 0, 0, 0, 0, -20, -40, -50, -40, -30, -30, -30, -30, -40, -50],
    "PS_BLACK_BISHOP": [19, 9, 9, 11, 10, 9, 9, 20, 10, 0, 0, 0, 0, 0, 0, 10, 10, 0, -5, -10, -10, -5, 0, 10, 9, -5, -4, -10, -10, -5, -5, 10, 10, 0, -11, -10, -10, -10, 0, 10, 9, -11, -10, -10, -11, -10, -10, 10, 11, -5, 0, 0, 0, 0, -4, 10, 20, 10, 10, 10, 10, 10, 10, 18],
    "PS_WHITE_BISHOP": [-20, -10, -10, -11, -10, -10, -11, -20, -10, 4, 0, 0, 0, 0, 7, -10, -10, 10, 10, 9, 9, 10, 10, -10, -10, 0, 9, 10, 10, 10, 0, -10, -10, 6, 6, 10, 10, 5, 5, -10, -10, 0, 5, 10, 10, 4, 0, -10, -10, 0, 0, 0, 0, 0, 0, -10, -20, -10, -10, -9, -10, -10, -10, -20],
    "PS_BLACK_ROOK": [0, 0, 0, 0, 0, 0, 0, 0, -3, -10, -11, -10, -10, -10, -11, -4, 6, 0, 0, 0, 0, 0, 0, 5, 6, 0, 0, 0, 0, 0, 0, 4, 5, 0, 0, 0, 0, 0, 0, 4, 5, 0, 0, 0, 0, 0, 0, 5, 5, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, -4, -5, 0, 0, 0],
    "PS_WHITE_ROOK": [0, 0, 0, 5, 5, 0, 0, 0, -4, 0, 0, 0, 0, 0, 0, -4, -5, 0, 0, 0, 0, 0, 0, -5, -4, 0, 0, 0, 0, 0, 0, -5, -6, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -4, 5, 10, 10, 10, 9, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0],
    "PS_BLACK_QUEEN": [20, 10, 10, 4, 5, 10, 10, 20, 10, 0, 0, 0, 0, 0, 0, 10, 10, 0, -5, -5, -4, -5, 0, 10, 5, 0, -5, -5, -4, -3, 0, 5, 0, 0, -4, -5, -5, -5, 0, 5, 10, -5, -5, -5, -5, -5, 0, 10, 10, 0, -5, 0, 0, 0, 0, 10, 20, 10, 9, 5, 5, 10, 10, 20],
    "PS_WHITE_QUEEN": [-21, -10, -11, -5, -5, -10, -10, -20, -10, 0, 5, 0, 0, 0, 0, -9, -10, 5, 5, 5, 5, 5, 0, -10, 0, 0, 6, 5, 5, 5, 0, -5, -5, 0, 5, 5, 5, 5, 0, -5, -9, 0, 5, 5, 5, 5, 0, -10, -10, 0, 0, 0, 0, 0, 0, -10, -20, -10, -10, -5, -5, -10, -10, -22],
    "PS_BLACK_KING": [30, 39, 39, 53, 50, 40, 40, 31, 30, 40, 43, 50, 50, 36, 40, 30, 27, 40, 40, 50, 47, 39, 40, 31, 30, 40, 33, 50, 50, 40, 40, 30, 20, 25, 30, 40, 40, 30, 30, 20, 10, 20, 21, 21, 20, 21, 20, 10, -20, -20, 0, 0, 0, 0, -20, -20, -20, -30, -9, 0, 0, -10, -32, -20],
    "PS_WHITE_KING": [20, 27, 11, 0, 0, 10, 28, 20, 18, 20, 0, 0, 0, 0, 20, 18, -10, -20, -20, -20, -21, -20, -22, -10, -21, -31, -31, -40, -40, -31, -30, -20, -30, -40, -40, -50, -50, -40, -40, -27, -27, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -51, -50, -40, -40, -30, -30, -37, -40, -50, -56, -40, -36, -29],
    "PS_BLACK_KING_ENDGAME": [50, 40, 30, 20, 20, 28, 40, 48, 30, 19, 11, 0, 0, 10, 20, 30, 31, 10, -20, -30, -30, -20, 10, 32, 30, 10, -29, -37, -40, -30, 9, 33, 30, 10, -31, -40, -40, -30, 9, 30, 30, 10, -21, -30, -30, -20, 11, 27, 30, 31, 0, 0, 0, 0, 30, 29, 50, 30, 27, 29, 32, 30, 30, 50],
    "PS_WHITE_KING_ENDGAME": [-50, -28, -30, -30, -29, -30, -30, -50, -30, -30, 0, 0, 0, 0, -30, -32, -30, -10, 20, 30, 30, 20, -11, -30, -32, -10, 30, 41, 40, 30, -10, -32, -30, -10, 30, 40, 39, 30, -10, -30, -30, -9, 19, 31, 30, 20, -10, -30, -30, -20, -10, 0, 0, -10, -20, -30, -50, -40, -30, -20, -21, -29, -40, -49],
}

ZOBRIST_CONSTANTS = """
// Zobrist hashing values
const uint64_t ZHASH_WHITE_KING[] = {0x190d3964b05efd1f, 0x76ba326c46c7b9f7, 0x43f25f9ef54d972a, 0xf4d2c72508000759, 0xe45cd2f16980e4b5, 0xe48bcad5b0f8c977, 0x809dce25ef0c7cf, 0x4b4c28bf4913b678, 0xb3f33a622e4bc220, 0x23c0b958a206d414, 0x6059184aae8db862, 0x80d72d5ff5e02e8f, 0x6f2247912cc1d5f5, 0xf45debf453ab3021, 0xcee7ea6ee2d16b31, 0x2d957f8a941544ab, 0x36fd2aea203e9cc, 0x488b189afb040e0e, 0xeb2b5b4793886f48, 0xc664a39aed1a74dc, 0xafa3d2221ef4e7f, 0xf036256846d3c6ad, 0x130e1850d09d24, 0x7cd2c5fd9354ffea, 0xaad00f1e48927873, 0x9bef315acf5f2496, 0x2426d06922d6eb8c, 0x83642d088b7673c7, 0x182c74fc92a92931, 0xa4121f3bbfbb2ffc, 0xb3c335e442e7e9b1, 0x5cb162ce7ed74aa5, 0x3fc825f57b178d13, 0x950961285684eee2, 0x95b8d90fdc2c3ce, 0x29563f564b9b18ca, 0x723ebfee5801da09, 0x33ded6e553cbfd13, 0xd741192ce74a7d6d, 0x712e6e92296a9e83, 0x8f85e00b298b96ac, 0xe48a0e91b37e09c5, 0x7651e2887c2a8a33, 0x72a22b5d6d1f0acf, 0x768a648a19731c12, 0x8d941992ddedc4b9, 0xfcc23484a16ee7c9, 0x59f8d1ea20218f58, 0xdd4a208352555f73, 0x78abe86569a45fed, 0xcba994ae95e58113, 0x978bb6a8f0c3e2ca, 0x3c1364b59c95f0ea, 0x45801c8500c7f4e4, 0x266de5331919292a, 0x7a6eae9d0f69f9e4, 0x32863caa2bf1675, 0xcfb3a18208681e85, 0xde0ccf4a9e5ea4b9, 0x171e274ec27d334d, 0xd43ad29140e71cf, 0x4eb06332eca60cfe, 0xd1ca58f5e017e68e, 0x53dc85952f5aeca4};
const uint64_t ZHASH_BLACK_KING[] = {0x1928d70da7513ef9, 0x18bbbcdf1da3c1f5, 0x665145d367ff7719, 0x9d1738ca88ba7cf1, 0x730dd76b944ccf7, 0x3551125162249e7, 0x8f58ba36ec4c169f, 0x553ec750e43fd6b1, 0xf6b44a4c50b67361, 0x553c1f8ede3b9e91, 0xe15a13641a2a1dcc, 0xe0f522362ae33072, 0xbea55f9c636492e8, 0x4f468d531a0246c9, 0x1dfe2f667d70fd6a, 0x768856d0efa8e097, 0x9b97148fe83bf821, 0xea0d901139ec7631, 0xe64b7b6002e2781a, 0x4de0b121ca159b21, 0xcbabb86b68e81069, 0xd6ac36b61bdbe7e4, 0x4173d5452f8d5cf, 0xe86a923ccdf3a0c2, 0x2c99ce0b7456511b, 0xd0e0823b76fcedf8, 0x189c2f70a5471050, 0xbaec9006ff6aafd1, 0xeb7950524b98c729, 0xc5ac4e8094d7ccdd, 0xff4b3e426dee7aaf, 0x7a78a7756299ab41, 0x52f19ce1acf0701e, 0xaaa4d59b46d78697, 0xd1422ab88f08960d, 0xacb12c658314290b, 0x3ab70aa62f2cecc5, 0x4bfc2b316a686803, 0x760244208a92044a, 0xe70e9b4d3f93c0df, 0x9b23975ec7752c08, 0xb26b2667235c3ca1, 0xe288f1b1c19c4192, 0x4eeda9b0bdd5bb15, 0x8b6ac11a31bb1f86, 0x92b087be2813851f, 0xb2721a5da6dca66e, 0x9c11c9b11c8d5e7, 0x12dfbe4681174078, 0x7fe0af69f25d720b, 0x209ae0d11a55af08, 0xb803fa2bcf20414d, 0xfce57f01caafb648, 0xed17ef89f203ab97, 0xb9a4f498d2c0ce55, 0x802c265b59cddfe7, 0xe4dda132e9f64b9, 0xf894168910ac6206, 0x6cdff5b65d5741c5, 0x31e750d6475554aa, 0x72d56accbfb19e46, 0x961b4ae82056fe0b, 0x65367b3a4b3d724a, 0x7358c69fde54b26b};
const uint64_t ZHASH_WHITE_QUEEN[] = {0xe740266307eda47b, 0x8431efb488379bc1, 0xd03c5dac3f5810e1, 0x1c443fcf1c240dbf, 0xa785b449528787af, 0xa4dbff039b7f65d3, 0x36fb997a3072bc84, 0x48641e5ffaa2efc0, 0x2ad04494a8e38a95, 0xc7b464a8e9bbbdf4, 0x4d5954ba79cc5a97, 0xf8befcb9e552df7a, 0xb4919a74a2cac0bb, 0x6f759775006bbf0b, 0xc119091938c076fe, 0x6d956354f6c37620, 0x7211b59f478103f8, 0xa19a5fb503341238, 0x11add79239c9405a, 0xeadf4a627af8d1d0, 0x9d3b46c6297eb4eb, 0x372978e109b96f7f, 0xf8fa192667fdd998, 0x93c6906ed24c76f5, 0x253c451848d5797b, 0x47fb3fc25ad8f67a, 0xd264d8aaf984c55, 0x219fef3dd3f34734, 0xef5a64eca270e784, 0x3e17e61c859fbfea, 0x888fdc2d37b26c22, 0xf3cb426758470784, 0xe6657d72758339cd, 0x50e1584c76ac6eac, 0xded16443eb2e84ea, 0x9d9b0f47f48f8e51, 0x47df7538999fffeb, 0x2d8c9c3831195ed4, 0x10dce73057a18537, 0x48905134d6e6c51e, 0x4b6073166d09a202, 0xb83d1fa874b9b43, 0x36a1f1cdc9579412, 0x769a28120c542590, 0x90ee5e22deb346a2, 0x3fd1840285f9692c, 0xa812512dbd85a3e, 0x1d85aa9ee41901d7, 0xf50998c5ee7c6443, 0x2d499150ef2a9252, 0xc1108811b322ef0a, 0x9256038ede412ac0, 0x43d25245dee0c12d, 0x597c9bd14cbfbffc, 0x249c8cb8294a6a5a, 0xd4706513ef5ca6af, 0x192b3365be83285f, 0xd3267d84383874a2, 0x6eb5c96de7439e05, 0xcdcfe5ccee949002, 0x92f6b39a9a0de001, 0x8bceb9a1660b4b7f, 0x4b2dc5884687f22, 0x2a21890621b26849};
const uint64_t ZHASH_BLACK_QUEEN[] = {0xdcb1ab611084fced, 0xb7e8c6b89fbc207e, 0x24a64ae81135bf7, 0x4558dcd718f57490, 0x795cb3f9a8720608, 0x8abd70312e51e0ab, 0xb7c4040d2268c80c, 0x1f68404e76de5ea6, 0x445aa7dd8b271426, 0xdd8bee65961248ac, 0x1b95885cfbb1e4fb, 0x2677d01c6f9a52b8, 0x33f864a98883a09b, 0xa9136c05d0c0ee35, 0xd2fe75a2d8dc321e, 0x60124798e9f04c1e, 0xcc1abd9d01bf8779, 0xd6213b7b1fa7899a, 0x151253a97ad7f260, 0x7f77311e8cf08672, 0x78607fd10633441e, 0xbacd65ba1350442b, 0x712b31896b2bcf1, 0x469a286c324c448a, 0x7a0929ee9e5de615, 0xb10f39150038546f, 0xfaa6d8a5dad0fe88, 0x48f0266f2591a335, 0x33b26f0567e090fc, 0x5c86d8f49dccc8f7, 0x72ce739db4404316, 0xa60d0fd20560dbd5, 0x17d08f6f91630033, 0xd0f91195946fdc0f, 0x64bf139600813656, 0x67db4023536d44b2, 0xaae695c8c59ba0e6, 0x7152bebf712be91, 0xdb55395b2052482b, 0x225c6d0f7f130f51, 0x82e5b6982679835a, 0x508cdf1597efecc8, 0x99d4258b9ec579e8, 0xdb3230e44ca0cf23, 0x4165e121547dfd8e, 0x9c3ed2dea6799ef3, 0xea996da67dd0c8d0, 0x999ed41fa4c80d5d, 0xe0898d39be007902, 0xf57edd278bdca62d, 0x77be9fb4cb08019e, 0xa6626d8100f40ca5, 0xc80d7e5b49ed0e44, 0x85b4dab58d97ca93, 0x9ef22ea382f9bf32, 0xa1e75bbfe223d551, 0x710846cc5342a038, 0x800251f682704bf3, 0x37c49c05873ddcf4, 0xc7b71866bd259885, 0x56bf4e84325558c1, 0x64a0ad29267bc5ef, 0xc49b4ce413b76d7b, 0xf7c20ec7c279d76e};
const uint64_t ZHASH_WHITE_ROOK[] = {0x608d15a55e1ff685, 0xec15373c6054ea78, 0x9a504586ff671f82, 0xe5d7f5d14a2f8768, 0xd9c6523c40b96d9e, 0xc51d4e99fcc80832, 0x2ee5b2f52796e5ab, 0xc535b3b6bc04d94b, 0x34a9d7466f3c74c5, 0xd38faacb98c60ddf, 0x1cd8cd79366e3651, 0x8532efa77c1e2764, 0x2780fc65da4b4a7b, 0xb62cc6334b3d18e6, 0x4b6d5ff416269135, 0x36df3dae3d962d1a, 0x14bf66b7700dddc3, 0x8784565b8d125714, 0x13bd00df72d9dce7, 0x3b241a20db1ffe85, 0xdd57d32388e41da5, 0xc0a2bc7d4fb0af7a, 0x776a3f299adc3797, 0x12c5c6f081648b55, 0xf842d6185a5d1186, 0x73e863bc4f80dc26, 0xad027d3b1bea5db, 0xee002a541a36478b, 0xba021f70d532e731, 0x64e468a727e41e6, 0x3309c7774c6cf2bb, 0xa94376e2efff921c, 0x2d2ef3c6a841fd75, 0x971e032f28a4cafc, 0x6f199f21936f84ba, 0xafa71be4185a64cf, 0xc8317a40f82a8321, 0x8474e88b016ef78c, 0xc1d83e6b8292c679, 0xc56dbc8a9e9fdbd7, 0x462e2465bd1869b3, 0x96bc94b368aee3b8, 0x7eabc6786c0fbbfb, 0x8f28240fbe800e0b, 0x1495b205236d63be, 0x2f161d581edf2574, 0xb761ca6a0f1d64c0, 0x1d46d4c721a61a0b, 0x23523736707ae311, 0xb102b7ae2becdb9, 0xc80332d5727f7d49, 0xe17b5799d7ba93df, 0x1b461faa5a3cbbcd, 0x6a271f65fc3c6322, 0xd597dea11aa10657, 0x64602443eb8f85dd, 0x815ea6e5d519eae9, 0x18280cd0d3c71404, 0x8399ee6bce79fa64, 0x70f9794a94d37032, 0x94ce6e8923675f5a, 0x6d97cd2730ef7181, 0x39e40b62f8fe8118, 0x2a059919fb6659a};
const uint64_t ZHASH_BLACK_ROOK[] = {0xaf3da9d8af2cd6e, 0xaff935489cea4872, 0x75a512689face917, 0x77a40f1ead80fba6, 0x72c53b5afe0d7302, 0xfff547a4249e011c, 0xdac4f8aa7c4f113, 0x3a0a2679a87ec2a1, 0x5da756d7cf085564, 0x73fb9029f469ad8c, 0x525b03bae425ffe3, 0xf32983bd014d7cc4, 0xa75c0c4a6bf35377, 0x38a687a51c7846bf, 0x4d9629f70fab3744, 0x87ce4f1ef87d8cee, 0x9b5f07ba08de174c, 0xe8ac039cf2cd0c3b, 0xf9743b543ee7b99, 0x872400fbbf6558da, 0xc74d7fdedd41577, 0x8f520ed00498bb65, 0x93729c4846886759, 0x5a9b3fbb37ba30f8, 0xa63b347a617df65b, 0x3cc80360b588f24f, 0xc836a2619704d0ca, 0x66ef81a63bf7bf8a, 0x14d7758e3fecf0bc, 0xcab220eb802b4fab, 0x47df2dff506d9b54, 0xf5ce1530221dd07a, 0xff2498e2fd8c245f, 0x1b36a3dad73790ac, 0xf374858a739e1372, 0xdd7298a7ba828862, 0x1b428329ccf610bb, 0x970beffdc850a3ff, 0x77a363b62d4cfa98, 0x6b278c7f9d45da43, 0x9477903206c8be5, 0x7547b645b2cc40db, 0x5e487a49d6f5ad11, 0xa72e6ed7e62fd158, 0xd24cb93c3eecb29e, 0xa44566ba60cbbb42, 0x29b83680e21c68b1, 0xd116f63e3ee93771, 0xc022dc777f171387, 0x1b463a8d59f2f5fc, 0xe81e3d03340eb9e8, 0xcf56cda1bd323271, 0xcd91e8ac36d85091, 0x89ff6503317d087d, 0xb706e0b16ad84978, 0x812f2ed3ed0481e7, 0x75480d16208f7d45, 0xb6bf191fa271be16, 0xe8fbfe36765d8d51, 0xb5854e78d070b17a, 0x2dc42ce51d0e8fd1, 0x39c1d0166bac20f8, 0x92e5b4cf71a90856, 0x8aecd4a134c97f0c};
const uint64_t ZHASH_WHITE_BISHOP[] = {0xfa74adad11072ade, 0x2e44433bf8f4ce4d, 0x200604fddfd40d74, 0x6dc267305a3b19b8, 0x7109112da97d6e7c, 0x84b6b76320c2a651, 0x4443bd78f0a93d59, 0x45b71dd3be2f0fae, 0xdcc8b09b7e69bcdc, 0xf851f89dbf9d1c6f, 0x10997632d18b4193, 0x66037494022c1aa1, 0xc1b0102ac21005aa, 0x1662cb1d5bd4b6a6, 0x3352f5f2f897a5bd, 0xb39ceac02b98106c, 0xd1e0d478aef3be96, 0xbeefd1575f4330a3, 0x46e1b9a88537b8ab, 0x5bfa8db84767652f, 0x5b2c2daffa11fb6c, 0xbf70d71d949ca756, 0x4263677cd328984e, 0xb1b8afa25b0adb61, 0xb28b868a4c00393a, 0xfdd4e2554de7eb6, 0x545f99e3d8a2e1fb, 0x81e0b287d763a269, 0xea17f36b19bfa831, 0xcf4aa7093132b3a7, 0xb1f885e00e3ef07f, 0x80889c8f7cd7ac26, 0x1fdac766b8b2a19c, 0xb589b7d7cedc781f, 0x9fe3d597e3700777, 0x6b855f8a4f8312d2, 0xa54d2c2fff3e3efa, 0x6054a6fb06477b5f, 0xfc529bfa7a48d980, 0xf83076e4a6adbbf2, 0xb919c67fd8761ce1, 0xf73473ec2903be5a, 0xfe6e276c685b2056, 0x76ee0ef0ab563793, 0x37b823ea4360a700, 0x5f1c2476326467c2, 0x52993d9f389de1ec, 0x68f9825fa08de949, 0x31768135345f5185, 0xf5fd5de5a1d14c1b, 0x1305b4f6620e37a2, 0x87b38a0c26f1b09d, 0xcfc948f24ecb9bcf, 0xd0aac9a9b6e50430, 0xa7e11ebcef0cc59a, 0xcf2a59e8b0ecdc4f, 0x9bf663a51d6208a0, 0xdb64e76df81cdda0, 0x2a04b2cce754d133, 0x6b8f540e063971ff, 0x54fc5c673083bdd8, 0xba561b5221e97157, 0x58916731b4b0f3d3, 0xeecc034200df9e18};
const uint64_t ZHASH_BLACK_BISHOP[] = {0x3e3c00d5be4037b, 0x6d11154da182c826, 0xb502b216c0d2cbaf, 0xa6b6c3c26fa9b3df, 0x7a79071cf1fbf386, 0x9940d0fc4d13096b, 0xb2c882708aaf1f24, 0xa044d3e6e79684b9, 0x6905336d000408d8, 0xbd836e702ca24d1c, 0x608ff93982f988f6, 0xf1b36e74248d7d12, 0xc6255236d394a104, 0x457b37eafca4a028, 0x1acb84e9afb1d992, 0x91544f3d710384c1, 0x18709f87090e2da0, 0x843c452ac8bc8072, 0xcd665fdc4fd25637, 0x40903c3d3f0d0df9, 0x9759985b600fc249, 0x8c5994be5770dfe2, 0x3b4187f87869b02d, 0xc82458d4b1595c63, 0x61cd686be47628bf, 0x9b41d27e0e402e1c, 0x6d67f208a6cd7393, 0xbfb13a86b6103ac0, 0x4658a1c595c13e45, 0x48717296a1038578, 0x2cfe0c95354390bc, 0xf2d0a14f21916a5, 0x72cafd713d6172d1, 0x8a45cc30e12b573c, 0xa8660ed20e59289b, 0xe08e75cfe221a647, 0xb69880329178ef5f, 0xf789b3e80776929b, 0xaec872f14a3ae356, 0x5856ca7c08ed1b31, 0xe607b3ebb3c37283, 0xab419b9e86e9e2a3, 0xb05f6035d0f74d13, 0x7fa7246c014043ad, 0x1c7e959963312495, 0x7ca472f644ff8028, 0x252c21e40bc7226a, 0x1705cfc290b8f38f, 0xa6fe4fd7cb3684a5, 0xbebc52d31afca543, 0xb6a3bf0f8b9a137, 0x6b0d36ca66e962c6, 0xa48b922a5884388e, 0x2f5f6ce852c371d9, 0xd40d57793270e700, 0x3be042ed5e60f4d8, 0xc0d5579860284b04, 0x2f898cf4bd3906d1, 0x9541738768ef77c5, 0x1e3d617429efea72, 0x83f477c1bbf3b6f6, 0xc769413f9ba7127b, 0x537e4d641fa8c8f, 0xede52256adfba331};
const uint64_t ZHASH_WHITE_KNIGHT[] = {0x609ca970ee205c42, 0x824c7d72e8b27148, 0x6ece98a637bbf3bb, 0x54b241579b4e74bf, 0x16bbe853a0c55483, 0x26fef27a964f736, 0x9aab522392d65a1f, 0x7212360b6fdea9f8, 0xdc270153ca3fd3af, 0x63f4d1ed3d51c268, 0x9d589ab971bb8470, 0xc7c8f521f08cf650, 0x3d29700da7c010e9, 0xfa51a924851ba334, 0x57857f0fc29c406e, 0xb6585e27d8f5ce83, 0x5854623f6ffd70d7, 0xd5e5d3d777c27e76, 0x9b228c083a30bc1a, 0xe1a3024145a5a535, 0xc1f1f2c5a58ab2b9, 0xac451ca715cea7fb, 0xd6e42371f1f14717, 0x612975ad8cd9eff9, 0xc6251a5a75fcf81c, 0x9a738adbc3b37758, 0x8abd06c1e2778cf1, 0x449156ed3f6eb850, 0x84fddddaa73cc789, 0x568cafa480b07ab9, 0x9940c00101fe1d75, 0x9a2de86b761e78fa, 0x770381ba46dbdbdf, 0x88a4309e085f6724, 0x15e32fd2d5e1c46f, 0x1426da205fe47c77, 0xc2f42b6aef14fa07, 0x402a9aa5aa30afc4, 0xf5d03f0aca95a225, 0xcfb23ef48d98a89, 0xbd56a9cc824e7cd3, 0x5fe354ae89b6f13a, 0xc1cf24d15242f16c, 0x6c6a856a4bd1c50c, 0x6035e1076d8bc44a, 0x6b73d0ab085b4517, 0x4cf11df413cdf9ad, 0x1889baa99d44daf5, 0x993b215fe47e44cd, 0x65149e9a76b7b56e, 0x3796920c731ce9c8, 0xee7aaa668dbf2c2a, 0x9ada56fed976166, 0x12cf1a00e0497d74, 0x5a23835f3dd6d896, 0x9a50ab1a5c7c95e8, 0x5395e9cff0fd75be, 0x6d6bb76e5358457d, 0xf675e99830e68555, 0xf8b7972a37473d6, 0x5c76ede588edc047, 0x6f333cc3b45ab561, 0x70dee893e674d784, 0x5de2474a3ba5f5e0};
const uint64_t ZHASH_BLACK_KNIGHT[] = {0xe00419be2433aa28, 0x895cb59885ae585e, 0x74618bb462e8008f, 0xb045cfe632e8c263, 0xdb89998c30b75b2a, 0x831e489ab08019e5, 0xf7f90e76ad338cdf, 0xd1f0dfe2cdaa388e, 0xe74c67ce67259064, 0x553ed0855b1dfe0e, 0x87aa2d8e6b1fe4c7, 0x3517f8e29efbc124, 0x4c9f7fa838a6dd8a, 0x9138aaa2105f6217, 0x5c92ae9ac59659c6, 0x6979607a883393e6, 0x78922264986555b7, 0x5721419bd782521c, 0x19e9080fc5f3bbcc, 0xc1fa9a447be758fb, 0x8c073a7b39449b2b, 0x8a46f14007250be9, 0x54430a67369cdde0, 0x323b6fa44a59cb0c, 0xd7aa59ca4b5be3da, 0xefcc8df07dfe3d2b, 0xa1e71b0509c253a9, 0x86106eff8c94fe64, 0x8b5bab2b6cb01ff, 0x61e5420e314de3d4, 0x6591bf02d2125105, 0x5f76097cf614c2f2, 0xf4ae07cace55ad04, 0xd53aba13727ecb12, 0xbedcd45ef26dc29a, 0x73939a4647013e1e, 0xd0f11ba8ec0891b8, 0x5ad43f10282b280d, 0xcbd8462ec377148c, 0x5839369c54317e78, 0x14c2d97bdec9e7f0, 0xacf2bd59461f24cd, 0xe6cfba3dfa79cc19, 0x990c98d5eccf5a06, 0xf2e76359a3e58b, 0xafce2e46af006384, 0x3d187c82c9b4e1b3, 0xf7bad241bb90e125, 0xbb2afef646449bd3, 0x8dc5098fb98706a3, 0xba955b8b59dc946f, 0x2b6fcac622ba6fbc, 0x4448446a29d01575, 0xdcd6a89d3989a1b8, 0x85ed35a6bc15a4b5, 0xdefe9f35f131932c, 0x4739c46560ef7dab, 0xee6b8e71e0e5a9e, 0x1f47464702efd121, 0xb610c6d30743dc7b, 0x611dfae93cc5aca2, 0x92ff9c1e87abb81c, 0x307c02fb4d52876a, 0xd3ee8ba84f2d598f};
const uint64_t ZHASH_WHITE_PAWN[] = {0x249c1c52e3244a31, 0x69307fd8826a8080, 0x2913d249bc674cec, 0x3f3b595d490e5696, 0xc53638c89e360f8a, 0x331fd89cc632cf7, 0xd26b86637bcce583, 0x894a2f44fa192706, 0x398f7ae86fd2e4ca, 0x74d995641787ec54, 0xf341a025da7aa965, 0x48dfb1da34adfd75, 0x3850d47e558f637d, 0x850ce8b83ebb1fe5, 0x4a3dc658be447f77, 0x3f73e4d4d35e7f1e, 0x5af97515f7ea2851, 0x89a6270c1d0fdb0a, 0xa651e9dc5bb1dba0, 0xd4eb1f0792fb0a0e, 0xccdc72607beb5584, 0x46ceb35f453923cd, 0x4cb6b21278919bf7, 0xebcb3fa512887fa5, 0x3e65e89f1ea5ebcb, 0x31b9c7edd002f6d3, 0xbab84b514e0c6938, 0x5dc082cf119c4c0c, 0xb8263c8d9f7df888, 0xb1ea0e3ef5006bef, 0xae87d6a9b5acab87, 0xe9a6fb48b7a2d40d, 0x8530f2dca88ccdc2, 0x348040ed18852a20, 0xffe1a2dbc26955a6, 0x6cc60ca50616e0d4, 0xb1e97bcefffbaf00, 0x8dc45ce94b2a6c27, 0x3b5fc7140b0a8231, 0x3000980b1c3f9108, 0x880db134aeddf59f, 0xbc322438c67cb73f, 0xeb5608a080139fac, 0x325e17e14877dc13, 0x901d20d4feb7764c, 0x84e09b231b7657ba, 0x5a8bfa0601dc157e, 0x41ea1d503a1a8ac5, 0x6dc50bbd68e422f6, 0x12445e5625963a46, 0x2968bd156d560da7, 0xfcb947d21705928b, 0xc968c54a4db6f30b, 0x496ec242e1d62d31, 0xd92e62583309a35e, 0x84c0c165368bcf3b, 0xc73b989c778d287, 0xc6eda516eb2f08c9, 0xef50cc4f2ef06edf, 0xbee95b6b6045f3fb, 0x23bf544c937dd7cc, 0x9031de9fbdad806b, 0x3c46050a3814acbe, 0x7254215c66143cb6};
const uint64_t ZHASH_BLACK_PAWN[] = {0x9152b9336d697c39, 0x9014b1137ede53de, 0xef418cf0aecbabf8, 0xdbadac9a1285ada, 0x857103b7bd70208a, 0xb09110929b58eaee, 0xc69316413c5d237e, 0x2e9a8cbdf87e7254, 0x912412aace1489bb, 0x7513593b80585003, 0x4da8b6c56ff4a7d5, 0xa55d37337bd2b4d4, 0x337179ac50e5b6be, 0x4c5b1c0ab221f7, 0x478d053c19257c24, 0x2c3a9e53cbfb0f7, 0x66c193478ae18c83, 0xced8920ede692e1b, 0x1dc852bae813f72, 0xc5c2d897e01036af, 0x91cdd9d1201c6e99, 0x7404670a6165151f, 0xafbcb4a19acded21, 0xeeb03425fe04947b, 0x38eafdd390b914be, 0x831ba7916d652e98, 0x112f9179ad23a08, 0x9ce46d2d5def8e22, 0xc290083f74fb9447, 0xbe6c2e3852ea1992, 0xc8f5525cf902b2e8, 0x5aaf3842ddf42ebc, 0xa6fd38f37bb2b574, 0xe3e33f9a9a22c677, 0xd2e7435848fad3b9, 0x5970bba88e627003, 0x2651faa2d0dd8252, 0x28536779fdc5aaf4, 0x11b7138287128c60, 0x4e43c7a04d07add0, 0xa6900d33a2f422d4, 0xb97a341eb864125e, 0x4edb2f32fb59cbef, 0x45aaecc73aa51807, 0xf51b346995b1997a, 0x1c65efe64908125d, 0xa07f019a3e73f2d9, 0x95ca1ff9562745f0, 0x9e7a55c92984a760, 0x7852f456895998df, 0xebedcbd072a564ba, 0xe9252809afd3251f, 0x4d6e57112a7ba23f, 0x34ae82dabf6ee722, 0xe568152babec2bfd, 0x354e9c62b471d061, 0xfa2146abb590b7ee, 0xb765ebea7bd8b4a4, 0x2f6c1b34347eb6f6, 0x669ec8e9b461e9bd, 0x16f751349198819b, 0xd7a449b91d7ef6e4, 0xd248fa3edcf95ffc, 0x1fff927c2779e01e};
const uint64_t ZHASH_STATE[256] = {0x119a61b56b2bc6c2, 0x24bcd8a0972e2a4b, 0x28da956e9242f336, 0xb73dd2020eba053, 0xe9b02d6e93e0618, 0x7b4eb1123ce39b5f, 0x10ad2fe1d84d82a9, 0xf66b2231297ec507, 0x4e74d1bd24c0a002, 0x7f2b54c04e43d8dc, 0xdc84c3c806df01f, 0x5d66450edfe7ce36, 0x3a5b2e621d17fd3c, 0x42a1e06df4f41b56, 0x21a1cb9d17ac9bfb, 0x666fff26ec58ec35, 0xe22663e32aea9ef7, 0x79f05c8d7d470da7, 0xe98c4cc0efbabddf, 0x810970ed06189e7e, 0x84ae8faf0ab9f364, 0xb356de94159e8236, 0xbb74ede183661c38, 0xab8854954da26896, 0x61eab10fca7d0e34, 0x6d7de4f86f3070d7, 0xd3b8b9a026494d49, 0x151a6a6725de5db5, 0xeba00ca0d0d02fc, 0x2b990eab56f24e1, 0x2e4296f290aca9d0, 0x696c49e21bfe3fd9, 0xdc42369f9cf5c365, 0x2d44c6899dbc762b, 0x269493713530b311, 0x58d8aaa869ab4966, 0x68f3ba2228ed72f, 0xbc2e8ea95f03d95a, 0x4f9a2532fc0a00cc, 0xd632d92496863079, 0x59e0f81fb456877b, 0x1d1a4fd3d01662a7, 0xb84cfcaf7304f241, 0xe34e966efc791df7, 0xaeac0ab5eb28859c, 0xad53421c3839a6e, 0x8c3278e55bede7e5, 0xb951e4e8b4590186, 0x7e703ee99c8202e4, 0x546e6c4e85f59abf, 0xe081a40d7d10e9c1, 0xe1d074bb9d7e1df4, 0x8b903bf4b6392979, 0x900b79ec86a08c74, 0x69c1f6e6e93eaea9, 0xd4b28f69c9af3ec3, 0x492bf7faaf3ff408, 0xc66641cd1586f4ca, 0xbdf9ba1457f607d4, 0x1b67fd61fe19b84d, 0xfc94573bdaf5ede5, 0x3efb6fa5617a49ad, 0xe1ac63f7b79ce293, 0x182c29d5b318c0d, 0x422804cc6d5a8024, 0x4a44a06cd10fbd9e, 0xbc416f684a2e9f4b, 0xfbe887ad0bff00a7, 0x2871d90e6e0350bf, 0xc16a7ee0defe7980, 0x68cf13e3b96c5f0e, 0x4421270e698bc0b8, 0x7d0ce4fab55155c6, 0x2f6392ca9e03f2de, 0xb92f80b24e19fb1, 0x884ef2cb3f503dd7, 0xd998de7b8a69e39c, 0xd6deacf2a1b39423, 0xaddf838e3d23b0b, 0x26948f7c55142e9c, 0x22170ad7a71e5dc6, 0xe0b785caf8fa227d, 0x93ca0a4163e28203, 0x37887f8f6c93efe2, 0xb19a82be828ab071, 0xe1e0c0183e45dea6, 0x4ad81d84acd59f6b, 0x564ab00b6d4fa94e, 0x62ec78735a47068a, 0x24619e3ad97ce6f4, 0x5462a8519c0c6b08, 0x9b0416d0c2c32d24, 0x469d15ad43ce745c, 0xd974b742cfeb5789, 0xa923ebc8ac21ebd1, 0xb7c6ee942423b096, 0x9186bd3a42c7ab71, 0x7120beb7ee792077, 0xd7bd189af96deedd, 0xcc4539df17c9bc88, 0x897eef745a128ee3, 0x80e223b6afec3409, 0x775817b419bb2009, 0x11a29a4e5e3793b0, 0x657ff5ca73fe020f, 0x8ee8773e3989897c, 0xd39c6857a6f79363, 0xd47f0337adccd9bc, 0x128bee7d049b834b, 0xc344088eb1f6810, 0x69a79ef0323cb6a, 0x96b5a9ef8448af4c, 0x433a4c52786fb3eb, 0x4cfbd280b2ed6222, 0x9b41b15ced8995f3, 0xae9ec5762cdf43f0, 0xddae3bb06ff81093, 0x6bb5a0bfeba4c21b, 0xde133e27a97d787d, 0xa4fd051769f4446c, 0x17269b4b9cc251e5, 0xfa3c63ae2413d9c1, 0xf21bba871553c23, 0x9560ea5628cb384a, 0xb1869e0c3ef81b8e, 0x754a7022cd666060, 0xeb9a607fac97f215, 0xa9a2eca76a892f61, 0xfc5bd9582b9f1615, 0x3948b05698a707b8, 0xabf6efc612e795a8, 0x5639b2ae8ea2d24a, 0xfb38c77ffea72bc4, 0x5c170f17d18e1aba, 0x1ec0f5ddd030abd7, 0x733df0790cf9456d, 0xfd59df1f7e2807df, 0xd78d355962c32d67, 0x6d8406b28896e463, 0x7e10982bd36e92e1, 0x5ed91b2d84eaa900, 0xb1b5a178973aa20e, 0x989a729861d33c22, 0xe627d8bd267c60ce, 0x8e13cbc162926e9f, 0x9b3f6b7243fa3886, 0x25dc46d344ab0152, 0x8259ece456028325, 0xb56e7da05169757, 0xa33f85c358faef12, 0xd66ef97a722eec39, 0xdb26c20f5737b37f, 0x21bfe2785df0f1d1, 0x7bddbe901418db40, 0x1e9b4b7b59ce6b4, 0x5b5e818a0c947528, 0x5c22506b059cf458, 0x5d070e7fe407ca2e, 0xe61baefb986c0969, 0x666dd6fc022671c, 0x3396ea50762ecdce, 0x77009d23399b428, 0x9817401dabceabc1, 0xb5197af242b71333, 0x2d7ad19e471d8b85, 0xc431fcdd6b0dccdf, 0xced4c0ecfaab751, 0x127ec5d8399ab8d6, 0x596a92ca5e9a0f2, 0x296a45d8edb0fc9f, 0xf80fb8a1ac303dab, 0xe9bab061374b1c7, 0x168b2104cc5e0588, 0xf0e94ebaae0922a7, 0x405f1b1d0ed821cf, 0xfe47262de4c9f8a5, 0x67b1c8e3a560d0d1, 0x5ae10187b316e617, 0x22742755be6efe96, 0x859fece72223ba1e, 0xf1f58f5ebe116262, 0x61d8c19ffc1d5240, 0x5dec382f86e31991, 0xf44d9c73a4f499a7, 0xcbcd69ede9039123, 0x24404871a4039d85, 0x3739efba7f2e2934, 0xa7298a80b6021b1b, 0xf18ad856a33542e9, 0x57b9f5503afd5185, 0x6d8d4bec4bb6ea85, 0xfac40f5163bcc2a1, 0x8d7553f5aea56b0, 0x66e57e8541964615, 0x30d1fdd890b8f546, 0x97f001885ec6f7a9, 0x8081b94ac3906be2, 0x80c9a112e4cfa62a, 0x2c58af855fd14687, 0xca5f119596e3685c, 0x3e6263e120432b7e, 0x75336fb847971ac7, 0xe12886f0026b453f, 0xf2eb7ffaa14d0b0, 0xbc3fab1a9f41cff2, 0xc5c2c3f67cfa4759, 0xffe407c9f6a4d1b5, 0x4b9ee9c6904c5267, 0xc6f6fd8ae1ff9eb5, 0x56284a9e72dfaf9a, 0xfffd38e36fafed8f, 0xe62ba62ec21a402f, 0x283891e36280c8de, 0xe64a0eca3ca6d08c, 0xab39c5c79f188120, 0xeced36ecf90ee054, 0xda6f04e6af5bf935, 0x34034f79d461b98, 0xb7b97b51916abbd7, 0x14df4a16593e2c01, 0x347d0cafff4da5b9, 0x5bd2c820912c9ce6, 0x9e6e4ac671fe4281, 0x5777396b69d3e67d, 0xa266cd975d761e06, 0x4fc1b1b53bc057e, 0x132dacd5e50441a7, 0x2f787d5ed1baac3, 0x88ad3b9f0e2302ad, 0xdb0e576a0e0f41d6, 0xc2ce6fe9fd60ecd1, 0x1a6c591e3b8afaa5, 0x6870e01562c0a875, 0xa1956949f3a42c62, 0x5f14a9c72eba85f, 0x7749d8404fcc4c, 0x193ebadc3f3e5d6e, 0xdf63912633738f9e, 0xd7b783d8744a74ff, 0x2a0fd65fb10f0459, 0x12dea8f5e024ba9a, 0x5ea7412af242ec00, 0x655060f908300e06, 0xf03530558a97e7c0, 0xc0c1831b86449772, 0xbee16648ede0b098, 0x314f899acc523e21, 0x6c4551b46174733, 0x448b9c4c11051416, 0xc8322fb51e5ae42b, 0x82e4dd6f0c4ea583, 0x7c660954814e2e5, 0xd8f4fa1bd6c3749f, 0xf149cf553d2f3b3f, 0x1a5842fa4b39d4e7, 0xb3298d4af99b4177};
const uint64_t ZHASH_WHITE = 0xe8554e0e45604657;
const uint64_t ZHASH_BLACK = 0x4c4a333dfffc947d;
""".strip()

def make_const_file(consts: dict) -> str:
    constant_params = ""
    ps_values = ""
    
    for key, value in consts.items():
        if isinstance(value, list):
            array_values = ", ".join(str(v) for v in value)
            ps_values += f"const int {key}[] = {{{array_values}}};\n"
        else:
            constant_params += f"#define {key} {value}\n"

    text = "#include <inttypes.h>\n\n"
    text += "// Constant parameters\n"
    text += constant_params + "\n"
    text += "// Piece square values\n"
    text += ps_values + "\n"
    text += ZOBRIST_CONSTANTS + "\n"
    return text

# Make sure consts.c is updated with the best constants
with open("src/consts.c", "r") as f:
    current_consts = f.read()
    if current_consts != make_const_file(best_consts):
        print("best constants are not updated in optimize_constants.py")
        sys.exit(1)
    

# Make a backup copy of consts.c
shutil.copyfile("src/consts.c", "src/consts_backup.c")


def round_up(n: float) -> int:
    if n > 0:
        return max(1, round(n))
    elif n < 0:
        return min(-1, round(n))
    else:
        return 0


def mutated_consts(consts: dict) -> dict:
    new_consts = copy.deepcopy(consts)
    for key in constants_to_optimize:
        if isinstance(consts[key], list):
            for i in range(len(consts[key])):
                if random.randint(1, math.ceil(len(constants_to_optimize) / 3)) == 1:
                    change_percent = random.uniform(-RAND_NOISE, RAND_NOISE)
                    change_amount = round_up(consts[key][i] * change_percent)
                    new_consts[key][i] = consts[key][i] + change_amount
                    
        elif random.randint(1, math.ceil(len(constants_to_optimize) / 5)) == 1:
            change_percent = random.uniform(-RAND_NOISE, RAND_NOISE)
            change_amount = round_up(consts[key] * change_percent)
            new_consts[key] = max(consts[key] + change_amount, 0)
    
    return new_consts


def executable(name: str) -> str:
    if os.name == "nt":
        return f"{name}.exe"
    else:
        return f"./{name}"


def what_draw(board: chess.Board) -> str:
    if board.is_stalemate():
        print("Draw by stalemate")
    elif board.is_insufficient_material():
        print("Draw by insufficient material")
    elif board.is_seventyfive_moves():
        print("Draw by 75-move rule")
    elif board.is_fivefold_repetition():
        print("Draw by fivefold repetition")
    elif board.is_fifty_moves():
        print("Draw by 50-move rule")
    elif board.can_claim_threefold_repetition():
        print("Draw by threefold repetition")
    else:
        print("Draw by agreement or unknown reason")


def illegal_move(board: chess.Board, move_uci: str, result: dict):
    print("Bot suggested an illegal move. Exiting.")
    print(f"{result=}")
    print(f"Board FEN:\n{board.fen()}")
    print(f"Illegal move UCI: {move_uci}")
    sys.exit(1)


def play_game(fen: str, is_white: bool) -> dict:
    results = {"score": 0, "time_old": 0, "time_new": 0, "avg_depth_new": 0, "avg_depth_old": 0}
    board = chess.Board(fen)
    number_of_moves = 0
    print(sigma_zero.EXE_FILE, "v", sigma_zero.OLD_EXE_FILE, f"({best_score_against_old})")
    sigma_zero.command("--version", exe=sigma_zero.EXE_FILE)
    sigma_zero.command("--version", exe=sigma_zero.OLD_EXE_FILE)

    while not board.is_game_over(claim_draw=True):
        if (board.turn == chess.WHITE and is_white) or (board.turn == chess.BLACK and not is_white):
            result = sigma_zero.play(board, TIME)
            results["time_new"] += result.get("time", 0)
            results["avg_depth_new"] += result.get("depth", 0)
        else:
            result = sigma_zero.old_play(board, TIME)
            results["time_old"] += result.get("time", 0)
            results["avg_depth_old"] += result.get("depth", 0)

        move_uci = result.get("move", "<unknown>")
        try:
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                board.push(move)
                number_of_moves += 1
            else:
                illegal_move(board, move_uci, result)
        except Exception:
            illegal_move(board, move_uci, result)

    if board.result(claim_draw=True) == "1-0":
        results["score"] = 1 if is_white else -1
    elif board.result(claim_draw=True) == "0-1":
        results["score"] = -1 if is_white else 1
    else:
        what_draw(board)

    results["end_fen"] = board.fen()
    if number_of_moves > 0:
        results["avg_depth_new"] /= number_of_moves / 2
        results["avg_depth_old"] /= number_of_moves / 2

    return results


def tournament(exec1: str, exec2: str, required_score: int) -> int:
    sigma_zero.EXE_FILE = exec1
    sigma_zero.OLD_EXE_FILE = exec2
    sigma_zero._COMPILED.add("make")
    
    start = time.perf_counter()
    results = {"wins": 0, "losses": 0, "draws": 0}
    for i, fen in enumerate(FENS):
        print(f"Game {i+1}/{len(FENS)}")
        result = play_game(fen, is_white=(i % 2 == 0))
        print("End FEN:", result.get("end_fen", "N/A"))
        print(f"Time SigmaZero: {result['time_new']:.2f}s, Old: {result['time_old']:.2f}s")
        print(f"Avg Depth SigmaZero: {result['avg_depth_new']:.2f}, Old: {result['avg_depth_old']:.2f}")
        if result["score"] == 1:
            results["wins"] += 1
            print("Result: Win", end=" ")
        elif result["score"] == -1:
            results["losses"] += 1
            print("Result: Loss", end=" ")
        else:
            results["draws"] += 1
            print("Result: Draw", end=" ")
        print(f"({results['wins']}W/{results['losses']}L/{results['draws']}D)\n")

        best_possible_score = 100 - i + results['wins'] - results['losses']
        if best_possible_score <= required_score:
            print("Stopping tournament, impossible to achieve required score at this point.")
            return False, 0

    print(f"Tournament completed in {time.perf_counter() - start:.2f} seconds.")
    print("Tournament Results:")
    print(f"Wins: {results['wins']}, Losses: {results['losses']}, Draws: {results['draws']}")
    score = results['wins'] - results['losses']
    return score > required_score, score


def log(message: str):
    with open("optimize_constants.log", "a") as log_file:
        log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    print(message)


def log_diff(old_consts: dict, new_consts: dict):
    for key in old_consts.keys():
        if type(old_consts[key]) is not list and old_consts[key] != new_consts[key]:
            log(f"  {key}: {old_consts[key]} -> {new_consts[key]}")


# Training loop:
# 1. Take the dict of constants 'best_consts' and select 1 in 5 for mutation
# 2. For each selected constant, randomly increase or decrease it by up to 10% (rounded to nearest non-zero integer)
# 3. Make sigma-zero with best_consts (rename executable to sigma-zero-best)
# 4. Make sigma-zero with mutated constants (rename executable to sigma-zero-mutated)
# 5. Run 100 games between the two versions, alternating colors
# 6. Give score to mutated version: # wins - # losses
# 7. If score <= 0, discard mutated constants and go back to step 1
# 8. Run 100 games between sigma-zero-mutated and old
# 9. If score <= best_score_against_old, discard mutated constants and go back to step 1
# 10. If score > best_score_against_old, keep mutated constants as best_consts and go back to step 1
best_score_against_old = 0
def training_step():
    global best_consts, best_score_against_old
    
    # Step 1 and 2
    mut_consts = mutated_consts(best_consts)
    
    # Step 3
    with open("src/consts.c", "w") as f:
        f.write(make_const_file(best_consts))
    os.system("make")
    shutil.move(executable("sigma-zero"), executable("sigma-zero-best"))
    
    # Step 4
    with open("src/consts.c", "w") as f:
        f.write(make_const_file(mut_consts))
    os.system("make")
    shutil.move(executable("sigma-zero"), executable("sigma-zero-mutated"))
    
    # Step 5 and 6
    print("Playing tournament between best and mutated constants...")
    won, score = tournament(executable("sigma-zero-mutated"), executable("sigma-zero-best"), 0)

    # Step 7
    if not won:
        log("Mutated constants did not outperform best constants. Discarding mutations.")
        return 0
    
    # Step 8
    log("Mutated constants outperformed best constants. Now testing against old...")
    sigma_zero.make(OLD)
    won, score = tournament(executable("sigma-zero-mutated"), executable("old"), best_score_against_old)
    
    # Step 9
    if not won:
        log(f"Mutated constants did not outperform best constants against {OLD}. Discarding mutations.")
        return 0
    
    # Step 10
    log(f"Mutated constants outperformed {OLD}! Updating best constants.")
    log_diff(best_consts, mut_consts)
    best_consts = mut_consts
    best_score_against_old = score
    with open("src/consts_best.c", "w") as f:
        f.write(make_const_file(best_consts))
    return 1

def get_average_cutoff(value_name: str) -> float:
    total_cutoff_index = 0
    
    for fen in CUTOFF_FENS:
        board = chess.Board(fen)
        result = sigma_zero.play(board, TIME_CUTOFF)
        cutoff_index = result[value_name]
        # print(f"FEN: {fen} -> {value_name}: {cutoff_index}")
        total_cutoff_index += cutoff_index
    
    average_cutoff = total_cutoff_index / len(CUTOFF_FENS)
    print(f"Average {value_name}: {average_cutoff:.2f}")
    return average_cutoff

# Training loop for beta cutoff constants only
# 1. Enable TRACK_BETA_CUTOFFS in main.c
# 2. Make an average of beta cutoff index over set of fens in data/puzzles.txt
# 3. Take the dict of constants 'best_consts' and select 1 in 5 for mutation (only those related to beta cutoffs)
# 4. For each selected constant, randomly increase or decrease it by up to 10% (rounded to nearest non-zero integer)
# 5. Make sigma-zero with mutated constants
# 6. Run over the set of fens and calculate average beta cutoff index
# 7. If average beta cutoff index is not improved, discard mutated constants and go back to step 3
# 8. If average beta cutoff index is improved, keep mutated constants as best_consts and go back to step 3
def train_cutoff(value_name: str, maximize: bool):
    global best_consts, RAND_NOISE
    with open("src/main.c", "rt") as f:
        main_c = f.read()
    with open("src/main.c", "wt") as f:
        f.write(main_c.replace("// #define TRACK_BETA_CUTOFFS", "#define TRACK_BETA_CUTOFFS"))
    
    best_value = get_average_cutoff(value_name)
    log(f"=== Training {value_name} constants ===")
    log(f"Initial average {value_name}: {best_value:.2f}")
    
    n_mutations = 0
    n_steps = 0
    og_rand_noise = 0.5
    
    try:
        while True:
            RAND_NOISE = max(0.1, og_rand_noise * (0.92 ** n_mutations))
            n_steps += 1
            log(f"=== {value_name} Training Step {n_steps} ===")
            log(f"{n_mutations} successful mutations so far.")
            
            # Step 3 and 4
            mut_consts = mutated_consts(best_consts)
            
            # Step 5
            with open("src/consts.c", "w") as f:
                f.write(make_const_file(mut_consts))
            os.system("make")
            
            # Step 6
            average_cutoff = get_average_cutoff(value_name)
            
            # Step 7
            if (maximize and average_cutoff <= best_value) or (not maximize and average_cutoff >= best_value):
                log(f"Mutated constants did not improve average {value_name}. Discarding mutations.")
                continue
            
            # Step 8
            log(f"Mutated constants improved average {value_name}: {average_cutoff:.2f} < {best_value:.2f}. Updating best constants.")
            log_diff(best_consts, mut_consts)
            best_consts = mut_consts
            best_value = average_cutoff
            n_mutations += 1
            
    except KeyboardInterrupt:
        log(f"{value_name} training interrupted.")
    except Exception as e:
        log(f"Error during {value_name} training: {e}")
    
    log(f"Final average {value_name}: {best_value:.2f} after {n_mutations} successful mutations in {n_steps} steps.")
    
    with open("src/main.c", "wt") as f:
        f.write(main_c)
    
    with open("src/consts.c", "w") as f:
        f.write(make_const_file(best_consts))
    print("Final best constants written to src/consts.c")
    
    # Cleanup
    if os.path.exists("src/consts_best.c"):
        os.remove("src/consts_best.c")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        constants_to_optimize = list(best_consts.keys())
    elif len(sys.argv) == 3 and sys.argv[1] == "--maximize":
        # Train a predefined set of constants to maximize a value
        value_name = sys.argv[2]
        if value_name not in MINMAX_VALUES:
            print(f"Value '{value_name}' is not recognized. Available values: {', '.join(MINMAX_VALUES)}")
            sys.exit(1)
        constants_to_optimize = CUTOFF_CONSTS
        train_cutoff(value_name, True)
        sys.exit(0)
    elif len(sys.argv) == 3 and sys.argv[1] == "--minimize":
        # Train a predefined set of constants to minimize a value        
        value_name = sys.argv[2]
        if value_name not in MINMAX_VALUES:
            print(f"Value '{value_name}' is not recognized. Available values: {', '.join(MINMAX_VALUES)}")
            sys.exit(1)
        constants_to_optimize = CUTOFF_CONSTS
        train_cutoff(value_name, False)
        sys.exit(0)
    else:
        constants_to_optimize = [key for key in sys.argv[1:] if key in best_consts]
        print("Optimizing only the following constants:", ", ".join(constants_to_optimize))
    
    # Calculate baseline score against old
    sigma_zero.make(OLD)
    os.system("make")
    print(f"Calculating baseline score against {OLD}...")
    won, best_score_against_old = tournament(executable("sigma-zero"), executable("old"), 0)
    
    # Clear log file
    with open("optimize_constants.log", "w") as log_file:
        log_file.write("=== Optimize Constants Log ===")
    
    n_mutations = 0
    n_steps = 0
    while True:
        n_steps += 1
        log(f"=== Training Step {n_steps} ===")
        log(f"{n_mutations} successful mutations so far.")
        try:
            n_mutations += training_step()
        except KeyboardInterrupt:
            log("Training interrupted.")
            break
        except Exception as e:
            log(f"Error during training step: {e}")
            break
    
    with open("src/consts.c", "w") as f:
        f.write(make_const_file(best_consts))
    print("Final best constants written to src/consts.c")
    
    # Cleanup
    os.remove(executable("sigma-zero-best"))
    os.remove(executable("sigma-zero-mutated"))
    if os.path.exists("src/consts_best.c"):
        os.remove("src/consts_best.c")
