CC=gcc
CFLAGS=-Wall -Werror -Wno-unused-function
OPTIMIZE=-O3 -march=native -mtune=native

# CFLAGS += -fsanitize=address -pthread -fno-omit-frame-pointer
# LDFLAGS += -fsanitize=thread -pthread

# Default target
all: sigma-zero

sigma-zero: src/main.c src/consts.c magicbb/moves.o
	$(CC) $(CFLAGS) $(OPTIMIZE) -o sigma-zero magicbb/moves.o src/main.c

quickly: src/main.c src/consts.c magicbb/moves.o
	$(CC) $(CFLAGS) -o sigma-zero magicbb/moves.o src/main.c

# Pattern rule to build any .c file as an executable
%: src/versions/%.c magicbb/moves.o
	$(CC) $(CFLAGS) $(OPTIMIZE) -o old magicbb/moves.o $<

magicbb_generator: magicbb/magicbb.c
	$(CC) $(CFLAGS) $(OPTIMIZE) -o magicbb_generator magicbb/magicbb.c

magicbb/moves.o: magicbb/moves.c_no_format
	$(CC) -x c -c magicbb/moves.c_no_format -o magicbb/moves.o

.PHONY: all