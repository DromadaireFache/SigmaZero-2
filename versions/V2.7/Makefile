CC=gcc
CFLAGS=-Wall -Werror -Wno-unused-function
OPTIMIZE=-O3 -march=native -mtune=native

all: engine

engine: src/main.c src/consts.c magicbb/moves.o
	$(CC) $(CFLAGS) $(OPTIMIZE) -o engine src/main.c magicbb/moves.o

magicbb/moves.o: magicbb/moves.c
	$(CC) $(CFLAGS) $(OPTIMIZE) -c magicbb/moves.c -o magicbb/moves.o
