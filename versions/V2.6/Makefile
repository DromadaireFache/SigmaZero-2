CC=gcc
CFLAGS=-Wall -Werror -Wno-unused-function
OPTIMIZE=-O3 -march=native -mtune=native

all: engine

engine: src/main.c src/consts.c
	$(CC) $(CFLAGS) $(OPTIMIZE) -o engine src/main.c
