CC=gcc
CFLAGS=-Wall -Werror -Wno-unused-function
OPTIMIZE=-O3 -march=native -mtune=native

all: sigma-zero

sigma-zero: src/main.c src/consts.c
	$(CC) $(CFLAGS) $(OPTIMIZE) -o sigma-zero src/main.c
