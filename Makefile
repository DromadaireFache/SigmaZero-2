CC=gcc
CFLAGS=-Wall -Werror -Wno-unused-function
OPTIMIZE=-O3 -march=native -mtune=native

# CFLAGS += -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer
# LDFLAGS += -fsanitize=thread -pthread

# Default target
all: engine

engine: src/main.c src/consts.c magicbb/moves.o
	$(CC) $(CFLAGS) $(OPTIMIZE) -o engine magicbb/moves.o src/main.c

quickly: src/main.c src/consts.c magicbb/moves.o
	$(CC) $(CFLAGS) -o engine magicbb/moves.o src/main.c

magicbb_generator: magicbb/magicbb.c
	$(CC) $(CFLAGS) $(OPTIMIZE) -o magicbb_generator magicbb/magicbb.c

magicbb/moves.o: magicbb/moves.c
	$(CC) -c magicbb/moves.c -o magicbb/moves.o

clean:
	rm -f engine magicbb/moves.o magicbb_generator