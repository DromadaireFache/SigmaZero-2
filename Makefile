CC=gcc
CFLAGS=-Wall -Werror -Wno-unused-function
OPTIMIZE=-O3 -march=native -mtune=native

# CFLAGS += -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer
# LDFLAGS += -fsanitize=thread -pthread

# Default target
all: sigma-zero

sigma-zero: src/main.c src/consts.c magicbb/moves.o
	$(CC) $(CFLAGS) $(OPTIMIZE) -o sigma-zero magicbb/moves.o src/main.c

quickly: src/main.c src/consts.c magicbb/moves.o
	$(CC) $(CFLAGS) -o sigma-zero magicbb/moves.o src/main.c

magicbb_generator: magicbb/magicbb.c
	$(CC) $(CFLAGS) $(OPTIMIZE) -o magicbb_generator magicbb/magicbb.c

magicbb/moves.o: magicbb/moves.c
	$(CC) -x c -c magicbb/moves.c -o magicbb/moves.o

clean:
	rm -f sigma-zero magicbb/moves.o magicbb_generator