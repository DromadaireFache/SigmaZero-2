CC=gcc
CFLAGS=-Wall -Werror
OPTIMIZE=-Ofast

# CFLAGS += -fsanitize=address -pthread -fno-omit-frame-pointer
# LDFLAGS += -fsanitize=thread -pthread

# Default target
all: sigma-zero

sigma-zero: src/main.c src/consts.c
	$(CC) $(CFLAGS) $(OPTIMIZE) -o sigma-zero src/main.c

# Pattern rule to build any .c file as an executable
%: src/versions/%.c
	$(CC) $(CFLAGS) $(OPTIMIZE) -o old $<

magicbb: src/magicbb.c
	$(CC) $(CFLAGS) -o magicbb src/magicbb.c

.PHONY: all