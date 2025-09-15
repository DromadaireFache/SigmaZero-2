CC=gcc
CFLAGS=-Wall -Werror -Ofast

# CFLAGS += -fsanitize=address -pthread -fno-omit-frame-pointer
# LDFLAGS += -fsanitize=thread -pthread

# Default target
all: sigma-zero

sigma-zero: src/main.c
	$(CC) $(CFLAGS) -o sigma-zero src/main.c

# Pattern rule to build any .c file as an executable
%: src/versions/%.c
	$(CC) $(CFLAGS) -o old $<

.PHONY: all