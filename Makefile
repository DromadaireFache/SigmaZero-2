CC=gcc
CFLAGS=-Wall -Werror -Wno-unused-function -MMD -MP
OPTIMIZE=-O3 -march=native -mtune=native
TARGET=engine
SRC_DIR=src
BUILD_DIR=.build
EXCLUDED=consts_backup.h
EXTRA_SRCS=magicbb/moves.c

SRCS := $(filter-out $(addprefix $(SRC_DIR)/,$(EXCLUDED)),$(wildcard $(SRC_DIR)/*.c))
SRCS += $(EXTRA_SRCS)
OBJS := $(patsubst %.c,$(BUILD_DIR)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OPTIMIZE) -o $@ $^

# Compile *.c -> .build/*.o
$(BUILD_DIR)/%.o: %.c | $(BUILD_DIR)
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(OPTIMIZE) -MF $(patsubst %.o,%.d,$@) -c $< -o $@

# Ensure build dir exists
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)
	echo '*' > $(BUILD_DIR)/.gitignore

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) $(TARGET) magicbb_generator

-include $(DEPS)