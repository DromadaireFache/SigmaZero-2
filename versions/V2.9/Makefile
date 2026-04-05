CC=gcc
CFLAGS=-Wall -Werror -Wno-unused-function -MMD -MP -O3 -march=native -mtune=native
TARGET=engine
DEBUG_TARGET=$(TARGET)_debug
SRC_DIR=src
BUILD_DIR=.build
DEBUG_MODE ?= full
DEBUG_ALLOWED_MODES := full symbols asan ubsan tsan lsan
EXCLUDED=consts_backup.h
EXTRA_SRCS=magicbb/moves.c

SRCS := $(filter-out $(addprefix $(SRC_DIR)/,$(EXCLUDED)),$(wildcard $(SRC_DIR)/*.c))
SRCS += $(EXTRA_SRCS)
OBJS := $(patsubst %.c,$(BUILD_DIR)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)
DEBUG_BUILD_DIR := $(BUILD_DIR)/debug/$(DEBUG_MODE)
DEBUG_OBJS := $(patsubst %.c,$(DEBUG_BUILD_DIR)/%.o,$(SRCS))
DEBUG_DEPS := $(DEBUG_OBJS:.o=.d)

# Allow `make debug <mode>` in addition to `make debug DEBUG_MODE=<mode>`.
ifneq ($(filter debug,$(MAKECMDGOALS)),)
DEBUG_GOAL_MODE := $(word 2,$(MAKECMDGOALS))
ifneq ($(DEBUG_GOAL_MODE),)
DEBUG_MODE := $(DEBUG_GOAL_MODE)
$(eval $(DEBUG_GOAL_MODE):;@:)
endif
endif

ifeq ($(filter $(DEBUG_MODE),$(DEBUG_ALLOWED_MODES)),)
$(error Invalid DEBUG_MODE '$(DEBUG_MODE)'. Use one of: $(DEBUG_ALLOWED_MODES))
endif

DEBUG_COMMON_CFLAGS := -Wall -Werror -Wno-unused-function -MMD -MP -O0 -g3 -ggdb3 -fno-omit-frame-pointer -fno-optimize-sibling-calls -fno-inline -DDEBUG
DEBUG_MODE_CFLAGS_full := -fsanitize=address,undefined -fno-sanitize-recover=all
DEBUG_MODE_CFLAGS_symbols :=
DEBUG_MODE_CFLAGS_asan := -fsanitize=address -fno-sanitize-recover=all
DEBUG_MODE_CFLAGS_ubsan := -fsanitize=undefined -fno-sanitize-recover=all
DEBUG_MODE_CFLAGS_tsan := -fsanitize=thread
DEBUG_MODE_CFLAGS_lsan := -fsanitize=leak

DEBUG_CFLAGS := $(DEBUG_COMMON_CFLAGS) $(DEBUG_MODE_CFLAGS_$(DEBUG_MODE))
DEBUG_LDFLAGS := $(DEBUG_MODE_CFLAGS_$(DEBUG_MODE))

all: $(TARGET)
debug: $(DEBUG_TARGET)
help:
	@echo "Usage:"
	@echo "  make                  Build release binary ($(TARGET))"
	@echo "  make all              Build release binary ($(TARGET))"
	@echo "  make debug            Build debug binary ($(DEBUG_TARGET)) with mode 'full'"
	@echo "  make debug <mode>     Build debug binary with mode: $(DEBUG_ALLOWED_MODES)"
	@echo "  make debug DEBUG_MODE=<mode>"
	@echo "  make clean            Remove build artifacts"
	@echo ""
	@echo "Debug modes:"
	@echo "  full     - symbols + strict debug flags + ASan + UBSan"
	@echo "  symbols  - debug symbols only (no sanitizer)"
	@echo "  asan     - AddressSanitizer"
	@echo "  ubsan    - UndefinedBehaviorSanitizer"
	@echo "  tsan     - ThreadSanitizer"
	@echo "  lsan     - LeakSanitizer"

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

$(DEBUG_TARGET): $(DEBUG_OBJS)
	$(CC) $(DEBUG_CFLAGS) $(DEBUG_LDFLAGS) -o $@ $^

# Compile *.c -> .build/*.o
$(BUILD_DIR)/%.o: %.c | $(BUILD_DIR)
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -MF $(patsubst %.o,%.d,$@) -c $< -o $@

# Compile *.c -> .build/debug/<mode>/*.o
$(DEBUG_BUILD_DIR)/%.o: %.c | $(DEBUG_BUILD_DIR)
	mkdir -p $(dir $@)
	$(CC) $(DEBUG_CFLAGS) -MF $(patsubst %.o,%.d,$@) -c $< -o $@

# Ensure build dir exists
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)
	echo '*' > $(BUILD_DIR)/.gitignore

$(DEBUG_BUILD_DIR):
	mkdir -p $(DEBUG_BUILD_DIR)

.PHONY: clean debug help
clean:
	rm -rf $(BUILD_DIR) $(TARGET) $(DEBUG_TARGET) magicbb_generator

-include $(DEPS) $(DEBUG_DEPS)