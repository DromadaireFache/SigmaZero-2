CC=gcc
CFLAGS=-Wall -Werror -Wno-unused-function -MMD -MP -O3 -march=native -mtune=native
EXE :=
TARGET=engine$(EXE)
DEBUG_TARGET=$(TARGET)_debug$(EXE)
SRC_DIR=src
BUILD_DIR=.build
DEBUG_MODE ?= full
DEBUG_ALLOWED_MODES := full symbols asan ubsan tsan lsan define
EXCLUDED=consts_backup.h
EXTRA_SRCS=magicbb/moves.c nnue/params.c
UNAME_S := $(shell uname -s)
MATH_LIB :=
RM_RF := rm -rf
MKDIR_P := mkdir -p
TOUCH_GITIGNORE := echo '*' > $(BUILD_DIR)/.gitignore

ifeq ($(OS),Windows_NT)
EXE := .exe
TARGET=engine$(EXE)
DEBUG_TARGET=$(TARGET)_debug$(EXE)
RM_RF := powershell -NoProfile -Command "Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"
MKDIR_P := powershell -NoProfile -Command "New-Item -ItemType Directory -Force -Path"
TOUCH_GITIGNORE := powershell -NoProfile -Command "Set-Content -Path $(BUILD_DIR)/.gitignore -Value '*'"
endif

ifeq ($(UNAME_S),Linux)
MATH_LIB := -lm
endif

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
DEBUG_MODE_CFLAGS_define := -DDEBUG
DEBUG_CFLAGS := $(DEBUG_COMMON_CFLAGS) $(DEBUG_MODE_CFLAGS_$(DEBUG_MODE))
DEBUG_LDFLAGS := $(DEBUG_MODE_CFLAGS_$(DEBUG_MODE))

ADDITIONAL_FLAGS :=
CFLAGS += $(ADDITIONAL_FLAGS)
DEBUG_CFLAGS += $(ADDITIONAL_FLAGS)

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
	@echo "  define   - Compile with -DDEBUG (no optimizations, no sanitizers)"

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(MATH_LIB)

$(DEBUG_TARGET): $(DEBUG_OBJS)
	$(CC) $(DEBUG_CFLAGS) $(DEBUG_LDFLAGS) -o $@ $^ $(MATH_LIB)

# Compile *.c -> .build/*.o
$(BUILD_DIR)/%.o: %.c | $(BUILD_DIR)
	$(MKDIR_P) "$(dir $@)"
	$(CC) $(CFLAGS) -MF $(patsubst %.o,%.d,$@) -c $< -o $@

# Compile *.c -> .build/debug/<mode>/*.o
$(DEBUG_BUILD_DIR)/%.o: %.c | $(DEBUG_BUILD_DIR)
	$(MKDIR_P) "$(dir $@)"
	$(CC) $(DEBUG_CFLAGS) -MF $(patsubst %.o,%.d,$@) -c $< -o $@

# Ensure build dir exists
$(BUILD_DIR):
	$(MKDIR_P) "$(BUILD_DIR)"
	$(TOUCH_GITIGNORE)

$(DEBUG_BUILD_DIR):
	$(MKDIR_P) "$(DEBUG_BUILD_DIR)"

.PHONY: clean debug help
clean:
	$(RM_RF) "$(BUILD_DIR)" "$(TARGET)" "$(DEBUG_TARGET)" magicbb_generator

-include $(DEPS) $(DEBUG_DEPS)