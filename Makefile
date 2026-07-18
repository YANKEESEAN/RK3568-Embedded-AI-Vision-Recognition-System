# 交叉编译器
CC = aarch64-linux-gcc
CXX = aarch64-linux-g++

# 系统根目录
SYSROOT := /usr/local/cross/aarch64-rk3568-linux-gnu/aarch64-buildroot-linux-gnu/sysroot
# 目录定义
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
INC_DIR = include
LIB_DIR = lib
# 目标程序
TARGET = $(BIN_DIR)/main

# 源文件与对象文件
SRCS_C = $(wildcard $(SRC_DIR)/*.c)
SRCS_CPP = $(wildcard $(SRC_DIR)/*.cpp)
OBJS_C = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS_C))
OBJS_CPP = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.opp, $(SRCS_CPP))
OBJS = $(OBJS_C) $(OBJS_CPP)

# RKNN库路径
RKNN_LIB_DIR = $(LIB_DIR)
RKNN_INC_DIR = $(INC_DIR)

# 编译选项（仅用于编译阶段）
CPPFLAGS += --sysroot=$(SYSROOT)
CPPFLAGS += -I $(SYSROOT)/usr/include/drm/
CPPFLAGS += -I $(INC_DIR)
CPPFLAGS += -I $(RKNN_INC_DIR)
CPPFLAGS += -Wall -g

# C++编译选项
CXXFLAGS += --sysroot=$(SYSROOT)
CXXFLAGS += -I $(INC_DIR)
CXXFLAGS += -I $(RKNN_INC_DIR)
CXXFLAGS += -Wall -g -std=c++11

# 链接选项（仅用于链接阶段）
# 修复：在末尾添加了 -lpng 以解决 PNG 库函数未定义引用的报错
LDFLAGS += -L$(LIB_DIR) -L$(SYSROOT)/usr/lib/ -L$(RKNN_LIB_DIR)
LDFLAGS += -ldrm -ljpeg -lpthread -lcamera_capture_yuyv -lrknnrt -lstdc++ -lm -lz -ldl -lpng

# 目标规则
$(TARGET): $(OBJS) | $(BIN_DIR)
	$(CXX) -o $@ $^ $(LDFLAGS) $(CPPFLAGS)

# 编译 C 源文件为 .o
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CPPFLAGS) -c $< -o $@

# 编译 C++ 源文件为 .opp
$(OBJ_DIR)/%.opp: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 创建必要的目录
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# 清理
.PHONY: clean distclean
clean:
	rm -rf $(OBJ_DIR) $(TARGET)

distclean: clean