CXX = g++
CXXFLAGS = -O3 -Wall -std=c++17 -fPIC -Ilib
LDFLAGS = -shared

LIB_SRCS = \
    lib/HighLevelSolver.cpp \
    lib/LowLevelSolver.cpp \
    lib/TreeNode.cpp \
    src/cbs_api.cpp

TARGET_LIB = libcbs.so

all: $(TARGET_LIB)

$(TARGET_LIB):
	$(CXX) $(CXXFLAGS) -c $(LIB_SRCS)
	$(CXX) $(LDFLAGS) -o $@ *.o
	rm -f *.o

clean:
	rm -f *.o $(TARGET_LIB)

.PHONY: all clean
