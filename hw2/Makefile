CC = gcc
CXX = g++
LDLIBS = -lpng
CFLAGS = -lm -O3 
hw2a: CFLAGS += -pthread
hw2b: CC = mpicc -cc=clang
hw2b: CXX = mpicxx -cxx=clang++
hw2b: CFLAGS += -fopenmp
CXXFLAGS = $(CFLAGS)
TARGETS = hw2seq hw2a hw2b 

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)
