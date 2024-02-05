NVFLAGS  := -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 
LDFLAGS  := -lm 
CFLAGS := -lm -O3 -fopenmp
OMPFLAGS := --default-stream per-thread
EXES     := hw3-1 hw3-2 hw3-3 
alls: $(EXES)

clean:
	rm -f $(EXES)

hw3-1: hw3-1.cc
	g++ $(CFLAGS) -o $@ $?
 
hw3-2: hw3-2.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?

hw3-3: hw3-3.cu
	nvcc $(NVFLAGS) -Xcompiler="$(CXXFLAGS)" $(LDFLAGS) $(OMPFLAGS) -o $@ $? 


