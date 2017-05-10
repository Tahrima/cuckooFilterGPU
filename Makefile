CC = /usr/local/cuda-7.5/bin/nvcc

cuckoo: *.cu *.cpp
	$(CC) -w -std=c++11 graph_test.cu GPUBatchInserts.cu mt19937ar.cpp -O3 -arch=sm_30 -o graph_test

clean:
	rm -f *.o graph_test
