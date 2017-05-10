CC = /usr/local/cuda-7.5/bin/nvcc

batch-insert: *.cu *.cpp
	$(CC) -w -std=c++11 GPUBatchInserts.cu mt19937ar.cpp -O3 -arch=sm_30 -o batch-insert

clean:
	rm -f *.o batch-insert
