CC = /usr/local/cuda-7.5/bin/nvcc

sssp: *.cu *.cpp *.c
	$(CC) -w -std=c++11 graph_test.cpp -O3 -arch=sm_30 -o graph_test

clean:
	rm -f *.o graph_test
