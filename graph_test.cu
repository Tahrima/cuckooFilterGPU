/*
 * Parallel Graph Preprocessing of cuckoo filter
 * This preprocesses a batch insertion into a cuckoo filter by creating a directed graph (V,E) where:
 *    V is a set of vertices that represent each bucket of the cuckoo filter
 *    E is a set of edges (u,v) with weight w where:
 *      w is the fingerprint of a specific entry
 *      u is the bucket number given by hash(entry)
 *      v is the bucket number given by hash(entry) xor hash(fingerprint)
 *      dir indicates the vertex pointed to by the edge. Also indicates
 *          which bucket number the fingerprint should be placed in.
 */

#include <cstring>
#include <stdexcept>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <climits>
#include <curand.h>
#include <curand_kernel.h>
#include <sys/time.h>

#define LARGE_THRESHOLD_VAL 10000

double preprocessTime = 0;
double insertTime = 0;

struct timeval StartingTime;

void setTime(){
	gettimeofday( &StartingTime, NULL );
}

double getTime(){
	struct timeval PausingTime, ElapsedTime;
	gettimeofday( &PausingTime, NULL );
	timersub(&PausingTime, &StartingTime, &ElapsedTime);
	return ElapsedTime.tv_sec*1000.0+ElapsedTime.tv_usec/1000.0;	// Returning in milliseconds.
}


__device__ void random(unsigned int seed, int* result, int max) {
  /* CUDA's random number library uses curandState_t to keep track of the seed value
     we will store a random state for every thread  */
  curandState_t state;

  /* we have to initialize the state */
  curand_init(seed, /* the seed controls the sequence of random values that are produced */
              0, /* the sequence number is only important with multiple cores */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &state);

  /* curand works like rand - except that it takes a state as a parameter */
  *result = curand(&state) % max;
}
template <typename T_file>
void openFileToAccess( T_file& input_file, std::string file_name ) {
	input_file.open( file_name.c_str() );
	if( !input_file )
		throw std::runtime_error( "Failed to open specified file: " + file_name + "\n" );
}

void * cudaMallocAndCpy(int size, void * hostMemory) {
  void * gpuMem;
  cudaMalloc((void**) &gpuMem, size);
  if (hostMemory != NULL) {
    cudaMemcpy(gpuMem, hostMemory, size, cudaMemcpyHostToDevice);
  }
  return gpuMem;
}

void cudaGetFromGPU(void * destination, void * gpuMemory, int size) {
  cudaMemcpy(destination, gpuMemory, size, cudaMemcpyDeviceToHost);
}

void cudaSendToGPU(void * destination, void * hostMemory, int size) {
  cudaMemcpy(destination, hostMemory, size, cudaMemcpyHostToDevice);
}

class Edge {
  public:
    unsigned int src; //hash(x) location
    unsigned int dst; //hash(x) xor hash(fp) location
    unsigned char fp; //fingerprint
    int dir; //0 to be src, 1 to be dst

 	__device__ __host__ Edge(){}
};

class Graph {
  public:
    int *buckets; //value at index i is the number of indegrees to a bucket i
  	Edge *edges;
  	unsigned int num_edges;
    unsigned int num_buckets;
    unsigned int max_bucket_size;

    __device__ __host__ Graph(unsigned int edges, unsigned int nb, unsigned int bucket_size) {
      num_edges = edges;
      num_buckets = nb;
      max_bucket_size = bucket_size;

      buckets = new int[num_buckets]();
      edges = NULL;
    }

    __device__ void printGraph() {
      int thread_id = blockDim.x * blockIdx.x + threadIdx.x; //real thread number
      if(thread_id == 0) {
        for(int i=0; i<num_edges; i++) {
          printf("Edge %d: %u \t src: %u \t dst: %u\n",i, edges[i].fp, edges[i].src, edges[i].dst);
        }
        printCollisions();

      }
    }

    __device__ void printCollisions() {
      int thread_id = blockDim.x * blockIdx.x + threadIdx.x; //real thread number
      if(thread_id == 0) {
        printf("\n\nBuckets\n");
        for(int i=0; i< num_buckets ; i++) {
          if(buckets[i] > max_bucket_size) {
            printf("Collisions for bucket %d: %d\n", i, buckets[i]);
          }
        }
      }
    }
};

// __global__ void setup_kernel (curandState * state, Graph *g)
// {
//   	int thread_id = blockDim.x * blockIdx.x + threadIdx.x; //real thread number
//    	// change sequence number to currIdx if values are too correlated
//   	// curand_init(1234, 0, 0, &state[currIdx]);
//     curand_init(1234, 0, 0, &state[currIdx]);

// }


/**
 * Parallel graph building
 * @param entries is a list of entries to enter
 * @param entryListSize is the size of the @param entries list
 * @param g is an address in the GPU to pla\ce result. Assumes g->edges has been given enough space for @param entryListSize items
 */
__global__ void findAllCollisions(int* entries, int entryListSize, Graph * g) {
  int total_threads = blockDim.x * gridDim.x; //total threads
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x; //real thread number
  int thread_id_block = threadIdx.x; //thread number in block

  // CHANGE BELOW LINE TO BE MORE EFFICIENT
  int rounds = entryListSize % total_threads == 0 ? (entryListSize/total_threads):((entryListSize/total_threads)+1);
  g->num_edges = entryListSize;

  for (size_t i = 0; i <rounds; i++) {
    int currIdx = i*total_threads + thread_id;
    if(currIdx < entryListSize) {
      int * entry = &entries[currIdx];

      //printf("KERNEL SPACE current Index %d, Thread id %d: %x\n", currIdx, thread_id, entry);
      unsigned int bucket1;
      hash_item((unsigned char*) entry,
                    4,
                    g->num_buckets,
      		      HASHFUN_NORM,
                    &bucket1);

      const uint64_t hash = TwoIndependentMultiplyShift(*entry);
      unsigned char fp = (unsigned char) hash;
      unsigned int fpHash;
      hash_item((unsigned char*) &fp,
                    1,
                    g->num_buckets,
      		      HASHFUN_NORM,
                    &fpHash);
      unsigned int bucket2 = ((bucket1 ^ fpHash) & 0b11111111) % g->num_buckets;

      //build edge

      g->edges[currIdx].fp = fp;
      g->edges[currIdx].src = bucket1 % g->num_buckets;
      g->edges[currIdx].dst = bucket2 % g->num_buckets;


  // 	Copy state to local memory for efficiency */
  //     curandState local_state = global_state[thread_id];
  // 	/* Generate pseudo - random unsigned ints
  //     g->edges[i].dir = curand_uniform(&local_state);

      //update bucket
      atomicAdd(&(g->buckets[bucket1]), 1);
    }
  }
}
__global__ void resetCollisions(Graph * g) {

  int total_threads = blockDim.x * gridDim.x; //total threads
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x; //real thread number
  int thread_id_block = threadIdx.x; //thread number in block

  int rounds = (g->num_buckets % total_threads == 0) ? (g->num_buckets/total_threads):(g->num_buckets/total_threads + 1);

  for (size_t iter = 0; iter < rounds; iter++) {
    int currIdx = iter*total_threads + thread_id;
    if(currIdx < g->num_buckets) {
      int * currBucket = &(g->buckets[currIdx]);
      *currBucket = 0;
    }

  }

  rounds = (g->num_edges % total_threads == 0) ? (g->num_edges/total_threads):(g->num_edges/total_threads + 1);

  for (size_t iter = 0; iter < rounds; iter++) {
    int currIdx = iter*total_threads + thread_id;

    if(currIdx < g->num_edges) {
      int b = (g->edges[currIdx].dir == 0) ? (g->edges[currIdx].src):(g->edges[currIdx].dst);
      atomicAdd(&(g->buckets[b]),1);
    }
  }
  //g->printCollisions();
}

/**
 * Edge Processing Kernel
 * Finds random edges to evict until capacity for each bucket is equal to 0
 *
 */
__global__ void processEdges(Graph * g, int* anyChange, unsigned int randNum) {
  int total_threads = blockDim.x * gridDim.x; //total threads
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x; //real thread number
  int thread_id_block = threadIdx.x; //thread number in block
  int num_edges = g->num_edges;

  int rounds = num_edges % total_threads == 0 ? (num_edges/total_threads):(num_edges/total_threads+1);

  for(int i=0; i<rounds; i++) {
  	int currIdx = total_threads*i + thread_id; //current edge to process
    if(currIdx < g->num_edges) {
      Edge *e = &g->edges[currIdx];

      //determine the bucket it's in
      int curr_bucket = e->dir == 0 ? e->src:e->dst;

      //check the bucket
      int * bucketCount = &(g->buckets[curr_bucket]);
      int tmp = *bucketCount;
      //decrement the bucket count if > 0
      //int rand;
      //random((unsigned int)clock() + thread_id, &rand, 50);
      if(*bucketCount > g->max_bucket_size) {
        int old = atomicDec((unsigned int *)bucketCount, INT_MAX);
        old--;
        int shift = randNum % tmp;
        int shiftedValue = old - shift;
        int bucketOffset = (shiftedValue < 0) ? shiftedValue + tmp : shiftedValue;
        //if (e->dir) {
        // } else {
        //   printf("tmp %d, old %d, shift %d, shiftedValue %d, bucketOffset %d \t Evicting %d from %d to %d\n", tmp, old, shift, shiftedValue, bucketOffset, e->fp, e->src, e->dst);
        // }
        //printf("tmp %d, old %d, shift %d, shiftedValue %d, bucketOffset %d\n", tmp, old, shift, shiftedValue, bucketOffset);
        if (bucketOffset >= g->max_bucket_size && old < LARGE_THRESHOLD_VAL){
        	e->dir = !e->dir; // flip the bit

            // if (e->dir)
            //     printf("Evicting %d from %d to %d\n", e->fp, e->src, e->dst);
            // else
            //     printf("Evicting %d from %d to %d\n", e->fp, e->dst, e->src);
           *anyChange = 1;
        }
      }
    }
  }
  //g->printCollisions();
}

void initGraphCPU(int entry_size) {
	Graph * graph;
  	cudaMalloc(&graph, sizeof(Graph));
    Edge * e;
    cudaMalloc(&e, sizeof(Edge)*entry_size);
}

__global__ void makeGraphCuckoo(Graph * g, CuckooFilter * c, int * globalByteMask) {

  int total_threads = blockDim.x * gridDim.x; //total threads
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x; //real thread number
  int thread_id_block = threadIdx.x; //thread number in block

  // if (thread_id==0) {
  //   c->printFilter();
  //   printf("\n");
  // }
  int rounds = (g->num_edges % total_threads == 0) ? (g->num_edges/total_threads):((g->num_edges/total_threads)+1);

  for (size_t i = 0; i < rounds; i++) {
    int currIdx = total_threads*i + thread_id;
    if(currIdx < g->num_edges) {
      Edge * e = &(g->edges[currIdx]);
      int currBucket = e->dir == 0 ? e->src:e->dst;

      int index = atomicAdd(&(globalByteMask[currBucket]), 1);
      c->insert(e->fp,currBucket,index);
    }
  }
  __syncthreads();

  // if (thread_id==0) {
  //   c->printFilter();
  // }
}
double transferToCuckooFilter(Graph * g, CuckooFilter * c) {
  Graph * h_graph = (Graph*)malloc(sizeof(Graph));
  cudaGetFromGPU(h_graph, g, sizeof(Graph));

  int * byteMask = new int[h_graph->num_buckets];
  for (size_t i = 0; i < h_graph->num_buckets; i++) {
    byteMask[i] = 0;
  }
  int * g_byteMask = (int*)cudaMallocAndCpy(sizeof(int)*h_graph->num_buckets,(void*) byteMask);

  setTime();
  makeGraphCuckoo<<<ceil((double)h_graph->num_buckets/1024), 1024>>>(g, c, g_byteMask);
  cudaDeviceSynchronize();
  double insertTime = getTime();

  delete byteMask;

  return insertTime;
}


int insert(int* entries, unsigned int num_entries, unsigned int num_buckets, unsigned int bucket_size, CuckooFilter * cf){
    std::cout << "Inserting " << num_entries << " entries"<< std::endl;

	const int fail_threshold = (int)(sqrt(num_buckets*bucket_size)*log2((float)(num_buckets*bucket_size)));

	int anychange = 1;
  	int * d_change = (int *) cudaMallocAndCpy(sizeof(int), &anychange);

  	Graph *h_graph = new Graph(num_entries, num_buckets, bucket_size);

  	//set up pointer
  	cudaMalloc((void**)&(h_graph->edges), sizeof(Edge)*num_entries);
    cudaMalloc((void**)&(h_graph->buckets), sizeof(int)*num_buckets);

  	Graph *d_graph = (Graph *) cudaMallocAndCpy(sizeof(Graph), h_graph);
  	int * d_entries = (int *) cudaMallocAndCpy(sizeof(int)*num_entries, entries);

    std::cout << "Calling kernel" << std::endl;
    setTime();
    findAllCollisions<<<2, 512>>>(d_entries, num_entries, d_graph);
	cudaDeviceSynchronize();
    preprocessTime = getTime();
    int count = 0;
  	while (anychange != 0){
      anychange = 0;
      cudaSendToGPU(d_change, &anychange, sizeof(int));

      // generate random number
      setTime();
      unsigned int randNum = rand() % (num_buckets * 8);
      //std::cout << "Found all collisions, rand num: "<< randNum << std::endl;
      processEdges<<<ceil((double)num_entries/1024), 1024>>>(d_graph, d_change, randNum);
      cudaDeviceSynchronize();
	  preprocessTime += getTime();

      //std::cout << "Proccessed edge using " << ceil((double)num_entries/1024) << "threads " << std::endl;

      cudaGetFromGPU(&anychange, d_change, sizeof(int));
      //std::cout << "Got value of anychange: " << anychange << std::endl;
      if(anychange == 1){
        setTime();
        resetCollisions<<<ceil((double)num_entries/1024), 1024>>>(d_graph);
		cudaDeviceSynchronize();

        preprocessTime += getTime();
      }

	  count++;
	  if (count >= fail_threshold)
	  	return count;
    }


    CuckooFilter * g_cf = (CuckooFilter *)cudaMallocAndCpy(sizeof(CuckooFilter), cf);
    setTime();
    insertTime = transferToCuckooFilter(d_graph, g_cf);
    cudaGetFromGPU(cf,g_cf, sizeof(CuckooFilter));

	printf("Preprocessing time %f\n", preprocessTime);
	printf("Insertion time: %f\n", insertTime);
    printf("Completed insertion with %d iterations\n",count);

	return 0;
}
