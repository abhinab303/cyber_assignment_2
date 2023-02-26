#include<stdio.h>
#include<string.h>
#include<algorithm>
#include<queue>
#include <cuda.h>
#include <cuda_runtime.h>
#include<omp.h>
#include<vector>
using namespace std;

#define BLOCK_SIZE 256
#define N_THREADS 8

int V,D,E,L,K,A,B,C,M,Q;
int* X;
int* edges;

int squared_l2_dist(int* x,int* y,int D){
	int sum2 = 0;
	for(int i = 0;i < D;++i)
		sum2 += (x[i] - y[i]) * (x[i] - y[i]);
	return sum2;
}

__global__ void squared_l2_dist_list(int* x,int* y, int* sum2, int D) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Handling arbitrary vector size
    if (tid < D){
        sum2[tid] = (x[tid] - y[tid]) * (x[tid] - y[tid]);
    }
}


__global__ void squared_l2_dist_reduce_simple(int* g_idata, int* g_odata, int n) {

	extern __shared__ int sdata[];
	// each thread loads one element from global to shared mem
	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
	sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	// write result for this block to global mem
	// if (tid == 0) g_odata[blockIdx.x] = sdata[0];
	if (tid == 0)
        atomicAdd(g_odata, sdata[0]);

}


__global__ void squared_l2_dist_reduce_simple_combined(int* x, int* y, int* g_odata) {

	extern __shared__ int sdata[];
	// each thread loads one element from global to shared mem
	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
	// sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];

	sdata[tid] = ((x[i] - y[i]) * (x[i] - y[i])) + ((x[i+blockDim.x] - y[i+blockDim.x]) * (x[i+blockDim.x] - y[i+blockDim.x]));
	
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	// write result for this block to global mem
	// if (tid == 0) g_odata[blockIdx.x] = sdata[0];
	if (tid == 0)
        atomicAdd(g_odata, sdata[0]);

}


__global__ void reduce_array(int* g_idata, int* g_odata) {

	extern __shared__ int sdata[];
	// each thread loads one element from global to shared mem
	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
	sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	// write result for this block to global mem
	// if (tid == 0) g_odata[blockIdx.x] = sdata[0];
	if (tid == 0)
        atomicAdd(g_odata, sdata[0]);

}

__global__ void get_array_to_sum(int* x, int* y, int* sum2, int D) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Handling arbitrary vector size
    if (tid < D){
        sum2[tid] = (x[tid] - y[tid]) * (x[tid] - y[tid]);
    }
}


__global__ void get_each_distance(vector<int> d, int* y, vector<int> dist_list) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	int block_size = BLOCK_SIZE;
	int grid_size = ((D + block_size) / block_size);
	int *d_sum_list, *d_d;

	cudaMalloc((void**)&d_sum_list, sizeof(int) * D);
	cudaMalloc(&d_d, sizeof(int));

	get_array_to_sum<<<grid_size,block_size>>>(X + d[tid] * D, y, d_sum_list, D);
	reduce_array<<<grid_size,block_size>>>(d_sum_list, d_d);

	if (tid < d.size()){
		dist_list[tid] = *d_d;
	}

}


int nearest_id(int start_point,int max_hop,int* query_data){
	std::queue<std::pair<int,int>> q;
	q.push(std::make_pair(start_point,0));
	int min_d = std::numeric_limits<int>::max();
	int min_id = -1;

	vector<int> dist_id;
	vector<int> dist_list;
	int count = 0;

	while(!q.empty()){
		auto now = q.front();
		q.pop();
		int id = now.first;
		int hop = now.second;
		int d = 0;
		
		dist_id.push_back(id);
		// d = squared_l2_dist(X + id * D,query_data,D);

		// if((d < min_d) || (d == min_d && id < min_id)){
		// 	min_d = d;
		// 	min_id = id;
		// }
		if(hop + 1 <= max_hop){
			int degree = edges[id * (L + 1)];
			for(int i = 1;i <= degree;++i){
				int v = edges[id * (L + 1) + i];
				q.push(std::make_pair(v,hop + 1));
			}
		}

		count++;

	}

	// printf("%d ", dist_id.size());


	get_each_distance<<<1,2>>>(dist_id, query_data, dist_list);



	return min_id;
}

int main(int argc,char** argv){
	FILE* fin = fopen(argv[1],"r");
	FILE* fout = fopen(argv[2],"w");
	fscanf(fin,"%d%d%d%d%d%d%d%d%d%d",&V,&D,&E,&L,&K,&A,&B,&C,&M,&Q);
	X = new int[V * D];
	for(int i = 0;i < K;++i)
		fscanf(fin,"%d",&X[i]);
	for(int i = K;i < V * D;++i)
		X[i] = ((long long)A * X[i - 1] + (long long)B * X[i - 2] + C) % M;
	edges = new int[V * (L + 1)];
	for(int i = 0;i < V;++i){
		edges[i * (L + 1)] = 0;
	}
	for(int i = 0;i < E;++i){
		int u,v;
		fscanf(fin,"%d%d",&u,&v);
		int degree = edges[u * (L + 1)];
		edges[u * (L + 1) + degree + 1] = v;
		++edges[u * (L + 1)];
	}
	int* query_data = new int[D];


	// can we convert all vars to cuda in here?
	// vars: query_data


	int nid;



	for(int i = 0;i < Q;++i){
		int start_point,hop;
		fscanf(fin,"%d%d",&start_point,&hop);
		for(int i = 0;i < D;++i){
			fscanf(fin,"%d",&query_data[i]);
		}

		nid = nearest_id(start_point,hop,query_data);

		fprintf(fout,"%d\n", nid);
	}
	fclose(fin);
	fclose(fout);

	delete[] X;
	delete[] edges;
	delete[] query_data;

	return 0;
}

