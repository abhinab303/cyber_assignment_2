#include<stdio.h>
#include<string.h>
#include<algorithm>
#include<queue>
#include <cuda.h>
#include <cuda_runtime.h>



int V,D,E,L,K,A,B,C,M,Q;
int* X;
int* edges;

int squared_l2_dist(int* x,int* y,int D){
	int sum2 = 0;
	for(int i = 0;i < D;++i)
		sum2 += (x[i] - y[i]) * (x[i] - y[i]);
	return sum2;
}

__global__ void squared_l2_dist_cuda(int* x,int* y, int* sum2, int D) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Handling arbitrary vector size
    // if (tid < n){
    //     out[tid] = a[tid] + b[tid];
    // }

	// for(int i = 0;i < D;++i)
	// 	sum2 += (x[i] - y[i]) * (x[i] - y[i]);
	// return sum2;

	if (tid < D){
		*sum2 += (x[tid] - y[tid]) * (x[tid] - y[tid]);
	}
}


int nearest_id(int start_point,int max_hop,int* query_data){
	std::queue<std::pair<int,int>> q;
	q.push(std::make_pair(start_point,0));
	int min_d = std::numeric_limits<int>::max();
	int min_id = -1;

	int *d_query_data, *d_X, *d_d; 

	while(!q.empty()){
		auto now = q.front();
		q.pop();
		int id = now.first;
		int hop = now.second;

		int d = 0;
		// int d = squared_l2_dist(X + id * D,query_data,D);

		// Allocate device memory 
		cudaMalloc((void**)&d_query_data, sizeof(int) * D);
		cudaMalloc((void**)&d_X, sizeof(int) * (V * D));
		cudaMalloc(&d_d, sizeof(int));


		// Transfer data from host to device memory
		cudaMemcpy(d_query_data, query_data, sizeof(int) * D, cudaMemcpyHostToDevice);
		cudaMemcpy(d_X, X + id * D, sizeof(int) * (V * D), cudaMemcpyHostToDevice);
		cudaMemcpy(d_d, &d, sizeof(int), cudaMemcpyHostToDevice);


		// Executing kernel 
		int block_size = 256;
		int grid_size = ((D + block_size) / block_size);
		squared_l2_dist_cuda<<<grid_size,block_size>>>(d_X,d_query_data,d_d,D);

		cudaMemcpy(&d, d_d, sizeof(int), cudaMemcpyDeviceToHost);

		// printf("%d %d\n", d, squared_l2_dist(X + id * D,query_data,D));
		printf("%d %d", block_size, grid_size);

		cudaFree(d_query_data);
		cudaFree(d_X);
		cudaFree(d_d);

		// int d = squared_l2_dist(X + id * D,query_data,D);

		if((d < min_d) || (d == min_d && id < min_id)){
			min_d = d;
			min_id = id;
		}
		if(hop + 1 <= max_hop){
			int degree = edges[id * (L + 1)];
			for(int i = 1;i <= degree;++i){
				int v = edges[id * (L + 1) + i];
				q.push(std::make_pair(v,hop + 1));
			}
		}
	}
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
	for(int i = 0;i < Q;++i){
		int start_point,hop;
		fscanf(fin,"%d%d",&start_point,&hop);
		for(int i = 0;i < D;++i){
			fscanf(fin,"%d",&query_data[i]);
		}
		fprintf(fout,"%d\n",nearest_id(start_point,hop,query_data));
	}
	fclose(fin);
	fclose(fout);

	delete[] X;
	delete[] edges;
	delete[] query_data;

	return 0;
}

