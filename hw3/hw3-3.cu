#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <omp.h>
#define Blockfactor 32
//======================
#define DEV_NO 0
cudaDeviceProp prop;
const int INF = ((1 << 30) - 1);
const int V = 50010;

void input(char* inFileName);
void output(char* outFileName);

// void block_FW(int B);
int ceil(int a, int b);
// void cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height);

int n, m;
// static int Dist[V][V];

extern __shared__ int d2[];
__global__ void p1_cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height, int n ,int* Dist) {
    
    __shared__ int transpose[Blockfactor * (Blockfactor + 1)];
    int b_i = block_start_x;
    int b_j = block_start_y;
    int block_internal_start_x = b_i * B;

    int block_internal_start_y = b_j * B;


    int i = threadIdx.y + block_internal_start_x;
    int j = threadIdx.x + block_internal_start_y;
    if(i >= n) return;
    if(j >= n) return;
    d2[threadIdx.y * B + threadIdx.x] = Dist[i * n + j];
    // __syncthreads();
    int a = 0;

    
    for (int k = Round * B; k < (Round + 1) * B && k < n; ++k) {     
        __syncthreads();
        if (d2[threadIdx.y * B + a] + d2[a * B + threadIdx.x] < d2[threadIdx.y * B + threadIdx.x]) {
            d2[threadIdx.y * B + threadIdx.x] = d2[threadIdx.y * B + a] + d2[a * B + threadIdx.x];
        }

        a++;
        
    }
    transpose[threadIdx.x * (B + 1) + threadIdx.y] = d2[threadIdx.y * B + threadIdx.x];
    

    Dist[i * n + j] = transpose[threadIdx.x * (B + 1) + threadIdx.y];
}


__global__ void p2_cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height, int n ,int* Dist, int c) {
    
    __shared__ int d1[Blockfactor];
    __shared__ int transpose[Blockfactor * (Blockfactor + 1)];
    int b_i = block_start_x + blockIdx.y;
    int b_j = block_start_y + blockIdx.x;

    int block_internal_start_x = b_i * B;

    int block_internal_start_y = b_j * B;

    //shared memory

    int i = threadIdx.y + block_internal_start_x;
    int j = threadIdx.x + block_internal_start_y;
    if(i >= n) return;
    if(j >= n) return;

    d2[threadIdx.y * B + threadIdx.x] = Dist[i * n + j];
    
    
    for (int k = Round * B; k < (Round + 1) * B && k < n; ++k) {  

        if(threadIdx.x == 0 && c == 1){
            d1[threadIdx.y] = Dist[i * n + k];
        }
        if(threadIdx.y == 0 && c == 2){
            d1[threadIdx.x] = Dist[k * n + j];
        }
        
        __syncthreads();
        // if (Dist[i * n + k] + Dist[k * n + j] < Dist[i * n + j]) {
        //     Dist[i * n + j] = Dist[i * n + k] + Dist[k * n + j];
        // }
        //d2[a * B + threadIdx.y]
        if(c == 1){
            if (d1[threadIdx.y] + d2[(k % B) * B + threadIdx.x] < d2[threadIdx.y * B + threadIdx.x]) {
                d2[threadIdx.y * B + threadIdx.x] = d1[threadIdx.y] + d2[(k % B) * B + threadIdx.x];
            }
        }
        if(c == 2){
            if (d2[threadIdx.y * B + (k % B)] + d1[threadIdx.x] < d2[threadIdx.y * B + threadIdx.x]) {
                d2[threadIdx.y * B + threadIdx.x] = d2[threadIdx.y * B + (k % B)] + d1[threadIdx.x];
            }
        }
        __syncthreads();
     
    }

    transpose[threadIdx.x * (B + 1) + threadIdx.y] = d2[threadIdx.y * B + threadIdx.x];
    
    Dist[i * n + j] = transpose[threadIdx.x * (B + 1) + threadIdx.y];
    

}

__global__ void p3_cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height, int n ,int* Dist) {
    __shared__ int d0[Blockfactor];
    __shared__ int d1[Blockfactor];
    __shared__ int transpose[Blockfactor * (Blockfactor + 1)];
    int b_i = block_start_x + blockIdx.y;
    int b_j = block_start_y + blockIdx.x;
    int block_internal_start_x = b_i * B;
    int block_internal_start_y = b_j * B;
    //shared memory

    int i = threadIdx.y + block_internal_start_x;
    int j = threadIdx.x + block_internal_start_y;
    if(i >= n) return;
    if(j >= n) return;

    d2[threadIdx.y * B + threadIdx.x] = Dist[i * n + j];
    // __syncthreads();  




    for (int k = Round * B; k < (Round + 1) * B && k < n; ++k) {     

        // if(threadIdx.y == 0){
            d0[threadIdx.y] = Dist[i * n + k];
        // }
        // if(threadIdx.x == 0){
            d1[threadIdx.x] = Dist[k * n + j];
        // }
        
        // __syncthreads();

        if (d0[threadIdx.y] + d1[threadIdx.x] < d2[threadIdx.y * B + threadIdx.x]) {  //d0[threadIdx.x] + d1[threadIdx.y]
            d2[threadIdx.y * B + threadIdx.x] = d0[threadIdx.y] + d1[threadIdx.x];
        }
        

        // __syncthreads();
     
    }

    transpose[threadIdx.x * (B + 1) + threadIdx.y] = d2[threadIdx.y * B + threadIdx.x];
    
    Dist[i * n + j] = transpose[threadIdx.x * (B + 1) + threadIdx.y];
}



__global__ void cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height, int n ,int* Dist) {

    int b_i = block_start_x + blockIdx.y;
    int b_j = block_start_y + blockIdx.x;
    int block_internal_start_x = b_i * B;
    int block_internal_start_y = b_j * B;
    //shared memory

    int i = threadIdx.x + block_internal_start_x;
    int j = threadIdx.y + block_internal_start_y;
    if(i >= n) return;
    if(j >= n) return;
        
    for (int k = Round * B; k < (Round + 1) * B && k < n; ++k) {     

        if (Dist[i * n + k] + Dist[k * n + j] < Dist[i * n + j]) {
            Dist[i * n + j] = Dist[i * n + k] + Dist[k * n + j];
        }
        

        __syncthreads();
     
    }


}

void block_FW(int B, int* Dist)
{


    int round = ceil(n, B);

    int* dev0Mem_Dist;
    int* dev1Mem_Dist;
    cudaSetDevice(0);
    cudaMalloc((void**)&dev0Mem_Dist, n * n * sizeof(int));

    cudaSetDevice(1);
    cudaMalloc((void**)&dev1Mem_Dist, n * n * sizeof(int));
    cudaMemcpy(dev0Mem_Dist, Dist, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev1Mem_Dist, Dist, n * n * sizeof(int), cudaMemcpyHostToDevice);



    // int *distance;
    // cudaMalloc(&distance, n * n * sizeof(int));
    // // cudaMallocHost(&Dist, n * n * sizeof(int));
    // cudaMemcpy(distance, Dist, n * n * sizeof(int), cudaMemcpyHostToDevice);

    // for(int i = 0; i < n * n; i++)
    // printf("Dist[%d] = %d\n", i, Dist[i]);

    dim3 num_threads(B, B);
    dim3 num_t(B, B + 1);
    
    
    // const int num_threads = B;
    const int sm_size = B * B * sizeof(int);
    for (int r = 0; r < round; ++r) {
        // printf("%d %d\n", r, round);
        //fflush(stdout);

        /* Phase 1*/
        cudaSetDevice(0);
        p1_cal<<<1 ,num_threads, sm_size>>>(B, r, r, r, 1, 1, n, dev0Mem_Dist);
        cudaMemcpyPeer(dev1Mem_Dist, 1, dev0Mem_Dist, 0, n * n * sizeof(int));
        dim3 num_block1(r, 1);
        dim3 num_block2(round - r - 1, 1);
        dim3 num_block3(1, r);
        dim3 num_block4(1, round - r - 1);

        #pragma omp parallel
        {
            
            unsigned int cpu_thread_id = omp_get_thread_num();
            unsigned int num_cpu_threads = omp_get_num_threads();
            cudaSetDevice(cpu_thread_id);
            int gpu_id = -1;
            cudaGetDevice(&gpu_id);
            printf("CPU thread %d (of %d) uses CUDA device %d\n",cpu_thread_id, num_cpu_threads, gpu_id);

            if(cpu_thread_id == 0){
                printf("%d", cpu_thread_id);
                cudaSetDevice(0);
                p2_cal<<<num_block1 ,num_threads, sm_size>>>(B, r, r, 0, r, 1, n, dev0Mem_Dist, 1);
                p2_cal<<<num_block3 ,num_threads, sm_size>>>(B, r, 0, r, 1, r, n, dev0Mem_Dist, 2); //**
                cudaMemcpyPeer(dev1Mem_Dist, 1, dev0Mem_Dist, 0, n * n * sizeof(int));
            }
            #pragma omp barrier
            if(cpu_thread_id == 1)
            {
                printf("%d", cpu_thread_id);
                cudaSetDevice(1);
                
                p2_cal<<<num_block2 ,num_threads, sm_size>>>(B, r, r, r + 1, round - r - 1, 1, n, dev1Mem_Dist, 1);
                p2_cal<<<num_block4 ,num_threads, sm_size>>>(B, r, r + 1, r, 1, round - r - 1, n, dev1Mem_Dist, 2); //**
                cudaMemcpyPeer(dev0Mem_Dist, 0, dev1Mem_Dist, 1,  n * n * sizeof(int));
            }
            
            



        }


        cudaSetDevice(0);
        dim3 num_block5(r, r);
        dim3 num_block6(round - r - 1, r);
        dim3 num_block7(r, round - r - 1);
        dim3 num_block8(round - r - 1, round - r - 1);
        /* Phase 3*/
        p3_cal<<<num_block5 ,num_threads, sm_size>>>(B, r, 0, 0, r, r, n, dev0Mem_Dist);
        p3_cal<<<num_block6 ,num_threads, sm_size>>>(B, r, 0, r + 1, round - r - 1, r, n, dev0Mem_Dist);
        p3_cal<<<num_block7 ,num_threads, sm_size>>>(B, r, r + 1, 0, r, round - r - 1, n, dev0Mem_Dist);
        p3_cal<<<num_block8 ,num_threads, sm_size>>>(B, r, r + 1, r + 1, round - r - 1, round - r - 1, n, dev0Mem_Dist);
    }

    cudaMemcpy(Dist, dev0Mem_Dist, n * n * sizeof(int), cudaMemcpyDeviceToHost);
    // cudaFree(distance);

}


int main(int argc, char* argv[]) {

    /*read the file*/
    FILE* file = fopen(argv[1], "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    int* Dist = (int *)malloc(n * n * sizeof(int));
    printf("%d", n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                Dist[i * n + j] = 0;
            } else {
                Dist[i * n + j] = INF;
            }
        }
    }
    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0] * n + pair[1]] = pair[2];
    }
    fclose(file);
    // input(argv[1]);

    // cudaGetDeviceProperties(&prop, DEV_NO);
    // printf("maxThreadsPerBlock = %d\nsharedMemPerBlock = %lu\n", prop.maxThreadsPerBlock, prop.sharedMemPerBlock);   //Shared memory available per block in bytes
    
    int B = Blockfactor;
    block_FW(B, Dist);
    
    // output(argv[2]);

    /*save the file*/
    FILE* outfile = fopen(argv[2], "w");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (Dist[i * n + j] >= INF) Dist[i * n + j] = INF;
        }
        fwrite(Dist + i * n, sizeof(int), n, outfile);
    }
    fclose(outfile);
   
}

int ceil(int a, int b) { return (a + b - 1) / b; }



// #pragma omp barrier
    
//     if(cpu_thread_id == 0){
//         printf("%d", cpu_thread_id);
//         cudaSetDevice(0);

//         cudaMemcpyPeer(dev1Mem_Dist, 1, dev0Mem_Dist, 0, n * n * sizeof(int));
//     }
//     if(cpu_thread_id == 1)
//     {
//         printf("%d", cpu_thread_id);
//         cudaSetDevice(1);

//         cudaMemcpyPeer(dev0Mem_Dist, 0, dev1Mem_Dist, 1,  n * n * sizeof(int));
//     }

// }

// #pragma omp parallel 
// {   
//     dim3 num_block1(r, 1);
//     dim3 num_block2(round - r - 1, 1);
//     dim3 num_block3(1, r);
//     dim3 num_block4(1, round - r - 1);
//     unsigned int cpu_thread_id = omp_get_thread_num();
//     if(cpu_thread_id == 0){
//         printf("%d", cpu_thread_id);
//         cudaSetDevice(0);
//         p2_cal<<<num_block1 ,num_threads, sm_size>>>(B, r, r, 0, r, 1, n, distance, 1);
//         p2_cal<<<num_block2 ,num_threads, sm_size>>>(B, r, r, r + 1, round - r - 1, 1, n, distance, 1);
//     }
//     if(cpu_thread_id == 1)
//     {
//         printf("%d", cpu_thread_id);
//         cudaSetDevice(0);
//         p2_cal<<<num_block3 ,num_threads, sm_size>>>(B, r, 0, r, 1, r, n, distance, 2); //**
//         p2_cal<<<num_block4 ,num_threads, sm_size>>>(B, r, r + 1, r, 1, round - r - 1, n, distance, 2); //**
//     }

// }