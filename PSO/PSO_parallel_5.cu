/*shared memory、Coalesced Memory Access、sequential addressing(handle bank conflict、handle warp divergent)、curand()、double->float*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
// #include "parameters.h"
#include <cuda.h>
#include <curand_kernel.h>
#define DEV_NO 0
#define c1 2
#define c2 2
#define w 1
#define sizepop 4096
#define popmax 100
#define popmin -100
#define Vmax (popmax-popmin)
#define Vmin -(popmax-popmin)
#define gen 1000000
#define r (float)rand() / RAND_MAX  //-1 + 2 * ((float)rand()) / RAND_MAX
#define poprange popmax - popmin
// #define func(a) (((a)*(a)*(a)))
#define func(x) fabs(8000.0 + x * (-10000.0 + x * (-0.8 + x)));
cudaDeviceProp prop;
float *pop=new float[sizepop];
float *V=new float[sizepop];
float *fitness=new float[sizepop];
float *pbest=new float[sizepop];
float *gbest=new float[1];
float *outputMax=new float[sizepop];
// float *answer=new float[1];
int best_index=0;
float *fitnesspbest=new float[sizepop];
float *fitnessgbest=new float[1];
// float elapsedTime;

// float func(float x) { return fabs(8000.0 + x * (-10000.0 + x * (-0.8 + x))); }

/*初始化粒子的位置和速度*/
void pop_init(float *pop){

        // #pragma unroll 32
        for(int i = 0; i < sizepop; i++){  

                pop[i] = r * poprange + popmin;  //position
                V[i] =  r * Vmax; 
                fitness[i] = func(pop[i]);   //value

        }
}

/*找出最大目標值*/
void max(float *fit,int size){
        int index = 0;
        float max = *fit; //set max value is first
     
        for(int i = 1;i < size; i++){
                if(*(fit+i) > max){
                    max = *(fit + i);
                    index = i;
                }
        }

        printf("max : %f\n",max);
        best_index = index;
        *fitnessgbest = max;
        outputMax[0] = max;
        
}


/*首先初始化粒子的位置和速度並求得「粒子自身最佳位置」、「群體最佳位置」、「個體最佳解」、「群體最佳解」*/
void PSO_init(void)
{
        pop_init(pop);
        max(fitness,sizepop);
        
        *gbest = pop[best_index];
        
        for(int i = 0; i < sizepop; i++){
                
                pbest[i] = pop[i];
                pbest[i] = pop[i];
                
        }
        for(int i = 0; i < sizepop; i++){
                fitnesspbest[i] = fitness[i];
        }

}

__global__ void init_kernel(curandState *state, long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; 
    curand_init(seed, idx, 0, state);
}


__global__ void kernel1(float *V, float *pop, float *pbest, float *gbest, curandState *state, float *fitness, float *fitnesspbest, float *outputMax){   //改用cuRand
        
        extern __shared__ float sdata[];
        int pno = blockIdx.x * blockDim.x +  threadIdx.x;
        int tid = threadIdx.x;
        
        
        if(fitnesspbest[pno] == outputMax[0]) *gbest = pbest[pno]; 
        __syncthreads(); 

        float r1 = curand_uniform(state);
        float r2 = curand_uniform(state);
        // printf("random: %lf  %lf\n", r1, r2);
        // if(tid == 0) printf("thread %d gbest: %lf\n",tid, *gbest);

        // printf(" %lf   %lf\n",random1,random2);
        // printf("V: %lf\n",V[i]);
        // printf("pno: %d \n",pno);
        /*update valocity*/
        V[pno] = w * V[pno] + c1 * (r1) * (pbest[pno] - pop[pno]) + c2 * (r2) * (*gbest - pop[pno]);   

        if (V[pno] > Vmax) V[pno] = Vmax;
        if (V[pno] < Vmin) V[pno] = Vmin;        

        // printf("pop: %lf\n",pop[i]);

        /*update position*/
        pop[pno] = pop[pno] + V[pno];
        
        if(pop[pno] > popmax) pop[pno] = popmax;
        if(pop[pno] < popmin) pop[pno] = popmin;

        /*update fitness*/
        fitness[pno] = func(pop[pno]); 

        /*update pbest、fitnesspbest*/
        if(fitness[pno] > fitnesspbest[pno]){       
                pbest[pno] = pop[pno];   
                fitnesspbest[pno] = fitness[pno]; 
        } 

        /*move fitnesspbest to shared memory data*/
        sdata[tid] = fitnesspbest[pno];
        __syncthreads();

        #pragma unroll 5
        for(int s = blockDim.x / 2; s > 0; s >>= 1){
                if(tid < s){
                        sdata[tid] = max(sdata[tid], sdata[tid + s]); 
                }
                __syncthreads();
        }



        if(tid == 0) outputMax[blockIdx.x] = sdata[tid]; 



        
        
}
__global__ void kernel2(float *outputMax){   //利用reduction

        extern __shared__ float srdata[];
        int tid = threadIdx.x;
        srdata[tid] = outputMax[tid];
        __syncthreads();

        /*second reduction*/
        for(int s = blockDim.x / 2; s > 0; s >>= 1){
                if(tid < s){
                        srdata[tid] = max(srdata[tid], srdata[tid + s]); 
                }
                __syncthreads();
        }

        if(tid == 0) outputMax[tid] = srdata[tid];
        // if(tid == 0) printf("fitness gbest per iter: %lf\n", outputMax[0]); 

}

int main()
{       
        /*count time*/
        struct timespec start, end, temp;
        float time_used;
        clock_gettime(CLOCK_MONOTONIC, &start);

        cudaGetDeviceProperties(&prop, DEV_NO);
        printf("maxThreadsPerBlock = %d\nsharedMemPerBlock = %lu\n", prop.maxThreadsPerBlock, prop.sharedMemPerBlock);   //Shared memory available per block in bytes
        
        PSO_init(); 

	float *d_pop,*d_V,*d_fitnesspbest,*d_pbest;
        float *d_gbest,*d_fitnessgbest,*d_fitness;
        float *d_outputMax;
        
        curandState *state; 
        cudaMalloc(&state, sizeof(curandState)); 
        init_kernel<<<1,1>>>(state, clock());         
        


	// Allocate Memory Space on Device
        cudaMalloc(&d_fitness,sizeof(float)*sizepop);
        cudaMalloc(&d_pop,sizeof(float)*sizepop);
        cudaMalloc(&d_V,sizeof(float)*sizepop);
        cudaMalloc(&d_fitnesspbest,sizeof(float)*sizepop);
        cudaMalloc(&d_pbest,sizeof(float)*sizepop);
        cudaMalloc(&d_gbest,sizeof(float));
        cudaMalloc(&d_fitnessgbest,sizeof(float));
        
        cudaMalloc(&d_outputMax,sizeof(float) * ceil(((float)sizepop) / 1024));
 
	// Copy Data to be Calculated
        cudaMemcpy(d_fitness, fitness, sizeof(float)*sizepop, cudaMemcpyHostToDevice);
        cudaMemcpy(d_pop, pop, sizeof(float)*sizepop, cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, V, sizeof(float)*sizepop, cudaMemcpyHostToDevice);
        cudaMemcpy(d_fitnesspbest, fitnesspbest, sizeof(float)*sizepop, cudaMemcpyHostToDevice);
        cudaMemcpy(d_pbest, pbest, sizeof(float)*sizepop, cudaMemcpyHostToDevice);
        cudaMemcpy(d_gbest, gbest, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_fitnessgbest, fitnessgbest, sizeof(float), cudaMemcpyHostToDevice);

        cudaMemcpy(d_outputMax, outputMax, sizeof(float), cudaMemcpyHostToDevice); 


        int num_block = ceil(((float)sizepop) / 1024);
        printf("number of block: %d\n",num_block);

        int num_threads = (sizepop < 1024)? sizepop : 1024;
        printf("number of threads: %d\n",num_threads);

        for(int i = 0; i < gen; i++){

                kernel1<<<num_block ,num_threads, num_threads * sizeof(float)>>>(d_V, d_pop, d_pbest, d_gbest, state, d_fitness, d_fitnesspbest, d_outputMax);
                kernel2<<<1, num_block, num_block * sizeof(float)>>>(d_outputMax);

        }

        /*final answer*/
        float answer;
        cudaMemcpy(&answer, d_outputMax, sizeof(float), cudaMemcpyDeviceToHost); 
        printf("The answer is: %f\n", answer);

        /*等待kernel執行完*/
        // cudaDeviceSynchronize();

        /*count time*/
        clock_gettime(CLOCK_MONOTONIC, &end);
        if ((end.tv_nsec - start.tv_nsec) < 0) {
        temp.tv_sec = end.tv_sec-start.tv_sec-1;
        temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
        } else {
        temp.tv_sec = end.tv_sec - start.tv_sec;
        temp.tv_nsec = end.tv_nsec - start.tv_nsec;
        }
        time_used = temp.tv_sec + (float) temp.tv_nsec / 1000000000.0;
        
        printf("%f second\n", time_used);

}
