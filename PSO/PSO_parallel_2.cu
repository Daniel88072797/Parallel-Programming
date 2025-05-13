/*shared memory、Coalesced Memory Access、interleaved  addressing(handle warp divergent)*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
// #include "parameters.h"
#include <cuda.h>
#include <curand.h>
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
#define r (double)rand() / RAND_MAX  //-1 + 2 * ((double)rand()) / RAND_MAX
#define poprange popmax - popmin
// #define func(a) (((a)*(a)*(a)))
#define func(x) fabs(8000.0 + x * (-10000.0 + x * (-0.8 + x)));
cudaDeviceProp prop;
double *pop=new double[sizepop];
double *V=new double[sizepop];
double *fitness=new double[sizepop];
double *pbest=new double[sizepop];
double *gbest=new double[1];
double *outputMax=new double[sizepop];
// double *answer=new double[1];
int best_index=0;
double *fitnesspbest=new double[sizepop];
double *fitnessgbest=new double[1];
// double elapsedTime;

// double func(double x) { return fabs(8000.0 + x * (-10000.0 + x * (-0.8 + x))); }

// extern double GPU_kernel(double *pop1, double *V1,double *fitnesspbest1,double *pbest1,double *gbest1,double *fitnessgbest1,double r1);


/*初始化粒子的位置和速度*/
void pop_init(double *pop){
        for(int i = 0; i < sizepop; i++){  

                pop[i] = r * poprange + popmin;  //position
                V[i] =  r * Vmax; 
                fitness[i] = func(pop[i]);   //value

        }
}

/*找出最大目標值*/
void max(double *fit,int size){
        int index = 0;
        double max = *fit; //set max value is first
     
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
                
        }
        for(int i = 0; i < sizepop; i++){
                fitnesspbest[i] = fitness[i];
        }

}

__global__ void kernel1(double *V, double *pop, double *pbest, double *gbest, double random1, double random2, double *fitness, double *fitnesspbest, double *outputMax){   //改用cuRand
        
        extern __shared__ double sdata[];
        int pno = blockIdx.x * blockDim.x +  threadIdx.x;
        int tid = threadIdx.x;

        if(fitnesspbest[pno] == outputMax[0]) *gbest = pbest[pno]; 
        __syncthreads(); 
        
        // if(tid == 0) printf("thread %d gbest: %lf\n",tid, *gbest);

        // printf(" %lf   %lf\n",random1,random2);
        // printf("V: %lf\n",V[i]);
        // printf("pno: %d \n",pno);
        /*update valocity*/
        V[pno] = w * V[pno] + c1 * (random1) * (pbest[pno] - pop[pno]) + c2 * (random2) * (*gbest - pop[pno]);   

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

        /*first interleaved reduction*/
        for(int s = 1; s < blockDim.x; s *= 2){

                int index = 2 * s * tid;
                if(index < blockDim.x){
                        sdata[index] = max(sdata[index], sdata[index + s]);
                }

                __syncthreads();
        }
        
        if(tid == 0) outputMax[blockIdx.x] = sdata[tid]; 



        
        
}
__global__ void kernel2(double *outputMax){   //利用reduction

        extern __shared__ double srdata[];
        int tid = threadIdx.x;
        srdata[tid] = outputMax[tid];
        __syncthreads();

        /*second reduction*/
        for(int s = 1; s < blockDim.x; s *= 2){

                int index = 2 * s * tid;
                if(index < blockDim.x){
                        srdata[index] = max(srdata[index], srdata[index + s]);
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
        double time_used;
        clock_gettime(CLOCK_MONOTONIC, &start);

        cudaGetDeviceProperties(&prop, DEV_NO);
        printf("maxThreadsPerBlock = %d\nsharedMemPerBlock = %lu\n", prop.maxThreadsPerBlock, prop.sharedMemPerBlock);   //Shared memory available per block in bytes
        
        PSO_init(); 

	double *d_pop,*d_V,*d_fitnesspbest,*d_pbest;
        double *d_gbest,*d_fitnessgbest,*d_fitness;
        double *d_outputMax;
        
	// Allocate Memory Space on Device
        cudaMalloc(&d_fitness,sizeof(double)*sizepop);
        cudaMalloc(&d_pop,sizeof(double)*sizepop);
        cudaMalloc(&d_V,sizeof(double)*sizepop);
        cudaMalloc(&d_fitnesspbest,sizeof(double)*sizepop);
        cudaMalloc(&d_pbest,sizeof(double)*sizepop);
        cudaMalloc(&d_gbest,sizeof(double));
        cudaMalloc(&d_fitnessgbest,sizeof(double));
        
        cudaMalloc(&d_outputMax,sizeof(double) * ceil(((float)sizepop) / 1024));
 
	// Copy Data to be Calculated
        cudaMemcpy(d_fitness, fitness, sizeof(double)*sizepop, cudaMemcpyHostToDevice);
        cudaMemcpy(d_pop, pop, sizeof(double)*sizepop, cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, V, sizeof(double)*sizepop, cudaMemcpyHostToDevice);
        cudaMemcpy(d_fitnesspbest, fitnesspbest, sizeof(double)*sizepop, cudaMemcpyHostToDevice);
        cudaMemcpy(d_pbest, pbest, sizeof(double)*sizepop, cudaMemcpyHostToDevice);
        cudaMemcpy(d_gbest, gbest, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_fitnessgbest, fitnessgbest, sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(d_outputMax, outputMax, sizeof(double), cudaMemcpyHostToDevice); 


        int num_block = ceil(((float)sizepop) / 1024);
        printf("number of block: %d\n",num_block);

        int num_threads = (sizepop < 1024)? sizepop : 1024;
        printf("number of threads: %d\n",num_threads);

        for(int i = 0; i < gen; i++){

                kernel1<<<num_block ,num_threads, num_threads * sizeof(double)>>>(d_V, d_pop, d_pbest, d_gbest, r, r, d_fitness, d_fitnesspbest, d_outputMax);
                kernel2<<<1, num_block, num_block * sizeof(double)>>>(d_outputMax);

        }

        /*final answer*/
        double answer;
        cudaMemcpy(&answer, d_outputMax, sizeof(double), cudaMemcpyDeviceToHost); 
        printf("The answer is: %lf\n", answer);

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
        time_used = temp.tv_sec + (double) temp.tv_nsec / 1000000000.0;
        
        printf("%f second\n", time_used);

}
