#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
// #include "parameters.h"
#include <cuda.h>
#define DEV_NO 0
#define c1 2
#define c2 2
#define w 1
#define sizepop 2048
#define popmax 100
#define popmin -100
#define Vmax (popmax-popmin)
#define Vmin -(popmax-popmin)
#define gen 500000
#define r (double)rand() / RAND_MAX  //-1 + 2 * ((double)rand()) / RAND_MAX
#define poprange popmax - popmin
// #define func(a) (((a)*(a)*(a)))

cudaDeviceProp prop;
double *pop=new double[sizepop];
double *V=new double[sizepop];
double *fitness=new double[sizepop];
double result[gen];
double *pbest=new double[sizepop];
double *gbest=new double[1];
int best_index=0;
double *fitnesspbest=new double[sizepop];
double *fitnessgbest=new double[1];
double genbest[gen];
// double elapsedTime;

double func(double x) { return fabs(8000.0 + x * (-10000.0 + x * (-0.8 + x))); }

// extern double GPU_kernel(double *pop1, double *V1,double *fitnesspbest1,double *pbest1,double *gbest1,double *fitnessgbest1,double r1);

/*copy*/
void function_1(double B[],double A[]){
	for(int i=0;i<sizepop;i++){
		B[i]=A[i];
	}
}


/*初始化粒子的位置和速度*/
void pop_init(double *pop){
        for(int i = 0; i < sizepop; i++){  
                pop[i] = r * poprange + popmin;  //position
                V[i] =  r * Vmax; 
                fitness[i] = func(pop[i]);   //value
                // printf("pop: %lf\n",pop[i]);
                // printf("V: %lf\n",V[i]);
                // printf("fitness: %lf\n",fitness[i]);
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
        
}
int final_bestfitness(double *fit,int size){
        int index = 0;
        double max = *fit; 
        for(int i = 1;i < size; i++){
                if(*(fit + i) > max)
                {
                        max = *(fit + i);
                        index = i;
                }

                // printf("%lf\n", *(fit+i));
        }

        return index;
        
}

/*更新粒子的位置和速度*/
void updateVelocity(double *V ){
        for(int i = 0;i < sizepop; i++){
                // printf("V: %lf\n",V[i]);
	        V[i] = w*V[i]+c1*(r)*(pbest[i]-pop[i])+(c2)*(r)*(*gbest-pop[i]);   

                if (V[i] > Vmax) V[i] = Vmax;
                if (V[i] < Vmin) V[i] = Vmin;                     
        }

}
void updatePosition(double *pop){           
         for(int i = 0;i < sizepop; i++){
                //  printf("pop: %lf\n",pop[i]);
                 pop[i] = pop[i] + V[i];
                 
                 if(pop[i] > popmax) pop[i] = popmax;
                 if(pop[i] < popmin) pop[i] = popmin; 
          }                     

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

/*更新粒子的位置和速度、找出每一iteration的最佳位置及最佳解*/
void PSO(void){ 
        for(int i = 0;i < gen; i++){
                updateVelocity( V );
                updatePosition( pop );
                for(int j = 0; j < sizepop; j++){
                        // printf("pop %d :%lf ",j,pop[j]);
                        // printf("fitness %d :%lf ",j,fitness[j]);
                        fitness[j] = func(pop[j]); //update fitness
                        // printf("%lf ",fitness[j]);
                }                            
                // printf("\n");
                for(int j = 0; j < sizepop; j++){
                        if(fitness[j] > fitnesspbest[j]){
                                pbest[j] = pop[j];
                                fitnesspbest[j] = fitness[j]; 
                        } 
                        if(fitness[j] > *fitnessgbest){
                                *gbest = pop[j];
                                *fitnessgbest = fitness[j];
                        }   
                }
               
                genbest[i]= *gbest;
                result[i] = *fitnessgbest;

        }
        
}
int main()
{       
        /*count time*/
        struct timespec start, end, temp;
        double time_used;
        clock_gettime(CLOCK_MONOTONIC, &start);

        cudaGetDeviceProperties(&prop, DEV_NO);
        printf("maxThreadsPerBlock = %d\nsharedMemPerBlock = %lu\n", prop.maxThreadsPerBlock, prop.sharedMemPerBlock);   //Shared memory available per block in bytes

        // srand(5);
        PSO_init(); 
        //gpu
        PSO(); 

        int best_gen_number = final_bestfitness(result,gen);
        // for(int i = 0; i < gen; i++){
        //     printf("result %lf\n",result[i]);
        // }
        // printf("best_gen_number %d",best_gen_number);
        // for(int i = 0; i < gen; i++){
        //         printf("genbest[%d] = %lf\n", i, genbest[i]);

        // }
        /*找出最後的最佳解*/
        printf("bestposition is (%lf).\n",genbest[best_gen_number]);
        printf("bestfitness for CPU : %lf.\n",result[best_gen_number]);

        // end = clock();
      
        // printf("CPU time : %5.2f ms\n",(double)(end,start)/1000);

        /*Please press any key to exit the program*/
        // getchar();


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
