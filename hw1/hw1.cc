#include <cstdio>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
//#include <time.h>
int cmpfunc(const void *a,const void *b) {   //compare function
  float fa = *(const float*) a;
  float fb = *(const float*) b;
  return (fa > fb) - (fa < fb);
}



int main(int argc, char** argv) {
//  struct timespec start, end, temp;
//  double time_used;
//  clock_gettime(CLOCK_MONOTONIC, &start);

  int n = atoi(argv[1]);
  
  //printf("%d",n);
	MPI_Init(&argc,&argv);
  //double starttime, endtime;
  //starttime = MPI_Wtime();
  
	int rank = 0, size = 0;
  
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  int num = n / size; //每個 rank floating point 個數

  int remain = n - num * size;
  
  if(rank < remain){
    num = num + 1;
  }
  
  
  //配置動態記憶體 
  float* data;
  data = (float*)malloc(num * sizeof(float) );

  
  /*讀取檔案*/
  MPI_File f;
  MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &f); 
  
  if(rank < remain){
    MPI_File_read_at(f, sizeof(float) * num * rank , data, num, MPI_FLOAT, MPI_STATUS_IGNORE);
  }
  else{
    MPI_File_read_at(f, sizeof(float) * (remain + rank * num) , data, num, MPI_FLOAT, MPI_STATUS_IGNORE);
  }
  
  MPI_File_close(&f);

  
  int rightnum;
  if(rank < remain && rank + 1 < remain){
    rightnum = num;
  }
  else if(rank < remain && rank + 1 >= remain){
    rightnum = num - 1;
  }
  else{
    rightnum = num;
  }


  //配置動態記憶體

  float* Buf ; 
  Buf = (float*)malloc( (num + rightnum) * sizeof(float) );
  float* srBuf ;
  srBuf= (float*)malloc( rightnum * sizeof(float) );

  int i ;
  int j ;
  int k ;
  int loopCount = 0; 
  qsort(data, num, sizeof(float), cmpfunc);
  if(size > 1){
      
      while(true){
 
        /*even phase*/
        /*------------------------------------------------------------------------------------------------------------*/
        /*將odd rank P 的 fp data send to rank P-1  */
        
        
        if(rank % 2 == 1){
          MPI_Send(data,num,MPI_FLOAT,rank - 1, 1, MPI_COMM_WORLD);
        }
        else if(rank % 2 == 0 && rank != size - 1){
          
          MPI_Recv(srBuf,rightnum,MPI_FLOAT,rank + 1,MPI_ANY_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
          i = 0;
          j = 0;
          k = 0;
          while(k < rightnum + num){      //merge two sorted array
            if(i < num && j == rightnum)
              Buf[k++] = data[i++];
            else if(i == num && j < rightnum) 
              Buf[k++] = srBuf[j++];
            else if(data[i] <= srBuf[j])
              Buf[k++] = data[i++];
            else
              Buf[k++] = srBuf[j++];
          }
          

        }

        
    
        if(rank % 2 == 0 && rank != size - 1){    //傳送較大的floating point numbers 到右邊rank
          
          for (i = 0; i < num; i++)
          {
            data[i] = Buf[i];
            if(i < rightnum){
              srBuf[i] = Buf[num + i];
            }
          }
          MPI_Send(srBuf,rightnum,MPI_FLOAT,rank + 1,1,MPI_COMM_WORLD);
          
        }
        else if(rank % 2 == 1){
        
          MPI_Recv(data,num,MPI_FLOAT,rank - 1,MPI_ANY_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);    //接收左邊rank資料並寫入data[]中

        }
        
       
        /*odd phase*/
        /*------------------------------------------------------------------------------------------------------------*/
        /*將even rank Q 的 fp data send to rank Q-1  */
        
        if(rank % 2 == 0){
          
          MPI_Send(data,num,MPI_FLOAT,rank - 1, 1, MPI_COMM_WORLD);
        }
        else if(rank % 2 == 1 && rank != size - 1){
          
          MPI_Recv(srBuf,rightnum,MPI_FLOAT,rank + 1,MPI_ANY_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
          i = 0;
          j = 0;
          k = 0;
          while(k < rightnum + num){            //merge two sorted array
            if(i < num && j == rightnum)
              Buf[k++] = data[i++];
            else if(i == num && j < rightnum) 
              Buf[k++] = srBuf[j++];
            else if(data[i] <= srBuf[j])
              Buf[k++] = data[i++];
            else
              Buf[k++] = srBuf[j++];
          }
        }
    
        if(rank % 2 == 1 && rank != size - 1){    //傳送較大的floating point numbers 到右邊rank
          
          for (i = 0; i < num; i++)
          {
            data[i] = Buf[i];
            if(i < rightnum){
              srBuf[i] = Buf[num + i];
            }
          }
          MPI_Send(srBuf,rightnum,MPI_FLOAT,rank + 1,1,MPI_COMM_WORLD);
          
        }
        else if(rank % 2 == 0){
        
          MPI_Recv(data,num,MPI_FLOAT,rank - 1,MPI_ANY_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);    //接收左邊rank資料並寫入data[]中

        }
        
        loopCount += 2;
    
        if (loopCount > size)
          break;
      
      
      }

    }
  

  free(Buf);
  free(srBuf);
    

/* write data in the file */    

    
  MPI_File f1;
  MPI_File_open(MPI_COMM_WORLD,argv[3], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &f1);    //open file

  if(rank < remain){
    MPI_File_write_at(f1, sizeof(float) * num * rank , data, num, MPI_FLOAT, MPI_STATUS_IGNORE);
  }
  else{
    MPI_File_write_at(f1, sizeof(float) * (remain + rank * num) , data, num, MPI_FLOAT, MPI_STATUS_IGNORE);
  }
  free(data);

  MPI_File_close(&f1);
  //endtime = MPI_Wtime();
  //printf("That took %f seconds\n",endtime-starttime);
  MPI_Finalize();




}


