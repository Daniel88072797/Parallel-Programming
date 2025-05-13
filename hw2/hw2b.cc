#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include <pthread.h>
#include <unistd.h>
#define min(a,b) ((a) < (b) ? (a) : (b))

//MPI_Request rq1 = MPI_REQUEST_NULL;
//MPI_Request rq2 = MPI_REQUEST_NULL;
//MPI_Request rq3 = MPI_REQUEST_NULL;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;


void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

struct data {
    int size;
    int height;
};
int workload = 1;
void* scheduler(void* srecv) {
      struct data *sw = (struct data*) srecv;
      int size = (sw -> size);
      int height = (sw -> height);
      int recvRank;
      int count = 0;
      int row = 500;
      int k;
      int terminate = 160010;


      for (k = 1; k < size; k++) { // send initial row to each processes
        //MPI_Isend(&row,1,MPI_INT,k,0,MPI_COMM_WORLD, &rq1);
        MPI_Send(&row,1,MPI_INT,k,0,MPI_COMM_WORLD);
        count++;
        row += workload;
        
      }
      
      do {

          //MPI_Irecv(&recvRank,1,MPI_INT, MPI_ANY_SOURCE,1,MPI_COMM_WORLD,&rq2);
          MPI_Recv(&recvRank,1,MPI_INT, MPI_ANY_SOURCE,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
          //printf("receive %d\n", recvRank);
          
          count--;
          
          if (row < height) { // keep sending until no new task 
          
            //MPI_Isend(&row,1,MPI_INT,recvRank,0,MPI_COMM_WORLD, &rq3);
            MPI_Send(&row,1,MPI_INT,recvRank,0,MPI_COMM_WORLD); // send next row 
            count++; 
            row += workload;
            
          } 
          else {
            //MPI_Isend(&t,1,MPI_INT,recvRank,0,MPI_COMM_WORLD, &rq3);
            MPI_Send(&terminate,1,MPI_INT,recvRank,0,MPI_COMM_WORLD); // terminate
          }        
   
      } while(count > 0);
      
      pthread_exit(NULL);
}



int main(int argc, char** argv) {


     

    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    
    int cpu_count = int(CPU_COUNT(&cpu_set));
    //printf("%d cpus available\n", cpu_count);
    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);
    
    /* MPI setting */
    int rank;
    int size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    double y0;
    double x0;
    int repeats;
    double x;
    double y;
    double length_squared;
    double temp;
    int rowNum; 
    /* allocate memory for image */
    int* image = (int*)malloc(width * height * sizeof(int));
    assert(image);
    for(int i = 0; i < width * height; i++){
      image[i] = 0;
    }
    
    if(rank == 0){
    
      /* create master */
      pthread_t thread;
      struct data d;
      d = {size, height};
      int rc;
      
      if(size > 1){
      
        rc = pthread_create(&thread, NULL, scheduler, (void*)&(d));
        if (rc) {
          printf("ERROR; return code from pthread_create() is %d\n", rc);
          exit(-1);
        }
        
      }

      rowNum = 0;
      
      int taskSize;
      
      if(size == 1){
        taskSize = height;
      }
      else{
        taskSize = 500;
      }
      
      #pragma omp parallel num_threads( cpu_count ) 
      { 
        
        #pragma omp for schedule(static, 1) private(y0, x0, repeats,  x, y, length_squared, temp) collapse(2)
        for (int j = rowNum; j < min(taskSize, height) ; ++j) {
          for (int i = 0; i < width; ++i) {
            y0 = j * ((upper - lower) / height) + lower;
            x0 = i * ((right - left) / width) + left;
    
            repeats = 0;
            x = 0;
            y = 0;
            length_squared = 0;
            
            while (repeats < iters && length_squared < 4) {
            
                temp = x * x - y * y + x0;
                y = 2 * x * y + y0;
                x = temp;
                length_squared = x * x + y * y;
                ++repeats;
                
            }
            image[j * width + i] = repeats;
            
          }    
          
        }
        
      }
      if(size > 1) pthread_join(thread,NULL);
        
        
      
      
    }
    else{      

      MPI_Recv(&rowNum,1,MPI_INT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      //MPI_Irecv(&rowNum, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,&rq1);
      while (true) { // keep receiving new task
  
      #pragma omp parallel num_threads(cpu_count) 
      { 
        #pragma omp for schedule(static, 1) private(y0, x0, repeats, x, y, length_squared, temp) collapse(2)
        for (int j = rowNum; j < min(rowNum + workload, height) ; ++j) {
          for (int i = 0; i < width; ++i) {
            y0 = j * ((upper - lower) / height) + lower;
            x0 = i * ((right - left) / width) + left;
    
            repeats = 0;
            x = 0;
            y = 0;
            length_squared = 0;
            
            while (repeats < iters && length_squared < 4) {
            
                temp = x * x - y * y + x0;
                y = 2 * x * y + y0;
                x = temp;
                length_squared = x * x + y * y;
                ++repeats;
                
            }

            image[j * width + i] = repeats;

          }    
        }
      }
        //MPI_Isend(&rank, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &rq2);
        //MPI_Recv(&rowNum, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //MPI_Irecv(&rowNum, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,&rq3);
      MPI_Sendrecv(&rank, 1, MPI_INT, 0, 1, &rowNum, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      //printf("rank %d receive rowNum%d\n", rank, rowNum);
        
      if(rowNum == 160010) break;
      }     
      
    }           

    int* imageRcv = (int*)malloc(width * height * sizeof(int));
    MPI_Reduce (image, imageRcv,width * height, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    free(image); 
    
    if(rank == 0){
      
    /* draw and cleanup */
      write_png(filename, iters, width, height, imageRcv);
      
    }

    free(imageRcv);
    MPI_Finalize(); 

}


      






















