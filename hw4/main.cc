#include <cstdio>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <pthread.h>
#include <string>
#include <iostream>
#include <ostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <unistd.h>
#include <time.h>
//#include <time.h>

using namespace std;

int count_total = 0;
int count_final_total = 0;

void scheduler(char** argv, int num_worker){
  
  string job_name = string(argv[1]);
  string input_filename = string(argv[4]);
  string locality_config_filename = string(argv[6]);
  string output_dir = string(argv[7]);
  
  int num_reducer = atoi(argv[2]);
  int delay = atoi(argv[3]);
  int chunk_size = atoi(argv[5]);
  
  cout<<"job_name: "<< job_name << endl;
  cout<<"input_filename: "<< input_filename << endl;
  cout<<"locality_config_filename: "<< locality_config_filename << endl;
  cout<<"output_dir: "<< output_dir << endl;
  printf("num_reducer: %d\n",num_reducer);
  printf("delay: %d\n",delay);
  printf("chunk_size: %d\n",chunk_size);
  
  string log_file_name = output_dir + "/" +job_name + ".log"; //"./"+output_dir+"/"
  ofstream write_log(log_file_name);
  
  ifstream loc_file(locality_config_filename);
  
  /*get cpu number*/
  cpu_set_t cpu_set;
  sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
  int cpu_num = CPU_COUNT(&cpu_set);
  
  string loc_line; 
  vector<int> chunkID;
  vector<int> nodeID;
    
  int start =  time(NULL); 
    
  write_log << start <<", Start_Job, " << argv[1] << ", ";
  write_log << (num_worker + 1) << ", " << cpu_num << ", " << argv[2] << ", ";
  write_log << argv[3] << ", " << argv[4] << ", " << argv[5] << ", ";
  write_log << argv[6] << ", " << argv[7] << "\n";
  
  
  /* initialize....get chunkID、nodeID (task tracker) */
  while(getline(loc_file, loc_line)){
  
      size_t pos = 0;
      string loc_params;
       
      while ((pos = loc_line.find(" ")) != string::npos){
      
          chunkID.push_back(stoi(loc_line.substr(0, pos)));
          loc_line.erase(0, pos + 1);
          nodeID.push_back(stoi(loc_line.substr(0, pos)) % num_worker);  
                                      
      }
      
  }
  
//  for(int i = 0; i < chunkID.size(); i++) printf("chunkID[%d] = %d nodeID[%d] = %d \n", i, chunkID[i], i, nodeID[i]);
  
//  int req_nodeID;
//  for(int i = 0; i < 2; i++) {
//    MPI_Recv(&req_nodeID,1,MPI_INT, MPI_ANY_SOURCE,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
//    cout << "req_nodeID = " << req_nodeID << endl; 
//  }
  

  /*scheduler(send map tasks)*/
  int* alloc_mark = (int*)malloc(chunkID.size() * sizeof(int));  //mark the allocated nodID array 
  for(int i = 0; i < chunkID.size(); i++) alloc_mark[i] = 0;
  int send_count = 0; 
  int endcount;
  int sendTask[2] = {0, 0};  //{chunkid, nodeid}
  int hit = 0;
  int request[2];
  int terminate = 0; //terminate send map tasks scheduler
  
  
  
  int* worker_first_request = (int*)malloc(num_worker * sizeof(int));
  int* record_start = (int*)malloc(num_worker * sizeof(int));
  
  
  for(int i = 0; i < num_worker; i++) worker_first_request[i] = 0;
  for(int i = 0; i < num_worker; i++) record_start[i] = 0;
  
  
  
  
  do{
    hit = 0;
    
    MPI_Recv(request,2,MPI_INT, MPI_ANY_SOURCE,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);  //receive request
    
    
    /*write log file*/
    if(worker_first_request[request[0]] == 1){  //if not first request
      
      write_log << time(NULL) << ", Complete_MapTask, "<< request[1] <<", "<<time(NULL) - record_start[request[0]]<<"\n"; 
    
    } 
     
    worker_first_request[request[0]] = 1;     
    
    
       
    for(int i = 0; i < chunkID.size(); i++){
      
      if(request[0] == nodeID[i] && alloc_mark[i] == 0){
      
        sendTask[0] = chunkID[i];
        sendTask[1] = nodeID[i];
        alloc_mark[i] = 1;
        send_count++;
        hit = 1;
        cout << "scheduler send chunk:" << chunkID[i] << endl;
        break;
          
      }
    }
        
    if(hit == 0){
    
      for(int i = 0; i < chunkID.size(); i++){
      
        if(alloc_mark[i] == 0){
        
          sendTask[0] = chunkID[i];
          sendTask[1] = nodeID[i];
          send_count++;
          alloc_mark[i] = 1;
          
          cout << "scheduler send chunk:" << chunkID[i] << endl;
          break;        

        }
        
      }
     
    }
    
    if(sendTask[0] != -1){
    
      write_log << time(NULL) << ", Dispatch_MapTask, " << sendTask[0] << ", " << request[0] << "\n";
      record_start[request[0]] = time(NULL);
    } 
    
    
    
    MPI_Send(sendTask,2,MPI_INT,request[0],0,MPI_COMM_WORLD); //send the task
    
    
    
    if(send_count == chunkID.size()){
    
      sendTask[0] = -1;  //-1 is terminate condition
      sendTask[1] = -1;
      
      endcount++;
    }
    
    

  } while(endcount <= num_worker);

  free(alloc_mark);
  MPI_Barrier(MPI_COMM_WORLD);
  
  cout<<"schedule map task end"<<endl;
  
  
  
  cout<<"chunkID.size()"<<chunkID.size()<<endl;

  /*do shuffle*/
  cout<<"start shuffling"<<endl;
  int start_shuffle = time(NULL);
  write_log << start_shuffle << ", Start_Shuffle, ";
  
  string reducer_num_str, chunk_str, line_num_str, filename, word;
  int countdata = 0;
  int countNum = 0;
  
  vector<string> key;
  vector<int> value; 
  
  for(int i = 0; i < num_reducer; i++){
  
    reducer_num_str = to_string(i);
    for(int j = 1; j <= chunkID.size(); j++){ 
    
      chunk_str = to_string(j);
      
      for(int k = (j - 1) * chunk_size + 1; k <= j * chunk_size; k++){
      
        line_num_str = to_string(k);
        
        filename = chunk_str + "_" + line_num_str + "_" + reducer_num_str + ".txt"; //"./intermediate_file/" + 
        ifstream input_file(filename);
        
        while (input_file >> word >> countNum) {
        
            key.push_back(word);
            value.push_back(countNum);
            countdata++;
          
       }
       
       input_file.close();
       
      }
      
    }
    
    filename = reducer_num_str + ".txt"; //"./intermediate_file/finish_shuffle/" +  
    ofstream intermediate_file(filename);

    //cout << "key size" << key.size() << endl;
    for (int i = 0; i < key.size(); i++) {
    
      intermediate_file << key[i] << " " << value[i] << "\n";
      
    }
    
    key.clear();
    value.clear();
    
  
  }
  write_log << countdata << "\n";
  cout<<"finish shuffling"<<endl;
  cout << "countdata = " << countdata << endl;
  
  
  write_log << time(NULL) << ", Finish_Shuffle, "<<(time(NULL) - start_shuffle) <<"\n";



  /*scheduler(send reducing tasks)*/ 
  int* alloc_mark_r = (int*)malloc(num_reducer * sizeof(int));
  int sendTask_r,send_count_r;
  int endcount_r = 0;
  int reducer_request[2];
  for(int i = 0; i < num_reducer; i++) alloc_mark_r[i] = 0;

  
  int* worker_first_request_r = (int*)malloc(num_worker * sizeof(int));
  int* record_start_r = (int*)malloc(num_worker * sizeof(int));
  
  for(int i = 0; i < num_worker; i++) worker_first_request_r[i] = 0;
  for(int i = 0; i < num_worker; i++) record_start_r[i] = 0;
  
  
  
  
  do{
    
    MPI_Recv(reducer_request,2,MPI_INT, MPI_ANY_SOURCE,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);  //receive request
    
    
    /*write log file*/
    if(worker_first_request_r[reducer_request[0]] == 1){  //if not first request
      
      write_log << time(NULL) << ", Complete_ReduceTask, "<< reducer_request[1] <<", "<<(time(NULL) - record_start_r[reducer_request[0]])<<"\n"; 
    
    } 
    worker_first_request_r[reducer_request[0]] = 1;     
      
    for(int i = 0; i < num_reducer; i++){
      
      if(alloc_mark_r[i] == 0){
      
        sendTask_r = i;
        
        alloc_mark_r[i] = 1;
        
        send_count_r++;
        
        cout << "scheduler send reducer task:" << i << endl;
        break;
          
      }
    }
    
    
    
    if(sendTask_r != -1){
    
      write_log << time(NULL) << ", Dispatch_ReduceTask, " << sendTask_r << ", " << reducer_request[0] << "\n";
      record_start_r[reducer_request[0]] = time(NULL);
    } 
    
    MPI_Send(&sendTask_r,1,MPI_INT,reducer_request[0],0,MPI_COMM_WORLD); //send the task
    
    if(send_count_r == num_reducer){
    
      sendTask_r = -1;  //-1 is terminate condition
      
      endcount_r++;
    }
    
    

  } while(endcount_r <= num_worker);  
  
  
  write_log << time(NULL) << ", Finish_Job" << ", " << time(NULL) - start << "\n";

  free(alloc_mark_r);


}

int partition_function(string word, int num_reducer){
   
  return (word.length() % num_reducer);

}

/*reads an input key-value pair record and output to a set of intermediate key-value pairs. */
void mapper_function(int lineNumber, string line, int num_reducer, int chunkID){

  int pos = 0;
  string word;
  vector<string> key;
  vector<int> value;
  vector<int> reducerID;
  
  /*generate key & value pair*/
  while ((pos = line.find(" ")) != string::npos) {   //少一個字
  
      word = line.substr(0, pos);

      key.push_back(word);
      value.push_back(1);
      reducerID.push_back( partition_function(word, num_reducer) );  //call partition function return reducerID
      
      count_total++; //count total data per node
//      cout<< word << endl;
      line.erase(0, pos + 1);
  }
  
  if (!line.empty()){  //補上最後一個字
  
      key.push_back(line);
      value.push_back(1);
      reducerID.push_back( partition_function(line, num_reducer) );  //call partition function return reducerID  
      
      count_total++; //count total data per node
//      cout << "line = " << line << endl;
  } 
        
        
        
        
  /*store intermediate data to file (chunk number 1 to k, reducer number 0 to n-1) */
  for (int i = 0; i < num_reducer; i++) {
  
    string chunk_str = to_string(chunkID);
    string reducer_num_str = to_string(i); 
    string line_num_str = to_string(lineNumber);
    
    string filename = chunk_str + "_" + line_num_str + "_" + reducer_num_str + ".txt"; //"./intermediate_file/" + 
    
    ofstream myfile(filename);
    
    for (int j = 0; j < reducerID.size(); j++) {
    
      if(i == reducerID[j]){
      
        myfile << key[j] << " " << value[j] << "\n";
        
      }
        
        
    }
    
    myfile.close();
    
  }
    
}

bool sort_comp(string a, string b){ 
  
  return a < b;

}

bool group_comp(string a, string b){
  
  if(a == b) return true;
  else return false;

}



void worker(char** argv, int num_worker, int workerID){

  string job_name = string(argv[1]);
  string input_filename = string(argv[4]);
  string locality_config_filename = string(argv[6]);
  string output_dir = string(argv[7]);
  
  int num_reducer = atoi(argv[2]);
  int delay = atoi(argv[3]); 
  int chunk_size = atoi(argv[5]);
  
  int recvTask[2];
  int schedulerID = num_worker;
  int lineNumber;
  int request[2];
  request[0] = workerID;
  
  do{ //if the task is not terminate message
  
    
    /*request new task*/
    MPI_Send(request,2,MPI_INT,schedulerID,0,MPI_COMM_WORLD); //initial request to scheduler
    
    MPI_Recv(recvTask,2,MPI_INT, schedulerID,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    
    if(recvTask[0] == -1) break;
    
    if(recvTask[1] != workerID){
    
      cout<<"worker:"<< workerID <<" ready to sleep"<<endl;
      sleep(delay); 
      cout<<"worker:"<< workerID <<" finish sleep"<<endl;
      
    }
    
    
    /*split function(split chunk)*/
    int start_pos = 1 + (recvTask[0] - 1) * chunk_size;
    ifstream input_file(input_filename);
    string line;
    
    for (int i = 1; i < start_pos; i++) {
    
        getline(input_file, line);
                
    }
    for (int i = 1; i <= chunk_size; i++) {
    
        getline(input_file, line);
        lineNumber = start_pos + i - 1;

        /*mapfunction (map each line)*/
        mapper_function(lineNumber, line, num_reducer, recvTask[0]);  
        
  
    }
    input_file.close();
    
    request[1] = recvTask[0];

    printf("worker:%d receive chunknum:%d nodeid:%d\n", workerID, recvTask[0], recvTask[1]);
//    sleep(1);
  }while(true);
  
  cout<<"the end"<<endl;
  
  cout<<"count total data per node = "<< count_total << endl;
  
  MPI_Barrier(MPI_COMM_WORLD);
    



//  /*creating and managing a set of reducer threads、execute the receiving reduce tasks */
  int recvTask_r;
  vector<string> key;
  vector<int> value;
  vector<string> final_key;
  vector<int> final_value;
  
  int reduce_request[2];
  reduce_request[0] = workerID;
  
  do{ 
  
  reduce_request[1] = recvTask_r;
  /*request new task*/
  MPI_Send(reduce_request,2,MPI_INT,schedulerID,0,MPI_COMM_WORLD); //initial request to scheduler
  
  MPI_Recv(&recvTask_r,1,MPI_INT, schedulerID,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  
  if(recvTask_r == -1) break;
  printf("worker:%d receive reducer:%d\n", workerID, recvTask_r);
  
  /*read reducer task file(intermediate file)*/
  
  string filename;
  string reducer_num_str = to_string(recvTask_r);
  string word;
  int count;

  filename = reducer_num_str + ".txt";//"./intermediate_file/finish_shuffle/" + 
  
  ifstream input_file(filename);
  while (input_file >> word >> count) {
  
      key.push_back(word);
      value.push_back(count);
//      cout << word <<" "<<count<<endl;
      
  }
  input_file.close();
  
  
  /*sort function*/
  sort(key.begin(), key.end(), sort_comp); 
//  for(int i = 0; i < 100; i++){
//  
//    cout<<key[i]<<endl;
//    
//  }

  /*group function*/
  int group = 1;
  int* group_mark = (int*)malloc(value.size() * sizeof(int));
  
  for(int i = 0; i < value.size(); i++) group_mark[i] = 0;
  
  for(int i = 0; i < value.size(); i++){
    
    if(group_mark[i] == 0){
    
      group_mark[i] = group;
      
      for(int j = i + 1; j < value.size(); j++){
      
        if(group_comp(key[i], key[j])){   //groupingComparator
        
          group_mark[j] = group;
        
        }
        
      }
      group++;
//      cout <<"group:"<< group<<endl;
    }

  }
  
//  for(int i = 0; i < 100; i++){
//  
//    cout<< "key:" << key[i] <<" group: "<< group_mark[i]<<endl;
//    
//  }

  
  /*reduce function*/
  int sum = 0;
  bool first = true;
  
  for(int i = 1; i < group; i++ ){
    
    for(int j = 0; j < value.size(); j++){
      
      if(group_mark[j] == i){
        
        if(first) final_key.push_back(key[j]);
        
        sum += value[j];
        first = false;
        
      }


    }
    
    final_value.push_back(sum);
    sum = 0;
    first = true;
  
  }
  

  
//  for(int i = 0; i < 10; i++){
//  
//    cout<< "final_key:" << final_key[i] <<" final_value: "<< final_value[i]<<endl;
//    
//  }  

  
  /*output function*/
  string output_filename = output_dir + "/" + job_name + "-" + reducer_num_str + ".out"; //"./" + output_dir + "/" 
  
  ofstream myfile(output_filename);

  for (int i = 0; i < final_key.size(); i++) {
  
      myfile << final_key[i] << " " << final_value[i] << "\n";
      count_final_total += final_value[i];
  }
  myfile.close();
  
  key.clear();
  value.clear();
  final_key.clear();
  final_value.clear();
  
  
//    sleep(1);
}while(true);

  cout<<"count final total:"<< count_final_total<<endl;
  
  
}


int main(int argc, char** argv) {

  int numnode, rank, rc; 
  rc = MPI_Init (&argc,&argv); 
  
  double starttime, endtime;
  starttime = MPI_Wtime();

  
  
  
  
  if (rc != MPI_SUCCESS) { 
  
  printf ("Error starting MPI program. Terminating.\n"); 
  MPI_Abort (MPI_COMM_WORLD, rc); 
  
  } 
  
  MPI_Comm_size (MPI_COMM_WORLD, &numnode); 
  MPI_Comm_rank (MPI_COMM_WORLD, &rank); 
  
  string job_name = string(argv[1]);
  string input_filename = string(argv[4]);
  string locality_config_filename = string(argv[6]);
  string output_dir = string(argv[7]);
  
  int num_reducer = atoi(argv[2]);
  int delay = atoi(argv[3]);
  int chunk_size = atoi(argv[5]);
  
  if (rank == numnode - 1) { // Scheduler
  
    scheduler(argv, numnode - 1);  //numnode - 1 (number of worker)
    
  } else { // worker
    cout << "rank" <<rank << endl;
    worker(argv, numnode - 1, rank);  //numnode - 1 (number of worker)、rank(nodeID)

  }
  endtime = MPI_Wtime();
  printf("That took %f seconds\n",endtime-starttime);

  MPI_Finalize (); 
  
  
  
  

}






//  cout<<"job_name: "<< job_name << endl;
//  cout<<"input_filename: "<< input_filename << endl;
//  cout<<"locality_config_filename: "<< locality_config_filename << endl;
//  cout<<"output_dir: "<< output_dir << endl;
//  printf("num_reducer: %d\n",num_reducer);
//  printf("delay: %d\n",delay);
//  printf("chunk_size: %d\n",chunk_size);


















