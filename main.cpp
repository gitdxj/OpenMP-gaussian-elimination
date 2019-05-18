#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "omp_lu.h"

using namespace std;
void Hello(void); /* Thread function */


int main()
{
    ofstream outFile;
    outFile.open("data.csv", ios::out);
    int start = 100;
    int num_iteration = 20;
    int *N = new int[num_iteration];
    for(int i = 0; i<num_iteration; i++){
        N[i] = start;
        start = start+100;
    }
    int repetition = 5;
    int numThreads = 2;
    long long time1=0, time2=0, time3=0, time4=0, time5=0, time6=0, time7=0, time8=0;
    long long mean1=0;
    long long mean2=0;
    long long mean3=0;
    long long mean4=0;
    long long mean5=0;
    long long mean6=0;
    long long mean7=0;
    long long mean8=0;
    for(int i = 0; i<num_iteration; i++)
    {
        for(int _ = 0; _<repetition; _++)
            {
                test(lu, N[i], numThreads, time1);  // 普通lu
                test(lu_col, N[i], numThreads, time2);  // 按列划分
                test(lu_col_omp, N[i], numThreads, time3);  // 按列划分+omp
                test(lu_omp, N[i], numThreads, time4);  // omp
                test(lu_omp_dynamic_thread, N[i], numThreads, time5);  // omp动态线程
                test(lu_omp_dynamic_schedule, N[i], numThreads, time6);  // omp动态划分
                test(lu_omp_guided, N[i], numThreads, time7);  // omp的guided动态划分
                test(lu_omp_sse, N[i], numThreads, time8);  // omp+sse

//                test_correctness(lu, N[i], numThreads);  // 普通lu
//                test_correctness(lu_col, N[i], numThreads);  // 按列划分
//                test_correctness(lu_col_omp, N[i], numThreads);  // 按列划分+omp
//                test_correctness(lu_omp, N[i], numThreads);  // omp
//                test_correctness(lu_omp_dynamic_thread, N[i], numThreads);  // omp动态线程
//                test_correctness(lu_omp_dynamic_schedule, N[i], numThreads);  // omp动态划分
//                test_correctness(lu_omp_guided, N[i], numThreads);  // omp的guided动态划分
//                test_correctness(lu_omp_sse, N[i], numThreads);  // omp+sse

                mean1 += time1/repetition;
                mean2 += time2/repetition;
                mean3 += time3/repetition;
                mean4 += time4/repetition;
                mean5 += time5/repetition;
                mean6 += time6/repetition;
                mean7 += time7/repetition;
                mean8 += time8/repetition;
            }
        cout << "N = " << N[i] <<endl;
        cout << "lu: "<<mean1<<endl
            << "col: " <<mean2<<endl
            << "col+omp: "<<mean3<<endl
            << "omp: "<<mean4<<endl
            << "d thread: " <<mean5<<endl
            << "d schedule: "<<mean6<<endl
            << "g schedule: "<<mean7<<endl
            << "omp+sse: " << mean8 <<endl;
        outFile<<N[i]<<','<<mean1<<','<<mean2<<','<<mean3<<','<<mean4<<','<<mean5
                <<','<<mean6<<','<<mean7<<','<<mean8<<endl;
    }
    outFile.close();


//    int N = 100;
//    int numThreads = 4;
//    test_correctness(lu_col_omp, N, numThreads);  // omp


    return 0;
}

// 用来测试omp的简单Hello函数
void Hello(void) {
  int my_rank = omp_get_thread_num();
  int thread_count = omp_get_num_threads();

  printf("Hello from thread %d of %d\n", my_rank, thread_count);

}  /* Hello */
