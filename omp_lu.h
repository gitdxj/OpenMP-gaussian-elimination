#ifndef OMP_LU_H_INCLUDED
#define OMP_LU_H_INCLUDED
/**为了防止编译器不支持OpenMP，如果编译器不支持OpenMP，就把_OPENMP的宏定义删去**/

#include <omp.h>
#include <windows.h>
#include "matrix.h"

void lu(float** Matrix, int N, int numThreads);
void lu_col(float** Matrix, int N, int numThreads);
void lu_col_omp(float** Matrix, int N, int numThreads);
void lu_omp_dynamic_thread(float** Matrix, int N, int numThreads);  // 动态线程
void lu_omp(float **Matrix, int N, int numThreads);  // 普通的omp
void lu_omp_45p(float** Matrix, int N, int numThreads);
void lu_omp_dynamic_schedule(float **Matrix, int N, int numThreads);  // 动态调度
void lu_omp_guided(float** Matrix, int N, int numThreads);  // guided划分
void lu_omp_sse(float** Matrix, int N, int numThreads);  // 结合sse的
void test(void (*algo)(float**, int, int), int N, int numThreads, long long &time_interval);  // 用来度量各个算法计算时间的程序
void test_correctness(void (*algo)(float**, int, int), int N, int numThreads);  // 用来测试一个算法得到的结果是否是正确的

// 下面这一开始写着用的，可以不用看
void test_lu_omp(int N, int numThreads, long long &time_interval);
void test_lu(int N, int numThreads, long long &time_interval);
#endif // OMP_LU_H_INCLUDED
