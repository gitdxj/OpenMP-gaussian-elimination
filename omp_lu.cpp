#include "omp_lu.h"

using namespace std;

int thread_count = 4;

void lu(float** Matrix, int N, int numThreads)
{
    for(int k = 0; k<N; k++)
    {
        if(0 == Matrix[k][k])  // 如果A(k,k)的位置为0的话，就从后面找一行不为0的互换
        {
            bool swapped = swap_rows(Matrix, N, k);

            if(!swapped)   // 如果下面任何一行的第k列都没有不是0打头的就直接跳下一个k
                continue;
        }
        for(int j = k+1; j<N; j++)
            Matrix[k][j] = Matrix[k][j] / Matrix[k][k];
        Matrix[k][k] = 1.0;
        for(int i = k+1; i<N; i++){
            for(int j = k+1; j<N; j++)
                Matrix[i][j] = Matrix[i][j] - Matrix[i][k] * Matrix[k][j];
            Matrix[i][k] = 0;
        }
    }
}

void lu_col(float** Matrix, int N, int numThreads)
{
    for(int k = 0; k<N; k++)
    {
        if(0 == Matrix[k][k])  // 如果A(k,k)的位置为0的话，就从后面找一行不为0的互换
        {
            bool swapped = swap_rows(Matrix, N, k);

            if(!swapped)   // 如果下面任何一行的第k列都没有不是0打头的就直接跳下一个k
                continue;
        }
        for(int j = k+1; j<N; j++)
            Matrix[k][j] = Matrix[k][j] / Matrix[k][k];
        Matrix[k][k] = 1.0;
        for(int j = k+1; j<N; j++){
            for(int i = k+1; i<N; i++)
                Matrix[i][j] = Matrix[i][j] - Matrix[i][k] * Matrix[k][j];
        }
        for(int i = k+1; i<N; i++)
            Matrix[i][k] = 0;
    }
}

void lu_col_omp(float** Matrix, int N, int numThreads)
{
    #pragma omp parallel num_threads(numThreads)
    for(int k = 0; k<N; k++)
    {
        if(0 == Matrix[k][k])  // 如果A(k,k)的位置为0的话，就从后面找一行不为0的互换
        {
            bool swapped = swap_rows(Matrix, N, k);

            if(!swapped)   // 如果下面任何一行的第k列都没有不是0打头的就直接跳下一个k
                continue;
        }
        #pragma omp critical
        {for(int j = k+1; j<N; j++)
            Matrix[k][j] = Matrix[k][j] / Matrix[k][k];
        Matrix[k][k] = 1.0;}
        #pragma omp for
        for(int j = k+1; j<N; j++){
            for(int i = k+1; i<N; i++)
                Matrix[i][j] = Matrix[i][j] - Matrix[i][k] * Matrix[k][j];
        }
        for(int i=k+1; i<N; i++)
            Matrix[i][k] = 0.0;
    }
}

void lu_omp(float** Matrix, int N, int numThreads)
{
    #pragma omp parallel num_threads(numThreads)
    for(int k = 0; k<N; k++)
    {
        if(0 == Matrix[k][k])  // 如果A(k,k)的位置为0的话，就从后面找一行不为0的互换
        {
            bool swapped = swap_rows(Matrix, N, k);

            if(!swapped)   // 如果下面任何一行的第k列都没有不是0打头的就直接跳下一个k
                continue;
        }
        #pragma omp critical
        {for(int j = k+1; j<N; j++)
            Matrix[k][j] = Matrix[k][j] / Matrix[k][k];
        Matrix[k][k] = 1.0;}
        #pragma omp for
        for(int i = k+1; i<N; i++){
            for(int j = k+1; j<N; j++)
                Matrix[i][j] = Matrix[i][j] - Matrix[i][k] * Matrix[k][j];
            Matrix[i][k] = 0;
        }
    }
}

void lu_omp_dynamic_thread(float** Matrix, int N, int numThreads)
{

    for(int k = 0; k<N; k++)
    {
        if(0 == Matrix[k][k])  // 如果A(k,k)的位置为0的话，就从后面找一行不为0的互换
        {
            bool swapped = swap_rows(Matrix, N, k);

            if(!swapped)   // 如果下面任何一行的第k列都没有不是0打头的就直接跳下一个k
                continue;
        }
        for(int j = k+1; j<N; j++)
            Matrix[k][j] = Matrix[k][j] / Matrix[k][k];
        Matrix[k][k] = 1.0;
        #pragma omp parallel for num_threads(numThreads)
        for(int i = k+1; i<N; i++){
            for(int j = k+1; j<N; j++)
                Matrix[i][j] = Matrix[i][j] - Matrix[i][k] * Matrix[k][j];
            Matrix[i][k] = 0;
        }
    }

}


void lu_omp_45p(float** Matrix, int N, int numThreads)
{
    #pragma omp parallel num_threads(numThreads)
    for(int k = 0; k<N; k++)
    {
        if(0 == Matrix[k][k])  // 如果A(k,k)的位置为0的话，就从后面找一行不为0的互换
        {
            bool swapped = swap_rows(Matrix, N, k);

            if(!swapped)   // 如果下面任何一行的第k列都没有不是0打头的就直接跳下一个k
                continue;
        }
        #pragma omp for
        for(int j = k+1; j<N; j++)
            Matrix[k][j] = Matrix[k][j] / Matrix[k][k];
        Matrix[k][k] = 1.0;
        #pragma omp for
        for(int i = k+1; i<N; i++){
            for(int j = k+1; j<N; j++)
                Matrix[i][j] = Matrix[i][j] - Matrix[i][k] * Matrix[k][j];
            Matrix[i][k] = 0;
        }
    }
}
void lu_omp_dynamic_schedule(float** Matrix, int N, int numThreads)
{
    #pragma omp parallel num_threads(numThreads)
    for(int k = 0; k<N; k++)
    {
        if(0 == Matrix[k][k])  // 如果A(k,k)的位置为0的话，就从后面找一行不为0的互换
        {
            bool swapped = swap_rows(Matrix, N, k);

            if(!swapped)   // 如果下面任何一行的第k列都没有不是0打头的就直接跳下一个k
                continue;
        }
        #pragma omp critical
        {for(int j = k+1; j<N; j++)
            Matrix[k][j] = Matrix[k][j] / Matrix[k][k];
        Matrix[k][k] = 1.0;}
        #pragma omp for schedule(dynamic, 20)
        for(int i = k+1; i<N; i++){
            for(int j = k+1; j<N; j++)
                Matrix[i][j] = Matrix[i][j] - Matrix[i][k] * Matrix[k][j];
            Matrix[i][k] = 0;
        }
    }
}

void lu_omp_guided(float** Matrix, int N, int numThreads)
{
    #pragma omp parallel num_threads(numThreads)
    for(int k = 0; k<N; k++)
    {
        if(0 == Matrix[k][k])  // 如果A(k,k)的位置为0的话，就从后面找一行不为0的互换
        {
            bool swapped = swap_rows(Matrix, N, k);

            if(!swapped)   // 如果下面任何一行的第k列都没有不是0打头的就直接跳下一个k
                continue;
        }
        #pragma omp critical
        {for(int j = k+1; j<N; j++)
            Matrix[k][j] = Matrix[k][j] / Matrix[k][k];
        Matrix[k][k] = 1.0;}
        #pragma omp for schedule(guided, 32)
        for(int i = k+1; i<N; i++){
            for(int j = k+1; j<N; j++)
                Matrix[i][j] = Matrix[i][j] - Matrix[i][k] * Matrix[k][j];
            Matrix[i][k] = 0;
        }
    }
}

void lu_omp_sse(float** Matrix, int N, int numThreads)
{
    #pragma omp parallel num_threads(numThreads)
    for(int k = 0; k<N; k++)
    {
        // 开始是解决Matrix(k,k)为0的问题
        if(0 == Matrix[k][k])  // 如果A(k,k)的位置为0的话，就从后面找一行不为0的互换
        {
            bool swapped = swap_rows(Matrix, N, k);

            if(!swapped)   // 如果下面任何一行的第k列都没有不是0打头的就直接跳下一个k
                continue;
        }

        #pragma omp critical
        {__m128 A_k_k = _mm_set_ps1(Matrix[k][k]);
//        __m128 A_k_k = _mm_load1_ps(Matrix[k]+k);
        for(int j = N-4; j>k; j-=4)
        {
            __m128 A_k_j = _mm_loadu_ps(Matrix[k]+j);
            A_k_j = _mm_div_ps(A_k_j, A_k_k);
            _mm_storeu_ps(Matrix[k]+j, A_k_j);
        }
        for(int j = k+1; j<k+1+(N-k-1)%4; j++)  // 不能被4整除的部分
            Matrix[k][j] = Matrix[k][j] / Matrix[k][k];
        Matrix[k][k] = 1.0;}

        #pragma omp for
        for(int i = k+1; i<N; i++)
        {
            __m128 A_i_k = _mm_set_ps1(Matrix[i][k]);
//             __m128 A_i_k = _mm_load1_ps(Matrix[i]+k);
            for(int j = N-4; j>k; j-=4)
                {
                    __m128 A_k_j = _mm_loadu_ps(Matrix[k]+j);
                    __m128 t = _mm_mul_ps(A_k_j, A_i_k);
                    __m128 A_i_j = _mm_loadu_ps(Matrix[i]+j);
                    A_i_j = _mm_sub_ps(A_i_j, t);
                    _mm_storeu_ps(Matrix[i]+j, A_i_j);
                }
            for(int j = k+1; j<k+1+(N-k-1)%4; j++)  // 不能被4整除的部分
                Matrix[i][j] = Matrix[i][j] - Matrix[i][k] * Matrix[k][j];
            Matrix[i][k] = 0.0;
        }
//        if(k == 2)
//            show_matrix(Matrix, N);
    }
}


void test(void (*algo)(float**, int, int), int N, int numThreads, long long &time_interval)
{
    long long head, tail, freq;  // 用于高精度计时
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);

    // 创建矩阵并初始化随机值
    float **Matrix = new float*[N];
    for(int i=0; i<N; i++)
        Matrix[i] = new float[N];
    matrix_initialize(Matrix, N);

    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    algo(Matrix, N, numThreads);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    time_interval = (tail - head) * 1000.0 / freq ;

    if(N<10)
        show_matrix(Matrix, N);

	// 回收内存
	for(int i=0; i<N; i++)
        delete []Matrix[i];
    delete []Matrix;
}

void test_correctness(void (*algo)(float**, int, int), int N, int numThreads)
{

    // 创建矩阵并初始化随机值
    float **A = new float*[N];
    for(int i=0; i<N; i++)
        A[i] = new float[N];
    matrix_initialize(A, N);

    // 初始化矩阵B使得B和A的值完全一样
    float **B = new float*[N];
    for(int i=0; i<N; i++)
        B[i] = new float[N];
    copy_matrix(B, A, N);

//    show_matrix(A,N);
//    show_matrix(B,N);

    lu(A, N, numThreads);
    algo(B, N, numThreads);

    if(ls_same(A, B, N))
        cout<<"correct!"<<endl;
    else
        {
            cout<<"incorrect!"<<endl;
        }
//    show_matrix(A, N);
//    show_matrix(B, N);

//    for(int i=0; i<N; i++)
//        for(int j=0; j<N; j++)
//            if(A[i][j] != B[i][j]){
//                cout << i <<"," <<j <<" 处不同"<<endl
//                    << "A: " << A[i][j] << endl
//                    << "B: " << B[i][j] << endl;
//            }


}
