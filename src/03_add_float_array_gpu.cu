#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define CHECK(call)                                                     \
    {                                                                   \
        const cudaError_t error = call;                                 \
        if(error != cudaSuccess)                                        \
        {                                                               \
            printf("Error: %s:%d    ", __FILE__, __LINE__);             \
            printf("code: %d, reason: %s\n", error,                     \
                   cudaGetErrorString(error));                          \
            exit(1);                                                    \
        }                                                               \
    }

double getTime(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1e-6;
}

void initializeArray(float * const array, const int num){

    for(int i = 0; i < num; i++){
        array[i] = (float)(rand() % 256) / 10.0f;
    }

    return;
}

// カーネル関数の返り値はvoidでなければならない。
__global__ void addArray(float * const array1, float * const array2,
                         float * const array3, const int num){

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(idx < num){
        array3[idx] = array1[idx] + array2[idx];
    }
    
    return;
}

int main(int argc, char** argv){

    if(argc != 3){
        printf("Usage: %s <size of arrays> <block size>\n", argv[0]);
        return 1;
    }

    srand(time(NULL));

    const int N = atoi(argv[1]);
    const int block_size = atoi(argv[2]);

    float* array1 = (float*)malloc(N * sizeof(float));
    float* array2 = (float*)malloc(N * sizeof(float));
    float* array3 = (float*)malloc(N * sizeof(float));

    initializeArray(array1, N);
    initializeArray(array2, N);

    float* d_array1 = nullptr;
    float* d_array2 = nullptr;
    float* d_array3 = nullptr;
    
    CHECK(cudaMalloc(&d_array1, N * sizeof(float)));
    CHECK(cudaMalloc(&d_array2, N * sizeof(float)));
    CHECK(cudaMalloc(&d_array3, N * sizeof(float)));

    CHECK(cudaMemcpy(d_array1, array1, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_array2, array2, N * sizeof(float), cudaMemcpyHostToDevice));

    const dim3 block(block_size);
    const dim3 grid((N + block.x - 1) / block.x);

    const double start_time = getTime();

    // カーネル呼び出し（非同期実行）
    addArray<<<grid, block>>>(d_array1, d_array2, d_array3, N);

    // カーネルの実行終了を待機
    // （後段のcudaMemcpyが暗黙的に同期するのでこの行は無くても正しく動く）
    CHECK(cudaDeviceSynchronize());

    const double elapsed_time = getTime() - start_time;

    printf("Elapsed time: %lf [ms]\n", elapsed_time * 1000.0);

    CHECK(cudaMemcpy(array3, d_array3, N * sizeof(float), cudaMemcpyDeviceToHost));

    // for(int i = 0; i < N; i++){
    //     printf("%3.1f+%3.1f=%3.1f  ", array1[i], array2[i], array3[i]);
    // }
    // printf("\n");

    CHECK(cudaFree(d_array1));
    CHECK(cudaFree(d_array2));
    CHECK(cudaFree(d_array3));
    free(array1);
    free(array2);
    free(array3);

    CHECK(cudaDeviceReset());

    return 0;
}
