#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

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

void addArray(float * const array1, float * const array2, float * const array3, const int num){

    for(int i = 0; i < num; i++){
        array3[i] = array1[i] + array2[i];
    }

    return;
}

int main(int argc, char** argv){

    if(argc != 2){
        printf("Usage: %s <size of arrays>\n", argv[0]);
        return 1;
    }

    srand(time(NULL));

    const int N = atoi(argv[1]);

    float* array1 = (float*)malloc(N * sizeof(float));
    float* array2 = (float*)malloc(N * sizeof(float));
    float* array3 = (float*)malloc(N * sizeof(float));

    initializeArray(array1, N);
    initializeArray(array2, N);

    const double start_time = getTime();

    addArray(array1, array2, array3, N);

    const double elapsed_time = getTime() - start_time;

    printf("Elapsed time: %lf [ms]\n", elapsed_time * 1000.0);

    /* for(int i = 0; i < N; i++){ */
    /*     printf("%3.1f+%3.1f=%3.1f  ", array1[i], array2[i], array3[i]); */
    /* } */
    /* printf("\n"); */

    free(array1);
    free(array2);
    free(array3);

    return 0;
}
