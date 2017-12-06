#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <stdio.h>
#include <vector>
#include <queue>

#define maxIter 3500
#define BLOCK_SIZE 32


// Learning rate policy
__device__ float step_fn(int t){
    float alpha = 0.012, beta = 0.01;
    return alpha/(1.0+beta*powf(t,1.5));
}

__global__ void SGD(float* dA, float* dW, float* dH, int* dI, int* dT, int m, int n, int k, float lambda){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i<m && j<n){
        if(dA[i*n+j]!=0 && dI[i]==j){
            dT[i*n+j]++;
            float step = step_fn(dT[i*n+j]), At = 0.0;

            // Compute inner product of <dWi, dHj>
            for(int l=0; l<k; l++){
                At += dW[i*k+l]*dH[l*n+j];
            }

            // SGD update for dWi
            for(int l=0; l<k; l++){
                dW[i*k+l] -= step*((At-dA[i*n+j])*dH[l*n+j] + lambda*dW[i*k+l]);
            }

            // Compute inner product of <dWi, dHj>
            At = 0.0;
            for(int l=0; l<k; l++){
                At += dW[i*k+l]*dH[l*n+j];
            }

            // SGD update for dHj
            for(int l=0; l<k; l++){
                dH[l*n+j] -= step*((At-dA[i*n+j])*dW[i*k+l] + lambda*dH[l*n+j]);
            }
        }
    }
}

__global__ void matrixMultiplication(float* dAt, float* dW, float* dH, int m, int n, int k){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i<m && j<n){
        float tmp = 0.0;
        for(int l=0; l<k; l++){
            tmp += dW[i*k+l] * dH[l*n+j];
        }
        dAt[i*n+j] = tmp;
    }
}

__global__ void computeColErrSum(float* dA, float* dAt, float* dErrSum, int* dNum, int m,int n){
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if(j<n){
        float errSum = 0.0;
        int num = 0;
        for(int i=0; i<m; i++){
            if(dA[i*n+j]>0){
                errSum += (dA[i*n+j]-dAt[i*n+j])*(dA[i*n+j]-dAt[i*n+j]);
                num++;
            }
        }
        dErrSum[j] = errSum;
        dNum[j] = num;
    }  
}

__global__ void computeRMSE(float* dErrSum, int* dNum, int n, int ite){
    float errSum = 0.0;
    int num = 0;
    for(int j=0; j<n; j++){
        errSum += dErrSum[j];
        num += dNum[j];
    }
    float RMSE = sqrt(errSum/num);
    //printf("Traning RMSE: %f\n", RMSE);
    if(ite%100==0) printf("Traning RMSE: %f\n", RMSE);
}

void readData(float* A, int m, int n){
    char file_name[50];
    sprintf(file_name, "data/data_%d",m);
    std::ifstream is(file_name);
    if(is.is_open()){
        float buf;
        for(int i=0; i<m*n; i++){
            is >> buf;
            A[i] = buf;
        }
        is.close();
    }
}

void initEmbedding(float* W, float* H, int m, int n, int k){
    float low = 0.0, high = 1.0/sqrt(k);

    for(int i=0; i<m; i++){
        for(int j=0; j<k; j++){
            W[i*k+j] = (high - low) * rand() / RAND_MAX + low;
        }
    }

    for(int i=0; i<k; i++){
        for(int j=0; j<n; j++){
            H[i*n+j] = (high - low) * rand() / RAND_MAX + low;
        }
    }

}

void initQueue(std::vector<std::queue<int> >& userQueue, int m, int n){
    for(int j=0; j<n; j++){
        int i = rand()%m;
        userQueue[i].push(j);
    }
}

void initTimes(int* T, int m, int n){
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            T[i*n+j] = 0;
        }
    }
}

void assignItemToUser(std::vector<std::queue<int> >& userQueue, int* I){
    for(int i=0; i<userQueue.size(); i++){
        if(userQueue[i].empty()) I[i] = -1;
        else{
            I[i] = userQueue[i].front();
            userQueue[i].pop();
        }
    }
}

void assignNextItem(std::vector<std::queue<int> >& userQueue, int* I){
    for(int i=0; i<userQueue.size(); i++){
        int ind = rand()%userQueue.size();
        userQueue[ind].push(I[i]);
    }
}

void printMatrix(float* A, int m, int n){
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            std::cout << A[i*n+j] << ' ';
        }
        std::cout << std::endl;
    }
}

int main(int argc, char** argv) {
    if(argc<3){
        std::cout << "Usage: ./mc_sgd m n" << std::endl;
        return 0;
    }

    //srand (time(NULL));

    int m = atoi(argv[1]), n = atoi(argv[2]), k = 100, ite = 0;
    float lambda = 0.05;

    float *A, *dA, *W, *dW, *H, *dH, *dAt, *dErrSum;
    int *I, *dI, *dNum, *T, *dT;

    std::vector<std::queue<int> > userQueue(m);

    dim3 numBlocks((m+BLOCK_SIZE-1)/BLOCK_SIZE, (n+BLOCK_SIZE-1)/BLOCK_SIZE);
    dim3 numThreads(BLOCK_SIZE, BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    A = (float*)malloc(sizeof(float)*m*n);
    W = (float*)malloc(sizeof(float)*m*k);
    H = (float*)malloc(sizeof(float)*k*n);
    I = (int*)malloc(sizeof(int)*m);
    T = (int*)malloc(sizeof(int)*m*n);

    cudaMalloc(&dA, sizeof(float)*m*n);
    cudaMalloc(&dW, sizeof(float)*m*k);
    cudaMalloc(&dH, sizeof(float)*k*n);
    cudaMalloc(&dI, sizeof(int)*m);
    cudaMalloc(&dAt, sizeof(float)*m*n);
    cudaMalloc(&dErrSum, sizeof(float)*n);
    cudaMalloc(&dNum, sizeof(int)*n);
    cudaMalloc(&dT, sizeof(int)*m*n);

    readData(A,m,n);
    initEmbedding(W, H, m, n, k);
    initQueue(userQueue,m,n);
    initTimes(T,m,n);

    cudaMemcpy(dA, A, sizeof(float)*m*n, cudaMemcpyHostToDevice);
    cudaMemcpy(dW, W, sizeof(float)*m*k, cudaMemcpyHostToDevice);
    cudaMemcpy(dH, H, sizeof(float)*k*n, cudaMemcpyHostToDevice);
    cudaMemcpy(dT, T, sizeof(int)*m*n, cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    while(ite<maxIter){
        assignItemToUser(userQueue,I);
        cudaMemcpy(dI, I, sizeof(int)*m, cudaMemcpyHostToDevice);
        
        
        SGD<<<numBlocks, numThreads>>>(dA,dW,dH,dI,dT,m,n,k,lambda);
        

        assignNextItem(userQueue,I);
        ite++;
        cudaDeviceSynchronize();

        /*
        // Compute RMSE
        matrixMultiplication<<<numBlocks, numThreads>>>(dAt,dW,dH,m,n,k);
        computeColErrSum<<<(n+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(dA,dAt,dErrSum,dNum,m,n);
        computeRMSE<<<1,1>>>(dErrSum,dNum,n,ite);
        cudaDeviceSynchronize();
        */
        
    }
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << milliseconds << "ms" << std::endl;

    return 0;
}
