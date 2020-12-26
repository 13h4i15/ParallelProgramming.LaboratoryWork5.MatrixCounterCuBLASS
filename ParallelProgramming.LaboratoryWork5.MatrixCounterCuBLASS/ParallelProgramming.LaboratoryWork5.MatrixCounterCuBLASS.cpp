#include <cstdlib>
#include <curand.h>
#include <cublas_v2.h>
#include <time.h>
#include <cuda.h>
#include <stdio.h>

#define N 512

void GPU_fill_rand(float* M, int rows, int cols)
{
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());
	curandGenerateUniform(prng, M, rows * cols);
}

void multiply_matrix(const float* A, const float* B, float* C, const int m, const int k, const int n)
{
	const float alf = 1;
	const float bet = 0;
	const float* alpha = &alf;
	const float* beta = &bet;
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, m, B, k, beta, C, m);
	cublasDestroy(handle);
}

void print_matrix(const float* M, int rows, int cols) {
	printf("\n");
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			printf("%f ", M[j * rows + i]);
		}
		printf("\n");
	}
	printf("\n");
}

int main()
{
	int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;
	nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = N;

	float* h_C = (float*)malloc(nr_rows_C * nr_cols_C * sizeof(float));

	float* d_A, * d_B, * d_C;
	cudaMalloc(&d_A, nr_rows_A * nr_cols_A * sizeof(float));
	cudaMalloc(&d_B, nr_rows_B * nr_cols_B * sizeof(float));
	cudaMalloc(&d_C, nr_rows_C * nr_cols_C * sizeof(float));

	GPU_fill_rand(d_A, N, N);
	GPU_fill_rand(d_B, N, N);

	clock_t begin = clock();

	multiply_matrix(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);

	double elapsed_ses = double(clock() - begin) * 1000 / CLOCKS_PER_SEC;
	printf("Dimension: %d\nTime: %f ms", N, elapsed_ses);

	cudaMemcpy(h_C, d_C, nr_rows_C * nr_cols_C * sizeof(float), cudaMemcpyDeviceToHost);

	printf("C:\n");
	print_matrix(h_C, nr_rows_C, nr_cols_C);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	free(h_C);
	return 0;
}

