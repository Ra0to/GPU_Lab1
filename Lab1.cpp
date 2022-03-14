#define _SILENCE_AMP_DEPRECATION_WARNINGS

#include <iostream>
#include <amp.h>
#include "timer.h"

using namespace std;
using namespace concurrency;


#pragma region arraySum

void INIT_arraySum(int size, int* a, int* b, int* result) {
	for (auto i = 0; i < size; i++) {
		a[i] = i;
		b[i] = i;
		result[i] = 0;
	}
}

void CPU_arraySum(int size, int* a, int* b, int* result) {
	for (auto i = 0; i < size; i++) {
		result[i] = a[i] + b[i];
	}
}

void PARALLEL_arraySum(int size, int* a, int* b, int* result) {
#pragma omp parallel for
	for (auto i = 0; i < size; i++) {
		result[i] = a[i] + b[i];
	}
}

void GPU_arraySum(int size, int* a, int* b, int* result) {
	array_view<int, 1> GPU_a(size, a);
	array_view<int, 1> GPU_b(size, b);
	array_view<int, 1> GPU_result(size, result);

	parallel_for_each(GPU_result.extent,
		[=](index<1> idx) restrict(amp) {
			GPU_result[idx] = GPU_a[idx] + GPU_b[idx];
		}
	);
}

#pragma endregion

#pragma region matrixNumberMultiply
void INIT_matrixNumberMultiply(int width, int height, int* matrix, int* result) {
	for (auto i = 0; i < width * height; i++) {
		matrix[i] = i;
		result[i] = 0;
	}
}

void CPU_matrixNumberMultiply(int width, int height, int multiplier, int* matrix, int* result) {
	for (auto i = 0; i < width * height; i++) {
		result[i] = matrix[i] * multiplier;
	}
}


void PARALLEL_matrixNumberMultiply(int width, int height, int multiplier, int* matrix, int* result) {
#pragma omp parallel for
	for (auto i = 0; i < width * height; i++) {
		result[i] = matrix[i] * multiplier;
	}
}

void GPU_matrixNumberMultiply(int width, int height, int multiplier, int* matrix, int* result) {
	array_view<int, 1> GPU_matrix(width * height, matrix);
	array_view<int, 1> GPU_result(width * height, result);
	
	parallel_for_each(GPU_result.extent, 
		[=](index<1> idx) restrict(amp) {
			GPU_result[idx] = GPU_matrix[idx] * multiplier;
		});
}
#pragma endregion

#pragma region matrixTranspose

void INIT_matrixTranspose(int width, int height, int* matrix, int* result) {
	for (auto i = 0; i < width * height; i++) {
		matrix[i] = i;
		result[i] = 0;
	}
}

void CPU_matrixTranspose(int width, int height, int* matrix, int* result) {
	for (auto i = 0; i < height; i++) {
		for (auto j = 0; j < width; j++) {
			result[j * height + i] = matrix[i * width + j];
		}
	}
}


void PARALLEL_matrixTranspose(int width, int height, int* matrix, int* result) {
#pragma omp parallel for
	for (auto i = 0; i < height; i++) {
		for (auto j = 0; j < width; j++) {
			result[j * height + i] = matrix[i * width + j];
		}
	}
}

void GPU_matrixTranspose(int width, int height, int* matrix, int* result) {
	array_view<int, 2> GPU_matrix(height, width, matrix);
	array_view<int, 2> GPU_result(width, height, result);

	parallel_for_each(GPU_matrix.extent,
		[=](index<2> idx) restrict(amp) {
			GPU_result[idx[1]][idx[0]] = GPU_matrix[idx[0]][idx[1]];
		});
}
#pragma endregion

#pragma region matrixMultiply

void INIT_matrixMultiply(int l, int m, int n, long* m1, long* m2, long* res) {
	for (auto i = 0; i < l * m; i++) {
		m1[i] = i;
	}

	for (auto i = 0; i < m * n; i++) {
		m2[i] = i;
	}

	for (auto i = 0; i < l * n; i++) {
		res[i] = 0;
	}
}

void CPU_matrixMultiply(int l, int m, int n, long* a, long* b, long* res) {
	for (auto i = 0; i < l; i++) {
		for (auto j = 0; j < n; j++) {
			long sum = 0;
			for (auto k = 0; k < m; k++) {
				sum += a[i * m + k] * b[k * n + j];
			}

			res[i * n + j] = sum;
		}
	}
}

void PARALLEL_matrixMultiply(int l, int m, int n, long* a, long* b, long* res) {
#pragma omp parallel for
	for (auto i = 0; i < l; i++) {
		for (auto j = 0; j < n; j++) {
			long sum = 0;
			for (auto k = 0; k < m; k++) {
				sum += a[i * m + k] * b[k * n + j];
			}

			res[i * n + j] = sum;
		}
	}
}

void GPU_matrixMultiply(int l, long m, long n, long* a, long* b, long* res) {
	array_view<long, 2> GPU_a(l, m, a);
	array_view<long, 2> GPU_b(m, n, b);
	array_view<long, 2> GPU_result(l, n, res);

	parallel_for_each(GPU_result.extent,
		[=](index<2> idx) restrict(amp) {

			auto i = idx[0];
			auto j = idx[1];
			long sum = 0;
			for (auto k = 0; k < m; k++) {
				sum += GPU_a[i][k] * GPU_b[k][j];
			}

			GPU_result[i][j] = sum;
		});
}

#pragma endregion

void printMatrix(int width, int height, int* matrix) {
	wcout << width << "x" << height << endl;

	for (auto i = 0; i < height; i++) {
		for (auto j = 0; j < width; j++) {
			wcout << matrix[i * width + j] << " ";
		}

		wcout << endl;
	}
}

void printAcceleratorsInfo() {
	auto accelerators = accelerator::get_all();

	for (auto& accel : accelerators)
	{
		wcout << accel.get_description() << endl;
		wcout << "	Path: " << accel.get_device_path() << endl;
		wcout << "	Memory: " << accel.get_dedicated_memory() << endl;
		wcout << "	Is Debug: " << accel.get_is_debug() << endl;
		wcout << "	Is Emulated: " << accel.get_is_emulated() << endl;
		wcout << "	Shared memory: " << accel.get_supports_cpu_shared_memory() << endl;
		wcout << "	Supports limited double precision: " << accel.get_supports_limited_double_precision() << endl;
		wcout << "	Supports double precision: " << accel.get_supports_double_precision() << endl;
		wcout << "	Has display: " << accel.get_has_display() << endl;

		wcout << "--------------------------------------------------" << endl << endl;
	}

	wcout << endl;
}

void testArraySum() {
	auto sw = new Timer();

	constexpr size_t size = 100000000;
	int* a = new int[size];
	int* b = new int[size];
	int* c = new int[size];

	wcout << "Test Array Sum" << endl << "Volume: " << size << endl << endl;

	INIT_arraySum(size, a, b, c);
	wcout << "CPU" << endl;

	sw->Start();
	CPU_arraySum(size, a, b, c);
	sw->Stop();

	wcout << "Time: " << sw->Elapsed() << endl;
	wcout << "Result:" << endl;

	wcout << "result[0]: " << c[0] << "; result[5]: " << c[5] << "; result[10]: " << c[10] << endl;

	wcout << endl << endl;


	INIT_arraySum(size, a, b, c);

	wcout << "Parallel CPU" << endl;

	sw->Start();
	PARALLEL_arraySum(size, a, b, c);
	sw->Stop();

	wcout << "Time: " << sw->Elapsed() << endl;
	wcout << "Result:" << endl;

	wcout << "result[0]: " << c[0] << "; result[5]: " << c[5] << "; result[10]: " << c[10] << endl;

	wcout << endl << endl;


	INIT_arraySum(size, a, b, c);

	wcout << "GPU" << endl;

	sw->Start();
	GPU_arraySum(size, a, b, c);
	sw->Stop();

	wcout << "Time: " << sw->Elapsed() << endl;
	wcout << "Result:" << endl;

	wcout << "result[0]: " << c[0] << "; result[5]: " << c[5] << "; result[10]: " << c[10] << endl;

	wcout << "--------------------------------------------------" << endl << endl;

	delete[] a;
	delete[] b;
	delete[] c;

	delete sw;
}

void testMatrixNumberMultiply() {
	auto sw = new Timer();

	const size_t width  = 5000;
	const size_t height = 5000;
	int multiplier = 729;

	int* matrix = new int[width * height];
	int* result = new int[width * height];
	
	wcout << "Test Matrix Number Multiply" << endl << "Width: " << width << "; Height: " << height << ";" << endl << "Multiplier: " << multiplier << ";" << endl << endl;

	INIT_matrixNumberMultiply(width, height, matrix, result);
	wcout << "CPU" << endl;

	sw->Start();
	CPU_matrixNumberMultiply(width, height, multiplier, matrix, result);
	sw->Stop();

	wcout << "Time: " << sw->Elapsed() << endl;
	wcout << "Result:" << endl;
	wcout << "Result[0][0]: " << result[0] << "; Result[1][0]: " << result[width] << "; Result[2][0]: " << result[2 * width] << ";" << endl;

	wcout << endl << endl;


	INIT_matrixNumberMultiply(width, height, matrix, result);
	wcout << "PARALLEL" << endl;

	sw->Start();
	PARALLEL_matrixNumberMultiply(width, height, multiplier, matrix, result);
	sw->Stop();

	wcout << "Time: " << sw->Elapsed() << endl;
	wcout << "Result:" << endl;
	wcout << "Result[0][0]: " << result[0] << "; Result[1][0]: " << result[width] << "; Result[2][0]: " << result[2 * width] << ";" << endl;

	wcout << endl << endl;


	INIT_matrixNumberMultiply(width, height, matrix, result);
	wcout << "GPU" << endl;

	sw->Start();
	GPU_matrixNumberMultiply(width, height, multiplier, matrix, result);
	sw->Stop();

	wcout << "Time: " << sw->Elapsed() << endl;
	wcout << "Result:" << endl;
	wcout << "Result[0][0]: " << result[0] << "; Result[1][0]: " << result[width] << "; Result[2][0]: " << result[2 * width] << ";" << endl;

	wcout << "--------------------------------------------------" << endl << endl;

	delete[] matrix;
	delete[] result;

	delete sw;
}

void testMatrixTranspose() {
	auto sw = new Timer();

	const size_t width = 5000;
	const size_t height = 5000;

	int* matrix = new int[width * height];
	int* result = new int[width * height];

	wcout << "Test Matrix Transpose" << endl << "Width: " << width << "; Height: " << height << ";" << endl << endl;

	INIT_matrixTranspose(width, height, matrix, result);
	wcout << "CPU" << endl;

	sw->Start();
	CPU_matrixTranspose(width, height, matrix, result);
	sw->Stop();

	wcout << "Time: " << sw->Elapsed() << endl;
	wcout << "Result:" << endl;
	wcout << "Result[0][0]: " << result[0] << "; Result[1][0]: " << result[height] << "; Result[2][0]: " << result[2 * height] << ";" << endl;

	wcout << endl << endl;

	INIT_matrixTranspose(width, height, matrix, result);
	wcout << "PARALLEL" << endl;

	sw->Start();
	PARALLEL_matrixTranspose(width, height, matrix, result);
	sw->Stop();

	wcout << "Time: " << sw->Elapsed() << endl;
	wcout << "Result:" << endl;
	wcout << "Result[0][0]: " << result[0] << "; Result[1][0]: " << result[height] << "; Result[2][0]: " << result[2 * height] << ";" << endl;

	wcout << endl << endl;

	INIT_matrixTranspose(width, height, matrix, result);
	wcout << "GPU" << endl;

	sw->Start();
	GPU_matrixTranspose(width, height, matrix, result);
	sw->Stop();

	wcout << "Time: " << sw->Elapsed() << endl;
	wcout << "Result:" << endl;
	wcout << "Result[0][0]: " << result[0] << "; Result[1][0]: " << result[height] << "; Result[2][0]: " << result[2 * height] << ";" << endl;

	wcout << endl << endl;

	wcout << "--------------------------------------------------" << endl << endl;

	delete[] matrix;
	delete[] result;

	delete sw;
}

void testMatrixMultiply() {
	auto sw = new Timer();

	const size_t size = 500;
	const size_t l = size;
	const size_t m = size;
	const size_t n = size;

	long* matrix1 = new long[l * m];
	long* matrix2 = new long[m * n];
	long* result  = new long[l * n];

	wcout << 
		"Test Matrix Multiply" << endl << 
		"First matrix size: "  << l << "x" << m << endl <<
		"Second matrix size: " << m << "x" << n << endl << endl;

	INIT_matrixMultiply(l, m, n, matrix1, matrix2, result);
	wcout << "CPU" << endl;

	sw->Start();
	CPU_matrixMultiply(l, m, n, matrix1, matrix2, result);
	sw->Stop();

	wcout << "Time: " << sw->Elapsed() << endl;
	wcout << "Result:" << endl;
	wcout << "Result[0][0]: " << result[0] << "; Result[1][0]: " << result[l] << "; Result[2][0]: " << result[2 * l] << ";" << endl;

	wcout << endl << endl;


	INIT_matrixMultiply(l, m, n, matrix1, matrix2, result);
	wcout << "PARALLEL" << endl;

	sw->Start();
	PARALLEL_matrixMultiply(l, m, n, matrix1, matrix2, result);
	sw->Stop();

	wcout << "Time: " << sw->Elapsed() << endl;
	wcout << "Result:" << endl;
	wcout << "Result[0][0]: " << result[0] << "; Result[1][0]: " << result[l] << "; Result[2][0]: " << result[2 * l] << ";" << endl;

	wcout << endl << endl;

	INIT_matrixMultiply(l, m, n, matrix1, matrix2, result);
	wcout << "GPU" << endl;

	sw->Start();
	GPU_matrixMultiply(l, m, n, matrix1, matrix2, result);
	sw->Stop();

	wcout << "Time: " << sw->Elapsed() << endl;
	wcout << "Result:" << endl;
	wcout << "Result[0][0]: " << result[0] << "; Result[1][0]: " << result[l] << "; Result[2][0]: " << result[2 * l] << ";" << endl;

	wcout << endl << endl;

	wcout << "--------------------------------------------------" << endl << endl;

	delete[] matrix1;
	delete[] matrix2;
	delete[] result;

	delete sw;
}

int main()
{
	//printAcceleratorsInfo();
	//testArraySum();
	//testMatrixNumberMultiply();
	//testMatrixTranspose();
	testMatrixMultiply();
}