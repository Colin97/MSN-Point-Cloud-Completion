#include <stdio.h>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void calc_penalty(int b, int n, int primitive_size, const float * xyz, int * idx, float * dist, float alpha, int * neighbor, float * cost, float * mean_mst_length) {
	const int batch = 512; // primitive_size should be less than 512
	__shared__ float xyz_buf[batch * 3];
	__shared__ bool vis[batch];
	__shared__ float cur_dis[batch];
	__shared__ int cur_idx[batch];
	__shared__ float min_dis[batch];
	__shared__ int min_idx[batch];
	__shared__ float sum_dis[batch];
	__shared__ int cnt[batch];
	__shared__ int degree[batch];

	for (int i = blockIdx.x; i < b; i += gridDim.x) {
		vis[threadIdx.x] = false;
		cur_dis[threadIdx.x] = 1e9;
		cnt[threadIdx.x] = 0;
		degree[threadIdx.x] = 0;
		
		for (int j = threadIdx.x; j < primitive_size * 3; j += blockDim.x) {
			xyz_buf[j] = xyz[(i * n + blockIdx.y * primitive_size) * 3 + j];
		}
		__syncthreads();

		__shared__ int last;
		__shared__ float x_last;
		__shared__ float y_last;
		__shared__ float z_last;
		
		if (threadIdx.x == 0) {
			vis[0] = true;
			sum_dis[0] = 0;
			last = 0;
			x_last = xyz_buf[0];
			y_last = xyz_buf[1];
			z_last = xyz_buf[2];
		}
		__syncthreads();

		for (int j = 0; j < primitive_size - 1; j++) {
			if (vis[threadIdx.x] == false) {
				float delta_x = xyz_buf[threadIdx.x * 3 + 0] - x_last;
				float delta_y = xyz_buf[threadIdx.x * 3 + 1] - y_last;
				float delta_z = xyz_buf[threadIdx.x * 3 + 2] - z_last;
				float d = sqrtf(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z);

				if (d < cur_dis[threadIdx.x]) {
					cur_dis[threadIdx.x] = d;
					cur_idx[threadIdx.x] = last;
				}
				min_dis[threadIdx.x] = cur_dis[threadIdx.x];
			}
			else {
				min_dis[threadIdx.x] = 1e9;
			}
			min_idx[threadIdx.x] = threadIdx.x;
			__syncthreads();
		
			int stride = 1;
			while(stride <= primitive_size / 2) {
				int index = (threadIdx.x + 1) * stride * 2 - 1; 
				if(index < primitive_size && min_dis[index - stride] < min_dis[index]) {
					min_dis[index] = min_dis[index - stride];
					min_idx[index] = min_idx[index - stride];
				}
				stride = stride * 2;
				__syncthreads(); 
			}
			__syncthreads();

			if (threadIdx.x == primitive_size - 1) {
				last = min_idx[threadIdx.x];
				int u = cur_idx[last];
				vis[last] = true;
				x_last = xyz_buf[last * 3 + 0];
				y_last = xyz_buf[last * 3 + 1];
				z_last = xyz_buf[last * 3 + 2];

				cnt[last] += 1;
				degree[last] += 1;
				neighbor[(i * n + blockIdx.y * primitive_size + last) * 512 + cnt[last]] = u;
				cost[(i * n + blockIdx.y * primitive_size + last) * 512 + cnt[last]] = cur_dis[last];
				cnt[u] += 1;
				degree[u] += 1;
				neighbor[(i * n + blockIdx.y * primitive_size + u) * 512 + cnt[u]] = last;
				cost[(i * n + blockIdx.y * primitive_size + u) * 512 + cnt[u]] = cur_dis[last];

				if (cnt[last] >= 512 || cnt[u] >= 512) {
					printf("MST Error: Too many neighbors! %d %d %d %d\n", cnt[last], cnt[u], last, u);
				}

				sum_dis[last] = cur_dis[last];
			}
			__syncthreads();
		}

		__syncthreads();
		int stride = 1;
		while(stride <= primitive_size / 2) {
			int index = (threadIdx.x + 1) * stride * 2 - 1; 
			if (index < primitive_size)
				sum_dis[index] += sum_dis[index - stride];
			stride = stride * 2;
			__syncthreads(); 
		}
		__syncthreads();

		__shared__ float mean_dis;
		if (threadIdx.x == 0) {
			mean_dis = sum_dis[primitive_size - 1] / (primitive_size - 1);
			atomicAdd(&mean_mst_length[i], mean_dis);
		}

		dist[i * n + blockIdx.y * primitive_size + threadIdx.x] = 0;
		idx[i * n + blockIdx.y * primitive_size + threadIdx.x] = -1;
		__syncthreads();

		while (true) {
			__shared__ int flag;
			flag = 0;
			int tmp = cnt[threadIdx.x];
			__syncthreads();
			if (tmp == 1) {
				atomicAdd(&flag, 1);
				for (int j = 1; j <= degree[threadIdx.x]; j++) {
					int u = neighbor[(i * n + blockIdx.y * primitive_size + threadIdx.x) * 512 + j];
					if (cnt[u] > 1 || (cnt[u] == 1 && threadIdx.x > u)) {
						float c = cost[(i * n + blockIdx.y * primitive_size + threadIdx.x) * 512 + j];
						atomicAdd(&cnt[threadIdx.x], -1);
						atomicAdd(&cnt[u], -1);
						if (c > mean_dis * alpha) {
							dist[i * n + blockIdx.y * primitive_size + threadIdx.x] = c;
							idx[i * n + blockIdx.y * primitive_size + threadIdx.x] = blockIdx.y * primitive_size + u;
						}
					}
				}
			}
			__syncthreads();
			if (flag == 0) break;
			__syncthreads();
		}
		__syncthreads();
	}
}

int expansion_penalty_cuda_forward(at::Tensor xyz, int primitive_size, at::Tensor idx, at::Tensor dist, double alpha, at::Tensor neighbor, at::Tensor cost, at::Tensor mean_mst_length) {

	const auto batch_size = xyz.size(0);
	const auto n = xyz.size(1); 
	
	calc_penalty<<<dim3(batch_size, n / primitive_size, 1), primitive_size>>>(batch_size, n, primitive_size, xyz.data<float>(), idx.data<int>(), dist.data<float>(),
																	 alpha, neighbor.data<int>(), cost.data<float>(), mean_mst_length.data<float>());

	cudaError_t err = cudaGetLastError();
	  if (err != cudaSuccess) {
	    printf("error in nnd Output: %s\n", cudaGetErrorString(err));
	    return 0;
	  }
	  return 1;
}

__global__ void calc_grad(int b, int n, const float * xyz, const float * grad_dist, const int * idx, float * grad_xyz) {
	for (int i = blockIdx.x; i < b; i += gridDim.x) {
		for (int j = threadIdx.x + blockIdx.y * blockDim.x; j < n; j += blockDim.x * gridDim.y) 
			if (idx[i * n + j] != -1) {
				float x1 = xyz[(i * n + j) * 3 + 0];
				float y1 = xyz[(i * n + j) * 3 + 1];
				float z1 = xyz[(i * n + j) * 3 + 2];
				int j2 = idx[i * n + j];
				float x2 = xyz[(i * n + j2) * 3 + 0];
				float y2 = xyz[(i * n + j2) * 3 + 1];
				float z2 = xyz[(i * n + j2) * 3 + 2];
				float g = grad_dist[i * n + j] * 2;
				atomicAdd(&(grad_xyz[(i * n + j) * 3 + 0]), g * (x1 - x2));
				atomicAdd(&(grad_xyz[(i * n + j) * 3 + 1]), g * (y1 - y2));
				atomicAdd(&(grad_xyz[(i * n + j) * 3 + 2]), g * (z1 - z2));
			}
	}
}

int expansion_penalty_cuda_backward(at::Tensor xyz, at::Tensor gradxyz, at::Tensor graddist, at::Tensor idx) {
	const auto batch_size = xyz.size(0);
	const auto n = xyz.size(1); 

	calc_grad<<<dim3(batch_size, 8, 1), 1024>>>(batch_size, n, xyz.data<float>(), graddist.data<float>(), idx.data<int>(), gradxyz.data<float>());
	
	cudaError_t err = cudaGetLastError();
	  if (err != cudaSuccess) {
	    printf("error in nnd get grad: %s\n", cudaGetErrorString(err));
	    return 0;
	  }
	  return 1;
}
