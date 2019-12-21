#include <stdio.h>
#include <stdlib.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#define TOTAL_THREADS 1024

inline int opt_n_threads(int work_size) {
  const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);

  return max(min(1 << pow_2, TOTAL_THREADS), 1);
}


#define CUDA_CHECK_ERRORS()                                           \
  do {                                                                \
    cudaError_t err = cudaGetLastError();                             \
    if (cudaSuccess != err) {                                         \
      fprintf(stderr, "CUDA kernel failed : %s\n%s at L:%d in %s\n",  \
              cudaGetErrorString(err), __PRETTY_FUNCTION__, __LINE__, \
              __FILE__);                                              \
      exit(-1);                                                       \
    }                                                                 \
  } while (0)


// input: points(b, c, n) idx(b, m)
// output: out(b, c, m)
__global__ void gather_points_kernel(int b, int c, int n, int m,
                                     const float *__restrict__ points,
                                     const int *__restrict__ idx,
                                     float *__restrict__ out) {
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int l = blockIdx.y; l < c; l += gridDim.y) {
      for (int j = threadIdx.x; j < m; j += blockDim.x) {
        int a = idx[i * m + j];
        out[(i * c + l) * m + j] = points[(i * c + l) * n + a];
      }
    }
  }
}

void gather_points_kernel_wrapper(int b, int c, int n, int npoints,
                                  const float *points, const int *idx,
                                  float *out) {
  gather_points_kernel<<<dim3(b, c, 1), opt_n_threads(npoints), 0,
                         at::cuda::getCurrentCUDAStream()>>>(b, c, n, npoints,
                                                             points, idx, out);

  CUDA_CHECK_ERRORS();
}

// input: grad_out(b, c, m) idx(b, m)
// output: grad_points(b, c, n)
__global__ void gather_points_grad_kernel(int b, int c, int n, int m,
                                          const float *__restrict__ grad_out,
                                          const int *__restrict__ idx,
                                          float *__restrict__ grad_points) {
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int l = blockIdx.y; l < c; l += gridDim.y) {
      for (int j = threadIdx.x; j < m; j += blockDim.x) {
        int a = idx[i * m + j];
        //atomicAdd(grad_points + (i * c + l) * n + a,
        //          grad_out[(i * c + l) * m + j]); 
        grad_points[(i * c + l) * n + a] += grad_out[(i * c + l) * m + j];
      }
    }
  }
}

void gather_points_grad_kernel_wrapper(int b, int c, int n, int npoints,
                                       const float *grad_out, const int *idx,
                                       float *grad_points) {
  gather_points_grad_kernel<<<dim3(b, c, 1), opt_n_threads(npoints), 0,
                              at::cuda::getCurrentCUDAStream()>>>(
      b, c, n, npoints, grad_out, idx, grad_points);

  CUDA_CHECK_ERRORS();
}

__device__ void __update(float *__restrict__ cnt, int *__restrict__ cnt_i,
                         int idx1, int idx2) {
  const float v1 = cnt[idx1], v2 = cnt[idx2];
  const int i1 = cnt_i[idx1], i2 = cnt_i[idx2];
  cnt[idx1] = min(v1, v2);
  cnt_i[idx1] = v2 < v1 ? i2 : i1;
}

// Input dataset: (b, n, 3), tmp: (b, n)
// Ouput idxs (b, m)
template <unsigned int block_size>
__global__ void minimum_density_sampling_kernel (
    int b, int n, int m, const float *__restrict__ dataset,
    float *__restrict__ temp, int *__restrict__ idxs, float *__restrict__ mean_mst_length) {
  if (m <= 0) return;
  __shared__ float cnt[block_size];
  __shared__ int cnt_i[block_size];

  int batch_index = blockIdx.x;
  dataset += batch_index * n * 3;
  temp += batch_index * n;
  idxs += batch_index * m;

  int tid = threadIdx.x;
  const int stride = block_size;

  int old = 0;
  if (threadIdx.x == 0) {
    idxs[0] = old;
    temp[old] = 1e9;
  }

  __syncthreads();
  float t = 5.0 * mean_mst_length[batch_index] * mean_mst_length[batch_index]; 

  float x1 = dataset[old * 3 + 0];
  float y1 = dataset[old * 3 + 1];
  float z1 = dataset[old * 3 + 2];
  for (int j = 1; j < m; j++) {
    int besti = 0;
    float best = 1e9;
    for (int k = tid; k < n; k += stride) {
      float x2, y2, z2;
      x2 = dataset[k * 3 + 0];
      y2 = dataset[k * 3 + 1];
      z2 = dataset[k * 3 + 2];

      float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);

      temp[k] += k < 8192 ? exp(-d / t) : exp(-d / t) * 2.0;
      besti = temp[k] < best ? k : besti;
      best = temp[k] < best ? temp[k] : best;
    }
    cnt[tid] = best;
    cnt_i[tid] = besti;
    __syncthreads();


    if (block_size >= 1024) {
      if (tid < 512) {
        __update(cnt, cnt_i, tid, tid + 512);
      }
      __syncthreads();
    }
    if (block_size >= 512) {
      if (tid < 256) {
        __update(cnt, cnt_i, tid, tid + 256);
      }
      __syncthreads();
    }
    if (block_size >= 256) {
      if (tid < 128) {
        __update(cnt, cnt_i, tid, tid + 128);
      }
      __syncthreads();
    }
    if (block_size >= 128) {
      if (tid < 64) {
        __update(cnt, cnt_i, tid, tid + 64);
      }
      __syncthreads();
    }
    if (block_size >= 64) {
      if (tid < 32) {
        __update(cnt, cnt_i, tid, tid + 32);
      }
      __syncthreads();
    }
    if (block_size >= 32) {
      if (tid < 16) {
        __update(cnt, cnt_i, tid, tid + 16);
      }
      __syncthreads();
    }
    if (block_size >= 16) {
      if (tid < 8) {
        __update(cnt, cnt_i, tid, tid + 8);
      }
      __syncthreads();
    }
    if (block_size >= 8) {
      if (tid < 4) {
        __update(cnt, cnt_i, tid, tid + 4);
      }
      __syncthreads();
    }
    if (block_size >= 4) {
      if (tid < 2) {
        __update(cnt, cnt_i, tid, tid + 2);
      }
      __syncthreads();
    }
    if (block_size >= 2) {
      if (tid < 1) {
        __update(cnt, cnt_i, tid, tid + 1);
      }
      __syncthreads();
    }

    old = cnt_i[0];
    if (tid == 0) {
      idxs[j] = old;
      temp[old] = 1e9;
    }

    x1 = dataset[old * 3 + 0];
    y1 = dataset[old * 3 + 1];
    z1 = dataset[old * 3 + 2];

  }
}

void minimum_density_sampling_kernel_wrapper(int b, int n, int m,
                                            const float *dataset, float *temp,
                                            int *idxs, float * mean_mst_length) {
  unsigned int n_threads = opt_n_threads(n);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (n_threads) {
    case 1024:
      minimum_density_sampling_kernel<1024>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs, mean_mst_length);
      break;
    case 512:
      minimum_density_sampling_kernel<512>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs, mean_mst_length);
      break;
    case 256:
      minimum_density_sampling_kernel<256>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs, mean_mst_length);
      break;
    case 128:
      minimum_density_sampling_kernel<128>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs, mean_mst_length);
      break;
    case 64:
      minimum_density_sampling_kernel<64>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs, mean_mst_length);
      break;
    case 32:
      minimum_density_sampling_kernel<32>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs, mean_mst_length);
      break;
    case 16:
      minimum_density_sampling_kernel<16>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs, mean_mst_length);
      break;
    case 8:
      minimum_density_sampling_kernel<8>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs, mean_mst_length);
      break;
    case 4:
      minimum_density_sampling_kernel<4>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs, mean_mst_length);
      break;
    case 2:
      minimum_density_sampling_kernel<2>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs, mean_mst_length);
      break;
    case 1:
      minimum_density_sampling_kernel<1>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs, mean_mst_length);
      break;
    default:
      minimum_density_sampling_kernel<1024>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs, mean_mst_length);
  }

  CUDA_CHECK_ERRORS();
}
