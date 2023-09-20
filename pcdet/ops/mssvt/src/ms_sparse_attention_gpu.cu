#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "ms_sparse_attention_gpu.h"
#include "ms_cuda_utils.h"

__device__ int murmur_hash(int k, int hash_size) {
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    //return k & (hash_size-1);
    return k % hash_size;
}

__device__ int hash(int k, int hash_size) {
    return k % hash_size;
}

__device__ void hash_table_insert(int &key, int &value, int &hash_size, int *xyz_to_vidx) {
    /*
        xyz_to_idx (hash_size, 2) NO BATCH SIZE
    */
    int hash_idx = hash(key, hash_size);
    int prob_cnt = 0;
    while(true) {
        int prev_key = atomicCAS(xyz_to_vidx + hash_idx*2 + 0, EMPTY_KEY, key); // insert key when empty
        if (prev_key == EMPTY_KEY || prev_key == key) {
            xyz_to_vidx[hash_idx*2 + 1] = value; // insert value
            break;
        }
        // linear probing
        hash_idx = (hash_idx + 1) % hash_size;

        // security in case of dead loop
        prob_cnt += 1;
        if (prob_cnt >= hash_size) break;
    }
}

__device__ int hash_table_find(int &key, int &hash_size, const int *xyz_to_vidx) {
    int hash_idx = key % hash_size;
    int v_idx = EMPTY_KEY;
    int prob_cnt = 0;
    while (true) {
        // found
        if (xyz_to_vidx[hash_idx * 2 + 0] == key) {
            v_idx = xyz_to_vidx[hash_idx * 2 + 1];
            break;
        }
        // empty, not found
        if (xyz_to_vidx[hash_idx * 2 + 0] == EMPTY_KEY) {
            break;
        }
        // linear probing
        hash_idx = (hash_idx + 1) % hash_size;
        // security in case of dead loop
        prob_cnt += 1;
        if (prob_cnt >= hash_size) break;
    }
    return v_idx;
}

__global__ void build_mapping_with_hash_kernel(int x_max, int y_max, int z_max, int num_voxels, int hash_size,
                                                const int *v_indices, const int *v_bs_cnt, int *xyz_to_vidx) {
    /*
        v_indices: [N1+N2, 4] bs zyx indices of voxels
        v_bs_cnt: [bs] num_voxels in each sample
        xyz_to_vidx: [B, hash_size, 2] hash table key-value for dim-2
    */

    int th_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (th_idx >= num_voxels) return;
    int bs_idx = v_indices[th_idx * 4 + 0];
    int z_idx = v_indices[th_idx * 4 + 1];
    int y_idx = v_indices[th_idx * 4 + 2];
    int x_idx = v_indices[th_idx * 4 + 3];

    int v_sum = 0;
    int bs_cnt = bs_idx - 1;
    while(bs_cnt >= 0){
        v_sum += v_bs_cnt[bs_cnt];
        bs_cnt--;
    }
    int v_idx = th_idx - v_sum; // v_idx for this sample

    xyz_to_vidx += bs_idx * hash_size * 2;
    if (x_idx >= x_max || x_idx < 0 || y_idx < 0 || y_idx >= y_max || z_idx < 0 || z_idx >= z_max) return; // out of bound

    // key -> [x_max, y_max, z_max] value -> v_idx
    int key = x_idx * y_max * z_max + y_idx * z_max + z_idx;
    hash_table_insert(key, v_idx, hash_size, xyz_to_vidx);

    return;
}

void build_mapping_with_hash_kernel_launcher(int x_max, int y_max, int z_max, int num_voxels, int hash_size,
                                                const int *v_indices, const int *v_bs_cnt, int *xyz_to_vidx){

    cudaError_t err;

    dim3 blocks(DIVUP(num_voxels, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    build_mapping_with_hash_kernel<<<blocks, threads>>>(x_max, y_max, z_max, num_voxels, hash_size,
                                                            v_indices, v_bs_cnt, xyz_to_vidx);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

__global__ void window_with_hash_kernel(
    int x_wgs, int y_wgs, int z_wgs, int x_ws, int y_ws, int z_ws,
    int num_voxels, int num_windows, int hash_size,
    const int *v_indices, int *w_indices, int *xyz_to_vidx, int *vcount
    ) {
    /*
        v_indices: [num_voxels, 4] bs + zyx indices of voxels
        w_indices: [bs, num_windows, 3] downsampled voxels, -1 if not unique
        xyz_to_vidx: [bs, hash_size, 2]
        vcount: [bs]
    */

    int th_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (th_idx >= num_voxels) return;

    int bs_idx = v_indices[th_idx * 4 + 0];
    int z_idx = v_indices[th_idx * 4 + 1];
    int y_idx = v_indices[th_idx * 4 + 2];
    int x_idx = v_indices[th_idx * 4 + 3];

    int w_z_idx = z_idx / z_ws;
    int w_y_idx = y_idx / y_ws;
    int w_x_idx = x_idx / x_ws;

    if (w_x_idx < 0 || w_x_idx >= x_wgs || w_y_idx < 0 || w_y_idx >= y_wgs || w_z_idx < 0 || w_z_idx >= z_wgs) return;

    xyz_to_vidx += bs_idx * hash_size * 2;
    w_indices += bs_idx * num_windows * 3;

    int key = w_x_idx * y_wgs * z_wgs + w_y_idx * z_wgs + w_z_idx;
    // hash table with force insert, reject duplicates
    int hash_idx = hash(key, hash_size);
    int prob_cnt = 0;
    while(true) {
        int prev_key = atomicCAS(xyz_to_vidx + hash_idx*2 + 0, EMPTY_KEY, key); // insert key when empty 
        if (prev_key == EMPTY_KEY) {
            int v_idx = atomicAdd(vcount + bs_idx, 1);
            w_indices[v_idx * 3 + 0] = w_z_idx; // insert zyx to w_indices
            w_indices[v_idx * 3 + 1] = w_y_idx;
            w_indices[v_idx * 3 + 2] = w_x_idx;
            xyz_to_vidx[hash_idx*2 + 1] = v_idx; // insert value to hash table
            break;
        } else if (prev_key == key) { // already occupied
            break;
        }
        // linear probing
        hash_idx = (hash_idx + 1) % hash_size;
        // security in case of dead loop
        prob_cnt += 1;
        if (prob_cnt >= hash_size) break;
    }
}

void window_with_hash_kernel_launcher(
    int x_wgs, int y_wgs, int z_wgs, int x_ws, int y_ws, int z_ws,
    int num_voxels, int num_windows, int hash_size,
    const int *v_indices, int *w_indices, int *xyz_to_vidx, int *vcount
    ) {

    cudaError_t err;

    dim3 blocks(DIVUP(num_voxels, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    window_with_hash_kernel<<<blocks, threads>>>(x_wgs, y_wgs, z_wgs, x_ws, y_ws, z_ws,
                                                num_voxels, num_windows, hash_size,
                                                v_indices, w_indices, xyz_to_vidx, vcount);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }

}

__global__ void gather_two_window_voxels_with_hash_kernel(
    int x_max, int y_max, int z_max, int x_ws, int y_ws, int z_ws, 
    int max_num_odd, int max_num_even, int max_num_win1, int max_num_win2,
    int num_wins, int hash_size, int num_odd, int num_even, int num_win1, int num_win2,
    int *vox_ind_odd, int *vox_ind_even, int *vox_ind_win1, int *vox_ind_win2,
    int *vox_coord_odd, int *vox_coord_even, int *vox_coord_win1, int *vox_coord_win2,
    const int *vox_query_odd, const int *vox_query_even, const int *vox_query_win1, const int *vox_query_win2,
    const int *v_indices, const int *xyz_to_vidx) {
    /*
        sparse voxel attention kernel
    */

    int th_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (th_idx >= num_wins) return;

    int bs_idx = v_indices[th_idx * 4 + 0];
    int z_idx = v_indices[th_idx * 4 + 1];
    int y_idx = v_indices[th_idx * 4 + 2];
    int x_idx = v_indices[th_idx * 4 + 3];

    xyz_to_vidx += bs_idx * hash_size * 2;

    int num_samples_odd = 0;
    int num_samples_even = 0;
    int num_samples_win1 = 0;
    int num_samples_win2 = 0;
    int x_radius = x_ws / 2;
    int y_radius = y_ws / 2;
    int z_radius = z_ws / 2;

    int center_x = x_idx * x_ws + x_radius;
    int center_y = y_idx * y_ws + y_radius;
    int center_z = z_idx * z_ws + z_radius;

    for (int query_idx = 0; query_idx < num_odd; ++query_idx){
        int offset_x = vox_query_odd[query_idx * 3 + 0];
        int offset_y = vox_query_odd[query_idx * 3 + 1];
        int offset_z = vox_query_odd[query_idx * 3 + 2];
        int sx_idx = center_x + offset_x;
        int sy_idx = center_y + offset_y;
        int sz_idx = center_z + offset_z;
        if (sx_idx >= x_max || sx_idx < 0 || sy_idx >= y_max || sy_idx < 0 || sz_idx >= z_max || sz_idx < 0) continue;
        int skey = sx_idx * y_max * z_max + sy_idx * z_max + sz_idx;
        int sv_idx = hash_table_find(skey, hash_size, xyz_to_vidx);
        if (sv_idx != EMPTY_KEY){
            if (num_samples_odd >= max_num_odd && num_samples_even >= max_num_even && num_samples_win1 >= max_num_win1 && num_samples_win2 >= max_num_win2) return;
            if (num_samples_odd < max_num_odd){
                vox_ind_odd[th_idx * max_num_odd + num_samples_odd] = sv_idx;
                vox_coord_odd[th_idx * max_num_odd * 3 + num_samples_odd * 3 + 0] = offset_x;
                vox_coord_odd[th_idx * max_num_odd * 3 + num_samples_odd * 3 + 1] = offset_y;
                vox_coord_odd[th_idx * max_num_odd * 3 + num_samples_odd * 3 + 2] = offset_z;
                num_samples_odd++;
            }
            if (num_samples_win1 < max_num_win1){
                vox_ind_win1[th_idx * max_num_win1 + num_samples_win1] = sv_idx;
                vox_coord_win1[th_idx * max_num_win1 * 3 + num_samples_win1 * 3 + 0] = offset_x;
                vox_coord_win1[th_idx * max_num_win1 * 3 + num_samples_win1 * 3 + 1] = offset_y;
                vox_coord_win1[th_idx * max_num_win1 * 3 + num_samples_win1 * 3 + 2] = offset_z;
                num_samples_win1++;
            }
            if (num_samples_win2 < max_num_win2){
                vox_ind_win2[th_idx * max_num_win2 + num_samples_win2] = sv_idx;
                vox_coord_win2[th_idx * max_num_win2 * 3 + num_samples_win2 * 3 + 0] = offset_x;
                vox_coord_win2[th_idx * max_num_win2 * 3 + num_samples_win2 * 3 + 1] = offset_y;
                vox_coord_win2[th_idx * max_num_win2 * 3 + num_samples_win2 * 3 + 2] = offset_z;
                num_samples_win2++;
            }
        }
    }

    for (int query_idx = 0; query_idx < num_even; ++query_idx){
        int offset_x = vox_query_even[query_idx * 3 + 0];
        int offset_y = vox_query_even[query_idx * 3 + 1];
        int offset_z = vox_query_even[query_idx * 3 + 2];
        int sx_idx = center_x + offset_x;
        int sy_idx = center_y + offset_y;
        int sz_idx = center_z + offset_z;
        if (sx_idx >= x_max || sx_idx < 0 || sy_idx >= y_max || sy_idx < 0 || sz_idx >= z_max || sz_idx < 0) continue;
        int skey = sx_idx * y_max * z_max + sy_idx * z_max + sz_idx;
        int sv_idx = hash_table_find(skey, hash_size, xyz_to_vidx);
        if (sv_idx != EMPTY_KEY){
            if (num_samples_even >= max_num_even && num_samples_win1 >= max_num_win1 && num_samples_win2 >= max_num_win2) return;
            if (num_samples_even < max_num_even){
                vox_ind_even[th_idx * max_num_even + num_samples_even] = sv_idx;
                vox_coord_even[th_idx * max_num_even * 3 + num_samples_even * 3 + 0] = offset_x;
                vox_coord_even[th_idx * max_num_even * 3 + num_samples_even * 3 + 1] = offset_y;
                vox_coord_even[th_idx * max_num_even * 3 + num_samples_even * 3 + 2] = offset_z;
                num_samples_even++;
            }
            if (num_samples_win1 < max_num_win1){
                vox_ind_win1[th_idx * max_num_win1 + num_samples_win1] = sv_idx;
                vox_coord_win1[th_idx * max_num_win1 * 3 + num_samples_win1 * 3 + 0] = offset_x;
                vox_coord_win1[th_idx * max_num_win1 * 3 + num_samples_win1 * 3 + 1] = offset_y;
                vox_coord_win1[th_idx * max_num_win1 * 3 + num_samples_win1 * 3 + 2] = offset_z;
                num_samples_win1++;
            }
            if (num_samples_win2 < max_num_win2){
                vox_ind_win2[th_idx * max_num_win2 + num_samples_win2] = sv_idx;
                vox_coord_win2[th_idx * max_num_win2 * 3 + num_samples_win2 * 3 + 0] = offset_x;
                vox_coord_win2[th_idx * max_num_win2 * 3 + num_samples_win2 * 3 + 1] = offset_y;
                vox_coord_win2[th_idx * max_num_win2 * 3 + num_samples_win2 * 3 + 2] = offset_z;
                num_samples_win2++;
            }
        }
    }

    for (int query_idx = 0; query_idx < num_win1; ++query_idx){
        int offset_x = vox_query_win1[query_idx * 3 + 0];
        int offset_y = vox_query_win1[query_idx * 3 + 1];
        int offset_z = vox_query_win1[query_idx * 3 + 2];
        int sx_idx = center_x + offset_x;
        int sy_idx = center_y + offset_y;
        int sz_idx = center_z + offset_z;
        if (sx_idx >= x_max || sx_idx < 0 || sy_idx >= y_max || sy_idx < 0 || sz_idx >= z_max || sz_idx < 0) continue;
        int skey = sx_idx * y_max * z_max + sy_idx * z_max + sz_idx;
        int sv_idx = hash_table_find(skey, hash_size, xyz_to_vidx);
        if (sv_idx != EMPTY_KEY){
            if (num_samples_win1 >= max_num_win1 && num_samples_win2 >= max_num_win2) return;
            if (num_samples_win1 < max_num_win1){
                vox_ind_win1[th_idx * max_num_win1 + num_samples_win1] = sv_idx;
                vox_coord_win1[th_idx * max_num_win1 * 3 + num_samples_win1 * 3 + 0] = offset_x;
                vox_coord_win1[th_idx * max_num_win1 * 3 + num_samples_win1 * 3 + 1] = offset_y;
                vox_coord_win1[th_idx * max_num_win1 * 3 + num_samples_win1 * 3 + 2] = offset_z;
                num_samples_win1++;
            }
            if (num_samples_win2 < max_num_win2){
                vox_ind_win2[th_idx * max_num_win2 + num_samples_win2] = sv_idx;
                vox_coord_win2[th_idx * max_num_win2 * 3 + num_samples_win2 * 3 + 0] = offset_x;
                vox_coord_win2[th_idx * max_num_win2 * 3 + num_samples_win2 * 3 + 1] = offset_y;
                vox_coord_win2[th_idx * max_num_win2 * 3 + num_samples_win2 * 3 + 2] = offset_z;
                num_samples_win2++;
            }
        }
    }

    for (int query_idx = 0; query_idx < num_win2; ++query_idx){
        int offset_x = vox_query_win2[query_idx * 3 + 0];
        int offset_y = vox_query_win2[query_idx * 3 + 1];
        int offset_z = vox_query_win2[query_idx * 3 + 2];
        int sx_idx = center_x + offset_x;
        int sy_idx = center_y + offset_y;
        int sz_idx = center_z + offset_z;
        if (sx_idx >= x_max || sx_idx < 0 || sy_idx >= y_max || sy_idx < 0 || sz_idx >= z_max || sz_idx < 0) continue;
        int skey = sx_idx * y_max * z_max + sy_idx * z_max + sz_idx;
        int sv_idx = hash_table_find(skey, hash_size, xyz_to_vidx);
        if (sv_idx != EMPTY_KEY){
            if (num_samples_win2 < max_num_win2){
                vox_ind_win2[th_idx * max_num_win2 + num_samples_win2] = sv_idx;
                vox_coord_win2[th_idx * max_num_win2 * 3 + num_samples_win2 * 3 + 0] = offset_x;
                vox_coord_win2[th_idx * max_num_win2 * 3 + num_samples_win2 * 3 + 1] = offset_y;
                vox_coord_win2[th_idx * max_num_win2 * 3 + num_samples_win2 * 3 + 2] = offset_z;
                num_samples_win2++;
            }
        }
    }

    return;
}

void gather_two_window_voxels_with_hash_kernel_launcher(
    int x_max, int y_max, int z_max, int x_ws, int y_ws, int z_ws,
    int max_num_odd, int max_num_even, int max_num_win1, int max_num_win2,
    int num_wins, int hash_size, int num_odd, int num_even, int num_win1, int num_win2, 
    int *vox_ind_odd, int *vox_ind_even, int *vox_ind_win1, int *vox_ind_win2,
    int *vox_coord_odd, int *vox_coord_even, int *vox_coord_win1, int *vox_coord_win2,
    const int *vox_query_odd, const int *vox_query_even, const int *vox_query_win1, const int *vox_query_win2,
    const int *v_indices, const int *xyz_to_vidx) {

    cudaError_t err;

    dim3 blocks(DIVUP(num_wins, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    gather_two_window_voxels_with_hash_kernel<<<blocks, threads>>>(
        x_max, y_max, z_max, x_ws, y_ws, z_ws, 
        max_num_odd, max_num_even, max_num_win1, max_num_win2,
        num_wins, hash_size, num_odd, num_even, num_win1, num_win2,
        vox_ind_odd, vox_ind_even, vox_ind_win1, vox_ind_win2,
        vox_coord_odd, vox_coord_even, vox_coord_win1, vox_coord_win2,
        vox_query_odd, vox_query_even, vox_query_win1, vox_query_win2, 
        v_indices, xyz_to_vidx
    );
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

__global__ void gather_one_window_voxels_with_hash_kernel(
    int x_max, int y_max, int z_max, int x_ws, int y_ws, int z_ws, 
    int max_num_win1, int num_wins, int hash_size, int num_win1, 
    int *vox_ind_win1, int *vox_coord_win1, const int *vox_query_win1, 
    const int *v_indices, const int *xyz_to_vidx) {
    /*
        sparse voxel attention kernel
    */

    int th_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (th_idx >= num_wins) return;

    int bs_idx = v_indices[th_idx * 4 + 0];
    int z_idx = v_indices[th_idx * 4 + 1];
    int y_idx = v_indices[th_idx * 4 + 2];
    int x_idx = v_indices[th_idx * 4 + 3];

    xyz_to_vidx += bs_idx * hash_size * 2;

    int num_samples_win1 = 0;
    int x_radius = x_ws / 2;
    int y_radius = y_ws / 2;
    int z_radius = z_ws / 2;

    int center_x = x_idx * x_ws + x_radius;
    int center_y = y_idx * y_ws + y_radius;
    int center_z = z_idx * z_ws + z_radius;

    for (int query_idx = 0; query_idx < num_win1; ++query_idx){
        int offset_x = vox_query_win1[query_idx * 3 + 0];
        int offset_y = vox_query_win1[query_idx * 3 + 1];
        int offset_z = vox_query_win1[query_idx * 3 + 2];
        int sx_idx = center_x + offset_x;
        int sy_idx = center_y + offset_y;
        int sz_idx = center_z + offset_z;
        if (sx_idx >= x_max || sx_idx < 0 || sy_idx >= y_max || sy_idx < 0 || sz_idx >= z_max || sz_idx < 0) continue;
        int skey = sx_idx * y_max * z_max + sy_idx * z_max + sz_idx;
        int sv_idx = hash_table_find(skey, hash_size, xyz_to_vidx);
        if (sv_idx != EMPTY_KEY){
            if (num_samples_win1 < max_num_win1){
                vox_ind_win1[th_idx * max_num_win1 + num_samples_win1] = sv_idx;
                vox_coord_win1[th_idx * max_num_win1 * 3 + num_samples_win1 * 3 + 0] = offset_x;
                vox_coord_win1[th_idx * max_num_win1 * 3 + num_samples_win1 * 3 + 1] = offset_y;
                vox_coord_win1[th_idx * max_num_win1 * 3 + num_samples_win1 * 3 + 2] = offset_z;
                num_samples_win1++;
            }
        }
    }

    return;
}

void gather_one_window_voxels_with_hash_kernel_launcher(
    int x_max, int y_max, int z_max, int x_ws, int y_ws, int z_ws,
    int max_num_win1, int num_wins, int hash_size, int num_win1, 
    int *vox_ind_win1, int *vox_coord_win1, const int *vox_query_win1, 
    const int *v_indices, const int *xyz_to_vidx) {

    cudaError_t err;

    dim3 blocks(DIVUP(num_wins, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    gather_one_window_voxels_with_hash_kernel<<<blocks, threads>>>(
        x_max, y_max, z_max, x_ws, y_ws, z_ws, 
        max_num_win1, num_wins, hash_size, num_win1, 
        vox_ind_win1, vox_coord_win1, vox_query_win1, 
        v_indices, xyz_to_vidx
    );
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}