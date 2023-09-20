#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "ms_sparse_attention_gpu.h"


#define CHECK_CUDA(x) do { \
  if (!x.type().is_cuda()) { \
    fprintf(stderr, "%s must be CUDA tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_CONTIGUOUS(x) do { \
  if (!x.is_contiguous()) { \
    fprintf(stderr, "%s must be contiguous tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)


int build_mapping_with_hash_wrapper(int x_max, int y_max, int z_max, int num_voxels, int hash_size,
                                        at::Tensor v_indices_tensor, at::Tensor v_bs_cnt_tensor, at::Tensor xyz_to_vidx_tensor) {
    CHECK_INPUT(v_indices_tensor);
    CHECK_INPUT(v_bs_cnt_tensor);
    CHECK_INPUT(xyz_to_vidx_tensor);

    const int *v_indices = v_indices_tensor.data<int>();
    const int *v_bs_cnt = v_bs_cnt_tensor.data<int>();
    int *xyz_to_vidx = xyz_to_vidx_tensor.data<int>();

    build_mapping_with_hash_kernel_launcher(x_max, y_max, z_max, num_voxels, hash_size, v_indices, v_bs_cnt, xyz_to_vidx);
    return 1;
}

int window_with_hash_wrapper(
    int x_wgs, int y_wgs, int z_wgs, int x_ws, int y_ws, int z_ws,
    int num_voxels, int num_windows, int hash_size,
    at::Tensor v_indices_tensor, at::Tensor w_indices_tensor,
    at::Tensor xyz_to_vidx_tensor, at::Tensor vcount_tensor
    ) {

    CHECK_INPUT(v_indices_tensor);
    CHECK_INPUT(w_indices_tensor);
    CHECK_INPUT(xyz_to_vidx_tensor);
    CHECK_INPUT(vcount_tensor);

    const int *v_indices = v_indices_tensor.data<int>();
    int *w_indices = w_indices_tensor.data<int>();
    int *xyz_to_vidx = xyz_to_vidx_tensor.data<int>();
    int *vcount = vcount_tensor.data<int>();

    window_with_hash_kernel_launcher(
        x_wgs, y_wgs, z_wgs, x_ws, y_ws, z_ws, 
        num_voxels, num_windows, hash_size,
        v_indices, w_indices, xyz_to_vidx, vcount);
    return 1;
}

int gather_two_window_voxels_with_hash_wrapper(
    int x_max, int y_max, int z_max, int x_ws, int y_ws, int z_ws,
    int max_num_odd, int max_num_even, int max_num_win1, int max_num_win2,
    int num_wins, int hash_size, int num_odd, int num_even, int num_win1, int num_win2, 
    at::Tensor vox_ind_odd_tensor,
    at::Tensor vox_ind_even_tensor,
    at::Tensor vox_ind_win1_tensor,
    at::Tensor vox_ind_win2_tensor,
    at::Tensor vox_coord_odd_tensor,
    at::Tensor vox_coord_even_tensor,
    at::Tensor vox_coord_win1_tensor,
    at::Tensor vox_coord_win2_tensor,
    at::Tensor vox_query_odd_tensor, 
    at::Tensor vox_query_even_tensor,
    at::Tensor vox_query_win1_tensor,
    at::Tensor vox_query_win2_tensor,
    at::Tensor v_indices_tensor, 
    at::Tensor xyz_to_vidx_tensor
    ) {

    CHECK_INPUT(vox_ind_odd_tensor);
    CHECK_INPUT(vox_ind_even_tensor);
    CHECK_INPUT(vox_ind_win1_tensor);
    CHECK_INPUT(vox_ind_win2_tensor);
    CHECK_INPUT(vox_coord_odd_tensor);
    CHECK_INPUT(vox_coord_even_tensor);
    CHECK_INPUT(vox_coord_win1_tensor);
    CHECK_INPUT(vox_coord_win2_tensor);
    CHECK_INPUT(vox_query_odd_tensor);
    CHECK_INPUT(vox_query_even_tensor);
    CHECK_INPUT(vox_query_win1_tensor);
    CHECK_INPUT(vox_query_win2_tensor);
    CHECK_INPUT(v_indices_tensor);
    CHECK_INPUT(xyz_to_vidx_tensor);

    int *vox_ind_odd = vox_ind_odd_tensor.data<int>();
    int *vox_ind_even = vox_ind_even_tensor.data<int>();
    int *vox_ind_win1 = vox_ind_win1_tensor.data<int>();
    int *vox_ind_win2 = vox_ind_win2_tensor.data<int>();
    int *vox_coord_odd = vox_coord_odd_tensor.data<int>();
    int *vox_coord_even = vox_coord_even_tensor.data<int>();
    int *vox_coord_win1 = vox_coord_win1_tensor.data<int>();
    int *vox_coord_win2 = vox_coord_win2_tensor.data<int>();
    const int *vox_query_odd = vox_query_odd_tensor.data<int>();
    const int *vox_query_even = vox_query_even_tensor.data<int>();
    const int *vox_query_win1 = vox_query_win1_tensor.data<int>();
    const int *vox_query_win2 = vox_query_win2_tensor.data<int>();
    const int *v_indices = v_indices_tensor.data<int>();
    const int *xyz_to_vidx = xyz_to_vidx_tensor.data<int>();

    gather_two_window_voxels_with_hash_kernel_launcher(
        x_max, y_max, z_max, x_ws, y_ws, z_ws, 
        max_num_odd, max_num_even, max_num_win1, max_num_win2,
        num_wins, hash_size, num_odd, num_even, num_win1, num_win2,
        vox_ind_odd, vox_ind_even, vox_ind_win1, vox_ind_win2,
        vox_coord_odd, vox_coord_even, vox_coord_win1, vox_coord_win2,
        vox_query_odd, vox_query_even, vox_query_win1, vox_query_win2, 
        v_indices, xyz_to_vidx);
    return 1;
}

int gather_one_window_voxels_with_hash_wrapper(
    int x_max, int y_max, int z_max, int x_ws, int y_ws, int z_ws,
    int max_num_win1, int num_wins, int hash_size, int num_win1, 
    at::Tensor vox_ind_win1_tensor,
    at::Tensor vox_coord_win1_tensor,
    at::Tensor vox_query_win1_tensor,
    at::Tensor v_indices_tensor, 
    at::Tensor xyz_to_vidx_tensor
    ) {

    CHECK_INPUT(vox_ind_win1_tensor);
    CHECK_INPUT(vox_coord_win1_tensor);
    CHECK_INPUT(vox_query_win1_tensor);
    CHECK_INPUT(v_indices_tensor);
    CHECK_INPUT(xyz_to_vidx_tensor);

    int *vox_ind_win1 = vox_ind_win1_tensor.data<int>();
    int *vox_coord_win1 = vox_coord_win1_tensor.data<int>();
    const int *vox_query_win1 = vox_query_win1_tensor.data<int>();
    const int *v_indices = v_indices_tensor.data<int>();
    const int *xyz_to_vidx = xyz_to_vidx_tensor.data<int>();

    gather_one_window_voxels_with_hash_kernel_launcher(
        x_max, y_max, z_max, x_ws, y_ws, z_ws, 
        max_num_win1, num_wins, hash_size, num_win1, 
        vox_ind_win1, vox_coord_win1, vox_query_win1, 
        v_indices, xyz_to_vidx);
    return 1;
}