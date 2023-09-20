#ifndef MS_GROUP_FEATURES_GPU_H
#define MS_GROUP_FEATURES_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>


int build_mapping_with_hash_wrapper(int x_max, int y_max, int z_max, int num_voxels, int hash_size,
                                        at::Tensor v_indices_tensor, at::Tensor v_bs_cnt_tensor, at::Tensor xyz_to_vidx_tensor);

void build_mapping_with_hash_kernel_launcher(int x_max, int y_max, int z_max, int num_voxels, int hash_size,
                                                const int *v_indices, const int *v_bs_cnt, int *xyz_to_vidx);

int window_with_hash_wrapper(
    int x_wgs, int y_wgs, int z_wgs, int x_ws, int y_ws, int z_ws,
    int num_voxels, int num_windows, int hash_size,
    at::Tensor v_indices_tensor, at::Tensor w_indices_tensor,
    at::Tensor xyz_to_vidx_tensor, at::Tensor vcount_tensor
    );

void window_with_hash_kernel_launcher(
    int x_wgs, int y_wgs, int z_wgs, int x_ws, int y_ws, int z_ws,
    int num_voxels, int num_windows, int hash_size,
    const int *v_indices, int *w_indices, int *xyz_to_vidx, int *vcount
    );

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
    );

void gather_two_window_voxels_with_hash_kernel_launcher(
    int x_max, int y_max, int z_max, int x_ws, int y_ws, int z_ws,
    int max_num_odd, int max_num_even, int max_num_win1, int max_num_win2,
    int num_wins, int hash_size, int num_odd, int num_even, int num_win1, int num_win2, 
    int *vox_ind_odd, int *vox_ind_even, int *vox_ind_win1, int *vox_ind_win2,
    int *vox_coord_odd, int *vox_coord_even, int *vox_coord_win1, int *vox_coord_win2,
    const int *vox_query_odd, const int *vox_query_even, const int *vox_query_win1, const int *vox_query_win2,
    const int *v_indices, const int *xyz_to_vidx
    );

int gather_one_window_voxels_with_hash_wrapper(
    int x_max, int y_max, int z_max, int x_ws, int y_ws, int z_ws,
    int max_num_win1, int num_wins, int hash_size, int num_win1, 
    at::Tensor vox_ind_win1_tensor,
    at::Tensor vox_coord_win1_tensor,
    at::Tensor vox_query_win1_tensor,
    at::Tensor v_indices_tensor, 
    at::Tensor xyz_to_vidx_tensor
    );

void gather_one_window_voxels_with_hash_kernel_launcher(
    int x_max, int y_max, int z_max, int x_ws, int y_ws, int z_ws,
    int max_num_win1, int num_wins, int hash_size, int num_win1, 
    int *vox_ind_win1, int *vox_coord_win1, const int *vox_query_win1, 
    const int *v_indices, const int *xyz_to_vidx);

#endif