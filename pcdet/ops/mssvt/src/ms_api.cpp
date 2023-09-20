#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "ms_sparse_attention_gpu.h"
#include "group_features_gpu.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_mapping_with_hash_wrapper", &build_mapping_with_hash_wrapper, "build_mapping_with_hash_wrapper");
    m.def("window_with_hash_wrapper", &window_with_hash_wrapper, "window_with_hash_wrapper");
    m.def("gather_two_window_voxels_with_hash_wrapper", &gather_two_window_voxels_with_hash_wrapper, "gather_two_window_voxels_with_hash_wrapper");
    m.def("gather_one_window_voxels_with_hash_wrapper", &gather_one_window_voxels_with_hash_wrapper, "gather_one_window_voxels_with_hash_wrapper");
    m.def("group_features_grad_wrapper", &group_features_grad_wrapper_stack, "group_features_grad_wrapper_stack");
    m.def("group_features_wrapper", &group_features_wrapper_stack, "group_features_wrapper_stack");
}