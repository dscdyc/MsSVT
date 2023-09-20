import torch
from torch.autograd import Function, Variable

from . import mssvt_ops_cuda


class BuildHashTable(Function):

    @staticmethod
    def forward(ctx, batch_size, hash_size, spatial_shape, voxel_indices, v_bs_cnt):

        x_max, y_max, z_max = spatial_shape
        x_max, y_max, z_max = int(x_max), int(y_max), int(z_max)
        num_voxels = voxel_indices.shape[0]
        assert voxel_indices.is_contiguous()
        dense_map = torch.zeros((batch_size, hash_size, 2)).int().fill_(-1)
        dense_map = dense_map.to(voxel_indices.device)

        mssvt_ops_cuda.build_mapping_with_hash_wrapper(x_max, y_max, z_max, num_voxels, hash_size, voxel_indices, v_bs_cnt, dense_map)
        return dense_map

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None

build_hash_table = BuildHashTable.apply


class WindowPartition(Function):
    @staticmethod
    def forward(ctx, win_size, max_num_wins, batch_size, hash_size, spatial_shape, voxel_indices):

        x_ws, y_ws, z_ws = win_size
        x_max, y_max, z_max = spatial_shape
        x_max, y_max, z_max = int(x_max), int(y_max), int(z_max)
        dense_map = torch.zeros((batch_size, hash_size, 2)).int().fill_(-1)
        dense_map = dense_map.to(voxel_indices.device)
        num_voxels = voxel_indices.shape[0]
        assert voxel_indices.is_contiguous()
        win_indices = torch.zeros((batch_size, max_num_wins, 3)).int().fill_(-1).to(voxel_indices.device)
        vcount = torch.zeros(batch_size).int().to(voxel_indices.device)
        mssvt_ops_cuda.window_with_hash_wrapper(x_max, y_max, z_max, x_ws, y_ws, z_ws,
                                                num_voxels, max_num_wins, hash_size,
                                                voxel_indices, win_indices, dense_map, vcount)
        win_list = []
        for i in range(batch_size):
            win = win_indices[i]
            win = win[win[:, 0] >= 0] # not -1
            bs_idx = torch.zeros((win.shape[0], 1)).int().fill_(i).to(voxel_indices.device)
            win = torch.cat([bs_idx, win], dim = 1)
            win_list.append(win)

        win_list = torch.cat(win_list, dim = 0)
        return win_list, dense_map

    @staticmethod
    def backward(ctx, a=None):
        return None, None

get_non_empty_window_center = WindowPartition.apply


class GatherTwoWindowVoxels(Function):

    @staticmethod
    def forward(ctx, spatial_shape, win_size, max_num_odd, max_num_even, max_num_win1, max_num_win2, 
                vox_query_odd, vox_query_even, vox_query_win1, vox_query_win2, win_indices, dense_map):
        
        x_max, y_max, z_max = spatial_shape
        x_max, y_max, z_max = int(x_max), int(y_max), int(z_max)
        x_ws, y_ws, z_ws = win_size
        batch_size, hash_size, _ = dense_map.shape
        num_odd, num_even, num_win1, num_win2 = vox_query_odd.shape[0], vox_query_even.shape[0], vox_query_win1.shape[0], vox_query_win2.shape[0] # here num_* means NUM OF VOXELS
        num_wins = win_indices.shape[0] # here num_win means NUM OF WINDOWS
        assert win_indices.is_contiguous()

        vox_ind_odd = torch.zeros((num_wins, max_num_odd)).int().fill_(-1).to(win_indices.device)
        vox_ind_even = torch.zeros((num_wins, max_num_even)).int().fill_(-1).to(win_indices.device)
        vox_ind_win1 = torch.zeros((num_wins, max_num_win1)).int().fill_(-1).to(win_indices.device)
        vox_ind_win2 = torch.zeros((num_wins, max_num_win2)).int().fill_(-1).to(win_indices.device)

        vox_coord_odd = torch.zeros((num_wins, max_num_odd, 3)).int().to(win_indices.device)
        vox_coord_even = torch.zeros((num_wins, max_num_even, 3)).int().to(win_indices.device)
        vox_coord_win1 = torch.zeros((num_wins, max_num_win1, 3)).int().to(win_indices.device)
        vox_coord_win2 = torch.zeros((num_wins, max_num_win2, 3)).int().to(win_indices.device)

        mssvt_ops_cuda.gather_two_window_voxels_with_hash_wrapper(
            x_max, y_max, z_max, x_ws, y_ws, z_ws, 
            max_num_odd, max_num_even, max_num_win1, max_num_win2,
            num_wins, hash_size, num_odd, num_even, num_win1, num_win2, 
            vox_ind_odd, vox_ind_even, vox_ind_win1, vox_ind_win2,
            vox_coord_odd, vox_coord_even, vox_coord_win1, vox_coord_win2,
            vox_query_odd, vox_query_even, vox_query_win1, vox_query_win2, 
            win_indices, dense_map)
        
        return vox_ind_odd, vox_ind_even, vox_ind_win1, vox_ind_win2, vox_coord_odd, vox_coord_even, vox_coord_win1, vox_coord_win2
    
    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None, None, None

gather_two_window_voxels = GatherTwoWindowVoxels.apply


class GatherOneWindowVoxels(Function):

    @staticmethod
    def forward(ctx, spatial_shape, win_size, max_num_win1, vox_query_win1, win_indices, dense_map):
        
        x_max, y_max, z_max = spatial_shape
        x_max, y_max, z_max = int(x_max), int(y_max), int(z_max)
        x_ws, y_ws, z_ws = win_size
        batch_size, hash_size, _ = dense_map.shape
        num_win1= vox_query_win1.shape[0] # here num_* means NUM OF VOXELS
        num_wins = win_indices.shape[0] # here num_win means NUM OF WINDOWS
        assert win_indices.is_contiguous()

        vox_ind_win1 = torch.zeros((num_wins, max_num_win1)).int().fill_(-1).to(win_indices.device)
        vox_coord_win1 = torch.zeros((num_wins, max_num_win1, 3)).int().to(win_indices.device)

        mssvt_ops_cuda.gather_one_window_voxels_with_hash_wrapper(
            x_max, y_max, z_max, x_ws, y_ws, z_ws, 
            max_num_win1, num_wins, hash_size, num_win1, 
            vox_ind_win1, vox_coord_win1, vox_query_win1, 
            win_indices, dense_map)
        
        return vox_ind_win1, vox_coord_win1
    
    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None

gather_one_window_voxels = GatherOneWindowVoxels.apply


class GroupingOperation(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, features_batch_cnt: torch.Tensor,
                idx: torch.Tensor, idx_batch_cnt: torch.Tensor):
        """
        Args:
            ctx:
            features: (N1 + N2 ..., C) tensor of features to group
            features_batch_cnt: (batch_size) [N1 + N2 ...] tensor containing the indicies of features to group with
            idx: (M1 + M2 ..., nsample) tensor containing the indicies of features to group with
            idx_batch_cnt: (batch_size) [M1 + M2 ...] tensor containing the indicies of features to group with

        Returns:
            output: (M1 + M2, C, nsample) tensor
        """
        assert features.is_contiguous()
        assert features_batch_cnt.is_contiguous()
        assert idx.is_contiguous()
        assert idx_batch_cnt.is_contiguous()

        assert features.shape[0] == features_batch_cnt.sum(), \
            'features: %s, features_batch_cnt: %s' % (str(features.shape), str(features_batch_cnt))
        assert idx.shape[0] == idx_batch_cnt.sum(), \
            'idx: %s, idx_batch_cnt: %s' % (str(idx.shape), str(idx_batch_cnt))

        M, nsample = idx.size()
        N, C = features.size()
        B = idx_batch_cnt.shape[0]
        output = torch.cuda.FloatTensor(M, C, nsample).zero_()

        mssvt_ops_cuda.group_features_wrapper(B, M, C, nsample, features, features_batch_cnt, idx, idx_batch_cnt, output)

        ctx.for_backwards = (B, N, idx, features_batch_cnt, idx_batch_cnt)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """
        Args:
            ctx:
            grad_out: (M1 + M2 ..., C, nsample) tensor of the gradients of the output from forward

        Returns:
            grad_features: (N1 + N2 ..., C) gradient of the features
        """
        B, N, idx, features_batch_cnt, idx_batch_cnt = ctx.for_backwards

        M, C, nsample = grad_out.size()
        grad_features = Variable(torch.cuda.FloatTensor(N, C).zero_())

        grad_out_data = grad_out.data.contiguous()
        mssvt_ops_cuda.group_features_grad_wrapper(B, M, C, N, nsample, grad_out_data, idx,
                                                   idx_batch_cnt, features_batch_cnt, grad_features.data)
        return grad_features, None, None, None

grouping_operation = GroupingOperation.apply