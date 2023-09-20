import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ...ops.votr_ops import votr_utils
from ...ops.voxel_ops import Voxelization
from ...ops.pointnet2.pointnet2_batch import pointnet2_utils
from ...ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_utils_stack
import pdb
from ...utils.spconv_utils import replace_feature, spconv
from functools import partial
from timm.models.layers import DropPath, trunc_normal_
# from .point_transformer import MultiheadPointTransformerLayer
from torch.cuda.amp import autocast as autocast
# from ..model_utils.attention import InstanceNormWithMask

DEBUG_MODE = False
PRINT = False
CHECK_TIME = False

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m

@torch.no_grad()
def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def scatter_nd(indices, updates, shape):
    """pytorch edition of tensorflow scatter_nd.
    this function don't contain except handle code. so use this carefully
    when indice repeats, don't support repeat add which is supported
    in tensorflow.
    """
    ret = torch.zeros(*shape, dtype=updates.dtype, device=updates.device)# B, D, H, W, C
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + shape[indices.shape[-1]:] # N, C  
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    ret[slices] = updates.view(*output_shape)
    return ret

class SparseTensor(object):
    def __init__(self, features, indices, spatial_shape, voxel_size, point_cloud_range, batch_size, hash_size, map_table = None, gather_dict = None):
        self.features = features # [N, C] notice that N is the sum of all batches voxels
        self.indices = indices #[N, 4] b,z,y,x
        self.spatial_shape = spatial_shape # [x, y, z] #[1408, 1600, 40]
        self.batch_size = batch_size
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.hash_size = hash_size #400,000
        self.gather_dict = gather_dict #None
        self.map_table = self.build_map_table() if not map_table else map_table # bs, hash_size, 2

    @torch.no_grad()
    def build_map_table(self):
        bs_cnt = torch.zeros(self.batch_size).int()
        for i in range(self.batch_size):
            bs_cnt[i] = (self.indices[:, 0] == i).sum().item() # [num1,num2,..] get voxel num of different batches
        bs_cnt = bs_cnt.to(self.indices.device)
        map_table = votr_utils.build_hash_table(
            self.batch_size,
            self.hash_size,
            self.spatial_shape,
            self.indices,
            bs_cnt,
        ) 
        # build hash table, input indices and return hashtable: (bs, hash_size, 2) {key is a mapping id e.g. 1456676 
        # from voxel position x, y, z and value is the index in N voxels, e.g. 0 indicates the first voxel in sp_tensor.features}
        return map_table

    def dense(self, channels_first=True):
        reverse_spatial_shape = self.spatial_shape[::-1] # (ZYX)
        output_shape = [self.batch_size] + list(
            reverse_spatial_shape) + [self.features.shape[1]]
        res = scatter_nd(
            self.indices.to(self.features.device).long(), self.features,
            output_shape)
        if not channels_first:
            return res
        ndim = len(reverse_spatial_shape)
        trans_params = list(range(0, ndim + 1))
        trans_params.insert(1, ndim + 1)
        return res.permute(*trans_params).contiguous() # b, c, d, h, w

class Attention3d(nn.Module):
    def __init__(self, input_channels, output_channels, ff_channels, dropout, num_heads, attention_modes, 
                 use_relative_position_encoding=False, use_pseudo_relative_position_encoding=False,
                 use_separte_multi_head=False):
        super(Attention3d, self).__init__()
        self.attention_modes = attention_modes
        self.num_heads = num_heads
        if (not use_relative_position_encoding) and (not use_pseudo_relative_position_encoding) \
            and (not use_separte_multi_head):
            self.mhead_attention = nn.MultiheadAttention(
                    embed_dim= input_channels,
                    num_heads= num_heads,
                    dropout= dropout,
                    )
        elif use_relative_position_encoding:
            from pcdet.models.model_utils.attention import WindowAttention3D
            self.mhead_attention = WindowAttention3D(
                    dim= input_channels,
                    num_heads= num_heads,
                    attn_drop= dropout,
                    )
        elif use_pseudo_relative_position_encoding:
            from pcdet.models.model_utils.attention import WindowAttention3DPos
            self.mhead_attention = WindowAttention3DPos(
                    dim= input_channels,
                    num_heads= num_heads,
                    attn_drop= dropout,
                    )
        elif use_separte_multi_head:
            self.mhead_attention = nn.ModuleList()
            # for i, nh in enumerate(num_heads):
            #     self.mhead_attention.append(
            #         nn.MultiheadAttention(
            #             embed_dim=input_channels//sum(num_heads)*nh,
            #             num_heads=nh,
            #             dropout=dropout,
            #         ))
            from pcdet.models.model_utils.attention import WindowAttention3DRelative3
            for i, nh in enumerate(num_heads):
                self.mhead_attention.append(
                    WindowAttention3DRelative3(
                        dim=input_channels//sum(num_heads)*nh,
                        num_heads=nh,
                        attn_drop=dropout,
                        batch_first=False
                    ))
        else:
            raise NotImplementedError
        self.drop_out = nn.Dropout(dropout)

        self.linear1 = nn.Linear(input_channels, ff_channels)
        self.linear2 = nn.Linear(ff_channels, input_channels)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

        # self.output_layer = nn.Sequential(
        #     nn.Linear(input_channels, output_channels),
        #     # nn.BatchNorm1d(output_channels),
        #     # nn.LayerNorm(output_channels),
        #     nn.ReLU()
        # )

    @torch.no_grad()
    def with_bs_cnt(self, indices, batch_size):
        bs_cnt = torch.zeros(batch_size).int()
        for i in range(batch_size):
            bs_cnt[i] = (indices[:, 0] == i).sum().item()
        bs_cnt = bs_cnt.to(indices.device)
        return bs_cnt

    @torch.no_grad()
    def with_coords(self, indices, point_cloud_range, voxel_size, shifts=None, strides=None, query_mode=False):
        voxel_size = torch.tensor(voxel_size).unsqueeze(0).to(indices.device)
        min_range = torch.tensor(point_cloud_range[0:3]).unsqueeze(0).to(indices.device)
        if not query_mode:
            coords = (indices[:, [3, 2, 1]].float() + 0.5) * voxel_size + min_range
        else:
            raw_voxel_size = voxel_size / torch.tensor(strides).unsqueeze(0).to(indices.device)
            coords = indices[:, [3, 2, 1]].float() * voxel_size + min_range + 0.5 * voxel_size -  torch.tensor(shifts).unsqueeze(0).to(indices.device \
                 ) * raw_voxel_size 
        return coords

    def forward(self, sp_tensor):
        raise NotImplementedError

class SparseAttention3d(Attention3d):
    def __init__(self, input_channels, output_channels, ff_channels, dropout, num_heads, attention_modes, strides, num_ds_voxels,
                 use_relative_coords = False, use_pooled_feature = False, use_no_query_coords = False):
        super(SparseAttention3d, self).__init__(input_channels, output_channels, ff_channels, dropout, num_heads, attention_modes)

        self.use_relative_coords = use_relative_coords
        self.use_pooled_features = use_pooled_feature
        self.use_no_query_coords = use_no_query_coords

        self.strides = strides
        self.num_ds_voxels = num_ds_voxels

        # self.norm = nn.BatchNorm1d(input_channels)
        self.norm1 = nn.LayerNorm(input_channels)
        self.norm2 = nn.LayerNorm(input_channels)
        if not self.use_no_query_coords:
            self.q_pos_proj = nn.Sequential(
                nn.Linear(3, input_channels),
                nn.ReLU(),
            )
        self.k_pos_proj = nn.Sequential(
            nn.Conv1d(3, input_channels, 1),
            nn.ReLU(),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(input_channels, output_channels),
            # nn.BatchNorm1d(output_channels),
            # nn.LayerNorm(output_channels),
            nn.ReLU()
        )

    @torch.no_grad()
    def create_gather_dict(self, attention_modes, map_table, voxel_indices, spatial_shape):
        _gather_dict = {}
        for attention_mode in attention_modes:
            if attention_mode.NAME == 'LocalAttention':
                attend_size = attention_mode.SIZE
                attend_range = attention_mode.RANGE
                _gather_indices = votr_utils.sparse_local_attention_hash_indices(spatial_shape, attend_size, attend_range, self.strides, map_table, voxel_indices)
            elif attention_mode.NAME == 'StridedAttention':
                attend_size = attention_mode.SIZE
                range_spec = attention_mode.RANGE_SPEC
                _gather_indices = votr_utils.sparse_strided_attention_hash_indices(spatial_shape, attend_size, range_spec, self.strides, map_table, voxel_indices)
            else:
                raise NotImplementedError

            _gather_mask = (_gather_indices < 0)
            #_gather_indices[_gather_indices < 0] = 0
            _gather_dict[attention_mode.NAME] = [_gather_indices, _gather_mask]

        return _gather_dict

    @torch.no_grad()
    def downsample(self, sp_tensor):
        x_shape = sp_tensor.spatial_shape[0] // self.strides[0]
        y_shape = sp_tensor.spatial_shape[1] // self.strides[1]
        z_shape = sp_tensor.spatial_shape[2] // self.strides[2]
        new_spatial_shape = [x_shape, y_shape, z_shape]
        new_indices, new_map_table = votr_utils.hash_table_down_sample(self.strides, self.num_ds_voxels, sp_tensor.batch_size, sp_tensor.hash_size, new_spatial_shape, sp_tensor.indices)
        return new_spatial_shape, new_indices, new_map_table

    def forward(self, sp_tensor):
        new_spatial_shape, new_indices, new_map_table = self.downsample(sp_tensor)# new_indices [M, 4] : (bs, z, y, x)
        vx, vy, vz = sp_tensor.voxel_size
        new_voxel_size = [vx * self.strides[0], vy * self.strides[1], vz * self.strides[2]]
        gather_dict = self.create_gather_dict(self.attention_modes, sp_tensor.map_table, new_indices, sp_tensor.spatial_shape)

        voxel_features = self.norm1(sp_tensor.features)
        v_bs_cnt = self.with_bs_cnt(sp_tensor.indices, sp_tensor.batch_size)
        k_bs_cnt = self.with_bs_cnt(new_indices, sp_tensor.batch_size)

        a_key_indices, a_key_mask = [], []
        for attention_idx, attetion_mode in enumerate(self.attention_modes):
            key_indices, key_mask = gather_dict[attetion_mode.NAME]
            a_key_indices.append(key_indices)
            a_key_mask.append(key_mask)

        key_indices = torch.cat(a_key_indices, dim = 1) # M, nsample : [200, 0, -1, -1,...] notice that key_indices are index while new_indices are position
        key_mask = torch.cat(a_key_mask, dim = 1)

        key_features = votr_utils.grouping_operation(voxel_features, v_bs_cnt, key_indices, k_bs_cnt) # M, C, nsample=size
        voxel_coords = self.with_coords(sp_tensor.indices, sp_tensor.point_cloud_range, sp_tensor.voxel_size)
        key_coords = votr_utils.grouping_operation(voxel_coords, v_bs_cnt, key_indices, k_bs_cnt)

        # debug
        # np.save("lihe_debug/voxel_branch_key_coords_votr.npy", key_coords.cpu().detach().numpy())

        query_coords = self.with_coords(new_indices, sp_tensor.point_cloud_range, new_voxel_size)

        if self.use_pooled_features:
            pooled_query_features = key_features.max(dim=-1)[0]
            pooled_query_features = pooled_query_features.unsqueeze(0) # 1, M, C
            if self.use_no_query_coords:
                query_features = pooled_query_features # 1, M, C
            else:
                query_features = self.q_pos_proj(query_coords).unsqueeze(0)
                query_features = query_features + pooled_query_features
        else:
            query_features = self.q_pos_proj(query_coords).unsqueeze(0)

        if self.use_relative_coords:
            key_coords = key_coords - query_coords.unsqueeze(-1) # (N, 3, size)

        key_pos_emb = self.k_pos_proj(key_coords)
        key_features = key_features + key_pos_emb
        key_features = key_features.permute(2, 0, 1).contiguous() # (size, N1+N2, C)

        attend_features, attend_weights = self.mhead_attention(
            query = query_features,
            key = key_features,
            value = key_features,
            key_padding_mask = key_mask,
        )

        attend_features = self.drop_out(attend_features.squeeze(0)) # 0

        new_features = attend_features # M, C
        act_features = self.linear2(self.dropout1(self.activation(self.linear1(self.norm2(new_features)))))
        new_features = new_features + self.dropout1(act_features)
        # new_features = self.norm(new_features)
        new_features = self.output_layer(new_features)

        # update sp_tensor
        sp_tensor.features = new_features
        sp_tensor.indices = new_indices
        sp_tensor.spatial_shape = new_spatial_shape
        sp_tensor.voxel_size = new_voxel_size

        del sp_tensor.map_table
        sp_tensor.gather_dict = None
        sp_tensor.map_table = new_map_table
        # print("========sparse attention map table========", new_map_table[...,1].max())
        # print("========sparse attention indices========", sp_tensor.features.shape)
        return sp_tensor

#######################################################################################################
# this class gathers two attention layers (wo shift / shift) into one group
class SwinSparseAttention3dGroup(Attention3d):
    def __init__(self, input_channels, output_channels, ff_channels, dropout, drop_path, num_heads, attention_modes, strides, num_ds_voxels,
                 use_relative_coords = False, use_pooled_feature = False, use_no_query_coords = False, shifts=None, use_point_branch=False,\
                 use_point_branch_debug=False, use_relative_position_encoding=False, use_amp=False, 
                 use_pseudo_relative_position_encoding=False, use_separte_multi_head=False):
        super(SwinSparseAttention3dGroup, self).__init__(
            input_channels, output_channels, ff_channels, dropout, num_heads, attention_modes, 
            use_relative_position_encoding=use_relative_position_encoding,
            use_pseudo_relative_position_encoding=use_pseudo_relative_position_encoding,
            use_separte_multi_head=use_separte_multi_head)

        self.use_relative_coords = use_relative_coords
        self.use_pooled_features = use_pooled_feature
        self.use_no_query_coords = use_no_query_coords
        self.use_point_branch = use_point_branch
        self.use_point_branch_debug = use_point_branch_debug

        self.strides = strides # window size
        self.num_ds_voxels = num_ds_voxels

        # self.norm = nn.BatchNorm1d(input_channels)
        # self.norm1 = nn.BatchNorm1d(input_channels)
        # self.norm2 = nn.BatchNorm1d(input_channels)
        self.norm1 = nn.LayerNorm(input_channels)
        self.norm2 = nn.LayerNorm(input_channels)
        # self.norm1_shift = nn.BatchNorm1d(input_channels)
        # self.norm2_shift = nn.BatchNorm1d(input_channels)
        # self.ins_norm1 = InstanceNormWithMask(input_channels//2)
        # self.ins_norm2 = InstanceNormWithMask(input_channels//2)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.shifts = shifts[1] # select valid shift

        self.pos_proj = nn.Sequential(
            nn.Conv1d(3, input_channels, 1),
            nn.ReLU(),
        )

        self.use_relative_position_encoding = use_relative_position_encoding
        self.use_pseudo_relative_position_encoding = use_pseudo_relative_position_encoding
        self.use_separte_multi_head = use_separte_multi_head
        self.use_real_relative_position_encoding = True
        # if self.use_separte_multi_head:
        #     self.smh_fc = nn.Linear(input_channels//sum(num_heads)*num_heads[0], input_channels)

        if self.use_pseudo_relative_position_encoding:
            self.max_window_x, self.max_window_y, self.max_window_z = self.attention_modes[0].MAXRANGE
            self.pos_emb_mat_qk = nn.Parameter(torch.zeros(self.attention_modes[0].MAXRANGE + [input_channels]))
            trunc_normal_(self.pos_emb_mat_qk, std=.02)
            self.pos_emb_mat_v = nn.Parameter(torch.zeros(self.attention_modes[0].MAXRANGE + [input_channels]))
            trunc_normal_(self.pos_emb_mat_v, std=.02)

        # relative position encoding
        if self.use_relative_position_encoding:
            self.max_window_x, self.max_window_y, self.max_window_z = self.attention_modes[0].MAXRANGE
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * self.max_window_x - 1) * (2 * self.max_window_y - 1) * (2 * self.max_window_z - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, , nH
            trunc_normal_(self.relative_position_bias_table, std=.02)
        
        # real relative position encoding
        elif self.use_real_relative_position_encoding:
            self.max_window_x, self.max_window_y, self.max_window_z = self.attention_modes[0].MAXRANGE
            for i in range(len(num_heads)):
                setattr(self, f'pos_emb_mat{i}', nn.ParameterDict())
                for m in ['q', 'k']:
                    for d in ['x', 'y', 'z']:
                        getattr(self, f'pos_emb_mat{i}')[m+d] = nn.Parameter(
                            torch.zeros(num_heads[i], input_channels//sum(num_heads), (2*getattr(self,'max_window_'+d)-1)))  # 2*Wh-1 * 2*Ww-1, , nH
                        trunc_normal_(getattr(self, f'pos_emb_mat{i}')[m+d], std=.02)

        self.point_sampler = pointnet2_utils_stack.QueryAndGroup(radius=3.0, nsample=256, use_xyz=True) if self.use_point_branch_debug else None
        # if self.use_point_branch_debug:
        #     self.point_attention_block = MultiheadPointTransformerLayer(dim=128)
        self.use_amp = use_amp

    @torch.no_grad()
    def create_gather_dict(self, attention_modes, map_table, voxel_indices, spatial_shape):
        _gather_dict = {}
        for attention_mode in attention_modes:
            if attention_mode.NAME == 'LocalAttention':
                attend_size = attention_mode.SIZE
                attend_size_out = attention_mode.SIZEOUT
                self.attend_size = attend_size
                self.attend_size_out = attend_size_out
                attend_range = attention_mode.RANGE
                attend_range_out_spec = attention_mode.RANGEOUT_SPEC
                # print("go=======================")
                _gather_indices_odd, _gather_indices_even, _gather_indices_all,  _gather_out_indices, \
                _gather_coords_odd, _gather_coords_even, _gather_coords_all, _gather_out_coords = \
                    votr_utils.gather_sampled_voxels_small_and_big_window_indices(
                    spatial_shape, self.strides, attend_size_out, attend_range, attend_range_out_spec, self.strides, map_table, voxel_indices)
                # print("done====================")
            elif attention_mode.NAME == 'StridedAttention':
                attend_size = attention_mode.SIZE
                self.attend_size = attend_size
                range_spec = attention_mode.RANGE_SPEC
                _gather_indices = votr_utils.sparse_strided_attention_hash_indices(spatial_shape, attend_size, range_spec, self.strides, map_table, voxel_indices)
                raise NotImplementedError
            else:
                raise NotImplementedError

            _gather_mask_odd = (_gather_indices_odd < 0)
            _gather_mask_even = (_gather_indices_even < 0)
            _gather_mask_other = (_gather_indices_all < 0)
            _gather_mask_out = (_gather_out_indices < 0)
            _gather_dict[attention_mode.NAME] = \
                [[_gather_indices_odd, _gather_mask_odd, _gather_coords_odd], \
                 [_gather_indices_even, _gather_mask_even, _gather_coords_even], \
                 [_gather_indices_all, _gather_mask_other, _gather_coords_all], \
                 [_gather_out_indices, _gather_mask_out, _gather_out_coords]]

        return _gather_dict

    @torch.no_grad()
    def window_partition(self, sp_tensor):
        x_shape = sp_tensor.spatial_shape[0] // self.strides[0]
        y_shape = sp_tensor.spatial_shape[1] // self.strides[1]
        z_shape = sp_tensor.spatial_shape[2] // self.strides[2]
        new_spatial_shape = [x_shape, y_shape, z_shape]
        
        center_indices, _ = \
                        votr_utils.hash_table_down_sample(self.strides, self.num_ds_voxels, sp_tensor.batch_size,\
                                                                sp_tensor.hash_size, new_spatial_shape, sp_tensor.indices)
        
        return center_indices

    def forward(self, sp_tensor, pv_set, v2p_dense_map, pv_set_bs_cnt=None, 
        pv_set_mask=None, stage_idx=None, points=None, points_bs_cnt=None, reuse_dict=None):
        """
        Args:
            sp_tensor: 
            pv_set: N_all, max_num_points, C(3,feat)
            v2p_dense_map: bs, hash_size, 2
        Returns:
            sp_tensor: 
            pv_set: N_all, max_num_points, C
        """
        if (reuse_dict is None) or (reuse_dict == {}):
            reuse_dict = {}
            reuse = False
        else:
            reuse = True
            # assert self.strides == [7,7,3]
        # process point input
        if self.use_point_branch:
            N_all, max_num_points, C_p = pv_set.shape
            pv_set = pv_set.permute(0, 2, 1) # [N_all, C, max_num_points]
            pv_set_mask = pv_set_mask.permute(0, 2, 1) # [N_all, C, max_num_points]
            short_cut = pv_set # save for residual connection
            pv_set = pv_set.reshape([N_all, -1]) # [N_all, C, max_points] --> [N_all, C*max_points]
        
        voxel_features_short_cut = sp_tensor.features
        voxel_features = self.norm1(voxel_features_short_cut)

        if reuse:
            new_indices = reuse_dict['new_indices']
            new_voxel_size = reuse_dict['new_voxel_size']
            gather_dict = reuse_dict['gather_dict']
            v_bs_cnt = reuse_dict['v_bs_cnt']
            k_bs_cnt = reuse_dict['k_bs_cnt']

            key_indices_in = reuse_dict['key_indices_in']
            _, nsample_in = key_indices_in.shape
            key_mask_in = reuse_dict['key_mask_in']
            key_grid_coords_in= reuse_dict['key_grid_coords_in']

            key_indices_out = reuse_dict['key_indices_out']
            key_mask_out = reuse_dict['key_mask_out']
            key_grid_coords_out= reuse_dict['key_grid_coords_out']

            query_indices = gather_dict[self.attention_modes[0].NAME][self.attention_modes[0].SAMPLE_KEY][0]
            M, nsample = query_indices.shape
            query_mask = gather_dict[self.attention_modes[0].NAME][self.attention_modes[0].SAMPLE_KEY][1]
            query_grid_coords = gather_dict[self.attention_modes[0].NAME][self.attention_modes[0].SAMPLE_KEY][2]

            key_indices_out_fps = reuse_dict['key_indices_out_fps']
            key_indices_in_fps = reuse_dict['key_indices_in_fps']
            if self.use_relative_position_encoding or self.use_pseudo_relative_position_encoding or self.use_real_relative_position_encoding:
                key_grid_coords_out_fps = reuse_dict['key_grid_coords_out_fps']
                key_grid_coords_in_fps = reuse_dict['key_grid_coords_in_fps']
            
            # relative position encoding
            if self.use_relative_position_encoding:
                with torch.no_grad():
                    relative_coords = query_grid_coords[:, :, None, :] - torch.cat([key_grid_coords_in_fps, key_grid_coords_out_fps], dim=1)[:, None, :, :]
                    # M, n_q, n_k, 3
                    relative_coords[..., 0] += self.max_window_x - 1
                    relative_coords[..., 1] += self.max_window_y - 1
                    relative_coords[..., 2] += self.max_window_z - 1
                    relative_coords[..., 1] *= 2*self.max_window_x - 1
                    relative_coords[..., 2] *= (2*self.max_window_x - 1)*(2*self.max_window_y - 1)
                    relative_position_index = relative_coords.sum(-1) # M, n_q, n_k
                    # print("========debug======= :", relative_position_index.max(), len(self.relative_position_bias_table))
                M, n_q, n_k = relative_position_index.shape
                # relative_position_bias = self.relative_position_bias_table[relative_position_index.view(-1).long()].view(M, n_q, n_k, -1)# num_heads
                # relative_position_bias = relative_position_bias.permute(0,3,1,2)

                relative_position_bias = pointnet2_utils.gather_operation(self.relative_position_bias_table.unsqueeze(0).repeat(M, 1, 1).permute(0,2,1).contiguous(), relative_position_index.reshape([M, -1]).int())
                # M, num_heads, nqxnk
                relative_position_bias = relative_position_bias.view(M, -1, n_q, n_k)
            
            elif self.use_real_relative_position_encoding:
                relative_coords_in = query_grid_coords[:, :, None, :] - key_grid_coords_in_fps[:, None, :, :] # (M, n_q, n_k, 3)
                relative_coords_in[..., 0] += self.max_window_x - 1
                relative_coords_in[..., 1] += self.max_window_y - 1
                relative_coords_in[..., 2] += self.max_window_z - 1
                relative_coords_in = relative_coords_in.long() # (M, n_q, n_k, 3)
                
                relative_coords_out = query_grid_coords[:, :, None, :] - key_grid_coords_out_fps[:, None, :, :] # (M, n_q, n_k, 3)
                relative_coords_out[..., 0] += self.max_window_x - 1
                relative_coords_out[..., 1] += self.max_window_y - 1
                relative_coords_out[..., 2] += self.max_window_z - 1
                relative_coords_out = relative_coords_out.long() # (M, n_q, n_k, 3)

                relative_coords = [relative_coords_in, relative_coords_out]

            with autocast(enabled=self.use_amp):
                q_features = votr_utils.grouping_operation(voxel_features, v_bs_cnt, query_indices, k_bs_cnt) # M, C, nsample=size
                key_features_in = votr_utils.grouping_operation(voxel_features, v_bs_cnt, key_indices_in_fps, k_bs_cnt) # M, C, nsample_out
                key_features_out = votr_utils.grouping_operation(voxel_features, v_bs_cnt, key_indices_out_fps, k_bs_cnt) # M, C, nsample_out
                voxel_coords = reuse_dict['voxel_coords']
                query_coords = votr_utils.grouping_operation(voxel_coords, v_bs_cnt, query_indices, k_bs_cnt) # M, 3, nsample

                key_coords_in_all = reuse_dict['key_coords_in_all']
                key_coords_in = reuse_dict['key_coords_in']
                key_coords_out = reuse_dict['key_coords_out']
                key_mask = reuse_dict['key_mask']

        else:
            new_indices = self.window_partition(sp_tensor)
            vx, vy, vz = sp_tensor.voxel_size
            new_voxel_size = [vx * self.strides[0], vy * self.strides[1], vz * self.strides[2]]
            gather_dict = self.create_gather_dict(
                self.attention_modes, sp_tensor.map_table, new_indices, sp_tensor.spatial_shape)
            v_bs_cnt = self.with_bs_cnt(sp_tensor.indices, sp_tensor.batch_size)# [batch_size] count num of each batch
            k_bs_cnt = self.with_bs_cnt(new_indices, sp_tensor.batch_size)

            reuse_dict['new_indices'] = new_indices
            reuse_dict['new_voxel_size'] = new_voxel_size
            reuse_dict['gather_dict'] = gather_dict
            reuse_dict['v_bs_cnt'] = v_bs_cnt
            reuse_dict['k_bs_cnt'] = k_bs_cnt

            a_key_indices, a_key_mask, a_key_coords = [], [], []
            a_key_indices_in, a_key_mask_in, a_key_coords_in = [], [], []
            cnt = [] # wo_shift, shift
            cnt_in = [] # wo_shift, shift
            
            for idx, gather_info in enumerate(gather_dict[self.attention_modes[0].NAME]):
                key_indices, key_mask, key_grid_coords = gather_info
                if idx in [self.attention_modes[0].SAMPLE_KEY, 3]:
                    cnt.append(key_indices)
                    a_key_indices.append(key_indices)
                    a_key_mask.append(key_mask)
                    a_key_coords.append(key_grid_coords)
                elif idx in [2]:
                    cnt_in.append(len(key_indices))
                    a_key_indices_in.append(key_indices)
                    a_key_mask_in.append(key_mask)
                    a_key_coords_in.append(key_grid_coords)

            key_indices_in = a_key_indices_in[0] # M, nsample1 + nsample2
            _, nsample_in = key_indices_in.shape
            key_mask_in = a_key_mask_in[0] # M, nsample1 + nsample2
            key_grid_coords_in= a_key_coords_in[0] # M, nsample1 + nsample2
            ##################################### wo shift attention #####################################
            query_indices = a_key_indices[0] # M, nsample
            M, nsample = query_indices.shape
            key_indices_out = a_key_indices[1] # M, nsample_out
            query_mask = a_key_mask[0]# M, nsample
            key_mask_out = a_key_mask[1]# M, nsample_out
            query_grid_coords = a_key_coords[0] # M, nsample, 3
            key_grid_coords_out = a_key_coords[1] # M, nsample_out, 3
            # print('DEBUG 0.0:', ((~key_mask).sum(dim=-1) == 0).sum(), M)
            # print('DEBUG 0.1:', ((~key_mask_in).sum(dim=-1) == 0).sum(), M)

            reuse_dict['key_indices_in'] = key_indices_in
            reuse_dict['key_mask_in'] = key_mask_in
            reuse_dict['key_grid_coords_in'] = key_grid_coords_in
            reuse_dict['key_indices_out'] = key_indices_out
            reuse_dict['key_mask_out'] = key_mask_out
            reuse_dict['key_grid_coords_out'] = key_grid_coords_out

            # fps
            npoint = self.attention_modes[0].FPS
            # t1 = time.time()
            # out_index = farthest_point_sample(key_coords_out.permute(0,2,1), 64)
            with torch.no_grad():
                key_out_index, key_mask_out_fps = pointnet2_utils.farthest_point_sample_new(key_grid_coords_out.float(), npoint) # M, npoint
                key_in_index, key_mask_in_fps = pointnet2_utils.farthest_point_sample_new(key_grid_coords_in.float(), npoint) # M, npoint
                # key_out_index = pointnet2_utils.furthest_point_sample(key_grid_coords_out.float(), npoint) # M, npoint
                # key_in_index = pointnet2_utils.furthest_point_sample(key_grid_coords_in.float(), npoint) # M, npoint
            
            key_indices_out_fps = key_indices_out.unsqueeze(1).float()
            key_indices_out_fps = pointnet2_utils.gather_operation(key_indices_out_fps, key_out_index).squeeze(1).int()
            key_indices_in_fps = key_indices_in.unsqueeze(1).float()
            key_indices_in_fps = pointnet2_utils.gather_operation(key_indices_in_fps, key_in_index).squeeze(1).int()
            if self.use_relative_position_encoding or self.use_pseudo_relative_position_encoding or self.use_real_relative_position_encoding:
                key_grid_coords_out_fps = pointnet2_utils.gather_operation(key_grid_coords_out.permute(0,2,1).contiguous().float(), key_out_index).int()# M, 3, npoint
                key_grid_coords_out_fps = key_grid_coords_out_fps.permute(0,2,1).contiguous().float()
                key_grid_coords_in_fps = pointnet2_utils.gather_operation(key_grid_coords_in.permute(0,2,1).contiguous().float(), key_in_index).int()# M, 3, npoint
                key_grid_coords_in_fps = key_grid_coords_in_fps.permute(0,2,1).contiguous().float()
                reuse_dict['key_grid_coords_out_fps'] = key_grid_coords_out_fps
                reuse_dict['key_grid_coords_in_fps'] = key_grid_coords_in_fps
            key_mask_out_fps = key_mask_out_fps | (key_indices_out_fps < 0)
            key_mask_in_fps = key_mask_in_fps | (key_indices_in_fps < 0)
            # key_mask_out_fps = (key_indices_out_fps < 0)
            # key_mask_in_fps = (key_indices_in_fps < 0)
            reuse_dict['key_indices_out_fps'] = key_indices_out_fps
            reuse_dict['key_indices_in_fps'] = key_indices_in_fps
            # print("--after fps-- :", key_indices_out_fps.shape)

            # relative position encoding
            if self.use_relative_position_encoding:
                with torch.no_grad():
                    relative_coords = query_grid_coords[:, :, None, :] - torch.cat([key_grid_coords_in_fps, key_grid_coords_out_fps], dim=1)[:, None, :, :]
                    # M, n_q, n_k, 3
                    relative_coords[..., 0] += self.max_window_x - 1
                    relative_coords[..., 1] += self.max_window_y - 1
                    relative_coords[..., 2] += self.max_window_z - 1
                    relative_coords[..., 1] *= 2*self.max_window_x - 1
                    relative_coords[..., 2] *= (2*self.max_window_x - 1)*(2*self.max_window_y - 1)
                    relative_position_index = relative_coords.sum(-1) # M, n_q, n_k
                    # print("========debug======= :", relative_position_index.max(), len(self.relative_position_bias_table))
                M, n_q, n_k = relative_position_index.shape
                # relative_position_bias = self.relative_position_bias_table[relative_position_index.view(-1).long()].view(M, n_q, n_k, -1)# num_heads
                # relative_position_bias = relative_position_bias.permute(0,3,1,2)

                relative_position_bias = pointnet2_utils.gather_operation(self.relative_position_bias_table.unsqueeze(0).repeat(M, 1, 1).permute(0,2,1).contiguous(), relative_position_index.reshape([M, -1]).int())
                # M, num_heads, nqxnk
                relative_position_bias = relative_position_bias.view(M, -1, n_q, n_k)
            
            elif self.use_real_relative_position_encoding:
                relative_coords_in = query_grid_coords[:, :, None, :] - key_grid_coords_in_fps[:, None, :, :] # (M, n_q, n_k, 3)
                relative_coords_in[..., 0] += self.max_window_x - 1
                relative_coords_in[..., 1] += self.max_window_y - 1
                relative_coords_in[..., 2] += self.max_window_z - 1
                relative_coords_in = relative_coords_in.long() # (M, n_q, n_k, 3)
                
                relative_coords_out = query_grid_coords[:, :, None, :] - key_grid_coords_out_fps[:, None, :, :] # (M, n_q, n_k, 3)
                relative_coords_out[..., 0] += self.max_window_x - 1
                relative_coords_out[..., 1] += self.max_window_y - 1
                relative_coords_out[..., 2] += self.max_window_z - 1
                relative_coords_out = relative_coords_out.long() # (M, n_q, n_k, 3)

                relative_coords = [relative_coords_in, relative_coords_out]
        
            t2 = time.time()
            # print(" STAGE {} FPS : {}s".format(stage_idx, t2 - t1))
            # print(" key indices : {}".format(key_indices.shape))
            with autocast(enabled=self.use_amp):
                q_features = votr_utils.grouping_operation(voxel_features, v_bs_cnt, query_indices, k_bs_cnt) # M, C, nsample=size
                key_features_in = votr_utils.grouping_operation(voxel_features, v_bs_cnt, key_indices_in_fps, k_bs_cnt) # M, C, nsample_in
                key_features_out = votr_utils.grouping_operation(voxel_features, v_bs_cnt, key_indices_out_fps, k_bs_cnt) # M, C, nsample_out

                voxel_coords = self.with_coords(sp_tensor.indices, sp_tensor.point_cloud_range, sp_tensor.voxel_size)
                query_coords = votr_utils.grouping_operation(voxel_coords, v_bs_cnt, query_indices, k_bs_cnt) # M, 3, nsample
                key_coords_in_all = votr_utils.grouping_operation(voxel_coords, v_bs_cnt, key_indices_in, k_bs_cnt) # M, 3, nsample
                key_coords_in = votr_utils.grouping_operation(voxel_coords, v_bs_cnt, key_indices_in_fps, k_bs_cnt)
                key_coords_out = votr_utils.grouping_operation(voxel_coords, v_bs_cnt, key_indices_out_fps, k_bs_cnt)
                win_center_coords = self.with_coords(new_indices, sp_tensor.point_cloud_range, new_voxel_size, \
                    shifts=[0., 0., 0.], strides=self.strides, query_mode=True)# window center

                # point branch
                # if points is not None:
                #     xyz_feats, _ = self.point_sampler(points[:, 1:4], points_bs_cnt, win_center_coords, k_bs_cnt) # xyz has been normalized
                #     points_xyz = xyz_feats[:, :3, :] # [M, 3, nsample]
                #     points_feats = xyz_feats[:, 3:, :]
                #     points_attend_feats = self.point_attention_block(points_feats.permute(0,2,1).contiguous(), points_xyz.permute(0,2,1).contiguous())
                # else:
                #     points_attend_feats = None

                # debug
                # if True:
                #     np.savez(f"debug/slide_window_qk_pts4.npz", win_center_coords=win_center_coords.cpu().detach().numpy(), \
                #             query_coords=query_coords.cpu().detach().numpy(), query_mask=query_mask.cpu().detach().numpy(), \
                #             key_coords_out = key_coords_out.cpu().detach().numpy(), key_coords_in = key_coords_in.cpu().detach().numpy(),
                #             k_bs_cnt=k_bs_cnt.cpu().detach().numpy(), key_coords_in_all=key_coords_in_all.cpu().detach().numpy(), \
                #             key_mask_in_fps=key_mask_in_fps.cpu().detach().numpy(), key_mask_out_fps=key_mask_out_fps.cpu().detach().numpy())
                # exit()

                # debug
                if DEBUG_MODE:
                    np.savez(f"lihe_debug/slide_window/slide_window_{stage_idx}.npz", query_coords=win_center_coords.cpu().detach().numpy(), key_coords=query_coords.cpu().detach().numpy(), \
                            key_coords_out = key_coords_out.cpu().detach().numpy(),
                            k_bs_cnt=k_bs_cnt.cpu().detach().numpy(), key_indices=query_indices.cpu().detach().numpy(), key_mask=key_mask.cpu().detach().numpy())


                if DEBUG_MODE:
                    np.savez(f"lihe_debug/voxel_branch_{stage_idx}.npz", query_coords=win_center_coords.cpu().detach().numpy(), key_coords=query_coords.cpu().detach().numpy(), \
                        k_bs_cnt=k_bs_cnt.cpu().detach().numpy(), key_indices=query_indices.cpu().detach().numpy(), key_mask=key_mask.cpu().detach().numpy())

                if self.use_relative_coords:
                    key_coords_in = key_coords_in - win_center_coords.unsqueeze(-1) # (N, 3, size)
                    key_coords_out = key_coords_out - win_center_coords.unsqueeze(-1) # (N, 3, size)
                
                # query_mask = key_mask
                key_mask = torch.cat([key_mask_in_fps, key_mask_out_fps], dim=-1)
                reuse_dict['voxel_coords'] = voxel_coords
                reuse_dict['key_coords_in_all'] = key_coords_in_all
                reuse_dict['key_coords_in'] = key_coords_in
                reuse_dict['key_coords_out'] = key_coords_out
                reuse_dict['key_mask'] = key_mask
            # end if reuse

        with autocast(enabled=self.use_amp):
            if self.use_pseudo_relative_position_encoding:
                query_pos_emb_ind_x = (query_grid_coords[..., 0] + self.max_window_x // 2).long()
                query_pos_emb_ind_y = (query_grid_coords[..., 1] + self.max_window_y // 2).long()
                query_pos_emb_ind_z = (query_grid_coords[..., 2] + self.max_window_z // 2).long()
                query_pos_emb = self.pos_emb_mat_qk[query_pos_emb_ind_x, query_pos_emb_ind_y, query_pos_emb_ind_z] # (M, pos_c, n_q)

                key_grid_coords_all = torch.cat([key_grid_coords_in_fps, key_grid_coords_out_fps], dim=1)
                key_pos_emb_ind_x = (key_grid_coords_all[..., 0] + self.max_window_x // 2).long()
                key_pos_emb_ind_y = (key_grid_coords_all[..., 1] + self.max_window_y // 2).long()
                key_pos_emb_ind_z = (key_grid_coords_all[..., 2] + self.max_window_z // 2).long()
                key_pos_emb = self.pos_emb_mat_qk[key_pos_emb_ind_x, key_pos_emb_ind_y, key_pos_emb_ind_z] # (M, n_k, pos_c)

                value_pos_emb = self.pos_emb_mat_v[key_pos_emb_ind_x, key_pos_emb_ind_y, key_pos_emb_ind_z] # (M, n_k, pos_c)

                query_features = q_features
                key_features = torch.cat([key_features_in, key_features_out], dim=-1)
            
            else:
                query_pos_emb = self.pos_proj(query_coords)
                key_pos_emb = self.pos_proj(torch.cat([key_coords_in, key_coords_out], dim=-1))
                query_features = q_features + query_pos_emb
                key_features = torch.cat([key_features_in, key_features_out], dim=-1) + key_pos_emb # (M,c,n)

            if (not self.use_relative_position_encoding) and (not self.use_pseudo_relative_position_encoding):
                query_features = query_features.permute(2, 0, 1).contiguous() # (size, N1+N2, C)
                key_features = key_features.permute(2, 0, 1).contiguous() # (size, N1+N2, C)
            else:
                query_features = query_features.permute(0, 2, 1).contiguous() # (N, C, size) -> (N, size, C)
                key_features = key_features.permute(0, 2, 1).contiguous() 

            if (not self.use_relative_position_encoding) and (not self.use_pseudo_relative_position_encoding) \
                and (not self.use_separte_multi_head):
                attend_features, attend_weights = self.mhead_attention(
                    query = query_features,# self_attn
                    key = key_features,
                    value = key_features,
                    key_padding_mask = key_mask.type(torch.bool),# M, 48
                    # attn_mask = attn_mask,# self_attn
                    need_weights=False
                )
                # ensure zeros still zeros
                attend_features = attend_features * (~query_mask).permute(1,0).unsqueeze(-1).type(torch.float)
                attend_features = self.drop_out(attend_features) # 0  (48, M, C)
            elif self.use_relative_position_encoding:
                attend_features = self.mhead_attention(
                    query = query_features,# self_attn
                    key = key_features,
                    relative_position_bias=relative_position_bias,
                    query_mask=query_mask,
                    key_mask=key_mask
                ) # attend_features: (nsample, win_num, C)
            elif self.use_pseudo_relative_position_encoding:
                attend_features = self.mhead_attention(
                    query = query_features,# self_attn
                    key = key_features,
                    query_pos_emb=query_pos_emb, 
                    key_pos_emb=key_pos_emb, 
                    value_pos_emb=value_pos_emb,
                    query_mask=query_mask,
                    key_mask=key_mask
                ) # attend_features: (nsample, win_num, C)
            elif self.use_separte_multi_head:
                attend_features = []
                num_points = self.attention_modes[0].FPS
                c_per_head = query_features.shape[-1] // sum(self.num_heads)
                head_c_list = [sum(([0] + self.num_heads)[:(_+1)]) for _ in range(len(self.num_heads)+1)]
                for i, mhead_attention in enumerate(self.mhead_attention):
                    query_features_ = query_features[:,:,c_per_head*head_c_list[i]:c_per_head*head_c_list[i+1]]
                    key_features_ = key_features[i*num_points:(i+1)*num_points,:,c_per_head*head_c_list[i]:c_per_head*head_c_list[i+1]] # (n,M,c)
                    key_mask_ = key_mask[...,i*num_points:(i+1)*num_points] # (M,n)
                    # query_features_ = self.ins_norm1(query_features_, mask=query_mask)
                    # key_features_ = self.ins_norm2(key_features_, mask=key_mask_)
                    if self.use_real_relative_position_encoding:
                        attend_features.append(mhead_attention(
                            query = query_features_, # (n_q,M,c)
                            key = key_features_,
                            value = key_features_,
                            pos_emb_mat=getattr(self, f'pos_emb_mat{i}'), 
                            relative_coords=relative_coords[i],
                            key_mask = key_mask_.type(torch.bool),# M, 48
                            # attn_mask = attn_mask,# self_attn
                            need_weights=False
                        ))
                    else:
                        attend_features.append(mhead_attention(
                            query = query_features_, # (n_q,M,c)
                            key = key_features_,
                            value = key_features_,
                            key_padding_mask = key_mask_.type(torch.bool),# M, 48
                            # attn_mask = attn_mask,# self_attn
                            need_weights=False
                        )[0])
                # attend_features = self.smh_fc(torch.cat(attend_features, dim=-1))
                attend_features = torch.cat(attend_features, dim=-1)
                # ensure zeros still zeros
                attend_features = attend_features * (~query_mask).permute(1,0).unsqueeze(-1).type(torch.float)
                attend_features = self.drop_out(attend_features) # 0  (48, M, C)
            
        # update raw tensor after self_attn
        # TODO(lihe): reimplement with cuda op
        # the reason of not using python implementation is the -1 returns
        # but we can use some trick to implement it with python
        
        bs = k_bs_cnt.shape[0] # (batch_size,)
        N, C = sp_tensor.features.shape[:2]
        padding = torch.zeros(1, C, device=sp_tensor.features.device)
        self.use_feature_interpolation = True
        if (not self.use_relative_position_encoding) and (not self.use_pseudo_relative_position_encoding):
            attend_features = attend_features.float().permute(1,0,2).contiguous()
            if self.use_feature_interpolation:
                know_features_for_inter = attend_features.permute(0,2,1).contiguous() # (win_num, C, nsample)
            attend_features = attend_features.reshape([-1,C]) # (size, M, C) --> (M, size, C) --> (-1, C)
        else:
            if self.use_feature_interpolation:
                know_features_for_inter = attend_features.float().permute(0,2,1).contiguous() # (win_num, C, nsample)
            attend_features = attend_features.reshape([-1,C])
        # attend_features = q_features.permute(0,2,1).contiguous().reshape([-1,C]) # NOTE

        # point interpolation
        if self.use_feature_interpolation:
            know_coords_for_inter = query_coords.float().permute(0,2,1).contiguous() # (win_num, nsample, 3)
            unknow_coords_for_inter = key_coords_in_all.float().permute(0,2,1).contiguous() # (win_num, nsample_unknown, 3)
            dists, idx = pointnet2_utils.three_nn(unknow_coords_for_inter, know_coords_for_inter) # (win_num, nsample_unknown, 3)
            dists = torch.clamp(dists, min=1e-10)
            weight = 1.0 / dists
            weight = weight / torch.sum(weight, -1, keepdim=True)
            features_in_all = torch.sum(
                pointnet2_utils.grouping_operation(know_features_for_inter, idx) * weight.unsqueeze(1), dim=-1) # (win_num, C, nsample_unknown)
            features_in_all = features_in_all.permute(0,2,1).contiguous().view(-1,C) # (win_num*nsample_unknown, C)

        v_cnt = 0
        k_cnt = 0
        # BUG: grad bug!
        features = sp_tensor.features.clone()
        for v_batch_num, index_batch_num in zip(v_bs_cnt, k_bs_cnt):
            select_v = sp_tensor.features[v_cnt:v_cnt + v_batch_num]
            select_v = torch.cat([select_v, padding], dim=0) # padding -1 index for each batch
            if self.use_feature_interpolation:
                select_feat_in_all = features_in_all[k_cnt*nsample_in:(k_cnt + index_batch_num)*nsample_in]
                select_update_index_in_all = key_indices_in[k_cnt:k_cnt+index_batch_num].reshape([-1]).type(torch.long)
                select_v[select_update_index_in_all] = select_feat_in_all
            else:
                select_feat = attend_features[k_cnt*nsample:(k_cnt + index_batch_num)*nsample]
                select_update_index = query_indices[k_cnt:k_cnt+index_batch_num].reshape([-1]).type(torch.long) # key_indices:(M, nsample)
                # update
                select_v[select_update_index] = select_feat
            # sp_tensor.features[v_cnt:v_cnt + v_batch_num] = select_v[:-1]
            features[v_cnt:v_cnt + v_batch_num] = select_v[:-1]
            v_cnt += v_batch_num
            k_cnt += index_batch_num
        
        # BUG
        # print("DEBUG++++++++:", (sp_tensor.features - features).sum())
        sp_tensor.features = features
        

        assert sp_tensor.features.shape[0] == N

        new_features = sp_tensor.features
        new_features = self.drop_path(new_features) + voxel_features_short_cut # add residual connection
        # new_features = self.norm2(new_features)
        act_features = self.linear2(self.dropout1(self.activation(self.linear1(self.norm2(new_features)))))
        new_features = new_features + self.drop_path(self.dropout1(act_features))
        # new_features = new_features.permute(1, 2, 0).contiguous() # (size, M, C) --> (M, C, size) self_attn
        # new_features = self.norm2(new_features)
        # new_features = self.output_layer(new_features)

        sp_tensor.features = new_features

        sp_tensor.gather_dict = None

        # print("finish one block")

        return sp_tensor, pv_set, reuse_dict

class SSTAttentionResBlock(nn.Module):
    def __init__(self, model_cfg, use_relative_coords = False, use_pooled_feature = False, use_no_query_coords = False, 
                       hash_size=None, spatial_shape=None, 
                       max_num_points=5, point_cloud_range=[0, -40, -3, 70.4, 40, 1], voxel_size=[0.1, 0.1, 0.1], max_voxels=(16000, 40000),
                       use_point_branch=False, out_layer=False, use_point_branch_debug=False, use_amp=False):
        super(SSTAttentionResBlock, self).__init__()
        sp_cfg = model_cfg.get('SP_CFGS', None)
        self.hash_size = hash_size
        self.spatial_shape = spatial_shape
        self.use_point_branch = use_point_branch
        self.use_point_branch_debug = use_point_branch_debug
        # in every block we need to revoxelize the points at least once
        if use_point_branch:
            self.voxel_layer = Voxelization(max_num_points=max_num_points,
                                            point_cloud_range=point_cloud_range,
                                            voxel_size=voxel_size,
                                            max_voxels=max_voxels) if self.use_point_branch else None# max voxels can be optimized
        if sp_cfg is not None:
            self.sp_attention = SparseAttention3d(
                input_channels = sp_cfg.CHANNELS[0],
                output_channels = sp_cfg.CHANNELS[2],
                ff_channels = sp_cfg.CHANNELS[1],
                dropout = sp_cfg.DROPOUT,#0
                num_heads = sp_cfg.NUM_HEADS,
                attention_modes = sp_cfg.ATTENTION,#StrideAttention
                strides = sp_cfg.STRIDE,#[2,2,2]
                num_ds_voxels = sp_cfg.NUM_DS_VOXELS,#90000
                use_relative_coords = use_relative_coords,
                use_pooled_feature = use_pooled_feature,
                use_no_query_coords= use_no_query_coords,
            )
        else:
            self.sp_attention = None
        if sp_cfg is not None:
            model_cfg.pop('SP_CFGS')
        # subm_cfg = model_cfg.SUBM_CFGS
        # self.points_embeding_ch = subm_cfg.POINTS_CHANNELS[1]
        # self.use_relative_position_encoding = subm_cfg.USE_RELATIVE_POSITION_ENCODING
        self.subm_attention_modules = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, 0.3, len(model_cfg.items()))]

        for i, (k, subm_cfg) in enumerate(model_cfg.items()):
            for j in range(subm_cfg.NUM_BLOCKS):
                self.subm_attention_modules.append(SwinSparseAttention3dGroup(
                    input_channels = subm_cfg.CHANNELS[0],
                    output_channels = subm_cfg.CHANNELS[2],
                    ff_channels = subm_cfg.CHANNELS[1],
                    dropout = subm_cfg.DROPOUT,#0
                    drop_path = dpr[i],
                    num_heads = subm_cfg.NUM_HEADS,
                    attention_modes = subm_cfg.ATTENTION,#LocalAttn
                    strides = subm_cfg.WINDOWSIZE,#[2,2,2]
                    num_ds_voxels = subm_cfg.NUM_DS_VOXELS,#90000
                    use_relative_coords = use_relative_coords,
                    use_pooled_feature = use_pooled_feature,
                    use_no_query_coords= use_no_query_coords,
                    shifts=subm_cfg.SHIFTS,
                    use_point_branch=self.use_point_branch,
                    use_point_branch_debug=self.use_point_branch_debug,
                    use_relative_position_encoding=subm_cfg.USE_RELATIVE_POSITION_ENCODING,
                    use_amp=use_amp,
                    use_pseudo_relative_position_encoding=subm_cfg.get('USE_PSEUDO_RELATIVE_POSITION_ENCODING', False),
                    use_separte_multi_head=subm_cfg.get('USE_SEPARTE_MULTI_HEAD', False)))
        
        # self.embed_layer = nn.Sequential(nn.Conv1d(subm_cfg.POINTS_CHANNELS[0], subm_cfg.POINTS_CHANNELS[1] - 3, 1), \
        #                                    nn.BatchNorm1d(subm_cfg.POINTS_CHANNELS[1] - 3), nn.ReLU())
        # self.point_branch_proj = nn.Sequential(nn.Conv1d(subm_cfg.CHANNELS[2], subm_cfg.CHANNELS[2], 1), \
        #                                    nn.BatchNorm1d(subm_cfg.CHANNELS[2]), nn.ReLU())
        # self.fusion_layer = nn.Sequential(nn.Linear(2 * subm_cfg.CHANNELS[2], subm_cfg.CHANNELS[2]), \
        #                                    nn.BatchNorm1d(subm_cfg.CHANNELS[2]), nn.ReLU())
    
    @torch.no_grad()
    def generate_pv_set(self, batch_size, bs_cnt, points):
        points_list = []
        voxels, coors, num_points = [], [], []
        count = 0
        for i in range(batch_size):
            cur_points = points[count:count+bs_cnt[i], 1:]
            cur_points = cur_points.contiguous() # N, C
            points_list.append(cur_points)
            res_voxels, res_coors, res_num_points = self.voxel_layer(cur_points)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
            count += bs_cnt[i]
        return voxels, coors, num_points
    
    @torch.no_grad()
    def get_bs_cnt(self, indices, batch_size):
        bs_cnt = torch.zeros(batch_size).int()
        for i in range(batch_size):
            bs_cnt[i] = (indices[:, 0] == i).sum().item()
        bs_cnt = bs_cnt.to(indices.device)
        return bs_cnt

    def forward(self, sp_tensor, points, idx=None, batch_size=None):
        # points bs_idx, x, y, z, intensity
        # voxelize points and build hash table
        pv_set = None
        v2p_dense_map = None
        pv_set_mask = None
        v_bs_cnt = None
        points_bs_cnt = self.get_bs_cnt(points, batch_size) if points is not None else None
        t1 = time.time()
        if self.sp_attention is not None:
            sp_tensor = self.sp_attention(sp_tensor)
        t2 = time.time()
        # print(" downsampling attention : {}s".format(t2 - t1))
        indentity_features = sp_tensor.features
        reuse_dict= None
        for i, subm_module in enumerate(self.subm_attention_modules):
            # only feed points in the last subm module
            sp_tensor, pv_set, reuse_dict = subm_module(sp_tensor, pv_set, v2p_dense_map, pv_set_bs_cnt=v_bs_cnt, pv_set_mask=pv_set_mask, stage_idx=idx, \
                                                points=points if i == len(self.subm_attention_modules) -1 else None,
                                                points_bs_cnt=points_bs_cnt if i == len(self.subm_attention_modules) -1 else None,
                                                reuse_dict=reuse_dict if i > 1 else None)
            if DEBUG_MODE and self.use_point_branch:
                np.savez(f'lihe_debug/pv_set_{idx}_subm{i}.npz', coors_batch=coors_batch.cpu().detach().numpy(), pv_set=pv_set.cpu().detach().numpy())
            #     # ATTENTION !!!!!! next line
        if len(self.subm_attention_modules) > 0: # only use residual connection when subm is not empty
            sp_tensor.features = sp_tensor.features + indentity_features
        t3 = time.time()
        # print(" subm attention : {}s".format(t3 - t2))

        return sp_tensor, points

class SlideWindowSSTDetectorDilatedSMH(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, use_conv_input=False, use_conv_output=False):
        super(SlideWindowSSTDetectorDilatedSMH, self).__init__()
        self.model_cfg = model_cfg
        self.input_channels = input_channels
        self.hash_size = model_cfg.get('HASH_SIZE', None)

        self.use_relative_coords = self.model_cfg.get('USE_RELATIVE_COORDS', False)#True
        self.use_pooled_feature = self.model_cfg.get('USE_POOLED_FEATURE', False)#True
        self.use_no_query_coords = self.model_cfg.get('USE_NO_QUERY_COORDS', False)#True
        # point-branch params
        self.max_num_points_list = self.model_cfg.get('MAX_NUM_POINTS', [3, 3, 10])
        self.num_point_features = self.model_cfg.NUM_OUTPUT_FEATURES #64
        self.use_point_branch = self.model_cfg.get('USE_POINT_BRANCH', False)
        self.use_point_branch_debug = self.model_cfg.get('USE_POINT_BRANCH_DEBUG', False)

        self.grid_size = grid_size #[1408, 1600, 40]
        self.voxel_size = voxel_size #[0.05, 0.05, 0.1]
        self.point_cloud_range = point_cloud_range #[0, -40, -3, 70.4, 40, 1]
        self.use_conv_input = use_conv_input
        self.use_conv_output = use_conv_output
        # self.input_transform = nn.Sequential(
        #     nn.Linear(input_channels, 16),# 4 --> 16
        #     nn.BatchNorm1d(16),
        #     nn.ReLU()
        # )
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        if self.use_conv_input:
            self.conv_input = spconv.SparseSequential(
                spconv.SubMConv3d(input_channels, input_channels, 3, padding=1, bias=False, indice_key='subm1'),
                norm_fn(input_channels),
                nn.ReLU(),
            )

            block = post_act_block

            self.conv1 = spconv.SparseSequential(
                block(input_channels, input_channels, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
            )

        if self.use_conv_output:
            last_pad = 0
            last_pad = self.model_cfg.get('last_pad', last_pad)
            self.conv_out = spconv.SparseSequential(
                # [200, 150, 5] -> [200, 150, 2]
                # [200, 150, 10] -> [200, 150, 2]
                spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                    bias=False, indice_key='spconv_down2'),
                norm_fn(128),
                nn.ReLU(),
            )

        self.backbone = nn.ModuleList()
        num_layers = len(self.model_cfg.PARAMS)
        for i, param in enumerate(self.model_cfg.PARAMS):
            # cur_block_voxel_size = [v * 2**(i + 1) for v in self.voxel_size]
            # cur_spatial_shape = [s // 2**(i+1) for s in self.grid_size]
            cur_block_voxel_size = self.voxel_size
            cur_spatial_shape = self.grid_size
            self.backbone.append(SSTAttentionResBlock(param, self.use_relative_coords, self.use_pooled_feature, self.use_no_query_coords, \
                hash_size=self.hash_size,
                spatial_shape=cur_spatial_shape,
                max_num_points=self.max_num_points_list[i], point_cloud_range=point_cloud_range, voxel_size=cur_block_voxel_size,
                use_point_branch=self.use_point_branch, out_layer=False if i < num_layers - 1 else True, 
                use_point_branch_debug=self.use_point_branch_debug,
                use_amp=self.model_cfg.AMP))

        self.num_point_features = self.model_cfg.NUM_OUTPUT_FEATURES #64

    @torch.no_grad()
    def rm_ground(self, points, batch_size, npoint=None, flatten=True):
        from pcdet.models.model_utils.ran_sec import rm_ground
        fore_ground_list = []
        for i in range(batch_size):
            cur_points = points[points[:, 0]==i]
            fore_ground_mask = rm_ground(cur_points[:, 1:4])
            fore_points = cur_points[fore_ground_mask]
            # fps downsample
            if npoint is not None:
                sampled_index = pointnet2_utils.furthest_point_sample(fore_points[:, 1:4].unsqueeze(0).contiguous(), 8192)
                fore_points = fore_points[sampled_index.squeeze(0).type(torch.long)] # [npoints, C]
            assert len(fore_points) > 0 
            fore_ground_list.append(fore_points)
        if flatten:
            points = torch.cat(fore_ground_list, dim=0)
        else:
            points = torch.stack([fore_ground_list])# [b,n,3]
        return points

    def forward(self, batch_dict):
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        points = None
        if self.use_point_branch or self.use_point_branch_debug:
            points = batch_dict['points']
            if DEBUG_MODE:
                np.save('lihe_debug/points_before_rm_ground.npy', points.cpu().detach().numpy()) # bs, x, y, z
            points = self.rm_ground(points, batch_size, npoint=8192)
        # points = self.rm_ground(points, batch_size)
        # save points to check ransec
        if DEBUG_MODE:
            if self.use_point_branch:
                np.save('lihe_debug/points_rm_ground_check.npy', points.cpu().detach().numpy()) # bs, x, y, z
            np.save('lihe_debug/slide_window/voxel_input_check.npy', voxel_coords.cpu().detach().numpy()) # bs z y x
        if self.use_conv_input:
            input_sp_tensor = spconv.SparseConvTensor(
                features=voxel_features,
                indices=voxel_coords.int(),
                spatial_shape=np.array(self.grid_size, dtype=np.int)[::-1] + [1, 0, 0],
                batch_size=batch_size
            )

            # voxel_features = self.input_transform(voxel_features)
            embed_sp_tensor = self.conv1(self.conv_input(input_sp_tensor))
            sp_tensor = SparseTensor(
                features = embed_sp_tensor.features,
                indices = embed_sp_tensor.indices,
                spatial_shape = self.grid_size,
                voxel_size = self.voxel_size,
                point_cloud_range = self.point_cloud_range,
                batch_size = batch_size,
                hash_size = self.model_cfg.HASH_SIZE,#400,000
                map_table = None,
                gather_dict = None,)
        else:
            sp_tensor = SparseTensor(
                features = voxel_features,
                indices = voxel_coords.int(),
                spatial_shape = self.grid_size,
                voxel_size = self.voxel_size,
                point_cloud_range = self.point_cloud_range,
                batch_size = batch_size,
                hash_size = self.model_cfg.HASH_SIZE,#400,000
                map_table = None,
                gather_dict = None,)

        for i, attention_block in enumerate(self.backbone):
            t1 = time.time()
            sp_tensor, points = attention_block(sp_tensor, points, idx=i, batch_size=batch_size)
            t2 = time.time()
            # print("========block {} time : {}s ========".format(i, t2 - t1))
            # debug check the points after attention block
            if DEBUG_MODE and self.use_point_branch:
                np.save('lihe_debug/points_{}.npy'.format(i), points.cpu().detach().numpy())# points [:, 1+3+61]

        if self.use_conv_output:
            out_tensor = spconv.SparseConvTensor(
                features=sp_tensor.features,
                indices=sp_tensor.indices,
                spatial_shape=np.array(sp_tensor.spatial_shape, dtype=np.int)[::-1],
                batch_size=batch_size
            )
            out_tensor = self.conv_out(out_tensor)
        else:
            out_tensor = sp_tensor
        # exit()

        batch_dict.update({
            'encoded_spconv_tensor': out_tensor,
            'encoded_spconv_tensor_stride': 1
        })
        return batch_dict
