import torch
from torch import nn

from timm.models.layers import DropPath

from ..model_utils.mssvt_utils import MixedScaleAttention, SparseTensor
from ...ops.pointnet2.pointnet2_batch import pointnet2_utils
from ...ops.mssvt import mssvt_ops


class MixedScaleSparseTransformerBlock(nn.Module):
    def __init__(self, 
                 cfg,
                 in_channels, 
                 ff_channels, 
                 out_channels, 
                 num_heads,
                 dropout=0.,
                 drop_path=None,
                 window_size=None,
                 max_num_win1=None,
                 max_num_win2=None,
                 cbs_mode='odd_even',
                 cbs_pattern=1,
                 key_num_sample=32,
                 use_feature_interpolation=True,
                 ):
        super().__init__()
        self.cfg = cfg
        self.ms_attn = MixedScaleAttention(
            embed_dim=in_channels, num_heads=num_heads, dropout=dropout)
        
        self.linear1 = nn.Linear(in_channels, ff_channels)
        self.linear2 = nn.Linear(ff_channels, in_channels)
        if out_channels != in_channels:
            self.out_linear = nn.Linear(in_channels, out_channels)
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.activation = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if len(window_size) == 2:
            self.pos_proj = nn.Sequential(
                nn.Conv1d(6, in_channels, 1),
                nn.ReLU(),
            )
        else:
            self.pos_proj = nn.Sequential(
                nn.Conv1d(6, in_channels, 1),
                nn.ReLU(),
                nn.Conv1d(in_channels, in_channels, 1),
                nn.ReLU(),
            )
        self.key_num_sample = key_num_sample
        self.max_num_wins = 90000
        self.use_feature_interpolation = use_feature_interpolation

        # window setup
        self.cbs_mode = cbs_mode
        self.cbs_pattern = cbs_pattern
        assert len(window_size) <= 2
        self.window_size = window_size
        self.win1_size = window_size[0]
        self.max_num_win1 = self.win1_size[0] * self.win1_size[1] * self.win1_size[2] if max_num_win1 is None else max_num_win1
        if len(window_size) == 2:
            self.win2_size = window_size[1]
            self.max_num_win2 = self.win2_size[0] * self.win2_size[1] * self.win2_size[2] if max_num_win2 is None else max_num_win2
        else:
            self.win2_size, self.max_num_win2 = None, None
        self.vox_query_table, self.max_num_odd, self.max_num_even = self.get_vox_query_table(self.win1_size, self.win2_size, self.cbs_mode)

    def get_vox_query_table(self, win1_size, win2_size=None, cbs_mode=None):
        if win2_size is not None:
            assert 1 not in [(win2_size[i] - win1_size[i]) % 2 for i in range(3)]
            x, y, z = torch.arange(win2_size[0]), torch.arange(win2_size[1]), torch.arange(win2_size[2])
            center = torch.tensor(win2_size, device='cuda').view(1, 3) // 2
        else:
            x, y, z = torch.arange(win1_size[0]), torch.arange(win1_size[1]), torch.arange(win1_size[2])
            center = torch.tensor(win1_size, device='cuda').view(1, 3) // 2
        x, y, z = torch.meshgrid(x, y, z)
        xyz = torch.stack([x, y, z], dim=-1).cuda().view(-1, 3)
        xyz = xyz - center
        distance, _ = torch.max(torch.abs(xyz), dim=-1)
        sorted_distance, sorted_indices = torch.sort(distance)
        xyz = xyz[sorted_indices]

        if win2_size is None:
            vox_query_table = {
                'win1': xyz.int(),
            }
            return vox_query_table, None, None

        offset = [1 - win1_size[i] % 2 for i in range(3)]
        mask_win1 = (xyz[:, 0] <= (win1_size[0] // 2 + offset[0])) & (xyz[:, 0] >= -(win1_size[0] // 2)) & \
                    (xyz[:, 1] <= (win1_size[1] // 2 + offset[1])) & (xyz[:, 1] >= -(win1_size[1] // 2)) & \
                    (xyz[:, 2] <= (win1_size[2] // 2 + offset[2])) & (xyz[:, 2] >= -(win1_size[2] // 2))
        xyz_win1 = xyz[mask_win1]
        xyz_win2_other = xyz[~mask_win1]

        if cbs_mode == 'odd_even':
            mask_odd = (xyz_win1[:, 0] % 2 == 1) & (xyz_win1[:, 1] % 2 == 1)
            mask_even = (xyz_win1[:, 0] % 2 == 0) & (xyz_win1[:, 1] % 2 == 0)
            xyz_odd = xyz_win1[mask_odd]
            xyz_even = xyz_win1[mask_even]
            xyz_win1_other = xyz_win1[~(mask_odd | mask_even)]
        else:
            raise NotImplementedError
        
        max_num_odd = xyz_odd.shape[0]
        max_num_even = xyz_even.shape[0]
        # max_num_win1 = xyz_win1.shape[0]
        # max_num_win2 = xyz.shape[0]

        vox_query_table = {
            'odd': xyz_odd.int(),
            'even': xyz_even.int(),
            'win1': xyz_win1_other.int(),
            'win2': xyz_win2_other.int()
        }

        return vox_query_table, max_num_odd, max_num_even
    
    @torch.no_grad()
    def with_bs_cnt(self, indices, batch_size):
        bs_cnt = torch.zeros(batch_size).int()
        for i in range(batch_size):
            bs_cnt[i] = (indices[:, 0] == i).sum().item()
        bs_cnt = bs_cnt.to(indices.device)
        return bs_cnt
    
    @torch.no_grad()
    def with_coords(self, indices, point_cloud_range, voxel_size):
        voxel_size = torch.tensor(voxel_size).unsqueeze(0).to(indices.device)
        min_range = torch.tensor(point_cloud_range[0:3]).unsqueeze(0).to(indices.device)
        coords = (indices[:, [3, 2, 1]].float() + 0.5) * voxel_size + min_range
        return coords
    
    def window_partition(self, sp_tensor):
        x_shape = sp_tensor.spatial_shape[0] // self.win1_size[0]
        y_shape = sp_tensor.spatial_shape[1] // self.win1_size[1]
        z_shape = sp_tensor.spatial_shape[2] // self.win1_size[2]
        new_spatial_shape = [x_shape, y_shape, z_shape]

        center_indices, new_map_table = mssvt_ops.get_non_empty_window_center(
            self.win1_size, 
            self.max_num_wins, 
            sp_tensor.batch_size, 
            sp_tensor.hash_size, 
            new_spatial_shape, 
            sp_tensor.indices)
        return new_spatial_shape, center_indices, new_map_table
    
    def mixed_scale_vox_sample(self, sp_tensor, win_ind):
        if len(self.window_size) == 1:
            vox_query_win1 = self.vox_query_table['win1']
            vox_ind_win1, vox_coord_win1 = mssvt_ops.gather_one_window_voxels(
                sp_tensor.spatial_shape, self.win1_size, self.max_num_win1, vox_query_win1, win_ind, sp_tensor.map_table)
            
            vox_mask_win1 = vox_ind_win1 < 0

            out = {
                'vox_ind_win1': vox_ind_win1,
                'vox_mask_win1': vox_mask_win1,
                'vox_coord_win1': vox_coord_win1,
            }
            
        else:
            vox_query_odd = self.vox_query_table['odd']
            vox_query_even = self.vox_query_table['even']
            vox_query_win1 = self.vox_query_table['win1']
            vox_query_win2 = self.vox_query_table['win2']
            vox_ind_win1_odd, vox_ind_win1_even, vox_ind_win1, vox_ind_win2, \
            vox_coord_win1_odd, vox_coord_win1_even, vox_coord_win1, vox_coord_win2 = \
            mssvt_ops.gather_two_window_voxels(
                sp_tensor.spatial_shape, self.win1_size, self.max_num_odd, self.max_num_even, self.max_num_win1, \
                self.max_num_win2, vox_query_odd, vox_query_even, vox_query_win1, vox_query_win2, win_ind, \
                sp_tensor.map_table)

            vox_mask_win1_odd = vox_ind_win1_odd < 0
            vox_mask_win1_even = vox_ind_win1_even < 0
            vox_mask_win1 = vox_ind_win1 < 0
            vox_mask_win2 = vox_ind_win2 < 0

            out = {
                'vox_ind_win1_odd': vox_ind_win1_odd,
                'vox_ind_win1_even': vox_ind_win1_even,
                'vox_ind_win1': vox_ind_win1,
                'vox_ind_win2': vox_ind_win2,
                'vox_mask_win1_odd': vox_mask_win1_odd,
                'vox_mask_win1_even': vox_mask_win1_even,
                'vox_mask_win1': vox_mask_win1,
                'vox_mask_win2': vox_mask_win2,
                'vox_coord_win1_odd': vox_coord_win1_odd,
                'vox_coord_win1_even': vox_coord_win1_even,
                'vox_coord_win1': vox_coord_win1,
                'vox_coord_win2': vox_coord_win2
            }
        return out
    
    def forward(self, sp_tensor, block_idx=None, recycle_dict=None):

        if (recycle_dict is None) or (recycle_dict == {}):
            recycle_dict = {}
            recycle = False
        else:
            recycle = True
        
        voxel_features_short_cut = sp_tensor.features
        voxel_features = self.norm1(voxel_features_short_cut)

        if not recycle:
            new_spatial_shape, win_ind, _ = self.window_partition(sp_tensor)
            vx, vy, vz = sp_tensor.voxel_size
            win_size = [vx * self.win1_size[0], vy * self.win1_size[1], vz * self.win1_size[2]]
            vox_win_map_dict = self.mixed_scale_vox_sample(sp_tensor, win_ind)
            v_bs_cnt = self.with_bs_cnt(sp_tensor.indices, sp_tensor.batch_size) # [batch_size] count num of each batch
            k_bs_cnt = self.with_bs_cnt(win_ind, sp_tensor.batch_size)

            if self.cbs_mode == 'odd_even':
                if self.cbs_pattern == 0:
                    q_ind = vox_win_map_dict['vox_ind_win1_even'] 
                    q_mask = vox_win_map_dict['vox_mask_win1_even']
                    q_grid_coord = vox_win_map_dict['vox_coord_win1_even']
                elif self.cbs_pattern == 1:
                    q_ind = vox_win_map_dict['vox_ind_win1_odd'] 
                    q_mask = vox_win_map_dict['vox_mask_win1_odd']
                    q_grid_coord = vox_win_map_dict['vox_coord_win1_odd']
                elif self.cbs_pattern == 2:
                    q_ind = vox_win_map_dict['vox_ind_win1'] 
                    q_mask = vox_win_map_dict['vox_mask_win1']
                    q_grid_coord = vox_win_map_dict['vox_coord_win1']
            else:
                raise NotImplementedError
            
            n_sample_q = q_ind.shape[1]
            
            win1_ind = vox_win_map_dict['vox_ind_win1'] 
            win1_mask = vox_win_map_dict['vox_mask_win1']
            win1_grid_coord = vox_win_map_dict['vox_coord_win1'] # (nw,ns,c)
            n_sample_win1 = win1_grid_coord.shape[1]

            win2_ind = vox_win_map_dict['vox_ind_win2']
            win2_mask = vox_win_map_dict['vox_mask_win2']
            win2_grid_coord = vox_win_map_dict['vox_coord_win2']

            k_fps_ind_win1 = pointnet2_utils.farthest_point_sample(win1_grid_coord.float(), self.key_num_sample)
            k_fps_mask_win1 = k_fps_ind_win1 == 0
            k_fps_mask_win1[:, 0] = False
            k_fps_ind_win2 = pointnet2_utils.farthest_point_sample(win2_grid_coord.float(), self.key_num_sample)
            k_fps_mask_win2 = k_fps_ind_win2 == 0
            k_fps_mask_win2[:, 0] = False
            k_ind_win1 = win1_ind.unsqueeze(1).float()
            k_ind_win1 = (pointnet2_utils.gather_operation(k_ind_win1, k_fps_ind_win1).squeeze(1) + 0.1).int()
            k_ind_win2 = win2_ind.unsqueeze(1).float()
            k_ind_win2 = (pointnet2_utils.gather_operation(k_ind_win2, k_fps_ind_win2).squeeze(1) + 0.1).int()
            k_fps_mask_win1 = k_fps_mask_win1 | (k_ind_win1 < 0)
            k_fps_mask_win2 = k_fps_mask_win2 | (k_ind_win2 < 0)

            q_fea = mssvt_ops.grouping_operation(voxel_features, v_bs_cnt, q_ind, k_bs_cnt)
            k_fea_win1 = mssvt_ops.grouping_operation(voxel_features, v_bs_cnt, k_ind_win1, k_bs_cnt)
            k_fea_win2 = mssvt_ops.grouping_operation(voxel_features, v_bs_cnt, k_ind_win2, k_bs_cnt)

            voxel_coord = self.with_coords(sp_tensor.indices, sp_tensor.point_cloud_range, sp_tensor.voxel_size)
            q_coord = mssvt_ops.grouping_operation(voxel_coord, v_bs_cnt, q_ind, k_bs_cnt)
            win1_coord = mssvt_ops.grouping_operation(voxel_coord, v_bs_cnt, win1_ind, k_bs_cnt)
            k_coord_win1 = mssvt_ops.grouping_operation(voxel_coord, v_bs_cnt, k_ind_win1, k_bs_cnt)
            k_coord_win2 = mssvt_ops.grouping_operation(voxel_coord, v_bs_cnt, k_ind_win2, k_bs_cnt)
            win_center_coord = self.with_coords(win_ind, sp_tensor.point_cloud_range, win_size)

            k_coord_win1 = k_coord_win1 - win_center_coord.unsqueeze(-1) # (nw,3,nk)
            k_coord_win2 = k_coord_win2 - win_center_coord.unsqueeze(-1) # (nw,3,nk)
            q_coord_rel = q_coord - win_center_coord.unsqueeze(-1) # (nw,3,nq)
            k_coord_win1 = k_coord_win1 * (~k_fps_mask_win1).unsqueeze(1)
            k_coord_win2 = k_coord_win2 * (~k_fps_mask_win2).unsqueeze(1)
            q_coord_rel = q_coord_rel * (~q_mask).unsqueeze(1)

            q_pos_emb = self.pos_proj(torch.cat((q_coord_rel, win_center_coord.unsqueeze(-1).expand_as(q_coord_rel)), dim=1))
            k_coord = torch.cat([k_coord_win1, k_coord_win2], dim=-1)
            k_pos_emb = self.pos_proj(torch.cat((k_coord, win_center_coord.unsqueeze(-1).expand_as(k_coord)), dim=1))
            q_fea = q_fea + q_pos_emb
            k_fea = torch.cat([k_fea_win1, k_fea_win2], dim=-1) + k_pos_emb # (nw,c,nk)

            q_fea = q_fea.permute(0, 2, 1).contiguous() # (nw,nq,c)
            k_fea = k_fea.permute(0, 2, 1).contiguous() # (nw,nk,c)

            k_mask = torch.cat([k_fps_mask_win1, k_fps_mask_win2], dim=-1) # (nw,nk)

        attn_fea = self.ms_attn(
            query=q_fea, 
            keys=k_fea, 
            query_mask=q_mask, 
            key_masks=k_mask,
            batch_first=True,
        ) # (b,nq,c)

        # feature interpolation
        C = sp_tensor.features.shape[1]
        padding = torch.zeros(1, C, device=sp_tensor.features.device)
        if self.use_feature_interpolation:
            know_fea_win1 = attn_fea.float().permute(0, 2, 1).contiguous() # (nw,c,nq)
            know_coord_win1 = q_coord.float().permute(0, 2, 1).contiguous() # (nw,nq,3)
            unknow_coord_win1 = win1_coord.float().permute(0, 2, 1).contiguous() # (nw, nsample_unknown, 3)
            dists, idx = pointnet2_utils.three_nn(unknow_coord_win1, know_coord_win1) # (nw, nsample_unknown, 3)
            dists = torch.clamp(dists, min=1e-10)
            weight = 1.0 / dists
            weight = weight / torch.sum(weight, -1, keepdim=True)
            win1_fea = torch.sum(
                pointnet2_utils.grouping_operation(know_fea_win1, idx) * weight.unsqueeze(1), dim=-1) # (nw, C, nsample_unknown)
            win1_fea = win1_fea.permute(0, 2, 1).contiguous().view(-1,C) # (nw*nsample_unknown, C)
        attn_fea = attn_fea.reshape(-1, C)

        v_cnt = 0
        k_cnt = 0
        # if self.revision:
        # BUG: grad bug!
        features = sp_tensor.features.clone()
        # print('\nDEBUG 0: ', attn_fea.shape, features.shape, v_bs_cnt, k_bs_cnt, n_sample_win1)
        for v_batch_num, index_batch_num in zip(v_bs_cnt, k_bs_cnt):
            select_v = sp_tensor.features[v_cnt:v_cnt + v_batch_num]
            select_v = torch.cat([select_v, padding], dim=0) # padding -1 index for each batch
            if self.use_feature_interpolation:
                select_feat_in_all = win1_fea[k_cnt * n_sample_win1:(k_cnt + index_batch_num) * n_sample_win1]
                select_update_index_in_all = win1_ind[k_cnt:k_cnt+index_batch_num].reshape([-1]).type(torch.long)
                select_v[select_update_index_in_all] = select_feat_in_all
            else:
                select_feat = attn_fea[k_cnt * n_sample_q:(k_cnt + index_batch_num) * n_sample_q]
                select_update_index = q_ind[k_cnt:k_cnt+index_batch_num].reshape([-1]).type(torch.long) # key_indices:(M, nsample)
                # update
                select_v[select_update_index] = select_feat
            # sp_tensor.features[v_cnt:v_cnt + v_batch_num] = select_v[:-1]
            features[v_cnt:v_cnt + v_batch_num] = select_v[:-1]
            v_cnt += v_batch_num
            k_cnt += index_batch_num
        
        sp_tensor.features = features
        new_features = sp_tensor.features
        new_features = self.drop_path(new_features) + voxel_features_short_cut # add residual connection
        act_features = self.linear2(self.dropout1(self.activation(self.linear1(self.norm2(new_features)))))
        new_features = new_features + self.drop_path(self.dropout1(act_features))
        if hasattr(self, 'out_linear'):
            new_features = self.out_linear(new_features)
        sp_tensor.features = new_features
        sp_tensor.gather_dict = None

        return sp_tensor


class MixedScaleSparseTransformerCompressBlock(MixedScaleSparseTransformerBlock):
        
    def forward(self, sp_tensor, block_idx=None, recycle_dict=None):
        voxel_features = self.norm1(sp_tensor.features)

        new_spatial_shape, win_ind, new_map_table = self.window_partition(sp_tensor)
        vx, vy, vz = sp_tensor.voxel_size
        win_size = [vx * self.win1_size[0], vy * self.win1_size[1], vz * self.win1_size[2]]
        vox_win_map_dict = self.mixed_scale_vox_sample(sp_tensor, win_ind)
        v_bs_cnt = self.with_bs_cnt(sp_tensor.indices, sp_tensor.batch_size) # [batch_size] count num of each batch
        k_bs_cnt = self.with_bs_cnt(win_ind, sp_tensor.batch_size)

        k_ind = vox_win_map_dict['vox_ind_win1'] 
        k_mask = vox_win_map_dict['vox_mask_win1']
        k_grid_coord = vox_win_map_dict['vox_coord_win1']
        
        k_fea = mssvt_ops.grouping_operation(voxel_features, v_bs_cnt, k_ind, k_bs_cnt)
        voxel_coord = self.with_coords(sp_tensor.indices, sp_tensor.point_cloud_range, sp_tensor.voxel_size)
        k_coord = mssvt_ops.grouping_operation(voxel_coord, v_bs_cnt, k_ind, k_bs_cnt)

        q_coord = self.with_coords(win_ind, sp_tensor.point_cloud_range, win_size)
        q_fea = k_fea.max(dim=-1)[0]
        q_fea = q_fea.unsqueeze(0) # (1,nw,c)
        k_coord = k_coord - q_coord.unsqueeze(-1) # (nw,3,ns)
        k_pos_emb = self.pos_proj(torch.cat((k_coord, q_coord.unsqueeze(-1).expand_as(k_coord)), dim=1))
        k_fea = k_fea + k_pos_emb
        k_fea = k_fea.permute(2, 0, 1).contiguous()

        attn_fea = self.ms_attn(
            query=q_fea, 
            keys=k_fea, 
            key_masks=k_mask
        ) # (1,nw,c)

        new_features = attn_fea.squeeze(0)
        act_features = self.linear2(self.dropout1(self.activation(self.linear1(self.norm2(new_features)))))
        new_features = new_features + self.dropout1(act_features)
        if hasattr(self, 'out_linear'):
            new_features = self.out_linear(new_features)
        # update sp_tensor
        sp_tensor.features = new_features
        sp_tensor.indices = win_ind
        sp_tensor.spatial_shape = new_spatial_shape
        sp_tensor.voxel_size = win_size

        del sp_tensor.map_table
        sp_tensor.gather_dict = None
        sp_tensor.map_table = new_map_table

        return sp_tensor


class MixedScaleSparseTransformer(nn.Module):

    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range):
        super().__init__()
        self.model_cfg = model_cfg
        self.input_channels = input_channels
        self.grid_size = grid_size #[1408, 1600, 40]
        self.voxel_size = voxel_size #[0.05, 0.05, 0.1]
        self.point_cloud_range = point_cloud_range #[0, -40, -3, 70.4, 40, 1]
        
        self.hash_size = model_cfg.get('HASH_SIZE', None)

        self.backbone = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, 0.3, len(model_cfg.PARAMS) - 1)]
        for i, param in enumerate(self.model_cfg.PARAMS):
            in_channels, ff_channels, out_channels = param.channels
            if param.name == 'MixedScaleSparseTransformerBlock':
                block = MixedScaleSparseTransformerBlock(
                    cfg=param,
                    in_channels=in_channels,
                    ff_channels=ff_channels,
                    out_channels=out_channels, 
                    num_heads=param.num_heads,
                    drop_path=dpr[i],
                    window_size=param.window_size,
                    max_num_win1=param.max_num_win1,
                    max_num_win2=param.max_num_win2,
                    cbs_mode=param.cbs_mode,
                    cbs_pattern=param.cbs_pattern,
                    key_num_sample=param.key_num_sample,
                    use_feature_interpolation=param.use_feature_interpolation,
                    )
            elif param.name == 'MixedScaleSparseTransformerCompressBlock':
                block = MixedScaleSparseTransformerCompressBlock(
                    cfg=param,
                    in_channels=in_channels,
                    ff_channels=ff_channels,
                    out_channels=out_channels, 
                    num_heads=param.num_heads,
                    drop_path=0.,
                    window_size=param.window_size,
                    max_num_win1=param.max_num_win1,
                    )
            else:
                raise NotImplementedError
            self.backbone.append(block)
        self.num_point_features = model_cfg.NUM_OUTPUT_FEATURES


    def forward(self, batch_dict):
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        sp_tensor = SparseTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.grid_size,
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range,
            batch_size=batch_size,
            hash_size=self.hash_size,#400,000
            map_table=None,
            gather_dict=None)
        
        for i, attention_block in enumerate(self.backbone):
            sp_tensor = attention_block(sp_tensor, block_idx=i)

        out_tensor = sp_tensor
        batch_dict.update({
            'encoded_spconv_tensor': out_tensor,
            'encoded_spconv_tensor_stride': 1
        })
        return batch_dict
