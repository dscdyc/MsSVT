import torch
import torch.nn as nn

try:
    import torch_scatter
except Exception as e:
    # Incase someone doesn't want to use dynamic pillar vfe and hasn't installed torch_scatter
    pass

from .vfe_template import VFETemplate


class DynamicVFE(VFETemplate):
    def __init__(self,
                 model_cfg,
                 num_point_features,
                 voxel_size,
                 grid_size,
                 point_cloud_range,
                 **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features_in = num_point_features

        self.grid_size = torch.tensor(grid_size).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]
        self.xyz_offset = torch.tensor([self.x_offset, self.y_offset, self.z_offset]).float().cuda().view(1, 3)

        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_z = grid_size[2]

        with_cluster_center = model_cfg.get('WITH_CLUSTER_CENTER', True)
        with_voxel_center = model_cfg.get('WITH_VOXEL_CENTER', True)
        with_distance = model_cfg.get('WITH_DISTANCE', False)
        in_channels = num_point_features
        if with_cluster_center:
            in_channels = in_channels + 3
        if with_voxel_center:
            in_channels = in_channels + 3
        if with_distance:
            in_channels = in_channels + 1
        self.in_channels = in_channels
        self.with_cluster_center = with_cluster_center
        self.with_voxel_center = with_voxel_center
        self.with_distance = with_distance

        filters = model_cfg.get('NUM_FILTERS', [64, 128])
        self.num_point_features = filters[-1]
        
        self.pfn = nn.ModuleList([])
        in_c = in_channels
        for out_c in filters:
            self.pfn.append(nn.Sequential(
                nn.Linear(in_c, out_c),
                nn.BatchNorm1d(out_c),
                nn.ReLU(inplace=True)
            ))
            in_c = out_c * 2

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        batch_size = batch_dict['batch_size']
        points = batch_dict['points'] # (batch_idx, x, y, z, i, e)

        point_coords = torch.floor((points[:, 1:4] - self.point_cloud_range[0:3]) / self.voxel_size).int()
        mask = ((point_coords >= 0) & (point_coords < self.grid_size)).all(dim=1)
        points = points[mask]
        point_coords = point_coords[mask]
        merge_coords = points[:, 0].int() * self.scale_xyz + \
                        point_coords[:, 0] * self.scale_yz + \
                        point_coords[:, 1] * self.scale_z + \
                        point_coords[:, 2]
        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True)

        xyz = points[:, 1:4]
        features_ls = [points[:, 1:self.num_point_features_in+1]]
        if self.with_cluster_center:
            xyz_mean = torch_scatter.scatter_mean(xyz, unq_inv, dim=0)
            f_cluster = xyz - xyz_mean[unq_inv]
            features_ls.append(f_cluster)
        if self.with_voxel_center:
            v_center = point_coords * self.voxel_size + self.xyz_offset
            f_center = xyz - v_center
            features_ls.append(f_center)
        if self.with_distance:
            points_dist = torch.norm(xyz, p=2, dim=2, keepdim=True)
            features_ls.append(points_dist)
        points_fea = torch.cat(features_ls, dim=-1)

        points_fea = self.get_points_fea(points_fea, unq_inv)
        voxel_fea = torch_scatter.scatter_max(points_fea, unq_inv, dim=0)[0]
        
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xyz,
                                    (unq_coords % self.scale_xyz) // self.scale_yz,
                                    (unq_coords % self.scale_yz) // self.scale_z,
                                    unq_coords % self.scale_z), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]
        
        batch_dict['voxel_features'] = voxel_fea.contiguous()
        batch_dict['voxel_coords'] = voxel_coords.contiguous()
        return batch_dict
    
    def get_points_fea(self, points, unq_inv):
        for i, blk in enumerate(self.pfn):
            points = blk(points)
            if i < len(self.pfn)-1:
                fea_v = torch_scatter.scatter_max(points, unq_inv, dim=0)[0]
                points = torch.cat((points, fea_v[unq_inv]), dim=-1)
        return points

