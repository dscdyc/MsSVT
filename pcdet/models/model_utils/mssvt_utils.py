import torch
from torch import nn
from ...ops.mssvt import mssvt_ops


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
        map_table = mssvt_ops.build_hash_table(
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


class MixedScaleAttention(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads # List
        self.num_head_groups = len(num_heads)
        self.tot_num_heads = sum(num_heads)
        assert self.embed_dim % self.tot_num_heads == 0
        self.per_head_dim = self.embed_dim // self.tot_num_heads
        self.group_c_idx = [self.per_head_dim * sum(num_heads[:i+1]) for i in range(self.num_head_groups)]
        self.scale_dims = [self.per_head_dim * h for h in num_heads]

        self.to_qs = nn.ModuleList([nn.Linear(sd, sd) for sd in self.scale_dims])
        self.to_kvs = nn.ModuleList([nn.Linear(sd, 2 * sd) for sd in self.scale_dims])
        self.scale = self.per_head_dim ** -0.5
        self.attn_drop = nn.Dropout(dropout)
        self.projs = nn.ModuleList([nn.Linear(sd, sd) for sd in self.scale_dims])
        self.proj_drop = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self,
                query,
                keys,
                batch_first=False,
                pos_emb_mat=None, 
                relative_coords=None,
                relative_position_bias=None,
                query_mask=None, 
                key_masks=None,
                need_weights=False):
        if batch_first:
            b, nq, _ = query.shape
            b, tot_nk, _ = keys.shape
        else:
            nq, b, _ = query.shape
            tot_nk, b, _ = keys.shape
            query = query.transpose(1, 0)
            keys = keys.transpose(1, 0)
        nk = tot_nk // self.num_head_groups
        # print('\nDEBUG 2:', query.shape, batch_first, self.group_c_idx)

        attn_weights = []
        attn_features = []
        start_c = 0
        for i in range(self.num_head_groups):
            end_c = self.group_c_idx[i]
            q = self.to_qs[i](query[:, :, start_c:end_c]) # (b,nq,c)
            q = q.reshape(b, nq, self.num_heads[i], self.per_head_dim).permute(0, 2, 1, 3) # (b,nh,nq,ch)

            kv = self.to_kvs[i](keys[:, i*nk:(i+1)*nk, start_c:end_c]) # (b,nk,c)
            kv = kv.reshape(b, nk, 2, self.num_heads[i], self.per_head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1] # (b,nh,nk,ch)

            start_c = end_c

            q = q * self.scale
            attn = (q @ k.transpose(-2, -1)) # (b,nh,nq,nk)

            if relative_position_bias is not None:
                attn = attn + relative_position_bias[i]
            
            if key_masks is not None:
                key_mask = key_masks[:, i*nk:(i+1)*nk]
                key_mask = key_mask.float().masked_fill(key_mask != 0, float(-100.0)) # (b,nk)
                key_mask = key_mask.view(b, 1, 1, nk)
                attn = attn.view(b, self.num_heads[i], nq, nk) + key_mask
                attn = self.softmax(attn)

            attn = self.attn_drop(attn)

            if batch_first:
                x = (attn @ v).transpose(1, 2).reshape(b, nq, -1) # (b,nh,nq,ch) --> (b,nq,nh*ch)
            else:
                x = (attn @ v).permute(2, 0, 1, 3).reshape(nq, b, -1) # (b,nh,nq,ch) --> (nq,b,nh*ch)
            
            x = self.projs[i](x)
            x = self.proj_drop(x)

            attn_features.append(x)
            if need_weights:
                attn_weights.append(attn)
        
        attn_features = torch.cat(attn_features, dim=-1)
        if query_mask is not None:
            attn_features = attn_features * (~query_mask).unsqueeze(-1).float()
        
        if need_weights:
            return attn_features, attn_weights
        else:
            return attn_features