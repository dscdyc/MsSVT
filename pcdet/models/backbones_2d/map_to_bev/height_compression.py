import torch.nn as nn
from torch.cuda.amp import autocast


class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.compress_layer_nums = self.model_cfg.get('COMPRESS_LAYER_NUMS', 3)
        self.layer_strides = self.model_cfg.get('LAYER_STRIDES', [1, 1, 1])
        self.layer_dialations = self.model_cfg.get('LAYER_DIALATIONS', [1, 1, 2])
        self.layer_paddings = self.model_cfg.get('LAYER_PADDINGS', [1, 1, 2])
        self.compress_layers = None
        if self.compress_layer_nums:
            conv = []
            for idx in range(self.compress_layer_nums):
                stride = self.layer_strides[idx]
                dialation = self.layer_dialations[idx]
                padding = self.layer_paddings[idx]
                conv_layer = nn.Conv2d(self.num_bev_features, self.num_bev_features, kernel_size=3, 
                    stride=stride, padding=padding, dilation=dialation, bias=False)
                conv.append(conv_layer)
                norm = nn.BatchNorm2d(self.num_bev_features)
                conv.append(norm)
                conv.append(nn.ReLU(inplace=True))
            self.compress_layers = nn.ModuleList(conv)

        self.use_amp = self.model_cfg.get('AMP', False)

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        with autocast(enabled=self.use_amp):
            encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
            spatial_features = encoded_spconv_tensor.dense()
            N, C, D, H, W = spatial_features.shape
            spatial_features = spatial_features.view(N, C * D, H, W)
            if self.compress_layers is not None:
                for conv_layer in self.compress_layers:
                    spatial_features = conv_layer(spatial_features)
        batch_dict['spatial_features'] = spatial_features.float()
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict
