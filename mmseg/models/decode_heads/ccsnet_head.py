# segformer_head_effitop20_proj_after_cls_42.6 D share qkpe
# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict

from mmseg.ops import resize
from ..builder import HEADS
from .cascade_decode_head import BaseCascadeDecodeHead
from mmseg.models.utils import *
from ..utils import SelfAttentionBlock as _SelfAttentionBlock

# import attr

# from IPython import embed

class Efficient_Attn(nn.Module):
    '''
    Args:
        in_channels(kv in_channel):             int
        channels(proj out_channels):            int
        coarse_size:                            [r, r]
        upsample(weather to upsample query):    bool
        window_size:                            int
    '''
    def __init__(self, in_channels, channels, conv_cfg, norm_cfg, act_cfg, upsample=False, window_size=1):
        super(Efficient_Attn, self).__init__()
        # self.key_proj = self.build_project(
        #     in_channels,
        #     channels,
        #     num_convs=1,
        #     use_conv_module=False,
        #     conv_cfg=conv_cfg,
        #     norm_cfg=norm_cfg,
        #     act_cfg=act_cfg)
        # self.value_proj = self.build_project(
        #     in_channels,
        #     channels,
        #     num_convs=1,
        #     use_conv_module=False,
        #     conv_cfg=conv_cfg,
        #     norm_cfg=norm_cfg,
        #     act_cfg=act_cfg)
        self.out_proj = self.build_project(
            channels,
            channels,
            num_convs=1,
            use_conv_module=True,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.upsample = upsample
        self.window_size = window_size
    
    def forward(self, key, value, query, coarse_size):
        '''
        input:
            spatial_context:    B*32(H)*32(W), C, K, 1
            query(projected):   B, C, H*r, W*r
        output:
            result:             B, C, H*r, W*r
        '''
        
        if self.upsample == True:
            query = resize(query, size=coarse_size, mode='bilinear', align_corners=False)
        b, c, h, w = query.size()                                       # B, C, H*r, W*r
                                                                        # h = H*r

        # q, k, v
        # k = self.key_proj(spatial_context)                              # BHW, C, K, 1
        k = key
        k = k.squeeze(-1)                                               # BHW, C, K
        # v = self.value_proj(spatial_context)                            # BHW, C, K, 1
        v = value
        v = v.squeeze(-1)                                               # BHW, C, K
        v = v.permute(0, 2, 1).contiguous()                             # BHW, K, C

        _query = query.permute(0, 2, 3, 1).contiguous()                      # B, H*r, W*r, C
        _query = self.window_partition(_query, self.window_size)                  # B*num_windows(HW), r*r, C

        # attn matrix
        sim_map = torch.bmm(_query, k)                                       # BHW, r*r, K
        sim_map = (c**-.5) * sim_map                                    # BHW, r*r, K
        sim_map = F.softmax(sim_map, dim=-1)                            # BHW, r*r, K

        result = torch.bmm(sim_map, v)                                  # BHW, r*r, C
        result = self.window_reverse(result, self.window_size, h, w)    # B, H*r, W*r, C
        result = result.permute(0, 3, 1, 2).contiguous()                # B, C, H*r, W*r

        result = self.out_proj(result)

        result = result + query
        return result
        
    def build_project(self, in_channels, channels, num_convs, use_conv_module,
                      conv_cfg, norm_cfg, act_cfg):
        """Build projection layer for key/query/value/out."""
        if use_conv_module:
            convs = [
                ConvModule(
                    in_channels,
                    channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
            ]
            for _ in range(num_convs - 1):
                convs.append(
                    ConvModule(
                        channels,
                        channels,
                        1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))
        else:
            convs = [nn.Conv2d(in_channels, channels, 1)]
            for _ in range(num_convs - 1):
                convs.append(nn.Conv2d(channels, channels, 1))
        if len(convs) > 1:
            convs = nn.Sequential(*convs)
        else:
            convs = convs[0]
        return convs


    def window_partition(self, x, window_size):
        '''
        input:
            x: B, H, W, C
            window_size: int
        
        output:
            windows: num_windows*B, window_size*window_size, C
        '''
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size*window_size, C)
        return windows
    
    def window_reverse(self, windows, window_size, H, W):
        '''
        input:
            windows: num_windows*B, window_size*window_size, C
            window_size: int
            H, W: int
        
        output:
            x: B, H, W, C
        '''
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x


class SpatialGatherModule(nn.Module):
    """Aggregate the context features according to the initial predicted
    probability distribution.

    Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, scale):
        super(SpatialGatherModule, self).__init__()
        self.scale = scale

    def forward(self, feats, probs):
        """Forward function."""
        batch_size, num_classes, height, width = probs.size()
        channels = feats.size(1)
        probs = probs.view(batch_size, num_classes, -1)
        feats = feats.view(batch_size, channels, -1)
        # [batch_size, height*width, num_classes]
        feats = feats.permute(0, 2, 1)
        # [batch_size, channels, height*width]
        probs = F.softmax(self.scale * probs, dim=2)
        # [batch_size, channels, num_classes]
        ocr_context = torch.matmul(probs, feats)  #b,n,c
        ocr_context = ocr_context.permute(0, 2, 1).contiguous()  #b,c,n
        return ocr_context.unsqueeze(-1) #b,c,n,1



@HEADS.register_module()
class CCSNetHead(BaseCascadeDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.c4_proj = nn.Conv2d(c4_in_channels, embedding_dim, 1)
        self.c3_proj = nn.Conv2d(c3_in_channels, embedding_dim, 1)
        self.c2_proj = nn.Conv2d(c2_in_channels, embedding_dim, 1)
        self.c1_proj = nn.Conv2d(c1_in_channels, embedding_dim, 1)

        self.k_proj = nn.Conv2d(c4_in_channels, embedding_dim, 1)
        self.v_proj = nn.Conv2d(c4_in_channels, embedding_dim, 1, bias=False)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        # ocr
        self.spatial_gather_module = SpatialGatherModule(1)
        self.object_context_block_c4 = Efficient_Attn(
            c4_in_channels,
            embedding_dim,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            upsample = True,
            window_size = 1)
        self.object_context_block_c3 = Efficient_Attn(
            c4_in_channels,
            embedding_dim,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            upsample = False,
            window_size = 1)
        self.object_context_block_c2 = Efficient_Attn(
            c4_in_channels,
            embedding_dim,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            upsample=False,
            window_size=2)
        self.object_context_block_c1 = Efficient_Attn(
            c4_in_channels,
            embedding_dim,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            upsample=False,
            window_size=4)
        
            

    def forward(self, inputs, prev_output):
        # prev_output = 1/16, num_classes
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        index = torch.topk(prev_output, 20, dim=1)[1]        # B, k, H, W

        _c1 = self.c1_proj(c1)
        _c2 = self.c2_proj(c2)
        _c3 = self.c3_proj(c3)
        _c4 = self.c4_proj(c4)  # B, C, H, W
        
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape
        coarse_size = prev_output.size()[2:]


        context = self.spatial_gather_module(c4, resize(prev_output, size=c4.size()[2:], mode='bilinear', align_corners=False))    # B, C, N, 1
        key = self.k_proj(context)
        global_key = key.squeeze(-1)
        global_key = global_key.permute(0, 2, 1).contiguous()
        value = self.v_proj(context)
        global_value = value.squeeze(-1)
        global_value = global_value.permute(0, 2, 1).contiguous()
        key_context = self.spatial_context_cacl(global_key, index)
        value_context = self.spatial_context_cacl(global_value, index)

        _c4 = self.object_context_block_c4(key_context, value_context, _c4, coarse_size)
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.object_context_block_c3(key_context, value_context, _c3, coarse_size)
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.object_context_block_c2(key_context, value_context, _c2, coarse_size)
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.object_context_block_c1(key_context, value_context, _c1, coarse_size)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x
            
    def spatial_context_cacl(self, context, index):
        '''
        input:
            context: B, N, C
            index: B, K, 32(H), 32(W)
        output:
            context: B*32(H)*32(W), C, K, 1
        '''
        n, c = context.size()[1:]
        b, k, h, w = index.size()
        # k,v generator
        context = context.unsqueeze(1)                                  # B, 1, N, C
        context = context.repeat(1, h*w, 1, 1)                          # B, HW, N, C
        context = context.view(b*h*w, n, c)                             # BHW, N, C
        index = index.view(b, k, -1)                                    # B, K, HW
        index = index.permute(0, 2, 1).contiguous()                     # B, HW, K
        index = index.view(b*h*w, k)                                    # BHW, K
        context = context[torch.arange(b*h*w).unsqueeze(1), index]      # BHW, K, C
        context = context.permute(0, 2, 1).contiguous().unsqueeze(-1)   # BHW, C, K, 1

        return context