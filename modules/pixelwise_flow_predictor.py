"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

from torch import nn
import torch.nn.functional as F
import torch
from modules.util import Hourglass, AntiAliasInterpolation2d, make_coordinate_grid, region2gaussian
from modules.util import to_homogeneous, from_homogeneous
from torchvision.utils import save_image
import numpy as np


class SegmentPixelFlowPredictor(nn.Module):

    def __init__(self, **pixelwise_flow_predictor_params):
        super(SegmentPixelFlowPredictor, self).__init__()
        
        self.cloth_pixelwise_flow_predictor = PixelwiseFlowPredictor(**pixelwise_flow_predictor_params)
        self.without_cloth_pixelwise_flow_predictor = PixelwiseFlowPredictor(**pixelwise_flow_predictor_params)
    
    def forward(self, source_image, driving_region_params, source_region_params, bg_params=None):
        out_dict = {}
        
        cloth_out_dict = self.cloth_pixelwise_flow_predictor(source_image, driving_region_params, source_region_params, bg_params, orig=False, cloth=True)
        without_cloth_dict = self.without_cloth_pixelwise_flow_predictor(source_image, driving_region_params, source_region_params, bg_params, orig=False, cloth=False)

        out_dict['source_without_cloth'] = without_cloth_dict['masked_source']
        out_dict['source_cloth'] = cloth_out_dict['masked_source']

        out_dict['source_without_cloth_optical_flow'] = without_cloth_dict['optical_flow']
        out_dict['source_cloth_optical_flow'] = cloth_out_dict['optical_flow']

        out_dict['deformed_source_without_cloth'] = without_cloth_dict['deformed_masked_source']
        out_dict['deformed_source_cloth'] = cloth_out_dict['deformed_masked_source']

        out_dict['occlusion_map_without_cloth'] = without_cloth_dict['occlusion_map']
        out_dict['occlusion_map_cloth'] = cloth_out_dict['occlusion_map']

        return out_dict

class PixelwiseFlowPredictor(nn.Module):
    """
    Module that predicts a pixelwise flow from sparse motion representation given by
    source_region_params and driving_region_params
    """

    def __init__(self, block_expansion, num_blocks, max_features, num_regions, num_channels, split,
                 estimate_occlusion_map=False, separate_training=False, scale_factor=1, region_var=0.01,
                 use_covar_heatmap=False, use_deformed_source=True, revert_axis_swap=False):
        super(PixelwiseFlowPredictor, self).__init__()
        self.split = split

        self.pre_split_hourglass = Hourglass(block_expansion=block_expansion,
                                   in_features=(self.split+1) * (num_channels * use_deformed_source + 1))
        self.pre_mask = nn.Conv2d(self.pre_split_hourglass.out_filters, self.split+1, kernel_size=(7,7), padding=(3,3))

        self.post_split_hourglass = Hourglass(block_expansion=block_expansion,
                                              in_features=(num_regions - self.split) * (num_channels * use_deformed_source + 1))
        self.post_mask = nn.Conv2d(self.post_split_hourglass.out_filters, (num_regions - self.split), kernel_size=(7,7), padding=(3,3))

        if estimate_occlusion_map:
            if not separate_training:
                self.occlusion = nn.Conv2d(self.pre_split_hourglass.out_filters + self.post_split_hourglass.out_filters, 1, kernel_size=(7, 7), padding=(3, 3))
            else:
                print("OM & OF Separate Training")
                self.om_hourglass = Hourglass(block_expansion=block_expansion,
                                   in_features=(num_regions + 1) * (num_channels * use_deformed_source + 1),
                                   max_features=max_features, num_blocks=num_blocks)
                self.occlusion = nn.Conv2d(self.om_hourglass.out_filters, 1, kernel_size=(7,7), padding=(3,3))
        else:
            self.occlusion = None
        self.separate_training = separate_training

        self.num_regions = num_regions
        self.scale_factor = scale_factor
        self.region_var = region_var
        self.use_covar_heatmap = use_covar_heatmap
        self.use_deformed_source = use_deformed_source
        self.revert_axis_swap = revert_axis_swap
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)
        
    def create_heatmap_representations(self, source_image, driving_region_params, source_region_params):
        """
        Eq 6. in the paper H_k(z)
        """
        spatial_size = source_image.shape[2:]
        covar = self.region_var if not self.use_covar_heatmap else driving_region_params['covar']
        gaussian_driving = region2gaussian(driving_region_params['shift'], covar=covar, spatial_size=spatial_size)
        covar = self.region_var if not self.use_covar_heatmap else source_region_params['covar']
        gaussian_source = region2gaussian(source_region_params['shift'], covar=covar, spatial_size=spatial_size)

        heatmap = gaussian_driving - gaussian_source

        # adding background feature
        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1])
        heatmap = torch.cat([zeros.type(heatmap.type()), heatmap], dim=1)
        heatmap = heatmap.unsqueeze(2)
        return heatmap

    def create_sparse_motions(self, source_image, driving_region_params, source_region_params, bg_params=None):
        bs, _, h, w = source_image.shape
        identity_grid = make_coordinate_grid((h, w), type=source_region_params['shift'].type())
        identity_grid = identity_grid.view(1, 1, h, w, 2)
        coordinate_grid = identity_grid - driving_region_params['shift'].view(bs, self.num_regions, 1, 1, 2)
        if 'affine' in driving_region_params:
            affine = torch.matmul(source_region_params['affine'], torch.inverse(driving_region_params['affine']))
            if self.revert_axis_swap:
                affine = affine * torch.sign(affine[:, :, 0:1, 0:1])
            affine = affine.unsqueeze(-3).unsqueeze(-3)
            affine = affine.repeat(1, 1, h, w, 1, 1)
            coordinate_grid = torch.matmul(affine, coordinate_grid.unsqueeze(-1))
            coordinate_grid = coordinate_grid.squeeze(-1)

        driving_to_source = coordinate_grid + source_region_params['shift'].view(bs, self.num_regions, 1, 1, 2)

        # adding background feature
        if bg_params is None:
            bg_grid = identity_grid.repeat(bs, 1, 1, 1, 1)
        else:
            bg_grid = identity_grid.repeat(bs, 1, 1, 1, 1)
            bg_grid = to_homogeneous(bg_grid)
            bg_grid = torch.matmul(bg_params.view(bs, 1, 1, 1, 3, 3), bg_grid.unsqueeze(-1)).squeeze(-1)
            bg_grid = from_homogeneous(bg_grid)

        sparse_motions = torch.cat([bg_grid, driving_to_source], dim=1)

        return sparse_motions

    def create_deformed_source_image(self, source_image, sparse_motions):
        bs, _, h, w = source_image.shape
        source_repeat = source_image.unsqueeze(1).unsqueeze(1).repeat(1, self.num_regions + 1, 1, 1, 1, 1)
        source_repeat = source_repeat.view(bs * (self.num_regions + 1), -1, h, w)
        sparse_motions = sparse_motions.view((bs * (self.num_regions + 1), h, w, -1))
        sparse_deformed = F.grid_sample(source_repeat, sparse_motions)
        sparse_deformed = sparse_deformed.view((bs, self.num_regions + 1, -1, h, w))
        return sparse_deformed

    def generate_mask(self, heatmap, shape, sharp=False):
        num_regions = heatmap.shape[1]
        
        weights = []
        parts = []

        for i in range(num_regions):
            part = heatmap[:,i:(i+1),:,:]
            part = part / torch.max(part)
            weights.append(part)
            parts.append(part)
            
        weight = torch.stack(weights, dim=0).sum(dim=0)
        weight = torch.maximum(torch.ones_like(weight), weight)
        mask = torch.stack(parts, dim=0).sum(dim=0) / weight
        mask = F.interpolate(mask, shape)
        if sharp:
            mask = 1 - torch.exp(-100. * mask.pow(2))
        return mask

    def generate_mask_from_heatmap(self, heatmap, img_shape):
        cloth_mask = self.generate_mask(heatmap[:,:self.split], img_shape, True)
        without_cloth_mask = 1 - cloth_mask
        return cloth_mask, without_cloth_mask

    def create_segmented_deformed_source_image(self, source_image, sparse_motions, heatmap, cloth=True):
        bs, _, h, w = source_image.shape
        img_shape = (h, w)
        
        predicted_cloth_mask, predicted_without_cloth_mask = self.generate_mask_from_heatmap(heatmap, img_shape)

        if cloth:
            mask = predicted_cloth_mask
        else:
            mask = predicted_without_cloth_mask

        source_masked = source_image * mask

        source_masked = source_masked.unsqueeze(1).repeat(1, self.num_regions + 1, 1, 1, 1)
        source_masked = source_masked.view((bs * (self.num_regions + 1), -1, h, w))
        sparse_motions = sparse_motions.view((bs * (self.num_regions + 1), h, w, -1))

        source_masked = F.grid_sample(source_masked, sparse_motions)

        source_masked = source_masked.view((bs, self.num_regions + 1, -1, h, w))

        return source_masked, mask

    @staticmethod
    def deform_input(inp, optical_flow):
        _, h_old, w_old, _ = optical_flow.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            optical_flow = optical_flow.permute(0, 3, 1, 2)
            optical_flow = F.interpolate(optical_flow, size=(h, w), mode='bilinear')
            optical_flow = optical_flow.permute(0, 2, 3, 1)
        return F.grid_sample(inp, optical_flow)
    
    def forward(self, source_image, driving_region_params, source_region_params, bg_params=None, orig=True, cloth=True):
        orig_source_image = source_image.clone()
        orig_shape = orig_source_image.shape[2:]
        if self.scale_factor != 1:
            source_image = self.down(source_image)

        bs, _, h, w = source_image.shape

        out_dict = dict()
        heatmap_representation = self.create_heatmap_representations(source_image, driving_region_params,
                                                                     source_region_params)
        # print(heatmap_representation.shape)
        sparse_motion = self.create_sparse_motions(source_image, driving_region_params,
                                                   source_region_params, bg_params=bg_params)
        if orig:
            deformed_source = self.create_deformed_source_image(source_image, sparse_motion)
            segment_mask = None
        else:
            deformed_source, segment_mask = self.create_segmented_deformed_source_image(source_image, sparse_motion, source_region_params['heatmap'], cloth=cloth)
        
        if self.use_deformed_source:
            pre_split_predictor_input = torch.cat([heatmap_representation[:,:self.split+1], deformed_source[:,:self.split+1]], dim=2)
            post_split_predictor_input = torch.cat([heatmap_representation[:,self.split+1:], deformed_source[:,self.split+1:]], dim=2)
        else:
            pre_split_predictor_input = heatmap_representation[:,:self.split+1]
            post_split_predictor_input = heatmap_representation[:,self.split+1:]

        pre_split_predictor_input = pre_split_predictor_input.view(bs, -1, h, w)
        post_split_predictor_input = post_split_predictor_input.view(bs, -1, h, w)

        pre_split_prediction = self.pre_split_hourglass(pre_split_predictor_input)
        post_split_prediction = self.post_split_hourglass(post_split_predictor_input)
        
        pre_split_dense_mask = self.pre_mask(pre_split_prediction)
        post_split_dense_mask = self.post_mask(post_split_prediction)
        mask = torch.softmax(torch.cat([pre_split_dense_mask, post_split_dense_mask], dim=1), dim=1)
        mask = mask.unsqueeze(dim=2)
        sparse_motion = sparse_motion.permute(0, 1, 4, 2, 3)

        deformation = (sparse_motion * mask).sum(dim=1)
        deformation = deformation.permute(0, 2, 3, 1)
        out_dict['optical_flow'] = deformation

        if self.occlusion:
            occlusion_map = torch.tanh(self.occlusion(torch.cat([pre_split_prediction, post_split_prediction], dim=1)))
            out_dict['occlusion_map'] = occlusion_map
        else:
            out_dict['occlusion_map'] = None
            
        if segment_mask is not None:
            segment_mask = F.interpolate(segment_mask, orig_shape)
            source_masked = orig_source_image * segment_mask
            deformed_source_masked = self.deform_input(source_masked, deformation)
            out_dict['masked_source'] = source_masked
            out_dict['deformed_masked_source'] = deformed_source_masked
            out_dict['segment_mask'] = segment_mask
            
        
        return out_dict
