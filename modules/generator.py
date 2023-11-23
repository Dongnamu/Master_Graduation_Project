"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

import torch
from torch import nn
import torch.nn.functional as F
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d, ResidualConv, Upsample, make_coordinate_grid
from modules.pixelwise_flow_predictor import PixelwiseFlowPredictor
from skimage import img_as_ubyte, img_as_float32
import cv2
import numpy as np


class ResUnet(nn.Module):
    def __init__(self, channel, num_channels, split, filters=[64, 128, 256, 512], detached=False):
        super(ResUnet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], num_channels, 1, 1),
            nn.Sigmoid(),
        )
        self.detached = detached
        self.num_channels = num_channels
        self.split = split


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

    def forward(self, input, deformed):
        
        output_dict = {}
        output_dict['deformed'] = deformed
        
        # driving_cloth_mask = input['driving_cloth_mask']
        # driving_whole_skin_mask = input['driving_whole_skin_mask']
        # driving_background_mask = input['driving_background_mask']

        driving_cloth_mask = self.generate_mask(input[:, :self.split], shape=deformed.shape[2:], sharp=True)
        driving_whole_skin_mask_with_duplicate = self.generate_mask(input[:,self.split:], shape=deformed.shape[2:], sharp=True)
        driving_background_mask = 1 - self.generate_mask(input, shape=deformed.shape[2:], sharp=True)
        driving_duplicated = driving_cloth_mask * driving_whole_skin_mask_with_duplicate
        driving_whole_skin_mask = driving_whole_skin_mask_with_duplicate - driving_duplicated

        background = deformed * driving_background_mask
        cloth = deformed * driving_cloth_mask
        whole_skin = deformed * driving_whole_skin_mask

        x = torch.concat((whole_skin, driving_whole_skin_mask, cloth, driving_cloth_mask, background), axis=1)
        
        # _ = self.input_skip(x)
        # _ = self.input_layer(x)
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        # Bridge
        x4 = self.bridge(x3)
        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        output_dict['prediction'] = self.output_layer(x10)
        output_dict['background'] = background
        output_dict['cloth'] = cloth
        output_dict['skin'] = whole_skin

        return output_dict

class Generators(nn.Module):
    def __init__(self, num_channels, **generator_params):
        super(Generators, self).__init__()

        self.num_channels = num_channels
        self.without_cloth_generator = Generator(num_channels=num_channels, **generator_params)
        self.cloth_generator = Generator(num_channels=num_channels, **generator_params)
        combiner_params = {}
        combiner_params.update(generator_params)
        combiner_params['no_first'] = True
        combiner_params['input_channel'] = 3
        self.combiner_om = nn.Conv2d(combiner_params['block_expansion'] * 2, 1, kernel_size=(7,7), padding=(3,3))
        self.combiner = Generator(num_channels=num_channels, **combiner_params)

    def forward(self, motion_params):
        def makeDictionary(segment):
            return motion_params['source_{}'.format(segment)], {'optical_flow':motion_params['source_{}_optical_flow'.format(segment)], 'occlusion_map':motion_params['occlusion_map_{}'.format(segment)]}
            
        source_without_cloth_image, without_cloth_motion_params = makeDictionary('without_cloth')
        source_cloth_image, cloth_motion_params = makeDictionary('cloth')

        predicted_without_cloth_latent, predicted_without_cloth = self.without_cloth_generator(source_without_cloth_image, motion_params=without_cloth_motion_params)
        predicted_cloth_latent, predicted_cloth = self.cloth_generator(source_cloth_image, motion_params=cloth_motion_params)
        predicted_cloth_numpy = torch.exp(-250 * torch.square(torch.sum(predicted_cloth[:,:self.num_channels].data, dim=1))).unsqueeze(1)
        combined = predicted_without_cloth[:,:self.num_channels].detach() * predicted_cloth_numpy + predicted_cloth[:,:self.num_channels].detach() * (1 - predicted_cloth_numpy)
        combined_latent = predicted_without_cloth_latent.detach() * predicted_cloth_numpy + predicted_cloth_latent.detach() * (1-predicted_cloth_numpy)
        
        b,_,h,w = predicted_cloth.shape
        identity_optical_flow = make_coordinate_grid((h,w), type=torch.float32).cuda().view(1,h,w,2).repeat((b,1,1,1))
        occlusion_map_combined = torch.tanh(self.combiner_om(torch.cat([predicted_cloth_latent, predicted_without_cloth_latent], dim=1)))
        predictions = {}
        predictions['occlusion_map_combined'] = occlusion_map_combined
        combined_motion_params = {'optical_flow':identity_optical_flow, 'occlusion_map':occlusion_map_combined}
        _, combined_image = self.combiner(combined, out=combined_latent, motion_params=combined_motion_params)
        predictions['prediction'] = {'without_cloth':predicted_without_cloth, 'cloth':predicted_cloth}
        predictions['combined'] = combined_image    
        
        return predictions


class Generator(nn.Module):
    """
    Generator that given source image and region parameters try to transform image according to movement trajectories
    induced by region parameters. Generator follows Johnson architecture.
    """

    def __init__(self, num_channels, block_expansion, max_features, num_down_blocks,
                 num_bottleneck_blocks, input_channel, skips=False, no_latent=False, no_first=False):
        super(Generator, self).__init__()

        if not no_first:
            self.first = SameBlock2d(input_channel, block_expansion, kernel_size=(7, 7), padding=(3, 3))
        else:
            self.first = None

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1), no_latent=no_latent))
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1), no_latent=no_latent))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.num_channels = num_channels
        self.skips = skips
        self.no_latent = no_latent

    @staticmethod
    def deform_input(inp, optical_flow):
        _, h_old, w_old, _ = optical_flow.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            optical_flow = optical_flow.permute(0, 3, 1, 2)
            optical_flow = F.interpolate(optical_flow, size=(h, w), mode='bilinear')
            optical_flow = optical_flow.permute(0, 2, 3, 1)
        return F.grid_sample(inp, optical_flow)

    def apply_optical(self, input_previous=None, input_skip=None, motion_params=None):
        if motion_params is not None:
            if 'occlusion_map' in motion_params:
                occlusion_map = motion_params['occlusion_map']
            else:
                occlusion_map = None
            deformation = motion_params['optical_flow']
            input_skip = self.deform_input(input_skip, deformation)

            if occlusion_map is not None:
                if input_skip.shape[2] != occlusion_map.shape[2] or input_skip.shape[3] != occlusion_map.shape[3]:
                    occlusion_map = F.interpolate(occlusion_map, size=input_skip.shape[2:], mode='bilinear')
                if input_previous is not None:
                    input_skip = input_skip * occlusion_map + input_previous * (1 - occlusion_map)
                else:
                    input_skip = input_skip * occlusion_map
            out = input_skip
        else:
            out = input_previous if input_previous is not None else input_skip
        return out

    def update_threshold(self):
        self.pixelwise_flow_predictor.update_threshold()
        
    def forward(self, source_image, out=None, motion_params=None):
        if self.first is not None:
            out = self.first(source_image)

        skips = [out]
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
            skips.append(out)

        if motion_params is not None:
            out = self.apply_optical(input_previous=None, input_skip=out, motion_params=motion_params)

        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            if self.skips:
                out = self.apply_optical(input_skip=skips[-(i + 1)], input_previous=out, motion_params=motion_params)
            out = self.up_blocks[i](out)
        if self.skips:
            out = self.apply_optical(input_skip=skips[0], input_previous=out, motion_params=motion_params)
        out_final = self.final(out)

        out_final = F.tanh(out_final)
        if self.skips:
            out_final = self.apply_optical(input_skip=source_image, input_previous=out_final, motion_params=motion_params)
        
        return out, out_final
