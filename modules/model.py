"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

from torch import nn
import torch
import torch.nn.functional as F
from modules.util import AntiAliasInterpolation2d, make_coordinate_grid
from torchvision import models
import numpy as np
from torch.autograd import grad
from skimage import img_as_float32, img_as_ubyte
import cv2

class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss.
    """

    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = (x - self.mean) / self.std
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class ImagePyramide(torch.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss.
    """

    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict


class Transform:
    """
    Random tps transformation for equivariance constraints.
    """

    def __init__(self, bs, **kwargs):
        noise = torch.normal(mean=0, std=kwargs['sigma_affine'] * torch.ones([bs, 2, 3]))
        self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
        self.bs = bs

        if ('sigma_tps' in kwargs) and ('points_tps' in kwargs):
            self.tps = True
            self.control_points = make_coordinate_grid((kwargs['points_tps'], kwargs['points_tps']),
                                                       type=self.theta.type())
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = torch.normal(mean=0,
                                               std=kwargs['sigma_tps'] * torch.ones([bs, 1, kwargs['points_tps'] ** 2]))
        else:
            self.tps = False

    def transform_frame(self, frame):
        grid = make_coordinate_grid(frame.shape[2:], type=frame.type()).unsqueeze(0)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.warp_coordinates(grid).view(self.bs, frame.shape[2], frame.shape[3], 2)
        return F.grid_sample(frame, grid, padding_mode="reflection")

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type())
        theta = theta.unsqueeze(1)
        transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
        transformed = transformed.squeeze(-1)

        if self.tps:
            control_points = self.control_points.type(coordinates.type())
            control_params = self.control_params.type(coordinates.type())
            distances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
            distances = torch.abs(distances).sum(-1)

            result = distances ** 2
            result = result * torch.log(distances + 1e-6)
            result = result * control_params
            result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result

        return transformed

    def jacobian(self, coordinates):
        new_coordinates = self.warp_coordinates(coordinates)
        grad_x = grad(new_coordinates[..., 0].sum(), coordinates, create_graph=True)
        grad_y = grad(new_coordinates[..., 1].sum(), coordinates, create_graph=True)
        jacobian = torch.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], dim=-2)
        return jacobian


def detach_kp(kp):
    return {key: value.detach() for key, value in kp.items()}


class ReconstructionModel(torch.nn.Module):
    """
    Merge all updates into single model for better multi-gpu usage
    """

    def __init__(self, region_predictor, bg_predictor, generator, pixelwise_flow_predictor, train_params):
        super(ReconstructionModel, self).__init__()
        self.region_predictor = region_predictor
        self.bg_predictor = bg_predictor
        self.generator = generator
        self.train_params = train_params
        self.scales = train_params['scales']
        self.pixelwise_flow_predictor = pixelwise_flow_predictor
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

        if 'perceptual' in self.loss_weights and sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()
            if torch.cuda.is_available():
                self.vgg = self.vgg.cuda()

    def update_threshold(self):
        self.generator.update_threshold()

    def coarse_image_generation(self, frames):
        coarse_frames = []
        small_shape = (np.array(frames.shape[1:3]) * 0.25).astype(np.int)
        
        for frame in frames:
            frame = img_as_ubyte(frame)
            frame = cv2.resize(frame, small_shape, interpolation=cv2.INTER_AREA)
            frame = cv2.edgePreservingFilter(frame, flags=cv2.RECURS_FILTER, sigma_s=100, sigma_r=0.5)
            frame = img_as_float32(frame)
            coarse_frames.append(frame)
        
        return np.array(coarse_frames)

    def pseudo_occlusion_map_generation(self, deformed, gt):
        pseudo_occlusion_map = torch.exp(-30. * torch.sum(deformed - gt, dim=1).pow(2)).unsqueeze(1)
        return pseudo_occlusion_map
        
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

    
    def forward(self, x):
        source_region_params = self.region_predictor(x['source'])
        driving_region_params = self.region_predictor(x['driving'])

        bg_params = self.bg_predictor(x['source'], x['driving'])
        motion_params = self.pixelwise_flow_predictor(source_image=x['source'], driving_region_params=driving_region_params, source_region_params=source_region_params, bg_params=bg_params)

        generated = self.generator(motion_params)

        generated.update({'source_region_params': source_region_params, 'driving_region_params': driving_region_params})
        generated.update(motion_params)
        loss_values = {}

        predictions = generated['prediction']
        img_shape = x['source'].shape[2:]
        

        if 'perceptual' in self.loss_weights and sum(self.loss_weights['perceptual']) != 0:
            value_total = 0
            pyramide_generated = {}
            pyramide_real = {}
            for k, v in predictions.items():
                pyramide_generated[k] = self.pyramid(v[:,:3])
                pyramide_real[k] = self.pyramid(x['driving_gt_{}'.format(k)])

            pyramide_generated['combined'] = self.pyramid(generated['combined'])
            pyramide_real['combined'] = self.pyramid(x['driving'])
            # deformed_value_total = 0
            # inpainted_value_total = 0
            for k, v in pyramide_generated.items():
                for scale in self.scales:
                    x_vgg = self.vgg(pyramide_generated[k]['prediction_' + str(scale)])
                    y_vgg = self.vgg(pyramide_real[k]['prediction_' + str(scale)])

                    for i, weight in enumerate(self.loss_weights['perceptual']):
                        value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                        value_total += self.loss_weights['perceptual'][i] * value

            loss_values['percep'] = value_total

        if (self.loss_weights['equivariance_shift'] + self.loss_weights['equivariance_affine']) != 0:
            transform = Transform(x['driving'].shape[0], **self.train_params['transform_params'])
            transformed_frame = transform.transform_frame(x['driving'])
            transformed_region_params = self.region_predictor(transformed_frame)

            generated['transformed_frame'] = transformed_frame
            generated['transformed_region_params'] = transformed_region_params

            if self.loss_weights['equivariance_shift'] != 0:
                value = torch.abs(driving_region_params['shift'] -
                                  transform.warp_coordinates(transformed_region_params['shift'])).mean()
                loss_values['es'] = self.loss_weights['equivariance_shift'] * value

            if self.loss_weights['equivariance_affine'] != 0:
                affine_transformed = torch.matmul(transform.jacobian(transformed_region_params['shift']),
                                                  transformed_region_params['affine'])

                normed_driving = torch.inverse(driving_region_params['affine'])
                normed_transformed = affine_transformed
                value = torch.matmul(normed_driving, normed_transformed)
                eye = torch.eye(2).view(1, 1, 2, 2).type(value.type())

                if self.pixelwise_flow_predictor.revert_axis_swap:
                    value = value * torch.sign(value[:, :, 0:1, 0:1])

                value = torch.abs(eye - value).mean()
                loss_values['ea'] = self.loss_weights['equivariance_affine'] * value

        if 'l1' in self.loss_weights and self.loss_weights['l1'] != 0:
            l1 = 0
            for k, v in predictions.items():
                # l1 += torch.abs(v - torch.concat([x['driving_gt_{}'.format(k)], x['driving_{}_mask'.format(k)]], dim=1)).mean()
                l1 += torch.abs(v - x['driving_gt_{}'.format(k)]).mean()

            l1 += torch.abs(generated['combined'] - x['driving']).mean()
            loss_values['l1'] = l1 * self.loss_weights['l1']
                
        if 'deformed_loss' in self.loss_weights and self.loss_weights['deformed_loss'] != 0:
            deformed_source_l1_loss = torch.abs(generated['deformed_source_without_cloth'] - x['driving_gt_without_cloth']).mean()
            deformed_source_l1_loss += torch.abs(generated['deformed_source_cloth'] - x['driving_gt_cloth']).mean()
            loss_values['deformed_l1'] = deformed_source_l1_loss * self.loss_weights['deformed_loss']

        if 'om_l1' in self.loss_weights and self.loss_weights['om_l1'] != 0:
            pseudo_om_background = self.pseudo_occlusion_map_generation(generated['deformed_source_background'], x['driving_gt_background'])
            pseudo_om_cloth = self.pseudo_occlusion_map_generation(generated['deformed_source_cloth'], x['driving_gt_cloth'])
            pseudo_om_skin = self.pseudo_occlusion_map_generation(generated['deformed_source_skin'], x['driving_gt_skin'])

            occlusion_l1_loss = torch.abs(pseudo_om_background - F.interpolate(generated['occlusion_map_background'], img_shape)).mean()
            occlusion_l1_loss += torch.abs(pseudo_om_cloth - F.interpolate(generated['occlusion_map_cloth'], img_shape)).mean()
            occlusion_l1_loss += torch.abs(pseudo_om_skin - F.interpolate(generated['occlusion_map_skin'], img_shape)).mean()

            loss_values['om_l1'] = occlusion_l1_loss * self.loss_weights['om_l1']

        if 'segmentation' in self.loss_weights and self.loss_weights['segmentation'] != 0:
            source_heatmap = source_region_params['heatmap']
            driving_heatmap = driving_region_params['heatmap']

            # Segmentation 1
            source_cloth_mask = x['source_cloth_mask']
            source_skin_mask = x['source_skin_mask']
            driving_cloth_mask = x['driving_cloth_mask']
            driving_skin_mask = x['driving_skin_mask']
            # Segmentation 2
            source_left_hand_mask = x['source_left_hand_mask']
            source_right_hand_mask = x['source_right_hand_mask']
            driving_left_hand_mask = x['driving_left_hand_mask']
            driving_right_hand_mask = x['driving_right_hand_mask']
            
            
            
            predicted_source_cloth_mask = self.generate_mask(source_heatmap[:,:3,:,:], img_shape, True)
            predicted_source_skin_mask = self.generate_mask(source_heatmap[:,3:6,:,:], img_shape, True)
            predicted_source_left_mask = self.generate_mask(source_heatmap[:,6:8,:,:], img_shape, True)
            predicted_source_right_mask = self.generate_mask(source_heatmap[:,8:10,:,:], img_shape, True)

            predicted_driving_cloth_mask = self.generate_mask(driving_heatmap[:,:3,:,:], img_shape, True)
            predicted_driving_skin_mask = self.generate_mask(driving_heatmap[:,3:6,:,:], img_shape, True)
            predicted_driving_left_mask = self.generate_mask(driving_heatmap[:,6:8,:,:], img_shape, True)
            predicted_driving_right_mask = self.generate_mask(driving_heatmap[:,8:10,:,:], img_shape, True)

                
            
            segmentation_loss = torch.abs(predicted_source_cloth_mask - source_cloth_mask).mean()
            segmentation_loss += torch.abs(predicted_source_skin_mask - source_skin_mask).mean()
            segmentation_loss += torch.abs(predicted_driving_cloth_mask - driving_cloth_mask).mean()
            segmentation_loss += torch.abs(predicted_driving_skin_mask - driving_skin_mask).mean()
            segmentation_loss += torch.abs(predicted_source_left_mask - source_left_hand_mask).mean()
            segmentation_loss += torch.abs(predicted_source_right_mask - source_right_hand_mask).mean()
            segmentation_loss += torch.abs(predicted_driving_left_mask - driving_left_hand_mask).mean()
            segmentation_loss += torch.abs(predicted_driving_right_mask - driving_right_hand_mask).mean()

            loss_values['seg'] = segmentation_loss * self.loss_weights['segmentation']

        if 'seg_op' in self.loss_weights and self.loss_weights['seg_op'] != 0:
            source_heatmap = source_region_params['heatmap']
            driving_heatmap = driving_region_params['heatmap']
            img_shape = x['source'].shape[2:]
            
            source_background = generated['deformed_source_background']
            source_skin = generated['deformed_source_skin']
            source_cloth = generated['deformed_source_cloth']

            value_total = torch.abs(source_skin - x['driving_gt_skin']).mean()
            value_total += torch.abs(source_cloth - x['driving_gt_cloth']).mean()
            value_total += torch.abs(source_background - x['driving_gt_background']).mean()

            loss_values['seg_op'] = value_total * self.loss_weights['seg_op']

                
            
        return loss_values, generated
