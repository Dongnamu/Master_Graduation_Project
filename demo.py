"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

import sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm

import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte, img_as_float32
import torch
from sync_batchnorm import DataParallelWithCallback

from modules.generator import Generator, ResUnet, Generators
from modules.region_predictor import RegionPredictor
from modules.avd_network import AVDNetwork
from modules.pixelwise_flow_predictor import PixelwiseFlowPredictor, SegmentPixelFlowPredictor
from animate import get_animation_region_params
import matplotlib
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
from skimage.draw import circle
import os

matplotlib.use('Agg')

def draw_colored_heatmap(heatmap, colormap, bg_color):
    parts = []
    # weights = []
    bg_color = np.array(bg_color).reshape((1, 1, 1, 3))
    num_regions = heatmap.shape[-1]
    for i in range(num_regions):
        color = np.array(colormap(i / num_regions))[:3]
        color = color.reshape((1, 1, 1, 3))
        part = heatmap[:, :, :, i:(i + 1)]
        part = part / np.max(part, axis=(1, 2), keepdims=True)
        # weights.append(part)

        color_part = part * color
        parts.append(color_part)
        # parts.append(part)

    # weight = sum(weights)
    # bg_weight = 1 - np.minimum(1, weight)
    # weight = np.maximum(1, weight)
    # color = np.zeros((1,384, 384,3))
    # color[:,:,:,0] = 1
    # result = (sum(parts) / weight) * color + bg_weight * bg_color
    # return result
    parts = np.array(parts)
    parts = np.transpose(parts, [1,0,2,3,4])
    return parts

def draw_combined_colored_heatmap(heatmap, colormap, bg_color):
    parts = []
    weights = []
    bg_color = np.array(bg_color).reshape((1, 1, 1, 3))
    num_regions = heatmap.shape[-1]
    for i in range(num_regions):
        color = np.array(colormap(i / num_regions))[:3]
        color = color.reshape((1, 1, 1, 3))
        part = heatmap[:, :, :, i:(i + 1)]
        part = part / np.max(part, axis=(1, 2), keepdims=True)
        weights.append(part)

        parts.append(part)
        # color_part = part * color
        # parts.append(color_part)

    weight = sum(weights)
    bg_weight = 1 - np.minimum(1, weight)
    weight = np.maximum(1, weight)
    result = sum(parts) / weight + bg_weight * bg_color
    # result = sum(parts)
    # result = np.repeat(result, 3, axis=3)
    return result

def generate_mask(heatmap, shape):
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
    mask = 1 - torch.exp(-100. * mask.pow(2))

    return mask

def draw_image_with_kp(image, kp_array):
    image = np.copy(image)
    spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]
    kp_array = spatial_size * (kp_array + 1) / 2
    num_regions = kp_array.shape[0]
    for kp_ind, kp in enumerate(kp_array):
        rr, cc = circle(kp[1], kp[0], 5, shape=image.shape[:2])
        image[rr, cc] = np.array(plt.get_cmap('gist_rainbow')(kp_ind / num_regions))[:3]
    return image

def create_optical_flow(flow):
    """
    flow: b, h, w, 2
    """
    of = []
    shape = list(flow.shape[:-1]) + [3]
    hsv = np.zeros(shape, dtype=np.uint8)
    hsv[..., 1] = 255
    flow = flow.permute(0, 3, 1, 2).numpy() # b, 2, h, w

    for idx, f in enumerate(flow):
        mag, ang = cv2.cartToPolar(f[0, ...], f[1, ...])
        hsv[idx, ..., 0] = ang * 180/ np.pi / 2
        hsv[idx, ..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        of.append(cv2.cvtColor(hsv[idx], cv2.COLOR_HSV2BGR))
    return np.stack(of)/255

if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.8 or Python 3.9")


def load_checkpoints(config, checkpoint_path, cpu=False):

    model_parameters = config['model_params']

    if 'generator_params' in model_parameters:
        if 'type' not in model_parameters or model_parameters['type'] == 'orig':
            generator = Generator(num_channels=model_parameters['num_channels'],
                                **model_parameters['generator_params'])
        elif model_parameters['type'] == 'resnet':
            generator = ResUnet(channel = model_parameters['generator_params']['channel'], num_channels=model_parameters['num_channels'], split=model_parameters['split'])
        elif model_parameters['type'] == 'generators':
            generator = Generators(num_channels=model_parameters['num_channels'], **model_parameters['generator_params'])
        else:
            raise NotImplementedError
    else:
        generator = None

    if not cpu and generator is not None:
        generator.cuda()

    pixelwise_flow_predictor = SegmentPixelFlowPredictor(num_regions=model_parameters['num_regions'], num_channels=model_parameters['num_channels'],
                                                      revert_axis_swap=model_parameters['revert_axis_swap'], split=model_parameters['split'],
                                                      **model_parameters['pixelwise_flow_predictor_params'])

    if not cpu:
        pixelwise_flow_predictor.cuda()

    region_predictor = RegionPredictor(num_regions=config['model_params']['num_regions'],
                                       num_channels=config['model_params']['num_channels'],
                                       estimate_affine=config['model_params']['estimate_affine'],
                                       **config['model_params']['region_predictor_params'])
    if not cpu:
        region_predictor.cuda()

    avd_network = AVDNetwork(num_regions=config['model_params']['num_regions'],
                             **config['model_params']['avd_network_params'])
    if not cpu:
        avd_network.cuda()

    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)

    generator.load_state_dict(checkpoint['generator'])
    region_predictor.load_state_dict(checkpoint['region_predictor'])
    pixelwise_flow_predictor.load_state_dict(checkpoint['pixelwise_flow_predictor'])
    if 'avd_network' in checkpoint:
        avd_network.load_state_dict(checkpoint['avd_network'])

    # if not cpu:
    #     if generator is not None:
    #         generator = DataParallelWithCallback(generator)
    #     region_predictor = DataParallelWithCallback(region_predictor)
    #     avd_network = DataParallelWithCallback(avd_network)
    #     pixelwise_flow_predictor = DataParallelWithCallback(pixelwise_flow_predictor)

    generator.eval()
    region_predictor.eval()
    avd_network.eval()
    pixelwise_flow_predictor.eval()

    return generator, region_predictor, avd_network, pixelwise_flow_predictor

def deform_input(inp, optical_flow):
    _, h_old, w_old, _ = optical_flow.shape
    _, _, h, w = inp.shape
    if h_old != h or w_old != w:
        optical_flow = F.interpolate(optical_flow, size=(h, w), mode='bilinear')
        optical_flow = optical_flow.permute(0, 2, 3, 1)
    return F.grid_sample(inp, optical_flow)

def draw_colored_heatmap(heatmap, colormap, bg_color):
    parts = []
    weights = []
    bg_color = np.array(bg_color).reshape((1, 1, 1, 3))
    num_regions = heatmap.shape[-1]
    for i in range(num_regions):
        color = np.array(colormap(i / num_regions))[:3]
        color = color.reshape((1, 1, 1, 3))
        part = heatmap[:, :, :, i:(i + 1)]
        part = part / np.max(part, axis=(1, 2), keepdims=True)
        weights.append(part)

        color_part = part * color
        parts.append(color_part)

    weight = sum(weights)
    bg_weight = 1 - np.minimum(1, weight)
    weight = np.maximum(1, weight)
    result = sum(parts) / weight + bg_weight * bg_color
    return result

def draw_image_with_kp(image, kp_array):
    image = np.copy(image)
    spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]
    kp_array = spatial_size * (kp_array + 1) / 2
    num_regions = kp_array.shape[0]
    for kp_ind, kp in enumerate(kp_array):
        rr, cc = circle(kp[1], kp[0], 5, shape=image.shape[:2])
        image[rr, cc] = np.array(plt.get_cmap('gist_rainbow')(kp_ind / num_regions))[:3]
    return image

def make_animation(config, source_image, driving_video, original_video, segmentation_video, generator, region_predictor, avd_network, pixelwise_flow_predictor,
                   animation_mode='standard', cpu=False):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        source_region_params = region_predictor(source)
        driving_region_params_initial = region_predictor(driving[:, :, 0].cuda())

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
            driving_region_params = region_predictor(driving_frame)
            new_region_params = get_animation_region_params(source_region_params, driving_region_params,
                                                            driving_region_params_initial, avd_network=avd_network,
                                                            mode=animation_mode)
            motion_params = pixelwise_flow_predictor(source_image=source, driving_region_params=new_region_params, source_region_params=source_region_params)
                    
            out = generator(motion_params)
            
            def fromTensor2Image(x):
                return x.cpu().permute(0,2,3,1).numpy()[0]
            
            optical_flow_mask = np.clip(motion_params['heatmap_cloth'].cpu().permute(0,1,3,4,2).numpy()[0], -1, 1)
            predictions.append(np.concatenate(optical_flow_mask, axis=1))
            # prediction_without_cloth = np.clip(fromTensor2Image(out['prediction']['without_cloth']), -1, 1)
            # prediction_with_cloth = np.clip(fromTensor2Image(out['prediction']['cloth']), -1, 1)
            # combined = np.clip(fromTensor2Image(out['combined']), -1, 1)
            # predictions.append(img_as_ubyte(combined))
            # occlusion_map_cloth = np.clip(fromTensor2Image(motion_params['occlusion_map_cloth']), -1, 1)
            # occlusion_map_without_cloth = np.clip(fromTensor2Image(motion_params['occlusion_map_without_cloth']), -1, 1)
            # occlusion_map_combine = np.clip(fromTensor2Image(out['occlusion_map_combined']), -1 ,1)
            # optical_flow_without_cloth = create_optical_flow(motion_params['source_without_cloth_optical_flow'].data.cpu())[0]
            # optical_flow_cloth = create_optical_flow(motion_params['source_cloth_optical_flow'].data.cpu())[0]

            # predictions.append((prediction_without_cloth, prediction_with_cloth, combined, occlusion_map_cloth, occlusion_map_without_cloth, occlusion_map_combine, optical_flow_without_cloth, optical_flow_cloth))
            
            # source_heatmap = F.interpolate(source_region_params['heatmap'], size=source.shape[2:])
            # source_heatmap = np.transpose(source_heatmap.data.cpu().numpy(), [0,2,3,1])
            # source_heatmap = draw_colored_heatmap(source_heatmap, plt.get_cmap('gist_rainbow'), np.array((0,0,0)))[0]
            
            # drive_heatmap = F.interpolate(driving_region_params['heatmap'], size=source.shape[2:])
            # drive_heatmap = np.transpose(drive_heatmap.data.cpu().numpy(), [0,2,3,1])
            # drive_heatmap = draw_colored_heatmap(drive_heatmap, plt.get_cmap('gist_rainbow'), np.array((0,0,0)))[0]

            # source_kp = source_region_params['shift'].data.cpu().numpy()[0]
            # drive_kp = driving_region_params['shift'].data.cpu().numpy()[0]
            # source_kp = draw_image_with_kp(source_image, source_kp)
            # drive_kp = draw_image_with_kp(driving_video[frame_idx], drive_kp)
            # predictions.append(img_as_ubyte(np.concatenate([source_kp, np.clip(source_heatmap, -1, 1), drive_kp, np.clip(drive_heatmap, -1, 1)], axis=1)))
            
            # source_without_cloth = fromTensor2Image(motion_params['source_without_cloth'])
            # source_cloth = fromTensor2Image(motion_params['source_cloth'])
                        
            # deformed_source_without_cloth = fromTensor2Image(motion_params['deformed_source_without_cloth'])
            # deformed_source_cloth = fromTensor2Image(motion_params['deformed_source_cloth'])

            # deforms = np.concatenate([deformed_source_without_cloth, deformed_source_cloth], axis=1)

            # source_pre_deformed = np.concatenate([source_without_cloth, source_cloth], axis=1)
            # source_deformed = np.concatenate([deformed_source_without_cloth, deformed_source_cloth], axis=1)

            # predictions.append(img_as_ubyte(np.concatenate([source_pre_deformed, source_deformed], axis=0)))

            # without_cloth_occlusion_map = fromTensor2Image(F.interpolate(motion_params['occlusion_map_without_cloth'], source.shape[2:]).repeat((1,3,1,1)))
            # cloth_occlusion_map = fromTensor2Image(F.interpolate(motion_params['occlusion_map_cloth'], source.shape[2:]).repeat((1,3,1,1)))
            # occlusion_maps = np.concatenate([without_cloth_occlusion_map, cloth_occlusion_map], axis=1)
            # predictions.append(img_as_ubyte(np.concatenate([deforms, occlusion_maps], axis=0)))
            
            # without_cloth_inpainted = fromTensor2Image(prediction['without_cloth'])
            # cloth_inpainted = fromTensor2Image(prediction['cloth'])
            # predictions.append(img_as_ubyte(np.concatenate([without_cloth_inpainted, cloth_inpainted], axis=1)))
            # predictions.append(img_as_ubyte(np.concatenate([fromTensor2Image(combined), driving_video[frame_idx]], axis=1)))
            # predictions.append(img_as_ubyte(fromTensor2Image(out['occlusion_map_boundary'].repeat((1,3,1,1)))))

        return predictions

def skin_cloth_segmentation(segmentation, opt):
    cloth = np.array([255, 0, 0])   
    skin = np.array([0, 255, 0])
    left_hand = np.array([128,0,255])
    right_hand = np.array([0,128,255])

    cloth_mask = cv2.inRange(segmentation, cloth, cloth)
    skin_mask = cv2.inRange(segmentation, skin, skin)
    left_hand_mask = cv2.inRange(segmentation, left_hand, left_hand)
    right_hand_mask = cv2.inRange(segmentation, right_hand, right_hand)
    cloth_mask = cv2.resize(cloth_mask, opt.img_shape)
    skin_mask = cv2.resize(skin_mask, opt.img_shape)
    left_hand_mask = cv2.resize(left_hand_mask, opt.img_shape)
    right_hand_mask = cv2.resize(right_hand_mask, opt.img_shape)
    whole_skin_mask = cv2.bitwise_or(skin_mask, left_hand_mask)
    whole_skin_mask = cv2.bitwise_or(whole_skin_mask, right_hand_mask)
    background_mask = 1 - whole_skin_mask

    cloth_mask = torch.tensor(np.transpose(img_as_float32(cloth_mask[...,None]), (2,0,1))).unsqueeze(0)
    skin_mask = torch.tensor(np.transpose(img_as_float32(skin_mask[...,None]), (2,0,1))).unsqueeze(0)
    left_hand_mask = torch.tensor(np.transpose(img_as_float32(left_hand_mask[...,None]), (2,0,1))).unsqueeze(0)
    right_hand_mask = torch.tensor(np.transpose(img_as_float32(right_hand_mask[...,None]), (2,0,1))).unsqueeze(0)
    whole_skin_mask = torch.tensor(np.transpose(img_as_float32(whole_skin_mask[...,None]), (2,0,1))).unsqueeze(0)
    background_mask = torch.tensor(np.transpose(img_as_float32(background_mask[...,None]), (2,0,1))).unsqueeze(0)

    return cloth_mask, skin_mask, left_hand_mask, right_hand_mask, whole_skin_mask, background_mask

def main(opt):
    source_image = imageio.imread(opt.source_image)

    if not os.path.isdir(opt.driving_video):
        reader = imageio.get_reader(opt.driving_video)
        fps = reader.get_meta_data()['fps']
        reader.close()
        driving_video = imageio.mimread(opt.driving_video, memtest=False)
    else:
        driving_video = [imageio.imread(os.path.join(opt.driving_video, x)) for x in sorted(os.listdir(opt.driving_video))]
        fps = 15

    segmentation_video = []
    if opt.segmentation_video is not None:
        for x in sorted(os.listdir(opt.segmentation_video)):
            segmentation = imageio.imread(os.path.join(opt.segmentation_video, x))
            driving_cloth_mask, driving_skin_mask, driving_left_hand_mask, driving_right_hand_mask, driving_whole_skin_mask, driving_background_mask = skin_cloth_segmentation(segmentation, opt)
            x = {}
            x['driving_cloth_mask'] = driving_cloth_mask
            x['driving_skin_mask'] = driving_skin_mask
            x['driving_left_hand_mask'] = driving_left_hand_mask
            x['driving_right_hand_mask'] = driving_right_hand_mask
            x['driving_whole_skin_mask'] = driving_whole_skin_mask
            x['driving_background_mask'] = driving_background_mask
            x['driving_segmentation'] = img_as_float32(segmentation)
            segmentation_video.append(x)


    # original_video = imageio.mimread(opt.original_video, memtest=False)
    original_video = None
    source_image = resize(source_image, opt.img_shape)[..., :3]
    driving_video = [resize(frame, opt.img_shape)[..., :3] for frame in driving_video]
    # source_image = driving_video[0]
    # original_video = [resize(frame[:,1152:], frame[:,1152:].shape)[..., :3] for frame in original_video]
    with open(opt.config) as f:
        config = yaml.full_load(f)

    generator, region_predictor, avd_network, pixelwise_flow_predictor = load_checkpoints(config=config,
                                                                checkpoint_path=opt.checkpoint, cpu=opt.cpu)
    predictions = make_animation(config, source_image, driving_video, original_video, segmentation_video, generator, region_predictor, avd_network, pixelwise_flow_predictor,
                                 animation_mode=opt.mode, cpu=opt.cpu)
    
    imageio.mimsave(opt.result_video, [img_as_ubyte(frame) for frame in predictions], fps=fps)
    # imageio.mimsave(opt.result_video, predictions, fps=fps)

    # optical_flow_without_cloth_dirs = opt.result_video + '_optical_flow_without_cloth'
    # optical_flow_cloth_dirs = opt.result_video + '_optical_flow_cloth'
    # occlusion_map_without_cloth_dirs = opt.result_video + '_occlusion_map_without_cloth'
    # occlusion_map_combine_dirs = opt.result_video + '_occlusion_map_combine'
    # occlusion_map_cloth_dirs = opt.result_video + '_occlusion_map_cloth'
    # without_cloth_inpainted_dirs = opt.result_video + '_without_cloth'
    # cloth_inpainted_dirs = opt.result_video + '_cloth'


    # os.makedirs(opt.result_video, exist_ok=True)
    # os.makedirs(optical_flow_without_cloth_dirs, exist_ok=True)
    # os.makedirs(optical_flow_cloth_dirs, exist_ok=True)
    # os.makedirs(occlusion_map_without_cloth_dirs, exist_ok=True)
    # os.makedirs(occlusion_map_combine_dirs, exist_ok=True)
    # os.makedirs(occlusion_map_cloth_dirs, exist_ok=True)
    # os.makedirs(without_cloth_inpainted_dirs, exist_ok=True)
    # os.makedirs(cloth_inpainted_dirs, exist_ok=True)

    # def saveImage(image, dir):
    #     image = cv2.cvtColor(img_as_ubyte(image), cv2.COLOR_RGB2BGR)
    #     cv2.imwrite(dir, image)

    # for i, (prediction_without_cloth, prediction_with_cloth, combined, occlusion_map_cloth, occlusion_map_without_cloth, occlusion_map_combine, optical_flow_without_cloth, optical_flow_cloth) in enumerate(predictions):

    #     saveImage(combined, os.path.join(opt.result_video, f'{i:05d}.png'))
    #     saveImage(prediction_without_cloth, os.path.join(without_cloth_inpainted_dirs, f'{i:05d}.png'))
    #     saveImage(prediction_with_cloth, os.path.join(cloth_inpainted_dirs, f'{i:05d}.png'))
    #     saveImage(occlusion_map_cloth, os.path.join(occlusion_map_cloth_dirs, f'{i:05d}.png'))
    #     saveImage(occlusion_map_without_cloth, os.path.join(occlusion_map_without_cloth_dirs, f'{i:05d}.png'))
    #     saveImage(occlusion_map_combine, os.path.join(occlusion_map_combine_dirs, f'{i:05d}.png'))
    #     saveImage(optical_flow_without_cloth, os.path.join(optical_flow_without_cloth_dirs, f'{i:05d}.png'))
    #     saveImage(optical_flow_cloth, os.path.join(optical_flow_cloth_dirs, f'{i:05d}.png'))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", default='ted384.pth', help="path to checkpoint to restore")

    parser.add_argument("--source_image", default='sup-mat/source.png', help="path to source image")
    parser.add_argument("--driving_video", default='sup-mat/driving.mp4', help="path to driving video")
    parser.add_argument("--segmentation_video", default=None)
    parser.add_argument("--original_video", default=None)
    parser.add_argument("--result_video", default='result.mp4', help="path to output")

    parser.add_argument("--mode", default='avd', choices=['standard', 'relative', 'avd'],
                        help="Animation mode")
    parser.add_argument("--img_shape", default="384,384", type=lambda x: list(map(int, x.split(','))),
                        help='Shape of image, that the model was trained on.')
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")

    main(parser.parse_args())
