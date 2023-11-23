"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from logger import Logger, Visualizer
import numpy as np
import imageio
import lpips
from torchjpeg.metrics import psnr, ssim
import pytorch_fid_wrapper as pfw
from skimage import img_as_ubyte
import cv2

def reconstruction(config, generator, region_predictor, bg_predictor, pixelwise_flow_predictor, checkpoint, log_dir, dataset):
    png_dir = os.path.join(log_dir, 'reconstruction/png')
    log_dir = os.path.join(log_dir, 'reconstruction')

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator, region_predictor=region_predictor, pixelwise_flow_predictor=pixelwise_flow_predictor, bg_predictor=bg_predictor)
    else:
        raise AttributeError("Checkpoint should be specified for mode='reconstruction'.")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    loss_list = []
    lpips_list = []
    psnr_list = []
    ssim_list = []
    fid_list = []

    generator.eval()
    region_predictor.eval()
    bg_predictor.eval()
    pixelwise_flow_predictor.eval()

    generator = generator.cuda()
    region_predictor = region_predictor.cuda()
    bg_predictor = bg_predictor.cuda()
    pixelwise_flow_predictor = pixelwise_flow_predictor.cuda()

    lpips_fn = lpips.LPIPS(net='vgg').cuda()
    pfw.set_config(batch_size=1, device='cuda')

    for it, x in tqdm(enumerate(dataloader)):
        if config['reconstruction_params']['num_videos'] is not None:
            if it > config['reconstruction_params']['num_videos']:
                break
        with torch.no_grad():
            predictions = []
            visualizations = []
            if torch.cuda.is_available():
                x['video'] = x['video'].cuda()
            # print(x['video'][:,:,0])
            source_region_params = region_predictor(x['video'][:, :, 0])
            for frame_idx in range(x['video'].shape[2]):
                source = x['video'][:, :, 0]
                driving = x['video'][:, :, frame_idx]
                driving_region_params = region_predictor(driving)

                bg_params = bg_predictor(source, driving)
                motion_params = pixelwise_flow_predictor(source_image=source, driving_region_params=driving_region_params, source_region_params=source_region_params, bg_params=bg_params)
                out = generator(motion_params)

                out['source_region_params'] = source_region_params
                out['driving_region_params'] = driving_region_params

                predictions.append(img_as_ubyte(np.transpose(out['combined'].clamp(-1,1).data.cpu().numpy(), [0, 2, 3, 1])[0]))

                inp = {'source':source, 'driving':driving}
                out.update(motion_params)
                
                visualization = Visualizer(**config['visualizer_params']).visualize(inp=inp, out=out)
                visualizations.append(visualization)
                loss_list.append(torch.abs(out['combined'] - driving).mean().cpu().numpy())
                lpips_list.append(lpips_fn.forward(driving, out['combined'].clamp(-1,1)).mean().cpu().detach().numpy())
                psnr_list.append(psnr(driving, out['combined'].clamp(-1,1)).cpu().detach().numpy())
                ssim_list.append(ssim(driving, out['combined'].clamp(-1,1)).cpu().detach().numpy())
                fid_list.append(pfw.fid(driving, out['combined'].clamp(-1,1)))

            predictions = np.concatenate(predictions, axis=1)
            predictions = cv2.cvtColor(predictions, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(png_dir, x['name'][0] + '.png'), predictions)

            image_name = x['name'][0] + config['reconstruction_params']['format']
            imageio.mimsave(os.path.join(log_dir, image_name), visualizations)

    print("L1 reconstruction loss: %s" % np.mean(loss_list))
    print("lpips loss: %s" % np.mean(lpips_list))
    print('psnr score: %s' % np.mean(psnr_list))
    print('ssim score: %s' % np.mean(ssim_list))
    print('fid score: %s' % np.mean(fid_list))