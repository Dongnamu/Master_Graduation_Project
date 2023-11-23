"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

import matplotlib

matplotlib.use('Agg')

import os
import sys
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy

from frames_dataset import FramesDataset

from modules.generator import Generator, ResUnet, Generators
from modules.bg_motion_predictor import BGMotionPredictor
from modules.region_predictor import RegionPredictor
from modules.avd_network import AVDNetwork
from modules.pixelwise_flow_predictor import SegmentPixelFlowPredictor

import torch

from train import train
from reconstruction import reconstruction
from animate import animate
from train_avd import train_avd
import torch.multiprocessing as mp
from logger import Logger
from torchvision import models

if __name__ == "__main__":
    
    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--mode", default="train", choices=["train", "train_avd", "reconstruction", "animate"])
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--orig_checkpoint", default=None)
    # parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
    #                     help="Names of the devices comma separated.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.set_defaults(verbose=False)
    parser.add_argument('--nodes', default=1, type=int)
    parser.add_argument('--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('--nr', default=0, type=int, help='ranking within the nodes')
    parser.add_argument('--master_addr', default='127.0.0.1', type=str)
    parser.add_argument('--master_port', default='8889', type=str)
    parser.add_argument('--only_kp', action='store_true')
    opt = parser.parse_args()

    opt.world_size = opt.gpus * opt.nodes
    os.environ['MASTER_ADDR'] = opt.master_addr
    os.environ['MASTER_PORT'] = opt.master_port

    # To download checkpoint in advance
    vgg19 = models.vgg19(pretrained=True)
    del vgg19

    with open(opt.config) as f:
        config = yaml.load(f)

    if opt.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])

    model_parameters = config['model_params']
    pixelwise_flow_predictor = SegmentPixelFlowPredictor(num_regions=model_parameters['num_regions'], num_channels=model_parameters['num_channels'],
                                                                revert_axis_swap=model_parameters['revert_axis_swap'], split=model_parameters['split'],
                                                                **model_parameters['pixelwise_flow_predictor_params'])

    if 'only_generator' in config['train_params'] and config['train_params']['only_generator']:
        pixelwise_flow_predictor.eval()
    

    region_predictor = RegionPredictor(num_regions=model_parameters['num_regions'],
                                       num_channels=model_parameters['num_channels'],
                                       estimate_affine=model_parameters['estimate_affine'],
                                       **model_parameters['region_predictor_params'])
    
    if 'only_generator' in config['train_params'] and config['train_params']['only_generator']:
        region_predictor.eval()

    if opt.verbose:
        print(region_predictor)

    bg_predictor = BGMotionPredictor(num_channels=model_parameters['num_channels'],
                                     **model_parameters['bg_predictor_params'])
    if opt.verbose:
        print(bg_predictor)

    if 'only_generator' in config['train_params'] and config['train_params']['only_generator']:
        bg_predictor.eval()

    if opt.only_kp:
        for k, v in list(bg_predictor.named_parameters()):
            v.requires_grad = False
                
    avd_network = AVDNetwork(num_regions=model_parameters['num_regions'],
                             **model_parameters['avd_network_params'])


    if opt.verbose:
        print(avd_network)

    if 'generator_params' in model_parameters:
        if 'type' not in model_parameters:
            generator = Generator(num_channels=model_parameters['num_channels'],
                                **model_parameters['generator_params'])
        elif model_parameters['type'] == 'resnet':
            generator = ResUnet(channel = model_parameters['generator_params']['channel'], num_channels=model_parameters['num_channels'], split=model_parameters['split'])
        elif model_parameters['type'] == 'generators':
            generator = Generators(num_channels=model_parameters['num_channels'],
                                **model_parameters['generator_params'])
        else:
            raise NotImplementedError
    else:
        generator = None

    if 'generator_freeze' in config['train_params'] and config['train_params']['generator_freeze']:
        print("Generator frozen")
        if config['train_params']['orig_checkpoint'] is not None:
            orig_checkpoint = torch.load(config['train_params']['orig_checkpoint'], map_location=torch.device('cpu'))
            generator.load_state_dict(orig_checkpoint, strict=False)

    if ('generator_freeze' in config['train_params'] and config['train_params']['generator_freeze']) or opt.only_kp:
        for k, v in list(generator.named_parameters()):
            v.requires_grad = False
    
    if 'only_generator' in config['train_params'] and config['train_params']['only_generator']:
        print("Only Generator")
        if config['train_params']['orig_checkpoint'] is not None:
            orig_checkpoint = torch.load(config['train_params']['orig_checkpoint'], map_location=torch.device('cpu'))
            pixelwise_flow_predictor.load_state_dict(orig_checkpoint['pixelwise_flow_predictor'], strict=False)
            region_predictor.load_state_dict(orig_checkpoint['region_predictor'])
            bg_predictor.load_state_dict(orig_checkpoint['bg_predictor'])
    
    if opt.verbose:
        print(generator)

    dataset = FramesDataset(is_train=(opt.mode.startswith('train')), **config['dataset_params'])
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    if opt.mode == 'train':
        print("Training...")
        # train(config, generator, region_predictor, bg_predictor, pixelwise_flow_predictor, opt.checkpoint, log_dir, dataset, opt)
        mp.spawn(train, nprocs=opt.gpus, args=(config, generator, region_predictor, bg_predictor, pixelwise_flow_predictor, opt.checkpoint, log_dir, dataset, opt))
    elif opt.mode == 'train_avd':
        print("Training Animation via Disentaglement...")
        train_avd(config, generator, region_predictor, bg_predictor, avd_network, opt.checkpoint, log_dir, dataset)
    elif opt.mode == 'reconstruction':
        print("Reconstruction...")
        reconstruction(config, generator, region_predictor, bg_predictor, pixelwise_flow_predictor, opt.checkpoint, log_dir, dataset)
    elif opt.mode == 'animate':
        print("Animate...")
        animate(config, generator, region_predictor, avd_network, opt.checkpoint, log_dir, dataset)
