"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

from tqdm import trange
import torch
from torch.utils.data import DataLoader
from logger import Logger
from modules.model import ReconstructionModel
from torch.optim.lr_scheduler import MultiStepLR
from sync_batchnorm import DataParallelWithCallback
from frames_dataset import DatasetRepeater
import torch.distributed as dist

def train(gpu, config, generator, region_predictor, bg_predictor, pixelwise_flow_predictor, checkpoint, log_dir, dataset, opt):

    train_params = config['train_params']

    rank = opt.nr * opt.gpus + gpu
    
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=opt.world_size,
        rank=rank
    )
      
    torch.cuda.set_device(gpu)
    region_predictor.to(gpu)
    if generator is not None:
        generator.to(gpu)
    bg_predictor.to(gpu)
    pixelwise_flow_predictor.to(gpu)

    if 'freeze_region' in train_params and train_params['freeze_region']:
        for p in region_predictor.parameters():
            p.requires_grad = False
        region_predictor.eval()

        optimizer = torch.optim.Adam(list(generator.parameters()) +
                                    list(bg_predictor.parameters()) +
                                    list(pixelwise_flow_predictor.parameters()), lr=train_params['lr'], betas=(0.5, 0.999))
    elif 'generator_freeze' in train_params and train_params['generator_freeze']:
        optimizer = torch.optim.Adam(list(pixelwise_flow_predictor.parameters()) + 
                                     list(region_predictor.parameters()) +
                                     list(bg_predictor.parameters()), lr=train_params['lr'], betas=(0.5, 0.999))
    elif opt.only_kp:
        if rank == 0:
            print("Region predictor optimizer")
        optimizer = torch.optim.Adam(list(region_predictor.parameters()), lr=train_params['lr'], betas=(0.5, 0.999))
    elif generator is None:
        if rank == 0:
            print("No Generator")
        optimizer = torch.optim.Adam(list(region_predictor.parameters()) +
                                    list(bg_predictor.parameters()) + 
                                    list(pixelwise_flow_predictor.parameters()), lr=train_params['lr'], betas=(0.5, 0.999))
    elif 'only_generator' in train_params and train_params['only_generator']:
        optimizer = torch.optim.Adam(list(generator.parameters()), lr=train_params['lr'], betas=(0.5, 0.999))
    else:
        optimizer = torch.optim.Adam(list(generator.parameters()) +
                                    list(region_predictor.parameters()) +
                                    list(bg_predictor.parameters()) + 
                                    list(pixelwise_flow_predictor.parameters()), lr=train_params['lr'], betas=(0.5, 0.999))

    model = ReconstructionModel(region_predictor, bg_predictor, generator, pixelwise_flow_predictor, train_params)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    if checkpoint is not None:
        if rank == 0:
            print('Loading from checkpoint')
        map_location = {'cuda:%d'%0:'cuda:%d'%rank}
        checkpoint = torch.load(checkpoint, map_location=map_location)
        if not opt.only_kp:
            start_epoch = checkpoint['epoch_reconstruction'] + 1
            model.module.region_predictor.load_state_dict(checkpoint['region_predictor'])
            optimizer.load_state_dict(checkpoint['optimizer_reconstruction'])
        else:
            if rank == 0:
                print("Not loading keypoint")
            start_epoch = 0
        
        if 'generator' in checkpoint:
            model.module.generator.load_state_dict(checkpoint['generator'])
        model.module.bg_predictor.load_state_dict(checkpoint['bg_predictor'])
        model.module.pixelwise_flow_predictor.load_state_dict(checkpoint['pixelwise_flow_predictor'])

        # Logger.load_cpk(opt.checkpoint, generator, region_predictor, bg_predictor, None,
        #                               optimizer, None)
    elif 'freeze_region' in train_params and train_params['freeze_region']:
        if rank == 0:
            print("Region Predictor frozen")
            
        map_location = {'cuda:%d'%0:'cuda:%d'%rank}
        checkpoint = torch.load(train_params['orig_checkpoint'], map_location=map_location)
        start_epoch = 0
        model.module.region_predictor.load_state_dict(checkpoint['region_predictor'])
    else:
        start_epoch = 0
    del checkpoint

    scheduler = MultiStepLR(optimizer, train_params['epoch_milestones'], gamma=0.1, last_epoch=start_epoch - 1)
    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
    
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=opt.world_size,
        rank=rank
    )
    
    # dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True,
    #                         num_workers=train_params['dataloader_workers'], drop_last=True)

    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], 
                            #####################################################
                            shuffle=False,
                            #####################################################
                            num_workers=train_params['dataloader_workers'], drop_last=True,
                            #####################################################
                            sampler=sampler
                            #####################################################
                            )


    # if torch.cuda.is_available():
    #     if ('use_sync_bn' in train_params) and train_params['use_sync_bn']:
    #         model = DataParallelWithCallback(model, device_ids=device_ids)
    #     else:
    #         model = torch.nn.DataParallel(model, device_ids=device_ids)

    if rank == 0:
        logger = Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'],
                checkpoint_freq=train_params['checkpoint_freq'])
    
    pbar = trange(start_epoch, train_params['num_epochs'])

    for epoch in pbar:
        for x in dataloader:
            losses, generated = model(x)
            loss_values = [val.mean() for val in losses.values()]
            loss = sum(loss_values)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
                
            losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses.items()}

            if rank == 0:
                logger.log_iter(losses=losses)
                pbar.set_description(" ".join([f"{k}:{v:.2f}" for k, v in losses.items()]))
        
            # break

        if 'threshold' in config['model_params']['pixelwise_flow_predictor_params'].keys():
            #! Thresholding for OM updated for every epoch (Linearly)
            model.module.update_threshold()
        scheduler.step()
        
        if rank == 0:
            logger.log_epoch(epoch, {'generator': generator,
                                    'bg_predictor': bg_predictor,
                                    'region_predictor': region_predictor,
                                    'optimizer_reconstruction': optimizer,
                                    'pixelwise_flow_predictor': pixelwise_flow_predictor}, inp=x, out=generated)
    