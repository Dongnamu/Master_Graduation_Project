"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

import copy
import os
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from skimage.transform import resize

from sklearn.model_selection import train_test_split
from imageio import mimread

import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from augmentation import AllAugmentationTransform
import glob
from functools import partial
import cv2

def read_video(name, frame_shape):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """

    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        # frames = [x for x in frames if 'npy' not in x.decode('utf-8')]
        num_frames = len(frames)
        
        if type(frames[0]) == bytes:
            video_array = [img_as_float32(io.imread(os.path.join(name, frames[idx].decode('utf-8')))) for idx in range(num_frames)]
        else:
            video_array = [img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(num_frames)]
        if frame_shape is not None:
            video_array = np.array([resize(frame, frame_shape) for frame in video_array])
    elif name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        image = io.imread(name)

        if frame_shape is None:
            raise ValueError('Frame shape can not be None for stacked png format.')

        frame_shape = tuple(frame_shape)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        video_array = np.moveaxis(image, 1, 0)

        video_array = video_array.reshape((-1,) + frame_shape + (3, ))
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        video = mimread(name)
        if len(video[0].shape) == 2:
            video = [gray2rgb(frame) for frame in video]
        if frame_shape is not None:
            video = np.array([resize(frame, frame_shape) for frame in video])
        video = np.array(video)
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array


class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None, seg_dir=None):
        self.root_dir = root_dir
        self.videos = os.listdir(root_dir)
        self.seg_dir = seg_dir
        self.frame_shape = frame_shape
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        if os.path.exists(os.path.join(root_dir, 'train')):
            assert os.path.exists(os.path.join(root_dir, 'test'))
            print("Use predefined train-test split.")
            if id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(root_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos = os.listdir(os.path.join(root_dir, 'train'))
            test_videos = os.listdir(os.path.join(root_dir, 'test'))
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
            if self.seg_dir is not None:
                self.seg_dir = os.path.join(self.seg_dir, 'train' if is_train else 'test')
        else:
            print("Use random train-test split.")
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

        if is_train:
            self.videos = train_videos
        else:
            self.videos = test_videos

        self.is_train = is_train

        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if self.is_train and self.id_sampling:
            name = self.videos[idx]
            try:
                path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))

            except ValueError:
                raise ValueError("File formatting is not correct for id_sampling=True. "
                                 "Change file formatting, or set id_sampling=False.")
        else:
            name = self.videos[idx]
            path = os.path.join(self.root_dir, name)

        if self.seg_dir is not None:
            seg_path = copy.deepcopy(path)
            seg_path = seg_path.replace(self.root_dir, self.seg_dir)

        video_name = os.path.basename(path)

        if self.is_train and os.path.isdir(path):
            frames = [x for x in os.listdir(path) if 'npy' not in x.decode('utf-8')]
            num_frames = len(frames)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))
            
            if self.frame_shape is not None:
                resize_fn = partial(resize, output_shape=self.frame_shape)
            else:
                resize_fn = img_as_float32

            # video_array = []
            # for idx in frame_idx:
            #     if type(frames[idx]) is bytes:
            #         if 'npy' not in frames[idx].decode('utf-8'):
            #             video_array.append(resize_fn(io.imread(os.path.join(path, frames[idx].decode('utf-8')))))
            #     else:
            #         video_array.append(resize_fn(io.imread(os.path.join(path, frames[idx]))))

            if type(frames[0]) is bytes:
                video_array = [resize_fn(io.imread(os.path.join(path, frames[idx].decode('utf-8')))) for idx in
                               frame_idx]
                if self.seg_dir is not None:
                    seg_array = [io.imread(os.path.join(seg_path, frames[idx].decode('utf-8'))) for idx in frame_idx]
            else:
                video_array = [resize_fn(io.imread(os.path.join(path, frames[idx]))) for idx in frame_idx]
                if self.seg_dir is not None:
                    seg_array = [io.imread(os.path.join(seg_path, frames[idx])) for idx in frame_idx]

        else:
            video_array = read_video(path, frame_shape=self.frame_shape)
            num_frames = len(video_array)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2)) if self.is_train else range(
                num_frames)
            video_array = video_array[frame_idx][..., :3]

        if self.transform is not None:
            if self.seg_dir is not None:
                video_array, seg_array = self.transform(video_array, seg_array)
            else:
                video_array, _ = self.transform(video_array)
        out = {}
        if self.is_train:
            source = np.array(video_array[0], dtype='float32')
            driving = np.array(video_array[1], dtype='float32')
            out['driving'] = driving.transpose((2, 0, 1))
            out['source'] = source.transpose((2, 0, 1))

            if self.seg_dir is not None:
                cloth = np.array([255, 0, 0])
                skin = np.array([0, 255, 0])
                left_hand = np.array([128,0,255])
                right_hand = np.array([0,128,255])

                source_segmentation, driving_segmentation = seg_array

                # green = np.zeros_like(driving)
                # green[:,:,1] = 1.

                def skin_cloth_segmentation(segmentation):
                    segmentation = cv2.resize(segmentation, self.frame_shape[:2])
                    cloth_mask = cv2.inRange(segmentation, cloth, cloth)
                    # skin_mask = cv2.inRange(segmentation, skin, skin)
                    # left_hand_mask = cv2.inRange(segmentation, left_hand, left_hand)
                    # right_hand_mask = cv2.inRange(segmentation, right_hand, right_hand)
                    # # cloth_mask = cv2.resize(cloth_mask, self.frame_shape[:2])
                    # # skin_mask = cv2.resize(skin_mask, self.frame_shape[:2])
                    # # left_hand_mask = cv2.resize(left_hand_mask, self.frame_shape[:2])
                    # # right_hand_mask = cv2.resize(right_hand_mask, self.frame_shape[:2])
                    # whole_skin_mask = cv2.bitwise_or(skin_mask, left_hand_mask)
                    # whole_skin_mask = cv2.bitwise_or(whole_skin_mask, right_hand_mask)
                    # # background_mask = 1 - whole_skin_mask
                    # foreground_mask = cv2.bitwise_or(whole_skin_mask, cloth_mask)
                    # background_mask = 255 - foreground_mask
                    without_cloth_mask = 255 - cloth_mask

                    # return cloth_mask, skin_mask, left_hand_mask, right_hand_mask, whole_skin_mask, background_mask, without_cloth_mask
                    return cloth_mask, without_cloth_mask


                # source_cloth_mask, source_skin_mask, source_left_hand_mask, source_right_hand_mask, source_whole_skin_mask, source_background_mask, _ = skin_cloth_segmentation(source_segmentation)
                # driving_cloth_mask, driving_skin_mask, driving_left_hand_mask, driving_right_hand_mask, driving_whole_skin_mask, driving_background_mask, driving_without_cloth_mask = skin_cloth_segmentation(driving_segmentation)
                driving_cloth_mask, driving_without_cloth_mask = skin_cloth_segmentation(driving_segmentation)

                # out['source_seg'] = np.transpose(img_as_float32(source_segmentation), (2,0,1))
                # out['driving_seg'] = np.transpose(img_as_float32(driving_segmentation), (2,0,1))
                # out['source_cloth_mask'] = np.transpose(img_as_float32(source_cloth_mask[...,None]), (2,0,1))
                # out['source_skin_mask'] = np.transpose(img_as_float32(source_skin_mask[...,None]), (2,0,1))
                # out['source_left_hand_mask'] = np.transpose(img_as_float32(source_left_hand_mask[...,None]), (2,0,1))
                # out['source_right_hand_mask'] = np.transpose(img_as_float32(source_right_hand_mask[...,None]), (2,0,1))
                # out['source_whole_skin_mask'] = np.transpose(img_as_float32(source_whole_skin_mask[...,None]), (2,0,1))
                # out['source_background_mask'] = np.transpose(img_as_float32(source_background_mask[...,None]), (2,0,1))
                # out['driving_cloth_mask'] = np.transpose(img_as_float32(driving_cloth_mask[...,None]), (2,0,1))
                # out['driving_without_cloth_mask'] = np.transpose(img_as_float32(driving_without_cloth_mask[...,None]), (2,0,1))
                # out['driving_skin_mask'] = np.transpose(img_as_float32(driving_skin_mask[...,None]), (2,0,1))
                # out['driving_left_hand_mask'] = np.transpose(img_as_float32(driving_left_hand_mask[...,None]), (2,0,1))
                # out['driving_right_hand_mask'] = np.transpose(img_as_float32(driving_right_hand_mask[...,None]), (2,0,1))
                # out['driving_whole_skin_mask'] = np.transpose(img_as_float32(driving_whole_skin_mask[...,None]), (2,0,1))
                # out['driving_background_mask'] = np.transpose(img_as_float32(driving_background_mask[...,None]), (2,0,1))
                out['driving_gt_cloth'] = np.transpose(driving * img_as_float32(driving_cloth_mask[...,None]),(2,0,1))
                # out['driving_gt_cloth'] = np.transpose((driving * img_as_float32(driving_cloth_mask[...,None]) + green * img_as_float32(255 - driving_cloth_mask[...,None])),(2,0,1))
                out['driving_gt_without_cloth'] = np.transpose(driving * img_as_float32(driving_without_cloth_mask[...,None]), (2,0,1))
        else:
            video = np.array(video_array, dtype='float32')
            out['video'] = video.transpose((3, 0, 1, 2))

        out['name'] = video_name
        out['id'] = idx
        
        return out


class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]


class PairedDataset(Dataset):
    """
    Dataset of pairs for animation.
    """

    def __init__(self, initial_dataset, number_of_pairs, seed=0):
        self.initial_dataset = initial_dataset
        pairs_list = self.initial_dataset.pairs_list

        np.random.seed(seed)

        if pairs_list is None:
            max_idx = min(number_of_pairs, len(initial_dataset))
            nx, ny = max_idx, max_idx
            xy = np.mgrid[:nx, :ny].reshape(2, -1).T
            number_of_pairs = min(xy.shape[0], number_of_pairs)
            self.pairs = xy.take(np.random.choice(xy.shape[0], number_of_pairs, replace=False), axis=0)
        else:
            videos = self.initial_dataset.videos
            name_to_index = {name: index for index, name in enumerate(videos)}
            pairs = pd.read_csv(pairs_list)
            pairs = pairs[np.logical_and(pairs['source'].isin(videos), pairs['driving'].isin(videos))]

            number_of_pairs = min(pairs.shape[0], number_of_pairs)
            self.pairs = []
            self.start_frames = []
            for ind in range(number_of_pairs):
                self.pairs.append(
                    (name_to_index[pairs['driving'].iloc[ind]], name_to_index[pairs['source'].iloc[ind]]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        first = self.initial_dataset[pair[0]]
        second = self.initial_dataset[pair[1]]
        first = {'driving_' + key: value for key, value in first.items()}
        second = {'source_' + key: value for key, value in second.items()}

        return {**first, **second}
