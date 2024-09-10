# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .coco import build as build_coco
#---遥感
from .DAcoco import build_dayclear,build_dayfoggy,build_duskrainy,build_nightrainy,build_nightclear



def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    if args.dataset_file == 'o365':
        from .o365 import build_o365_combine
        return build_o365_combine(image_set, args)
    if args.dataset_file == 'vanke':
        from .vanke import build_vanke
        return build_vanke(image_set, args)

    #------training domain
    if args.dataset_file == 'dayclear':
        return build_dayclear(image_set, args)

    #----only test
    if args.dataset_file == 'dayfoggy':
        return build_dayfoggy(image_set, args)
    if args.dataset_file == 'duskrainy':
        return build_duskrainy(image_set, args)
    if args.dataset_file == 'nightrainy':
        return build_nightrainy(image_set, args)
    if args.dataset_file == 'nightclear':
        return build_nightclear(image_set, args)



    raise ValueError(f'dataset {args.dataset_file} not supported')
