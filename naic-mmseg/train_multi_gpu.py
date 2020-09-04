import argparse
import copy
import os
import os.path as osp
import time

import mmcv
import torch
from mmcv.runner import init_dist
from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__
from mmseg.apis import set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor


def main():

    cfg = mmcv.Config.fromfile('./configs/hrnet/fcn_hr18s_512x1024_40k_cityscapes.py')
    cfg.work_dir = "./output"
    cfg.load_from = "./checkpoints/fcn_hr18s_512x1024_40k_cityscapes_20200601_014216-93db27d0.pth"
    cfg.gpu_ids = [0, 1, 2]
    distributed = True
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    meta = dict()
    set_random_seed(2020)
    cfg.seed = 2020
    meta['seed'] = 2020
    meta['exp_name'] = osp.basename("./configs/hrnet/fcn_hr18s_512x1024_40k_cityscapes.py")

    model = build_segmentor(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmseg version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE)
    # add an attribute for visualization convenience
    model.CLASSES = [datasets[0].CLASSES]
    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not False),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
