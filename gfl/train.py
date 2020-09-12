import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import init_dist
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger


def main():

    # config路径
    config = './configs/gfl_x101_32x4d_fpn_dconv.py'
    # 权重路径
    checkpoint_path = "/raid/ours_data/Competition/seed-advertising-detection/checkpoints/gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco_20200630_102002-14a2bf25.pth"

    cfg = Config.fromfile(config)
    # 输出目录
    cfg.work_dir = '/raid/ours_data/Competition/seed-advertising-detection/output2'

    cfg.load_from = checkpoint_path
    # 设置gpu
    cfg.gpu_ids = [3]
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(config)))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    meta = dict()
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    set_random_seed(2020, deterministic=False)
    cfg.seed = 2020
    meta['seed'] = 2020

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    datasets = [build_dataset(cfg.data.train)]
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=(not False),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
