import copy
import os.path as osp
import time

import mmcv
from mmcv.utils import get_git_hash

from mmseg import __version__
from mmseg.apis import set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor


def main():

    # 定义config路径
    config = './configs/hrnet/fcn_hr18_512x1024_40k_cityscapes.py'
    cfg = mmcv.Config.fromfile(config)
    # 定义输出目录
    cfg.work_dir = "./output"
    cfg.model.pretrained = None
    # 定义加载预训练权重路径
    cfg.load_from = './checkpoints/fcn_hr18_512x1024_40k_cityscapes_20200601_014216-f196fb4e.pth'
    # 使用哪块gpu
    cfg.gpu_ids = [0]
    # 非分布式训练
    distributed = False
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    meta = dict()
    # 定义随机种子
    set_random_seed(2020)
    cfg.seed = 2020
    meta['seed'] = 2020
    meta['exp_name'] = osp.basename(config)

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
