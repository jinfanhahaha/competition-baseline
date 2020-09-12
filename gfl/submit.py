from mmdet.apis import submit_to_save
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.datasets import build_dataloader, build_dataset
from mmcv import Config
from mmdet.models import build_detector


# 定义config
config = './configs/gfl_x101_32x4d_fpn_dconv.py'
# 模型路径
checkpoint_path = "/raid/ours_data/Competition/seed-advertising-detection/output1/epoch_14.pth"
# 定义输出目录
output_dir = './result'

cfg = Config.fromfile(config)
cfg.model.pretrained = None
# 定义使用哪块gpu
cfg.gpu_ids = [2]
if cfg.model.get('neck'):
    if isinstance(cfg.model.neck, list):
        for neck_cfg in cfg.model.neck:
            if neck_cfg.get('rfp_backbone'):
                if neck_cfg.rfp_backbone.get('pretrained'):
                    neck_cfg.rfp_backbone.pretrained = None
    elif cfg.model.neck.get('rfp_backbone'):
        if cfg.model.neck.rfp_backbone.get('pretrained'):
            cfg.model.neck.rfp_backbone.pretrained = None

cfg.data.test.test_mode = True
distributed = False
samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)
model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
checkpoint = load_checkpoint(model, checkpoint_path)
model = MMDataParallel(model, device_ids=cfg.gpu_ids)
submit_to_save(model, data_loader, output_dir)
