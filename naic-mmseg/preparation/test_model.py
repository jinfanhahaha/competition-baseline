import mmcv
from mmseg.models import build_segmentor
from mmcv.runner import load_checkpoint


config = "../configs/hrnet/fcn_hr18s_512x1024_40k_cityscapes.py"
checkpoint_path = "../checkpoints/fcn_hr18s_512x1024_40k_cityscapes_20200601_014216-93db27d0.pth"
cfg = mmcv.Config.fromfile(config)
cfg.model.pretrained = None
model = build_segmentor(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
print(model)

'''
   
'''
