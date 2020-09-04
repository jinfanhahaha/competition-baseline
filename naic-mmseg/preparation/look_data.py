from mmseg.datasets import build_dataset
import mmcv

config = "../configs/hrnet/fcn_hr18_512x1024_40k_cityscapes.py"
cfg = mmcv.Config.fromfile(config)
train_dataset = build_dataset(cfg.data.train)
val_dataset = build_dataset(cfg.data.val)
test_dataset = build_dataset(cfg.data.test)
# print(train_dataset[0])
# print("\n"+"#"*100+"\n")
# print(val_dataset[0])
# print("\n"+"#"*100+"\n")
# print(test_dataset[0])
a = val_dataset[0]['gt_semantic_seg']
print(a)