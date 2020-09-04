import mmcv
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmseg.apis import single_gpu_test_and_save
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor


def main():
    # 定义 config 文件路径
    cfg = mmcv.Config.fromfile('./configs/hrnet/fcn_hr18_512x1024_40k_cityscapes.py')
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    distributed = False
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_segmentor(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    # 定义 加载预训练路径
    checkpoint = load_checkpoint(model, './output/iter_20000.pth', map_location='cuda')
    model.CLASSES = checkpoint['meta']['CLASSES']
    model.PALETTE = checkpoint['meta']['PALETTE']

    # if not distributed:
    model = MMDataParallel(model, device_ids=[1])
    # ./result 为输出路径
    single_gpu_test_and_save(model, data_loader, True, './result')


if __name__ == '__main__':
    main()
