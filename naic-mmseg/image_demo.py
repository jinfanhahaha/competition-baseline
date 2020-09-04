from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette


def main():
    model = init_segmentor("./configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_40k_cityscapes.py",
                           "./checkpoints/deeplabv3plus_r50-d8_512x1024_40k_cityscapes_20200605_094610-d222ffcd.pth",
                           device='cpu')
    # test a single image
    result = inference_segmentor(model, 'demo.png')
    # show the results
    show_result_pyplot(model, "demo.png", result, get_palette('cityscapes'))


if __name__ == '__main__':
    main()
