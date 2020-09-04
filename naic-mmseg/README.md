# NAIC-遥感图像分割 baseline

## 使用说明

### 环境搭建

```
可参考 https://mmsegmentation.readthedocs.io/en/latest/install.html#installation
或者，可以先pip一个mmcv，版本要1.0.5 然后直接开始跑，根据报错信息缺少包补啥包
```

### 制作数据集

**一、把比赛训练数据集做成如下格式：**

```markdown
├── data
│   ├── my_dataset
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   │   ├── xxx{img_suffix}
│   │   │   │   ├── yyy{img_suffix}
│   │   │   │   ├── zzz{img_suffix}
│   │   │   ├── val
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   │   ├── xxx{seg_map_suffix}
│   │   │   │   ├── yyy{seg_map_suffix}
│   │   │   │   ├── zzz{seg_map_suffix}
│   │   │   ├── val

操作：
	1、将 train_val_split.py  中的数据集路径改成自己的
	2、python train_val_split.py
```

**二、修改config文件，以训练 Cityscapes 数据集上的 HRNetV2p-W18 模型 为例，将下载好的模型放入checkpoints目录中**

```markdown
相关模型参见： https://github.com/open-mmlab/mmsegmentation/blob/master/docs/model_zoo.md
1、打开 ./configs/_base_/naic.py 文件
	将data_root， img_dir， ann_dir 改成相应的路径
	samples_per_gpu 表示每个gpu训练多少图片，注意是‘每个’，不是全部
	workers_per_gpu 表示使用多少gpu进行训练
	修改相关数据增强方法见：
			https://mmsegmentation.readthedocs.io/en/latest/tutorials/data_pipeline.html
2、打开 ./configs/_base_/models/fcn_hr18.py 文件
	如果使用单个gpu进行训练，在第二行，将 type='SyncBN' 改成 type='BN'，上传文件已修改，若使用多gpu训练，改回 type='SyncBN' 即可
	将 num_classes 设置为 8
3、打开 ./configs/_base_/schedules/schedule_40k.py 文件，即找到和模型对应的schedule文件
	total_iters： 总共迭代的次数
	checkpoint_config = dict(by_epoch=False, interval=2000) 中的 interval 表示迭代多少次保存模型
	evaluation = dict(interval=500, metric='mIoU') 中的 interval 表示迭代多少次测试指标 mIoU 
	举个例子：
		总共的训练图片是100000张，如果设置一个gpu，每个gpu跑4张，那么当total_iters设置为100000时，就相当于4个epoch。
4、打开 ./configs/_base_/default_runtime.py
	第三行的 interval 表示迭代多少次输出信息
5、打开 ./configs/hrnet/fcn_hr18_512x1024_40k_cityscapes.py 文件
	将 '../_base_/datasets/cityscapes.py' 改为 '../_base_/datasets/naic.py'
6、进行测试 python ./preparation/test_dataset.py 若能正常输出，就OK了
```

### 训练

```markdown
## 单gpu ##
$ python train_single_gpu.py 
## 多gpus ##
$ python train_multi_gpu.py
```

### 提交文件

```
$ python submit.py
```

