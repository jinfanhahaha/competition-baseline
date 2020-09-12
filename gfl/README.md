# gfl模型

## 使用说明

### 一、准备数据

```
1、划分训练验证集，打开train_val_split.py，将DATA_PIC修改成存放违法广告图片的目录
	ANN修改成，存放违法广告标注信息的目录
	python train_val_split.py
2、制作pkl数据，供mmdet训练
	打开get_pkl.py,将第六行ILL_ANN改成违法广告标注训练集的目录，第59行为train_ann.pkl
	接着制作验证集，将第六行ILL_ANN改成违法广告标注验证集的目录，第59行为val_ann.pkl
	接着制作测试集，打开get_test_pkl.py,将第六行ILL_ANN改成违法广告标注测试集的目录
3、修改config文件里的路径
	打开./configs/gfl_x101_32x4d_fpn_dconv.py, 修改82行的路径，改成对应的目录，格式可参考：
		data
			train/
			val/
			test/
			train_ann.pkl
			val_ann.pkl
			test_ann.pkl
```

### 二、运行

```
1、训练
	python train.py  ## 需要修改一些路径
2、生成结果
  python submit.py  ## 需要修改一些路径
3、修改结果中的bbox个数
  python select_result.py  ## 需要修改一些路径
```

