import numpy as np
import shutil
import os


# 定义原始存放数据的路径
DATA_PIC = "/raid/ours_data/Competition/NAIC-Segmentation/train/image/"
DATA_TAG = "/raid/ours_data/Competition/NAIC-Segmentation/train/label/"

# 定义存放划分后数据的路径
TRAIN_PIC = "/raid/ours_data/Competition/NAIC-Segmentation/data/my_dataset/img_dir/train/"
TRAIN_TAG = "/raid/ours_data/Competition/NAIC-Segmentation/data/my_dataset/ann_dir/train/"
VAL_PIC = "/raid/ours_data/Competition/NAIC-Segmentation/data/my_dataset/img_dir/val/"
VAL_TAG = "/raid/ours_data/Competition/NAIC-Segmentation/data/my_dataset/ann_dir/val/"

data_pic_list = np.array(sorted(os.listdir(DATA_PIC)))
data_tag_list = np.array(sorted(os.listdir(DATA_TAG)))

# shuffle
p = np.random.permutation(len(data_pic_list))
data_pic_list = data_pic_list[p]
data_tag_list = data_tag_list[p]

# 划分训练验证集
split_k = len(data_pic_list) // 8
train_pic_list = data_pic_list[split_k:]
train_tag_list = data_tag_list[split_k:]
val_pic_list = data_pic_list[: split_k]
val_tag_list = data_tag_list[: split_k]

# 创建目录
if not os.path.exists(TRAIN_PIC):
    os.makedirs(TRAIN_PIC)
if not os.path.exists(TRAIN_TAG):
    os.makedirs(TRAIN_TAG)
if not os.path.exists(VAL_PIC):
    os.makedirs(VAL_PIC)
if not os.path.exists(VAL_TAG):
    os.makedirs(VAL_TAG)

# do it
for train_pic in train_pic_list:
    shutil.move(DATA_PIC + train_pic, TRAIN_PIC + train_pic)

for train_tag in train_tag_list:
    shutil.move(DATA_TAG + train_tag, TRAIN_TAG + train_tag)

for val_pic in val_pic_list:
    shutil.move(DATA_PIC + val_pic, VAL_PIC + val_pic)

for val_tag in val_tag_list:
    shutil.move(DATA_TAG + val_tag, VAL_TAG + val_tag)

os.rmdir(DATA_PIC)
os.rmdir(DATA_TAG)
print("finished!")
